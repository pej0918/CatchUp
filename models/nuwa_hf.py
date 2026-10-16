# models/nuwa_hf.py
import math
import torch
import torch.nn.functional as F


TOKEN_CONFIGS = {
    64: {"top_n_per_region": 1},
    128: {"top_n_per_region": 2},
    192: {"top_n_per_region": 3},
    160: {"top_n_per_region": 1},
    320: {"top_n_per_region": 3},
    640: {"top_n_per_region": 4},
}

STAGE1_CONFIGS = {
    64: 112,
    128: 224,
    192: 336,
    160: 290,
    320: 582,
    640: 1162,
}

STAGE2_PRESERVE = {
    112: 16,
    224: 32,
    336: 48,
}


def create_distance_penalty_matrix(
    grid_size=(24, 24),
    distance_threshold=math.sqrt(280),
    device="cpu",
):
    H, W = grid_size
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).float()
    coords_flat = coords.view(-1, 2)
    dist_matrix = torch.cdist(coords_flat, coords_flat, p=2.0)
    normalized_dist = dist_matrix / distance_threshold
    clipped_dist = torch.clamp(normalized_dist, max=1.0)
    return 1.0 - clipped_dist


@torch.no_grad()
def nuwa_stage1_aggregate(
    vision_outputs,
    select_layer: int,
    matain_token: int = 192,
    distance: int = 280,
    penalty_threshold_percentile: float = 0.55,
):
    """
    HF LLaVA vision outputs -> Nuwa Stage-1 aggregated visual tokens.

    Returns:
        aggregated_tokens: [B, K1, D_v]
        keep_idxs: [B, K1], original 24x24 patch indices
    """
    all_hidden_states = vision_outputs.hidden_states
    attentions = vision_outputs.attentions[select_layer]

    hidden_states_for_aggregation = all_hidden_states[select_layer]  # [B, 577, D]
    B, N, D = hidden_states_for_aggregation.shape
    H, W = 24, 24
    num_patches = N - 1

    if num_patches != H * W:
        raise ValueError(f"Nuwa expects 24x24=576 patches, got {num_patches}")

    config = TOKEN_CONFIGS[matain_token]
    total_tokens_to_keep = STAGE1_CONFIGS[matain_token]
    top_n_per_region = config["top_n_per_region"]
    num_tokens_per_image = min(
        math.ceil(total_tokens_to_keep / B),
        144 * top_n_per_region,
    )

    # CLS attention map: [B, 576]
    cls_attention = attentions[:, :, 0, 1:]
    metric_map = cls_attention.sum(dim=1)

    # 2x2 regional top-k
    metric_2d = metric_map.view(B, H, W)
    unfolded_attn = metric_2d.unfold(1, 2, 2).unfold(2, 2, 2)
    num_regions = (H // 2) * (W // 2)
    regions = unfolded_attn.reshape(B, num_regions, 4)

    _, local_topk_indices = regions.topk(k=top_n_per_region, dim=-1)
    local_y = local_topk_indices // 2
    local_x = local_topk_indices % 2

    region_ids = torch.arange(num_regions, device=metric_map.device).unsqueeze(0)
    regions_per_row = W // 2
    region_y = region_ids // regions_per_row
    region_x = region_ids % regions_per_row

    global_y = region_y.unsqueeze(-1) * 2 + local_y
    global_x = region_x.unsqueeze(-1) * 2 + local_x

    candidate_indices = (global_y * W + global_x).view(B, -1)
    candidate_scores = torch.gather(metric_map, 1, candidate_indices)

    _, top_candidate_indices = torch.topk(
        candidate_scores,
        num_tokens_per_image,
        dim=1,
    )
    keep_idxs = torch.gather(candidate_indices, 1, top_candidate_indices)
    keep_idxs = keep_idxs.sort(dim=1).values

    # Similarity graph from averaged layers 16~24
    if len(all_hidden_states) < 25:
        raise ValueError("Nuwa requires vision hidden_states[16:25].")

    stacked_hs = torch.stack(all_hidden_states[16:25], dim=0)
    avg_hidden_states_for_sim = stacked_hs.mean(dim=0)
    patch_tokens_for_sim = avg_hidden_states_for_sim[:, 1:, :]
    patch_tokens_norm = F.normalize(patch_tokens_for_sim, p=2, dim=-1)
    sim_matrix = torch.bmm(
        patch_tokens_norm,
        patch_tokens_norm.transpose(1, 2),
    )

    dist_penalty = create_distance_penalty_matrix(
        grid_size=(H, W),
        distance_threshold=math.sqrt(distance),
        device=metric_map.device,
    )

    aggregation_weights = F.relu(sim_matrix) * dist_penalty

    # Selective penalty
    benchmark_scores = torch.gather(metric_map, 1, keep_idxs)
    score_threshold = torch.quantile(
        benchmark_scores.float(),
        penalty_threshold_percentile,
        dim=1,
        keepdim=True,
    )
    is_high_attention_token = benchmark_scores >= score_threshold

    selective_penalty_matrix = torch.ones(
        B,
        num_tokens_per_image,
        num_patches,
        device=metric_map.device,
        dtype=aggregation_weights.dtype,
    )
    selective_penalty_matrix[is_high_attention_token] = 0.0
    selective_penalty_matrix.scatter_(
        dim=2,
        index=keep_idxs.unsqueeze(-1),
        value=1.0,
    )

    patch_tokens_for_aggregation = hidden_states_for_aggregation[:, 1:, :]
    keep_expanded = keep_idxs.unsqueeze(-1).expand(-1, -1, num_patches)

    benchmark_weights = torch.gather(aggregation_weights, 1, keep_expanded)
    benchmark_weights = benchmark_weights * selective_penalty_matrix

    benchmark_weights_norm = benchmark_weights / (
        benchmark_weights.sum(dim=-1, keepdim=True) + 1e-8
    )
    benchmark_weights_norm.scatter_(
        dim=2,
        index=keep_idxs.unsqueeze(-1),
        value=1.0,
    )

    aggregated_tokens = torch.bmm(
        benchmark_weights_norm.to(patch_tokens_for_aggregation.dtype),
        patch_tokens_for_aggregation,
    )

    return aggregated_tokens, keep_idxs


# ---------------------------------------------------------------------
# Nuwa Stage-2 router for mllm-reroute-qver
# ---------------------------------------------------------------------
from .router import TokenRouter, RoutingDecision


class NuwaRouter(TokenRouter):
    """
    Nuwa-style Stage-2 decoder pruning router.

    Stage-1 aggregation:
        576 visual tokens -> stage1_tokens, e.g. 336 for matain_token=192

    Stage-2 decoder pruning:
        stage1_tokens -> stage2_tokens at decoder layer stage2_layer,
        e.g. 336 -> 48 at layer 16.

    Note:
        This router handles Stage-2 pruning. Stage-1 aggregation must be
        called separately from the LLaVA image-feature path.
    """

    _STAGE1_CONFIGS = {
        64: 112,
        128: 224,
        192: 336,
        160: 290,
        320: 582,
        640: 1162,
    }

    _STAGE2_PRESERVE = {
        112: 16,
        224: 32,
        336: 48,
        290: 32,
        582: 64,
        1162: 128,
    }

    def __init__(
        self,
        matain_token: int = 192,
        distance: int = 280,
        penalty_threshold_percentile: float = 0.55,
        stage1_tokens: int | None = None,
        stage2_layer: int = 16,
        stage2_tokens: int | None = None,
    ):
        self.matain_token = int(matain_token)
        self.distance = int(distance)
        self.penalty_threshold_percentile = float(penalty_threshold_percentile)

        if stage1_tokens is None:
            stage1_tokens = self._STAGE1_CONFIGS.get(self.matain_token)

        if stage1_tokens is None:
            raise ValueError(f"Unsupported Nuwa matain_token={self.matain_token}")

        if stage2_tokens is None:
            stage2_tokens = self._STAGE2_PRESERVE.get(int(stage1_tokens))

        if stage2_tokens is None:
            raise ValueError(f"Unsupported Nuwa stage1_tokens={stage1_tokens}")

        self.stage1_tokens = int(stage1_tokens)
        self.stage2_layer = int(stage2_layer)
        self.stage2_tokens = int(stage2_tokens)

        # compatibility with patching.py
        self.drop_layers = [self.stage2_layer]
        self.keep_ratios = None
        self.monotonic = True

    def should_route(self, layer_idx: int) -> bool:
        return int(layer_idx) == self.stage2_layer

    def is_decision_layer(self, layer_idx: int) -> bool:
        return int(layer_idx) == self.stage2_layer

    def compute_scores(
        self,
        layer_idx: int,
        attn_weights,
        hidden_states,
        visual_token_range,
        **kwargs,
    ) -> RoutingDecision:
        """
        Nuwa Stage-2 pruning score.

        Uses last-token attention over current visual tokens and keeps
        top stage2_tokens visual tokens.
        """
        if not self.should_route(layer_idx):
            raise RuntimeError(
                f"NuwaRouter.compute_scores called at non-routing layer {layer_idx}"
            )

        if attn_weights is None:
            raise ValueError("NuwaRouter requires attention weights for Stage-2 pruning.")

        vis_start, vis_end = visual_token_range

        # attn_weights: [B, H, T, T] or [B, T, T]
        if attn_weights.dim() == 4:
            attn_avg = attn_weights.mean(dim=1)
        else:
            attn_avg = attn_weights

        # last generated/query token attention to visual tokens
        scores = attn_avg[:, -1, vis_start:vis_end].float()  # [B, V]

        B, V = scores.shape
        k = min(self.stage2_tokens, V)

        selected_indices = torch.topk(scores, k=k, dim=-1).indices

        selected_mask = torch.zeros(
            B,
            V,
            device=scores.device,
            dtype=torch.bool,
        )
        selected_mask.scatter_(1, selected_indices, True)

        return RoutingDecision(
            selected_mask=selected_mask,
            scores=scores,
            selected_indices=selected_indices,
        )

    def __repr__(self) -> str:
        return (
            f"NuwaRouter("
            f"matain_token={self.matain_token}, "
            f"stage1_tokens={self.stage1_tokens}, "
            f"stage2_layer={self.stage2_layer}, "
            f"stage2_tokens={self.stage2_tokens}, "
            f"distance={self.distance}, "
            f"penalty_threshold_percentile={self.penalty_threshold_percentile}"
            f")"
        )
    def reset(self):
        """
        Reset per-sample router state.

        NuwaRouter is currently stateless, but the evaluation wrapper calls
        router.reset() before each generation, matching other routers.
        """
        return None