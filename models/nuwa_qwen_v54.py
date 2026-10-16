# models/nuwa_qwen_v54.py
"""
Qwen2.5-VL Nüwa port for the project's Transformers 5.4.0 environment.

Design:
  * Qwen2.5-VL stays on the SAME HF model/environment used by ReRoute.
  * Stage 1 follows the official Nüwa Qwen implementation:
      - last vision-block attention metric
      - 2x2 regional candidate selection
      - ratio-based selection on the native post-merge grid
      - hidden-state similarity from vision blocks 7..23
      - dynamic spatial distance threshold (0.8 * max grid distance)
      - selective penalty percentile 0.1
      - information aggregation into selected image tokens
  * Stage 2 uses the official Qwen Nüwa score:
      mean over the last 3 query tokens, then max over heads.
  * Project-specific choice:
      Stage-2 routing/scoring boundary = decoder layer 16.
      The project's generic physical_delete path performs the deletion
      immediately before layer 16 after computing layer-16 QK scores.

This is therefore an environment-compatible port of Nüwa, NOT the upstream
custom modeling file byte-for-byte.  The Stage-2 layer/timing is deliberately
the project's L16 choice.
"""

from __future__ import annotations

import math
import types
from typing import Optional

import torch
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_vision,
)

from .router import TokenRouter, RoutingDecision
from .dispatcher import TokenDispatcher
from .patching import patch_model_for_routing


def _distance_penalty_matrix(
    grid_size: tuple[int, int],
    distance_threshold: float,
    device,
) -> torch.Tensor:
    """Official Qwen-Nüwa spatial penalty."""
    h, w = grid_size
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).float()
    coords_flat = coords.view(-1, 2)
    dist_matrix = torch.cdist(coords_flat, coords_flat, p=2.0)
    normalized_dist = dist_matrix / float(distance_threshold)
    clipped_dist = torch.clamp(normalized_dist, max=1.0)
    return 1.0 - clipped_dist


@torch.no_grad()
def _final_vision_attention_metric(
    blk,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings,
    spatial_merge_unit: int,
    reverse_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Reproduce the official final vision-block attention metric without
    depending on the old custom Transformers attention API.

    Returns one score per POST-MERGE visual token, in original spatial order.
    """
    attn = blk.attn
    normed = blk.norm1(hidden_states)

    seq_len = normed.shape[0]
    q, k, _v = (
        attn.qkv(normed)
        .reshape(seq_len, 3, attn.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

    # Official custom source uses [H, T, D] and a block-diagonal attention mask.
    q = q.permute(1, 0, 2)  # [H, T, D]
    k = k.permute(1, 0, 2)  # [H, T, D]

    logits = torch.matmul(q.float(), k.float().transpose(-1, -2))
    logits = logits * float(attn.scaling)

    min_value = torch.finfo(logits.dtype).min
    attention_mask = torch.full(
        (1, seq_len, seq_len),
        min_value,
        device=logits.device,
        dtype=logits.dtype,
    )
    for i in range(1, len(cu_seqlens)):
        s = int(cu_seqlens[i - 1].item())
        e = int(cu_seqlens[i].item())
        attention_mask[..., s:e, s:e] = 0

    attn_weights = torch.softmax(logits + attention_mask, dim=-1)

    # Official:
    #   mean over heads -> sum over all query tokens
    #   -> average each spatial_merge_unit group (normally 4)
    metric = attn_weights.mean(dim=0).sum(dim=0)
    if metric.numel() % spatial_merge_unit != 0:
        raise ValueError(
            f"Nüwa vision metric length {metric.numel()} is not divisible "
            f"by spatial_merge_unit={spatial_merge_unit}"
        )
    metric = metric.view(metric.numel() // spatial_merge_unit, spatial_merge_unit).mean(dim=-1)
    metric = metric[reverse_indices]
    return metric


@torch.no_grad()
def _run_visual_for_nuwa(
    visual,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
):
    """
    Run the HF v5.4 Qwen2.5-VL vision tower while collecting exactly the
    quantities used by official Qwen-Nüwa Stage 1.

    Returns:
      image_embeds : post-merge final features, original spatial order [V, D]
      metric       : official final-block attention metric [V]
      middle       : merger(mean(block outputs 7..23)), original order [V, D]
    """
    hidden_states = visual.patch_embed(pixel_values)
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(grid_thw)

    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)

    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    reverse_indices = torch.argsort(window_index)
    sim_states = []
    metric = None
    last_idx = len(visual.blocks) - 1

    for layer_num, blk in enumerate(visual.blocks):
        cu_now = (
            cu_seqlens
            if layer_num in visual.fullatt_block_indexes
            else cu_window_seqlens
        )

        if layer_num == last_idx:
            metric = _final_vision_attention_metric(
                blk=blk,
                hidden_states=hidden_states,
                cu_seqlens=cu_now,
                position_embeddings=position_embeddings,
                spatial_merge_unit=visual.spatial_merge_unit,
                reverse_indices=reverse_indices,
            )

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_now,
            position_embeddings=position_embeddings,
        )

        # Official Qwen Nüwa averages vision block outputs 7..23 inclusive.
        if 7 <= layer_num <= 23:
            sim_states.append(hidden_states.unsqueeze(0))

    if metric is None:
        raise RuntimeError("Failed to compute Qwen-Nüwa final vision attention metric.")
    if not sim_states:
        raise RuntimeError("No vision states collected for Nüwa blocks 7..23.")

    middle = visual.merger(torch.vstack(sim_states).mean(dim=0))
    image_embeds = visual.merger(hidden_states)

    image_embeds = image_embeds[reverse_indices, :]
    middle = middle[reverse_indices, :]

    return image_embeds, metric, middle


@torch.no_grad()
def qwen_nuwa_stage1(
    image_embeds: torch.Tensor,
    metric: torch.Tensor,
    middle_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    keep_ratio: float,
    top_n_per_region: int,
    penalty_threshold_percentile: float = 0.1,
):
    """
    Official Qwen-Nüwa Stage-1 selection + aggregation for a single image.

    Returns:
      updated_image_embeds: SAME length V, but selected slots contain
                            aggregated features.
      select_mask: [V] bool; only these image tokens enter the LLM.
    """
    if image_grid_thw is None or image_grid_thw.shape[0] != 1:
        raise ValueError(
            "This Nüwa port follows the official Qwen code's single-image "
            "evaluation path. Use eval.batch_size=1 and one image per sample."
        )

    _t, h_pre, w_pre = image_grid_thw[0]
    h_post = int(h_pre.item()) // 2
    w_post = int(w_pre.item()) // 2

    v = int(image_embeds.shape[0])
    if v != h_post * w_post:
        raise ValueError(
            f"Qwen-Nüwa token/grid mismatch: V={v}, "
            f"post-merge grid={h_post}x{w_post}={h_post*w_post}"
        )
    if metric.numel() != v or middle_embeds.shape[0] != v:
        raise ValueError("Nüwa Stage-1 feature/metric lengths do not match.")

    pad_h = (2 - h_post % 2) % 2
    pad_w = (2 - w_post % 2) % 2
    padded_h, padded_w = h_post + pad_h, w_post + pad_w

    metric_2d = metric.view(h_post, w_post)
    sim_2d = middle_embeds.clone().view(h_post, w_post, -1)
    agg_2d = image_embeds.clone().view(h_post, w_post, -1)

    if pad_h > 0 or pad_w > 0:
        metric_2d = F.pad(metric_2d, (0, pad_w, 0, pad_h), value=0)
        padding_3d = (0, 0, 0, pad_w, 0, pad_h)
        sim_2d = F.pad(sim_2d, padding_3d, value=0)
        agg_2d = F.pad(agg_2d, padding_3d, value=0)

    b = 1
    padded_metric = metric_2d.flatten().unsqueeze(0)
    padded_sim = sim_2d.view(b, padded_h * padded_w, -1)
    padded_agg = agg_2d.view(b, padded_h * padded_w, -1)

    # 2x2 regional candidate selection.
    regions_h, regions_w = padded_h // 2, padded_w // 2
    num_regions = regions_h * regions_w
    regions = metric_2d.unfold(0, 2, 2).unfold(1, 2, 2).reshape(num_regions, 4)

    top_n = int(top_n_per_region)
    if not 1 <= top_n <= 4:
        raise ValueError(f"top_n_per_region must be 1..4, got {top_n}")

    _, local_topk = regions.topk(k=top_n, dim=-1)
    local_y, local_x = local_topk // 2, local_topk % 2

    region_ids = torch.arange(num_regions, device=image_embeds.device)
    region_y = (region_ids // regions_w) * 2
    region_x = (region_ids % regions_w) * 2
    global_y = region_y.unsqueeze(-1) + local_y
    global_x = region_x.unsqueeze(-1) + local_x

    candidate_indices = (global_y * padded_w + global_x).view(1, -1)

    num_tokens_to_keep = math.ceil(v * float(keep_ratio))
    candidate_scores = torch.gather(padded_metric, 1, candidate_indices)
    k1 = min(num_tokens_to_keep, candidate_indices.shape[1])

    _, top_candidate_indices = torch.topk(candidate_scores, k1, dim=1)
    benchmark_indices = torch.gather(
        candidate_indices, 1, top_candidate_indices
    ).sort().values

    # Remove padded-grid candidates.
    coords_y = benchmark_indices // padded_w
    coords_x = benchmark_indices % padded_w
    valid_mask = (coords_y < h_post) & (coords_x < w_post)
    benchmark_indices = benchmark_indices[valid_mask].unsqueeze(0)

    # Official information aggregation.
    max_dist = math.sqrt((padded_h - 1) ** 2 + (padded_w - 1) ** 2)
    dynamic_distance_threshold = max_dist * 0.8 if max_dist > 0 else 1.0
    dist_penalty = _distance_penalty_matrix(
        (padded_h, padded_w),
        dynamic_distance_threshold,
        image_embeds.device,
    )

    sim_norm = F.normalize(padded_sim, p=2, dim=-1)
    sim_matrix = torch.bmm(sim_norm, sim_norm.transpose(1, 2))
    aggregation_weights = F.relu(sim_matrix) * dist_penalty

    benchmark_scores = torch.gather(padded_metric, 1, benchmark_indices)
    score_threshold = torch.quantile(
        benchmark_scores.float(),
        float(penalty_threshold_percentile),
        dim=1,
        keepdim=True,
    )
    is_high_attention = (benchmark_scores >= score_threshold).squeeze(0)

    benchmark_weights_expanded = torch.gather(
        aggregation_weights,
        1,
        benchmark_indices.unsqueeze(-1).expand(
            -1, -1, padded_h * padded_w
        ),
    )

    selective_penalty = torch.ones_like(benchmark_weights_expanded)
    high_idx = torch.where(is_high_attention)[0]
    if high_idx.numel() > 0:
        selective_penalty[:, high_idx, :] = 0.0
    selective_penalty.scatter_(
        2,
        benchmark_indices.unsqueeze(-1),
        1.0,
    )

    benchmark_weights = benchmark_weights_expanded * selective_penalty
    benchmark_weights_norm = (
        benchmark_weights
        / (benchmark_weights.sum(dim=-1, keepdim=True) + 1e-8)
    ).to(padded_agg.dtype)

    aggregated_tokens = torch.bmm(benchmark_weights_norm, padded_agg)

    padded_mask = torch.zeros(
        padded_h * padded_w,
        dtype=torch.bool,
        device=image_embeds.device,
    )
    padded_mask[benchmark_indices.squeeze(0)] = True
    select_mask = padded_mask.view(padded_h, padded_w)[:h_post, :w_post].flatten()

    if int(select_mask.sum().item()) != int(aggregated_tokens.shape[1]):
        raise RuntimeError(
            "Nüwa Stage-1 selected-token count does not match aggregated tokens."
        )

    updated = image_embeds.clone()
    updated[select_mask] = aggregated_tokens.squeeze(0)

    return updated, select_mask


class QwenNuwaRouter(TokenRouter):
    """
    Nüwa Stage-2 router for the project's Qwen2.5-VL path.

    Official Qwen-Nüwa score:
      mean(attention over last 3 query tokens) -> max over heads.

    Project choice:
      score/route at layer 16 using the generic QK capture path.
    """

    def __init__(
        self,
        stage1_keep_ratio: float,
        stage2_keep_ratio: float,
        top_n_per_region: int = 3,
        stage2_layer: int = 16,
        penalty_threshold_percentile: float = 0.1,
    ):
        self.stage1_keep_ratio = float(stage1_keep_ratio)
        self.stage2_keep_ratio = float(stage2_keep_ratio)
        self.top_n_per_region = int(top_n_per_region)
        self.stage2_layer = int(stage2_layer)
        self.penalty_threshold_percentile = float(
            penalty_threshold_percentile
        )

        self.drop_layers = [self.stage2_layer]
        # patching.py will pick this up after the tiny compatibility patch.
        self.scoring_layers = [self.stage2_layer]
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
        if not self.should_route(layer_idx):
            raise RuntimeError(
                f"QwenNuwaRouter called at non-routing layer {layer_idx}"
            )
        if attn_weights is None:
            raise ValueError("Qwen-Nüwa Stage 2 requires attention weights.")
        if attn_weights.dim() != 4:
            raise ValueError(
                f"Expected attention [B,H,Q,K], got {tuple(attn_weights.shape)}"
            )

        vis_start, vis_end = visual_token_range
        if vis_end <= vis_start:
            raise ValueError(f"Invalid visual range: {visual_token_range}")

        # Official Qwen-Nüwa:
        #   attn[:, :, -3:, :].mean(dim=2).max(dim=1).values
        last_n = min(3, attn_weights.shape[2])
        all_scores = (
            attn_weights[:, :, -last_n:, :]
            .mean(dim=2)
            .max(dim=1)
            .values
        )
        scores = all_scores[:, vis_start:vis_end].float()

        b, v = scores.shape
        k = max(1, int(v * self.stage2_keep_ratio))
        k = min(k, v)

        if not getattr(self, "_printed_stage2", False):
            print(
                f"[Qwen-Nüwa S2] "
                f"layer={layer_idx} "
                f"K1={v} -> K2={k} "
                f"(r2={self.stage2_keep_ratio:.6f})",
                flush=True,
            )
            self._printed_stage2 = True

        selected_indices = torch.topk(
            scores,
            k=k,
            dim=-1,
            sorted=False,
        ).indices

        selected_mask = torch.zeros(
            b,
            v,
            device=scores.device,
            dtype=torch.bool,
        )
        selected_mask.scatter_(1, selected_indices, True)

        return RoutingDecision(
            selected_mask=selected_mask,
            scores=scores,
            selected_indices=selected_indices,
        )

    def reset(self):
        self._printed_stage2 = False


def _patch_stage1_image_features(model, router: QwenNuwaRouter):
    """
    Patch Qwen2.5-VL's base-model get_image_features.

    Important: return ORIGINAL-LENGTH image features so stock HF can compute
    full MRoPE positions first.  Actual token removal is applied by a
    pre-hook on the text model immediately before decoder layer 0.
    """
    raw_model = getattr(model, "_model", model)
    base_model = raw_model.model

    if getattr(base_model, "_qwen_nuwa_stage1_patched", False):
        return

    original_get_image_features = base_model.get_image_features

    def nuwa_get_image_features(
        self,
        pixel_values,
        image_grid_thw=None,
        **kwargs,
    ):
        if image_grid_thw is None:
            return original_get_image_features(
                pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )

        # Official code path is image-only / single-image / batch=1.
        pixel_values_local = pixel_values.type(self.visual.dtype)

        image_embeds, metric, middle = _run_visual_for_nuwa(
            self.visual,
            pixel_values_local,
            image_grid_thw,
        )

        updated, select_mask = qwen_nuwa_stage1(
            image_embeds=image_embeds,
            metric=metric,
            middle_embeds=middle,
            image_grid_thw=image_grid_thw,
            keep_ratio=router.stage1_keep_ratio,
            top_n_per_region=router.top_n_per_region,
            penalty_threshold_percentile=router.penalty_threshold_percentile,
        )

        # The base model needs V image features now; Stage-1 sequence deletion
        # is delayed until AFTER full position_ids are computed.
        self._qwen_nuwa_pending_select_mask = select_mask.detach()

        print(
            "[Qwen-Nüwa S1] "
            f"V={select_mask.numel()} -> K1={int(select_mask.sum())} "
            f"(r1={router.stage1_keep_ratio:.6f}, "
            f"top_n={router.top_n_per_region})",
            flush=True,
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=updated,
            pooler_output=(updated,),
        )

    base_model.get_image_features = types.MethodType(
        nuwa_get_image_features,
        base_model,
    )
    base_model._qwen_nuwa_stage1_patched = True
    base_model._qwen_nuwa_original_get_image_features = original_get_image_features


def _register_stage1_text_pre_hook(model, router: QwenNuwaRouter, ctx):
    """
    Apply the Stage-1 select mask immediately before the Qwen text decoder.

    By this point the stock Qwen base model has already computed full
    multimodal position_ids, matching the ordering used by official Nüwa.
    """
    raw_model = getattr(model, "_model", model)
    base_model = raw_model.model
    language_model = base_model.language_model

    if hasattr(base_model, "_qwen_nuwa_stage1_hook_handle"):
        return

    def pre_hook(module, args, kwargs):
        select_mask = getattr(
            base_model,
            "_qwen_nuwa_pending_select_mask",
            None,
        )
        if select_mask is None:
            return

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None or inputs_embeds.shape[1] <= 1:
            return

        old_start, old_end = ctx.visual_token_range
        v = old_end - old_start

        if v != select_mask.numel():
            raise RuntimeError(
                "Qwen-Nüwa Stage-1 visual-range mismatch: "
                f"ctx V={v}, select_mask V={select_mask.numel()}, "
                f"range={ctx.visual_token_range}"
            )

        seq_len = inputs_embeds.shape[1]
        full_keep = torch.ones(
            seq_len,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        full_keep[old_start:old_end] = select_mask.to(inputs_embeds.device)

        kwargs["inputs_embeds"] = inputs_embeds[:, full_keep, :].contiguous()

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                kwargs["attention_mask"] = attention_mask[:, full_keep].contiguous()
            else:
                raise ValueError(
                    "Qwen-Nüwa Stage-1 expected a 2D prefill attention_mask."
                )

        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            if position_ids.ndim == 3:
                kwargs["position_ids"] = position_ids[:, :, full_keep].contiguous()
            elif position_ids.ndim == 2:
                kwargs["position_ids"] = position_ids[:, full_keep].contiguous()
            else:
                raise ValueError(
                    f"Unsupported Qwen position_ids rank: {position_ids.ndim}"
                )

        k1 = int(select_mask.sum().item())
        ctx.visual_token_range = (old_start, old_start + k1)

        # One-shot per visual prefill.
        base_model._qwen_nuwa_pending_select_mask = None

        print(
            "[Qwen-Nüwa S1 sequence] "
            f"seq={seq_len}->{int(full_keep.sum())}, "
            f"visual={v}->{k1}, "
            f"new_range={ctx.visual_token_range}",
            flush=True,
        )

        return args, kwargs

    handle = language_model.register_forward_pre_hook(
        pre_hook,
        with_kwargs=True,
    )
    base_model._qwen_nuwa_stage1_hook_handle = handle


def patch_qwen25vl_nuwa(
    model,
    *,
    stage1_keep_ratio: float,
    stage2_keep_ratio: float,
    top_n_per_region: int = 3,
    stage2_layer: int = 16,
    penalty_threshold_percentile: float = 0.1,
):
    """
    Main API. Call this instead of constructing the LLaVA NuwaRouter.

    Returns the project's RoutingContext.
    """
    router = QwenNuwaRouter(
        stage1_keep_ratio=stage1_keep_ratio,
        stage2_keep_ratio=stage2_keep_ratio,
        top_n_per_region=top_n_per_region,
        stage2_layer=stage2_layer,
        penalty_threshold_percentile=penalty_threshold_percentile,
    )

    # Generic project routing handles Qwen MRoPE + real physical deletion.
    ctx = patch_model_for_routing(
        model,
        router,
        TokenDispatcher(),
        action="physical_delete",
        model_family="qwen25vl",
    )

    _patch_stage1_image_features(model, router)
    _register_stage1_text_pre_hook(model, router, ctx)

    print(
        "[Qwen-Nüwa] patched: "
        f"official-style S1 r1={stage1_keep_ratio}, "
        f"official S2 score, L{stage2_layer} routing, "
        f"r2={stage2_keep_ratio}",
        flush=True,
    )

    return ctx
