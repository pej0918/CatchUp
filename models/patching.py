"""Model patching for visual token routing.

Patches HuggingFace VLM decoder layers to inject token routing before
attention. Supports LLaVA-1.5 (LLaMA backbone), Qwen2.5-VL, and Qwen3-VL.

Action semantics:
- delete: mask-based surrogate for token removal. Column masking blocks
  attention to unselected tokens as K,V targets, but unselected tokens
  still keep active Q rows and therefore can attend to selected/text
  tokens. Their final layer output keeps the normal masked-attention +
  FFN result. Decode-time KV masking prevents new tokens from attending
  to "deleted" entries in the cache.
- physical_delete: real physical token removal. Unselected visual tokens
  are removed from the sequence at routing layers. Subsequent layers see
  a shorter sequence. KV cache only contains remaining tokens. Position
  IDs are preserved (GAP-style: original RoPE positions with gaps).
- residual_skip: mask-based routing. Column masking blocks attention to
  unselected tokens as K,V targets, but unselected tokens still compute
  masked attention internally before their layer output is overwritten
  with the input (residual bypass — full layer skip). Tokens can
  re-enter at later routing layers. Decode-time KV masking blocks stale
  cache entries.
- attn_skip: mask-based attention-only skip. Column masking again only
  blocks K,V access to unselected tokens. The full masked-attention pass
  still runs, but for unselected tokens its output is discarded and
  replaced with FFN-only updates from the original input:
  x + MLP(post_attention_layernorm(x)). Decode-time KV masking blocks
  stale attention cache entries.
- full_kv: mask-based, diagnostic only. Unselected tokens don't attend
  but ARE attended to as K,V targets. Requires bfloat16.
- compact_route: gather selected+text → compact K×K attention → FFN →
  scatter back to full sequence. Unselected tokens take residual (input
  passthrough). Combines physical_delete efficiency with recoverability.
"""

from __future__ import annotations

import functools
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from .router import TokenRouter, RoutingDecision
from .dispatcher import TokenDispatcher, DispatchInfo

def _patch_nuwa_stage1_get_image_features(model, router):
    """
    Patch HF LLaVA get_image_features() to apply Nuwa Stage-1 aggregation.

    Target schedule for matain_token=192:
      raw visual tokens: 576
      Stage-1: 576 -> 336 before LLM
      Stage-2: 336 -> 48 at decoder layer 16
    """
    try:
        from .nuwa_hf import NuwaRouter, nuwa_stage1_aggregate
    except Exception:
        return

    if not isinstance(router, NuwaRouter):
        return

    # lmms-eval wrapper may pass either the HF model itself or a wrapper.
    raw_model = getattr(model, "_model", model)

    # For HF LlavaForConditionalGeneration:
    # top-level model has .model.get_image_features(...)
    if hasattr(raw_model, "model") and hasattr(raw_model.model, "get_image_features"):
        llava_model = raw_model.model
        top_config = getattr(raw_model, "config", None)
    elif hasattr(raw_model, "get_image_features"):
        llava_model = raw_model
        top_config = getattr(raw_model, "config", None)
    else:
        raise AttributeError(
            "Nuwa Stage-1 patch failed: cannot find get_image_features() target."
        )

    if getattr(llava_model, "_nuwa_stage1_patched", False):
        return

    import types

    def nuwa_get_image_features(
        self,
        pixel_values,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        **kwargs,
    ):
        # Follow HF LLaVA defaults when args are omitted.
        cfg = top_config if top_config is not None else getattr(self, "config", None)

        if vision_feature_layer is None:
            vision_feature_layer = getattr(cfg, "vision_feature_layer", -2)

        if vision_feature_select_strategy is None:
            vision_feature_select_strategy = getattr(
                cfg,
                "vision_feature_select_strategy",
                "default",
            )

        # Nuwa Stage-1 needs CLIP hidden states and attentions.
        # Force these on regardless of caller kwargs.
        vision_outputs = self.vision_tower(
            pixel_values,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        # For LLaVA-1.5, vision_feature_layer is usually -2.
        # nuwa_stage1_aggregate supports negative indices because it indexes
        # hidden_states/attentions directly.
        aggregated_features, keep_idxs = nuwa_stage1_aggregate(
            vision_outputs=vision_outputs,
            select_layer=vision_feature_layer,
            matain_token=router.matain_token,
            distance=router.distance,
            penalty_threshold_percentile=router.penalty_threshold_percentile,
        )

        # CLIP dim -> LLM hidden dim
        image_features = self.multi_modal_projector(aggregated_features)

        # Optional debug handles
        self._nuwa_stage1_last_keep_idxs = keep_idxs
        self._nuwa_stage1_last_num_tokens = int(image_features.shape[1])

        from transformers.modeling_outputs import BaseModelOutputWithPooling

        # HF LlavaModel.forward expects:
        #   image_features = self.get_image_features(...).pooler_output
        #   image_features = torch.cat(image_features, dim=0)
        #
        # Therefore pooler_output must be a tuple/list of per-image tensors,
        # each shaped [num_visual_tokens, hidden_dim].
        image_features_per_image = tuple(image_features[b] for b in range(image_features.shape[0]))

        return BaseModelOutputWithPooling(
            last_hidden_state=image_features,
            pooler_output=image_features_per_image,
            hidden_states=None,
            attentions=None,
        )


    llava_model.get_image_features = types.MethodType(
        nuwa_get_image_features,
        llava_model,
    )
    llava_model._nuwa_stage1_patched = True

    print(
        f"[Nuwa] patched get_image_features: "
        f"576 -> {router.stage1_tokens} before LLM, "
        f"then -> {router.stage2_tokens} at layer {router.stage2_layer}",
        flush=True,
    )

@dataclass
class RoutingContext:
    """Shared state across decoder layers during a single forward pass.

    Attributes:
        visual_token_range: (start, end) absolute positions of visual
            tokens in the current sequence. Must be set before the first
            routing layer.
        router: the scoring strategy.
        dispatcher: the dispatch strategy (kept for API compat).
        action: "delete" | "residual_skip" | "full_kv".
        attn_weights_cache: attention weights stored from a prior layer,
            used by the router for scoring.
        routing_log: per-layer routing decisions, for analysis.
    """
    visual_token_range: tuple[int, int]
    router: TokenRouter
    dispatcher: TokenDispatcher
    action: str
    attn_weights_cache: Optional[Tensor] = None
    routing_log: dict[int, RoutingDecision] = field(default_factory=dict)
    # Layer module references (used by some helpers to detect attn type).
    _layer_modules: dict = field(default_factory=dict)
    # Physical-delete state propagated to downstream-layer decode-step
    # intercept. After physical_delete shrinks the seq at the decision
    # layer, downstream layers receive the compacted hidden_states but
    # the top-level model still passes the original full-L PE/pos_ids;
    # _compact_position_{embeddings,ids} cache the compacted versions.
    _compacted: bool = False
    _compact_from_layer: Optional[int] = None
    _compact_position_embeddings: Optional[tuple] = None
    _compact_position_ids: Optional[Tensor] = None
    # Model family: "llava" (default) or "qwen25vl" / "qwen3vl" (for
    # future extension — Qwen hooks are kept end-to-end).
    _model_family: str = "llava"

    # Stagewise reroute state (compact_route_stagewise action).
    # Inside a "stage" (between two routing decisions) the sequence
    # stays compact across all layers; only at stage boundaries do we
    # scatter back to full L. Avoids the per-layer gather/scatter that
    # vanilla compact_route does.
    #
    # We snapshot ONLY the deferred (L-K) positions, not full L —
    # the K kept positions get overwritten by the in-stage compact
    # output at the next decision layer, so caching them is wasted
    # memory. Position embeddings / IDs are re-read from the outer
    # model on each forward (no cache needed).
    _stage_compact_kept_indices: Optional[Tensor] = None
    _stage_deferred_indices: Optional[Tensor] = None
    _stage_deferred_hidden: Optional[Tensor] = None
    _stage_full_seq_len: int = 0

    # ------------------------------------------------------------------
    # Oracle diagnostic state for representation staleness.
    #
    # REROUTE_ORACLE_MODE controls this diagnostic:
    #   none             : normal Reroute
    #   current_depth    : when a deferred visual token is re-admitted,
    #                      replace its hidden state with the same token's
    #                      full-token current-depth shadow state.
    #   shuffled_current : same as above, but use a different visual token's
    #                      full-token current-depth shadow state as a control.
    #
    # This is NOT a deployable acceleration path. It intentionally runs a
    # shadow full-token forward at routed layers to test whether stale hidden
    # states limit recoverable routing. Routing decisions and active-token
    # budget are kept unchanged.
    # ------------------------------------------------------------------
    oracle_mode: str = field(
        default_factory=lambda: os.environ.get("REROUTE_ORACLE_MODE", "none")
    )
    _oracle_shadow_hidden: Optional[Tensor] = None
    _oracle_prev_selected_mask: Optional[Tensor] = None
    _oracle_num_replaced: int = 0
    _oracle_num_layers: int = 0

    # ------------------------------------------------------------------
    # Hidden-state alignment diagnostic.
    # ------------------------------------------------------------------
    diag_alignment: bool = field(
        default_factory=lambda: os.environ.get("REROUTE_ALIGNMENT_DIAG", "0") == "1"
    )
    diag_alignment_out_path: str = field(
        default_factory=lambda: os.environ.get(
            "REROUTE_ALIGNMENT_OUT", "diagnostics/hidden_state_alignment.jsonl"
        )
    )
    diag_alignment_method: str = field(
        default_factory=lambda: os.environ.get("REROUTE_ALIGNMENT_METHOD", "auto")
    )
    _diag_alignment_prev_selected_mask: Optional[Tensor] = None
    _diag_alignment_last_active_layer: Optional[Tensor] = None
    _diag_alignment_file_initialized: bool = False
    _diag_alignment_num_events: int = 0

    # Optional hidden-vector dump for PCA/UMAP diagnostics.
    # Leave REROUTE_ALIGNMENT_VECTOR_DIR empty to disable vector dumping.
    diag_alignment_vector_dir: str = field(
        default_factory=lambda: os.environ.get(
            "REROUTE_ALIGNMENT_VECTOR_DIR", ""
        )
    )
    diag_alignment_vector_max: int = field(
        default_factory=lambda: int(
            os.environ.get("REROUTE_ALIGNMENT_VECTOR_MAX", "2000")
        )
    )
    _diag_alignment_vector_count: int = 0

    # ------------------------------------------------------------------
    # Diagnostic: stale-key routing vs. current-depth-oracle-key routing.
    #
    # This diagnostic isolates the question:
    #   "Is a deferred token's stale representation still sufficient for
    #    deciding whether the token should be re-admitted?"
    #
    # IMPORTANT:
    #   - The text query Q is ALWAYS taken from the actual current-depth
    #     routing state.
    #   - Only the VISUAL K states are replaced by same-token full-shadow
    #     current-depth states for the oracle score.
    #   - The oracle scores are diagnostic-only: they NEVER change the real
    #     routing decision, hidden states, or active-token budget.
    # ------------------------------------------------------------------
    diag_routing_agreement: bool = field(
        default_factory=lambda: os.environ.get(
            "REROUTE_ROUTING_AGREEMENT_DIAG", "0"
        ) == "1"
    )
    diag_routing_agreement_out_path: str = field(
        default_factory=lambda: os.environ.get(
            "REROUTE_ROUTING_AGREEMENT_OUT",
            "diagnostics/routing_agreement.jsonl",
        )
    )
    _diag_routing_prev_selected_mask: Optional[Tensor] = None
    _diag_routing_last_active_layer: Optional[Tensor] = None
    _diag_routing_file_initialized: bool = False

    # ------------------------------------------------------------------
    # Deployable correction state for stale re-admitted tokens.
    #
    # REROUTE_CORRECTION_MODE controls this path:
    #   none             : normal Reroute
    #   local            : state-mixing local correction
    #   global           : global depth-anchor correction
    #   local_global     : combine local state and global anchor correction
    #   local_transition : identity-preserving local depth-transition correction
    #   query_local_transition / qver : query-relevant local transition correction
    #
    # Unlike oracle modes, this path does NOT run full-token shadow forward.
    # It keeps routing decisions and active-token budget unchanged, and only
    # modifies hidden states of re-admitted tokens before compact dispatch.
    # ------------------------------------------------------------------
    correction_mode: str = field(
        default_factory=lambda: os.environ.get("REROUTE_CORRECTION_MODE", "none")
    )
    correction_topm: int = 8
    correction_spatial_lambda: float = 0.15
    correction_tau_l: float = 0.07
    # Anchor-selection ablation mode for LTC:
    #   random / spatial / semantic / semantic_spatial
    # Default preserves the original Q-VER behavior.
    correction_anchor_mode: str = "semantic_spatial"
    correction_tau_g: float = 0.07
    correction_local_weight: float = 0.0
    correction_global_weight: float = 0.0
    correction_max_anchors: int = 3
    correction_min_layer_gap: int = 1
    correction_gamma_max: float = 0.8
    correction_norm_preserve: bool = True
    # Post-LTC norm target:
    #   cached           : preserve the re-admitted token's cached/base norm
    #   current_anchor   : weighted current-layer norm of selected LTC anchors
    #   transition_ratio : transfer the anchors' weighted norm-change ratio
    #   none             : no post-correction norm rescaling
    correction_norm_target: str = "cached"
    # Query-relevant LTC: add the router/query relevance score of fresh
    # neighbor tokens to the local transition neighbor-selection score.
    correction_query_eta: float = 0.0
    correction_query_norm: str = "minmax"

    # Historical-state cache policy used by LTC.
    #
    #   fifo (default): preserve the original behavior and retain only the
    #                   most recent N full visual snapshots (N=8 by default).
    #   live          : retain the current snapshot plus only historical
    #                   snapshots that are still referenced by currently
    #                   deferred tokens. This supports arbitrarily long gaps
    #                   without keeping every decoder-depth snapshot.
    #   full          : keep every snapshot (diagnostic / 32-snapshot control).
    #
    # Select from shell via:
    #   REROUTE_CORRECTION_SNAPSHOT_POLICY=live
    correction_snapshot_policy: str = "fifo"
    correction_snapshot_keep: int = 8
    correction_snapshot_debug: bool = False

    _correction_prev_selected_mask: Optional[Tensor] = None
    _correction_depth_anchors: list[Tensor] = field(default_factory=list)
    _correction_layer_snapshots: dict[int, Tensor] = field(default_factory=dict)
    _correction_last_active_layer: Optional[Tensor] = None
    _correction_num_corrected: int = 0
    _correction_num_gap_filtered: int = 0

    # Lightweight cache statistics for profiling/debugging.
    _correction_snapshot_current_count: int = 0
    _correction_snapshot_peak_count: int = 0
    _correction_snapshot_current_bytes: int = 0
    _correction_snapshot_peak_bytes: int = 0

    # ------------------------------------------------------------------
    # Diagnostic: all-layer relevance shift / hypothetical re-activation.
    #
    # This does not change routing decisions. At every prefill layer, it
    # computes a hypothetical Top-K visual-token set from last-text-token
    # attention and compares it with the previous layer's Top-K set.
    # ------------------------------------------------------------------
    diag_all_layer_react: bool = field(
        default_factory=lambda: os.environ.get("REROUTE_DIAG_ALL_LAYER_REACT", "0") == "1"
    )
    diag_topk: int = field(
        default_factory=lambda: int(os.environ.get("REROUTE_DIAG_TOPK", "64"))
    )
    diag_out_path: str = field(
        default_factory=lambda: os.environ.get(
            "REROUTE_DIAG_OUT",
            "diagnostics/all_layer_react.jsonl",
        )
    )
    _diag_prev_layer_mask: Optional[Tensor] = None
    # Reference layer for lost-and-recovered diagnostic.
    # Recoverable(l) compares TopK(r^l) with TopK(r^l0).
    diag_ref_layer: int = field(
        default_factory=lambda: int(os.environ.get("REROUTE_DIAG_REF_LAYER", "0"))
    )
    _diag_ref_layer_mask: Optional[Tensor] = None
    _diag_ref_layer_actual: Optional[int] = None
    _diag_sample_id: int = 0
    _diag_file_initialized: bool = False

    # ------------------------------------------------------------------
    # Diagnostic: trace specific visual token relevance trajectory.
    #
    # This is for qualitative analysis. Given one or more visual-token
    # indices, it logs their layer-wise last-token-attention score, rank,
    # percentile, and Top-K membership. Token indices are relative visual
    # token indices in [0, V-1]. For LLaVA-1.5, V=576 on a 24x24 grid.
    # ------------------------------------------------------------------
    diag_token_trace: bool = field(
        default_factory=lambda: os.environ.get("REROUTE_DIAG_TOKEN_TRACE", "0") == "1"
    )
    diag_token_indices: str = field(
        default_factory=lambda: os.environ.get("REROUTE_DIAG_TOKEN_INDICES", "")
    )
    diag_token_topk: int = field(
        default_factory=lambda: int(os.environ.get("REROUTE_DIAG_TOKEN_TOPK", "64"))
    )
    diag_token_out_path: str = field(
        default_factory=lambda: os.environ.get(
            "REROUTE_DIAG_TOKEN_OUT",
            "diagnostics/token_trajectory.jsonl",
        )
    )
    diag_token_sample_filter: int = field(
        default_factory=lambda: int(os.environ.get("REROUTE_DIAG_TOKEN_SAMPLE_ID", "-1"))
    )
    _diag_token_file_initialized: bool = False
    
    diag_token_map_path: str = field(
        default_factory=lambda: os.environ.get("REROUTE_DIAG_TOKEN_MAP", "")
    )
    _diag_token_map_cache: Optional[dict] = None


    # === CATCHUP_QUAL_TRACE_INSTALLED ===
    # Diagnostic-only routing trace for qualitative analysis.
    # This records actual selected / re-admitted visual-token indices and
    # never changes routing decisions or hidden states.
    diag_qual_trace: bool = field(
        default_factory=lambda: os.environ.get("REROUTE_QUAL_TRACE", "0") == "1"
    )
    diag_qual_out_path: str = field(
        default_factory=lambda: os.environ.get(
            "REROUTE_QUAL_TRACE_OUT", "diagnostics/qual_routing_trace.jsonl"
        )
    )
    _diag_qual_prev_selected_mask: Optional[Tensor] = None
    _diag_qual_last_sample_id: int = -1
    _diag_qual_file_initialized: bool = False



def _qual_trace_log(
    ctx: RoutingContext,
    layer_idx: int,
    selected_mask: Tensor,
) -> None:
    """Log actual routing states for qualitative visualization only.

    selected_mask uses visual-token-relative indices. For LLaVA-1.5 this is
    the 24x24 (=576) visual grid. The function is intentionally read-only.
    """
    if not getattr(ctx, "diag_qual_trace", False):
        return

    # The existing diagnostic sample counter is incremented once per prefill.
    # run_qualitative_cherrypick.sh enables the already-present token-trace
    # diagnostic with an empty token list, so this counter advances without
    # adding an extra attention computation.
    sample_id = int(getattr(ctx, "_diag_sample_id", 0))

    if int(getattr(ctx, "_diag_qual_last_sample_id", -1)) != sample_id:
        ctx._diag_qual_prev_selected_mask = None
        ctx._diag_qual_last_sample_id = sample_id

    out_path = str(getattr(ctx, "diag_qual_out_path", "")).strip()
    if not out_path:
        return

    if not getattr(ctx, "_diag_qual_file_initialized", False):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Truncate once per process. The shell runner also removes old traces.
        with open(out_path, "w", encoding="utf-8") as f:
            pass
        ctx._diag_qual_file_initialized = True

    mask = selected_mask.detach().to(device="cpu", dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    prev = ctx._diag_qual_prev_selected_mask
    if prev is not None:
        prev = prev.detach().to(device="cpu", dtype=torch.bool)
        if prev.shape != mask.shape:
            prev = None

    router_name = type(ctx.router).__name__
    rows = []
    for b in range(mask.shape[0]):
        curr_b = mask[b]
        selected = curr_b.nonzero(as_tuple=True)[0].tolist()

        if prev is None:
            readmitted = []
            newly_deferred = []
        else:
            prev_b = prev[b]
            readmitted = (curr_b & (~prev_b)).nonzero(as_tuple=True)[0].tolist()
            newly_deferred = ((~curr_b) & prev_b).nonzero(as_tuple=True)[0].tolist()

        rows.append({
            "sample_id": sample_id,
            "batch": int(b),
            "layer": int(layer_idx),
            "router": router_name,
            "action": str(ctx.action),
            "num_visual": int(curr_b.numel()),
            "num_selected": int(curr_b.sum().item()),
            "selected": [int(x) for x in selected],
            "newly_readmitted": [int(x) for x in readmitted],
            "newly_deferred": [int(x) for x in newly_deferred],
        })

    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    ctx._diag_qual_prev_selected_mask = mask.clone()


def _get_scoring_layers(router: TokenRouter) -> list[int]:
    """Determine which layers need to output attention weights for scoring."""
    from .router import FastVRouter, PDropRouter, SparseVLMRouter
    from .nuwa_hf import NuwaRouter

    # Model-specific routers may explicitly specify scoring layers.
    # QwenNuwaRouter uses this to capture QK attention at layer 16.
    if hasattr(router, "scoring_layers"):
        return [int(x) for x in router.scoring_layers]

    if isinstance(router, FastVRouter):
        return [router.scoring_layer]
    elif isinstance(router, (PDropRouter, SparseVLMRouter, NuwaRouter)):
        # Capture attention at each drop layer for re-scoring
        return list(router.drop_layers)
    return [2]  # default


def _make_patched_forward(original_forward, layer_idx: int, ctx: RoutingContext):
    """Create a patched forward function for a decoder layer."""

    @functools.wraps(original_forward)
    def patched_forward(
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings=None,
        **kwargs,
    ) -> Tensor:
        seq_len = hidden_states.shape[1]

        # --- Stagewise compact intercept (release-only) ---
        # If we're inside an active stage (compact kept-set carried from
        # upstream routing layer), short-circuit BEFORE the is_prefill
        # check (which uses seq_len >= vis_end and fails for K << L).
        # Two sub-cases:
        #   - non-decision layer in stage: stay compact via _forward_compact_in_stage
        #   - decision layer (PDrop L=7/15/23 etc): scatter back to full L,
        #     re-route, gather new compact via _forward_compact_route_stagewise.
        if (
            ctx.action == "compact_route_stagewise"
            and ctx._stage_compact_kept_indices is not None
            and seq_len == ctx._stage_compact_kept_indices.shape[0]
        ):
            if ctx.router.is_decision_layer(layer_idx):
                return _forward_compact_route_stagewise(
                    original_forward, layer_idx, ctx,
                    hidden_states, attention_mask, position_ids,
                    past_key_values, use_cache, position_embeddings,
                    **kwargs,
                )
            return _forward_compact_in_stage(
                original_forward, layer_idx, ctx,
                hidden_states, attention_mask, position_ids,
                past_key_values, use_cache, position_embeddings,
                **kwargs,
            )

        # --- Physical compaction intercept ---
        # After physical delete at a prior layer, hidden_states is shorter
        # than the attention_mask/position_embeddings from the top-level model.
        # Use the compacted versions stored in ctx.
        if ctx._compacted and seq_len > 1 and position_embeddings is not None:
            # cos shape: (batch, seq, dim) for LLaMA, (3, batch, seq, dim) for Qwen MROPE
            cos = position_embeddings[0]
            pe_seq = cos.shape[2] if cos.ndim == 4 else cos.shape[1]
            if seq_len != pe_seq:
                position_embeddings = ctx._compact_position_embeddings
                position_ids = ctx._compact_position_ids
                attention_mask = None  # sdpa handles causal for compacted seq

        vis_start, vis_end = ctx.visual_token_range

        # Skip routing during autoregressive decode steps (seq_len=1)
        # or when the current sequence doesn't span the visual range.
        is_prefill = seq_len > 1 and seq_len >= vis_end and vis_end > vis_start

        # Diagnostic-only: all-layer relevance shift analysis.
        # This computes hypothetical Top-K visual evidence at every layer
        # and does NOT affect routing decisions or hidden states.
        if is_prefill and getattr(ctx, "diag_all_layer_react", False):
            _maybe_log_all_layer_reactivation(
                layer_idx=layer_idx,
                ctx=ctx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        # Diagnostic-only: trace selected token relevance trajectory.
        # This does NOT affect routing decisions or hidden states.
        if is_prefill and getattr(ctx, "diag_token_trace", False):
            _maybe_log_token_trajectory(
                layer_idx=layer_idx,
                ctx=ctx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        if not is_prefill:
            # During autoregressive decode after physical_delete, layers before
            # the deletion layer still have full-length KV cache, while layers
            # at and after the deletion layer have compact KV cache. The
            # top-level attention_mask is built for the full sequence, so it
            # mismatches compact KV layers. Drop it only for compacted
            # downstream layers.
            if ctx._compacted:
                compact_from = getattr(ctx, "_compact_from_layer", None)
                if compact_from is not None and int(layer_idx) >= int(compact_from):
                    attention_mask = None

            return original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        scoring_layers = _get_scoring_layers(ctx.router)

        # --- Case 1: Scoring-only layer (e.g. FastV layer 2) ---
        if layer_idx in scoring_layers and not ctx.router.should_route(layer_idx):
            return _forward_scoring_layer(
                original_forward, layer_idx, ctx,
                hidden_states, attention_mask, position_ids,
                past_key_values, use_cache, position_embeddings, **kwargs,
            )

        # --- Case 2: Routing layer ---
        if ctx.router.should_route(layer_idx):
            if layer_idx in scoring_layers:
                try:
                    _capture_attention_weights(layer_idx, ctx, hidden_states,
                                               attention_mask, position_embeddings)
                except torch.cuda.OutOfMemoryError:
                    import warnings
                    seq_len_val = hidden_states.shape[1]
                    warnings.warn(
                        f"[OOM] _capture_attention_weights skipped at layer "
                        f"{layer_idx} (seq_len={seq_len_val}). "
                        f"Sample will run without routing.",
                        stacklevel=2,
                    )
                    torch.cuda.empty_cache()
                    ctx.attn_weights_cache = None
                    ctx._oom_skipped = True

            if ctx.attn_weights_cache is not None:
                if ctx.action == "physical_delete":
                    # Physical delete fires at the decision layer; later
                    # layers see the already-shrunk sequence and fall
                    # through to the vanilla forward.
                    if ctx.router.is_decision_layer(layer_idx):
                        return _forward_physical_delete(
                            original_forward, layer_idx, ctx,
                            hidden_states, attention_mask, position_ids,
                            past_key_values, use_cache, position_embeddings,
                            **kwargs,
                        )
                elif ctx.action == "compact_route":
                    # Per-layer compact gather + KÃK attention + scatter
                    # back to full L. Inner k_proj/v_proj run on K tokens,
                    # so KV cache writes K K,V (matches the original FastV/PDrop semantics for KV cache writes).
                    return _forward_compact_route(
                        original_forward, layer_idx, ctx,
                        hidden_states, attention_mask, position_ids,
                        past_key_values, use_cache, position_embeddings,
                        **kwargs,
                    )
                elif ctx.action == "compact_route_stagewise":
                    # Routing-layer entry; in-stage non-routing layers
                    # are caught by the early intercept at top of
                    # patched_forward (above).
                    if ctx.router.is_decision_layer(layer_idx):
                        return _forward_compact_route_stagewise(
                            original_forward, layer_idx, ctx,
                            hidden_states, attention_mask, position_ids,
                            past_key_values, use_cache, position_embeddings,
                            **kwargs,
                        )
                else:
                    raise ValueError(
                        f"Unsupported routing action: {ctx.action!r}. "
                        f"Release supports: physical_delete, compact_route, "
                        f"compact_route_stagewise."
                    )


        # --- Case 3: Normal layer ---
        # Stagewise reroute: when we're inside an active stage (compact
        # tokens carried over from upstream routing layer), keep this
        # non-routing layer compact too — gather PE / pos_ids and run
        # original_forward on the compact tensor without scattering back.
        # Decode steps (hidden_states.shape[1] == 1) take vanilla path.
        if (
            ctx.action == "compact_route_stagewise"
            and ctx._stage_compact_kept_indices is not None
            and hidden_states.shape[1] == ctx._stage_compact_kept_indices.shape[0]
        ):
            return _forward_compact_in_stage(
                original_forward, layer_idx, ctx,
                hidden_states, attention_mask, position_ids,
                past_key_values, use_cache, position_embeddings,
                **kwargs,
            )

        return original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    return patched_forward


def _capture_attention_weights(
    layer_idx: int, ctx: RoutingContext,
    hidden_states: Tensor, attention_mask: Tensor | None,
    position_embeddings,
) -> None:
    """Compute and cache attention weights from Q,K projection hooks.

    Runs the Q and K projections (plus layernorm) without running the
    full layer. Works with any attention backend by directly computing
    Q*K in fp32.
    """
    layer_module = ctx._layer_modules[layer_idx]
    attn = layer_module.self_attn
    head_dim = attn.head_dim
    batch, seq_len = hidden_states.shape[:2]

    # Compute Q, K from the layer's input (through layernorm + projection)
    normed = layer_module.input_layernorm(hidden_states)
    q_raw = attn.q_proj(normed)
    k = attn.k_proj(normed)

    # Some Qwen variants use a gated q_proj: output is (H * head_dim * 2), chunked
    # into (query, gate). We only need query for attention scoring;
    # gate is applied after attn_output in the real forward path.
    if ctx._model_family == "qwen35":
        q_raw = q_raw.view(batch, seq_len, -1, head_dim * 2)
        q, _gate = torch.chunk(q_raw, 2, dim=-1)  # (B, S, H, D)
    else:
        q = q_raw.view(batch, seq_len, -1, head_dim)
    k = k.view(batch, seq_len, -1, head_dim)

    # Apply QK normalization if present (Qwen3-VL)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    # Apply RoPE (LLaMA standard or Qwen MROPE)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if ctx._model_family == "qwen25vl":
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                apply_multimodal_rotary_pos_emb,
            )
            mrope_section = attn.config.rope_parameters["mrope_section"]
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        else:
            # LLaVA (standard RoPE) and Qwen3-VL (interleaved MROPE
            # already baked into cos/sin by rotary_emb.forward())
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    # Expand KV heads for GQA models
    num_kv_groups = attn.num_key_value_groups
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)

    # Compute attention weights in fp32
    scaling = attn.scaling
    attn_weights = torch.matmul(q.float(), k.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1)

    ctx.attn_weights_cache = attn_weights.detach()


def _maybe_log_all_layer_reactivation(
    layer_idx: int,
    ctx: RoutingContext,
    hidden_states: Tensor,
    attention_mask: Tensor | None,
    position_embeddings,
) -> None:
    """Diagnostic-only all-layer relevance shift analysis.

    At every prefill layer, compute a hypothetical Top-K visual-token set
    using last-text-token attention.

    It logs two metrics:
      1) volatility:
         newly entered Top-K tokens compared to the previous layer.
      2) recoverable:
         tokens not in the early reference Top-K but recovered in the current Top-K.

    This function does NOT change routing decisions or hidden states.
    """
    if not getattr(ctx, "diag_all_layer_react", False):
        return

    vis_start, vis_end = ctx.visual_token_range
    if vis_end <= vis_start:
        return

    seq_len = hidden_states.shape[1]
    if seq_len <= 1 or seq_len < vis_end:
        return

    # Do not overwrite the routing cache used by the real router.
    old_cache = ctx.attn_weights_cache

    try:
        _capture_attention_weights(
            layer_idx=layer_idx,
            ctx=ctx,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        attn_weights = ctx.attn_weights_cache
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        ctx.attn_weights_cache = old_cache
        return
    except Exception as e:
        warnings.warn(
            f"[DIAG] Failed to compute all-layer relevance metrics at layer {layer_idx}: {e}",
            stacklevel=2,
        )
        ctx.attn_weights_cache = old_cache
        return

    # Restore original routing cache.
    ctx.attn_weights_cache = old_cache

    if attn_weights is None:
        return

    # attn_weights: (B, H, S, S)
    # Query relevance = last text token -> visual token attention.
    attn_avg = attn_weights.mean(dim=1)  # (B, S, S)
    scores = attn_avg[:, -1, vis_start:vis_end].float()  # (B, V)

    B, V = scores.shape
    k = min(max(1, int(ctx.diag_topk)), V)

    topk_idx = scores.topk(k, dim=-1).indices  # (B, K)
    curr_mask = torch.zeros_like(scores, dtype=torch.bool)
    curr_mask.scatter_(1, topk_idx, True)

    # ================================================================
    # 1) Relevance volatility
    #    Vol(l) = 1 - |TopK(l) ∩ TopK(l-1)| / K
    #           = |TopK(l) \ TopK(l-1)| / K
    # ================================================================
    prev_mask = ctx._diag_prev_layer_mask

    if prev_mask is None or prev_mask.shape != curr_mask.shape:
        volatility_count = None
        volatility_ratio = None
        prev_overlap_count = None
        prev_overlap_ratio = None
    else:
        prev_mask = prev_mask.to(device=curr_mask.device)

        newly_entered = curr_mask & (~prev_mask)
        prev_overlap = curr_mask & prev_mask

        volatility_count_tensor = newly_entered.sum(dim=1).float()
        prev_overlap_count_tensor = prev_overlap.sum(dim=1).float()

        volatility_ratio_tensor = volatility_count_tensor / float(k)
        prev_overlap_ratio_tensor = prev_overlap_count_tensor / float(k)

        volatility_count = float(volatility_count_tensor.mean().item())
        volatility_ratio = float(volatility_ratio_tensor.mean().item())
        prev_overlap_count = float(prev_overlap_count_tensor.mean().item())
        prev_overlap_ratio = float(prev_overlap_ratio_tensor.mean().item())

    ctx._diag_prev_layer_mask = curr_mask.detach().clone()

    # ================================================================
    # 2) Lost-and-recovered token ratio
    #    Recoverable(l) = |TopK(l) \ TopK(l0)| / K
    #
    #    l0 is the early reference layer. Default: layer 0.
    # ================================================================
    ref_layer = int(getattr(ctx, "diag_ref_layer", 0))

    if (
        ctx._diag_ref_layer_mask is None
        or ctx._diag_ref_layer_mask.shape != curr_mask.shape
    ):
        if layer_idx >= ref_layer:
            ctx._diag_ref_layer_mask = curr_mask.detach().clone()
            ctx._diag_ref_layer_actual = int(layer_idx)

    ref_mask = ctx._diag_ref_layer_mask

    if ref_mask is None or ref_mask.shape != curr_mask.shape:
        recoverable_count = None
        recoverable_ratio = None
        ref_overlap_count = None
        ref_overlap_ratio = None
        ref_layer_actual = None
    else:
        ref_mask = ref_mask.to(device=curr_mask.device)

        recovered = curr_mask & (~ref_mask)
        ref_overlap = curr_mask & ref_mask

        recoverable_count_tensor = recovered.sum(dim=1).float()
        ref_overlap_count_tensor = ref_overlap.sum(dim=1).float()

        recoverable_ratio_tensor = recoverable_count_tensor / float(k)
        ref_overlap_ratio_tensor = ref_overlap_count_tensor / float(k)

        recoverable_count = float(recoverable_count_tensor.mean().item())
        recoverable_ratio = float(recoverable_ratio_tensor.mean().item())
        ref_overlap_count = float(ref_overlap_count_tensor.mean().item())
        ref_overlap_ratio = float(ref_overlap_ratio_tensor.mean().item())
        ref_layer_actual = int(ctx._diag_ref_layer_actual)

    out_path = ctx.diag_out_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    row = {
        "sample_id": int(ctx._diag_sample_id),
        "layer": int(layer_idx),
        "topk": int(k),
        "num_visual": int(V),

        # Backward-compatible names used by the previous summarizer.
        "react_count": volatility_count,
        "react_ratio": volatility_ratio,
        "overlap_count": prev_overlap_count,
        "overlap_ratio": prev_overlap_ratio,

        # Explicit names for the two motivation diagnostics.
        "volatility_count": volatility_count,
        "volatility_ratio": volatility_ratio,
        "prev_overlap_count": prev_overlap_count,
        "prev_overlap_ratio": prev_overlap_ratio,

        "ref_layer": ref_layer,
        "ref_layer_actual": ref_layer_actual,
        "recoverable_count": recoverable_count,
        "recoverable_ratio": recoverable_ratio,
        "ref_overlap_count": ref_overlap_count,
        "ref_overlap_ratio": ref_overlap_ratio,
    }

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _parse_token_indices(token_indices: str) -> list[int]:
    """Parse comma-separated visual-token indices."""
    if token_indices is None:
        return []
    token_indices = str(token_indices).strip()
    if token_indices == "":
        return []

    out: list[int] = []
    for x in token_indices.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _maybe_log_token_trajectory(
    layer_idx: int,
    ctx: RoutingContext,
    hidden_states: Tensor,
    attention_mask: Tensor | None,
    position_embeddings,
) -> None:
    """Diagnostic-only token trajectory logging.

    For selected visual token indices, log their layer-wise query relevance:
      - last-token attention score
      - rank among all visual tokens (rank=1 is highest)
      - importance percentile (1.0 is highest, 0.0 is lowest)
      - whether the token is in the hypothetical Top-K set

    This function does NOT change routing decisions or hidden states.
    """
    if not getattr(ctx, "diag_token_trace", False):
        return

    sample_filter = int(getattr(ctx, "diag_token_sample_filter", -1))
    if sample_filter >= 0 and int(ctx._diag_sample_id) != sample_filter:
        return

    try:

        token_indices = _parse_token_indices(getattr(ctx, "diag_token_indices", ""))

        token_map_path = getattr(ctx, "diag_token_map_path", "")
        if token_map_path:
            if ctx._diag_token_map_cache is None:
                with open(token_map_path, "r", encoding="utf-8") as f:
                    ctx._diag_token_map_cache = json.load(f)

            sample_key = str(int(ctx._diag_sample_id))
            if sample_key in ctx._diag_token_map_cache:
                token_indices = ctx._diag_token_map_cache[sample_key]["token_indices"]
            else:
                token_indices = []
    except Exception as e:
        warnings.warn(
            f"[DIAG] Failed to parse REROUTE_DIAG_TOKEN_INDICES: {e}",
            stacklevel=2,
        )
        return

    if len(token_indices) == 0:
        return

    vis_start, vis_end = ctx.visual_token_range
    if vis_end <= vis_start:
        return

    seq_len = hidden_states.shape[1]
    if seq_len <= 1 or seq_len < vis_end:
        return

    # Do not overwrite the routing cache used by the real router.
    old_cache = ctx.attn_weights_cache

    try:
        _capture_attention_weights(
            layer_idx=layer_idx,
            ctx=ctx,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        attn_weights = ctx.attn_weights_cache
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        ctx.attn_weights_cache = old_cache
        return
    except Exception as e:
        warnings.warn(
            f"[DIAG] Failed to compute token trajectory at layer {layer_idx}: {e}",
            stacklevel=2,
        )
        ctx.attn_weights_cache = old_cache
        return

    ctx.attn_weights_cache = old_cache

    if attn_weights is None:
        return

    # Query relevance = last text token -> visual token attention.
    # attn_weights: (B, H, S, S)
    attn_avg = attn_weights.mean(dim=1)  # (B, S, S)
    scores = attn_avg[:, -1, vis_start:vis_end].float()  # (B, V)

    B, V = scores.shape
    topk = min(max(1, int(getattr(ctx, "diag_token_topk", 64))), V)

    out_path = ctx.diag_token_out_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    for b in range(B):
        scores_b = scores[b]

        for token_idx in token_indices:
            if token_idx < 0 or token_idx >= V:
                continue

            score = scores_b[token_idx]

            # rank: 1 means highest relevance among all visual tokens.
            rank = int((scores_b > score).sum().item()) + 1

            # Higher percentile means more important.
            # rank=1 -> 1.0, rank=V -> 0.0.
            if V > 1:
                importance_percentile = 1.0 - float(rank - 1) / float(V - 1)
            else:
                importance_percentile = 1.0

            in_topk = bool(rank <= topk)

            rows.append({
                "sample_id": int(ctx._diag_sample_id),
                "batch_idx": int(b),
                "layer": int(layer_idx),
                "token_idx": int(token_idx),
                "num_visual": int(V),
                "topk": int(topk),
                "score": float(score.detach().cpu().item()),
                "rank": int(rank),
                "importance_percentile": float(importance_percentile),
                "in_topk": in_topk,
            })

    if len(rows) == 0:
        return

    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _forward_scoring_layer(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """Run a scoring-only layer: capture attention weights, then run normally.

    Used for FastV where the scoring layer is separate from the decision
    layer (e.g., scoring_layer=2, decision_layer=3).
    """
    _capture_attention_weights(
        layer_idx, ctx, hidden_states, attention_mask, position_embeddings,
    )

    return original_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )


def _apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to Q and K tensors.

    Compatible with LLaMA / Qwen2.5-VL / Qwen3-VL (full rotary) and
    Variants with partial rotary (rotary_dim = head_dim * partial_rotary_factor,
    trailing pass-through dims). Detected by comparing cos last-dim to q
    last-dim — if shorter, only rotate the leading slice.

    cos, sin: (batch, seq_len, rotary_dim) from the model's rotary_emb.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    if rotary_dim == head_dim:
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed

    # Partial rotary: rotate only the first rotary_dim, pass rest through.
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _forward_physical_delete(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """Apply routing via physical token removal.

    Physically removes unselected visual tokens from the sequence.
    After this layer, hidden_states is shorter. Subsequent layers
    receive the compacted sequence via ctx's stored position embeddings.
    KV cache naturally only contains remaining tokens — no decode-time
    masking needed.

    Position IDs are preserved (GAP-style: original RoPE positions with
    gaps where tokens were removed).
    """
    vis_start, vis_end = ctx.visual_token_range
    batch = hidden_states.shape[0]
    device = hidden_states.device
    seq_len = hidden_states.shape[1]

    # 1. Compute routing decision
    decision = ctx.router.compute_scores(
        layer_idx=layer_idx,
        attn_weights=ctx.attn_weights_cache,
        hidden_states=hidden_states,
        visual_token_range=ctx.visual_token_range,
    )
    ctx.routing_log[layer_idx] = decision

    # 2. Build full-sequence keep indices
    # Keep all text tokens + selected visual tokens
    selected_mask = decision.selected_mask  # (batch, num_visual)
    _qual_trace_log(
        ctx=ctx,
        layer_idx=layer_idx,
        selected_mask=selected_mask,
    )
    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    unsel_relative = (~selected_mask[0]).nonzero(as_tuple=True)[0]
    unsel_abs = unsel_relative + vis_start
    keep_mask[unsel_abs] = False
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]  # sorted ascending
    num_kept_visual = int(selected_mask[0].sum().item())

    # 3. Physically compact hidden_states
    hidden_states = hidden_states[:, kept_indices].contiguous()

    # 4. GAP position strategy: preserve original position IDs with gaps
    #    so kept tokens retain their original RoPE positions (paper §3
    #    GAP mechanism). Release supports only GAP.
    if position_ids is not None:
        position_ids = position_ids[:, kept_indices].contiguous()
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 4:
            # Qwen MROPE: (3, batch, seq, dim).
            cos = cos[:, :, kept_indices].contiguous()
            sin = sin[:, :, kept_indices].contiguous()
        else:
            # LLaMA / standard rotary: (batch, seq, dim).
            cos = cos[:, kept_indices].contiguous()
            sin = sin[:, kept_indices].contiguous()
        position_embeddings = (cos, sin)

    # 6. Drop the attention mask — sdpa handles causal masking for the
    # compacted sequence (tokens remain in original causal order).
    attention_mask = None

    # 7. Run the original layer forward with compacted inputs
    output = original_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    # 8. Update context for subsequent layers: post-delete decode-step
    #    intercept (line ~190) uses these to feed the compacted PE/pos_ids
    #    when the top-level model's outer cache mask is sized for full L.
    ctx._compacted = True
    ctx._compact_from_layer = int(layer_idx)
    ctx._compact_position_embeddings = position_embeddings
    ctx._compact_position_ids = position_ids
    ctx.visual_token_range = (vis_start, vis_start + num_kept_visual)

    return output



# =====================================================================
# Oracle diagnostic helpers for representation staleness
# =====================================================================


def _oracle_enabled(ctx: RoutingContext) -> bool:
    """Return True when oracle state replacement is enabled."""
    return ctx.oracle_mode in ("current_depth", "shuffled_current")


def _shadow_reference_enabled(ctx: RoutingContext) -> bool:
    """Whether a full current-depth shadow reference is needed."""
    return (
        _oracle_enabled(ctx)
        or bool(getattr(ctx, "diag_alignment", False))
        or bool(getattr(ctx, "diag_routing_agreement", False))
    )


def _oracle_get_shadow_input(
    ctx: RoutingContext,
    hidden_states: Tensor,
) -> Tensor:
    """Return full-token shadow hidden state at the current layer input.

    The first routed layer is reached before any routing has modified the
    sequence, so its hidden_states are also the full-token reference at that
    depth. From that point on, _oracle_advance_shadow runs the original layer
    on the full sequence at each compact_route layer, keeping a shadow
    current-depth state for the next routed layer.
    """
    if ctx._oracle_shadow_hidden is None:
        ctx._oracle_shadow_hidden = hidden_states.detach().clone()
    return ctx._oracle_shadow_hidden


@torch.no_grad()
def _oracle_advance_shadow(
    original_forward,
    ctx: RoutingContext,
    oracle_input: Tensor,
    attention_mask,
    position_ids,
    position_embeddings,
    **kwargs,
) -> None:
    """Advance the shadow full-token hidden state by one decoder layer.

    Diagnostic-only and intentionally expensive: this keeps a full-token
    current-depth reference while the actual Reroute path still computes only
    selected visual tokens.
    """
    if not _shadow_reference_enabled(ctx):
        return

    shadow_out = original_forward(
        oracle_input,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    shadow_hidden = shadow_out[0] if isinstance(shadow_out, tuple) else shadow_out
    ctx._oracle_shadow_hidden = shadow_hidden.detach()
    ctx._oracle_num_layers += 1


def _make_shuffled_perm(num_visual: int, device: torch.device) -> Tensor:
    """Deterministic non-identity permutation for shuffled oracle control."""
    if num_visual <= 1:
        return torch.arange(num_visual, device=device)
    return torch.roll(torch.arange(num_visual, device=device), shifts=1)


def _oracle_replace_readmitted_tokens(
    ctx: RoutingContext,
    hidden_states: Tensor,
    selected_mask: Tensor,
    visual_token_range: tuple[int, int],
    oracle_input: Tensor,
) -> Tensor:
    """Replace only re-admitted visual-token hidden states.

    A re-admitted token is selected in the current routing decision but was
    not selected in the previous routing decision. We keep the routing mask
    and the number of active tokens fixed; only the hidden state at those
    re-admitted positions is swapped for an oracle/control state.
    """
    # Always update prev-selected state so mode can be toggled per run.
    if not _oracle_enabled(ctx):
        ctx._oracle_prev_selected_mask = selected_mask.detach().clone()
        return hidden_states

    # First routing decision has no prior deferred set to re-admit from.
    if ctx._oracle_prev_selected_mask is None:
        ctx._oracle_prev_selected_mask = selected_mask.detach().clone()
        return hidden_states

    prev_mask = ctx._oracle_prev_selected_mask
    if prev_mask.shape != selected_mask.shape:
        ctx._oracle_prev_selected_mask = selected_mask.detach().clone()
        return hidden_states

    readmit_mask = selected_mask & (~prev_mask)  # (B, num_visual)
    ctx._oracle_prev_selected_mask = selected_mask.detach().clone()

    if not readmit_mask.any():
        return hidden_states

    vis_start, vis_end = visual_token_range
    num_visual = vis_end - vis_start
    oracle_vis = oracle_input[:, vis_start:vis_end, :]  # (B, V, D)

    if ctx.oracle_mode == "current_depth":
        source_vis = oracle_vis
    elif ctx.oracle_mode == "shuffled_current":
        perm = _make_shuffled_perm(num_visual, oracle_vis.device)
        source_vis = oracle_vis[:, perm, :]
    else:
        return hidden_states

    new_hidden = hidden_states.clone()
    target_vis = new_hidden[:, vis_start:vis_end, :]
    target_vis[readmit_mask] = source_vis[readmit_mask].to(
        dtype=target_vis.dtype,
        device=target_vis.device,
    )
    new_hidden[:, vis_start:vis_end, :] = target_vis

    ctx._oracle_num_replaced += int(readmit_mask.sum().item())
    return new_hidden





# =====================================================================
# Routing-agreement diagnostic helpers
# =====================================================================


def _routing_agreement_spearman(x: Tensor, y: Tensor) -> Optional[float]:
    """Spearman rank correlation for one paired score vector.

    Routing scores are continuous in practice, so ties are extremely rare.
    We therefore use a deterministic double-argsort rank assignment.
    """
    if x.numel() < 2 or y.numel() != x.numel():
        return None

    xf = x.float().flatten()
    yf = y.float().flatten()

    rx = torch.empty_like(xf)
    ry = torch.empty_like(yf)

    rx[torch.argsort(xf)] = torch.arange(
        xf.numel(), device=xf.device, dtype=xf.dtype
    )
    ry[torch.argsort(yf)] = torch.arange(
        yf.numel(), device=yf.device, dtype=yf.dtype
    )

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = rx.norm() * ry.norm()
    if float(denom.item()) <= 1e-12:
        return None

    return float(((rx * ry).sum() / denom).item())


@torch.no_grad()
def _routing_agreement_oracle_visual_scores(
    ctx: RoutingContext,
    layer_idx: int,
    hidden_states: Tensor,
    oracle_input: Tensor,
    attention_mask: Optional[Tensor],
    position_embeddings,
) -> Tensor:
    """Compute oracle visual-routing scores with the SAME current query.

    The real CatchUp routing score at layer l uses:
        Q_current  x  K_latest-available

    For this diagnostic we form a paired oracle score:
        Q_current  x  K_current-depth-oracle

    Only the visual K vectors are replaced. Text/system keys and, crucially,
    the query vector are kept from the actual routing trajectory. Therefore
    the comparison isolates the effect of visual-state staleness on routing.

    Returns:
        (B, V) post-softmax, head-averaged last-query attention scores,
        matching Q-VER/PDrop's normal scoring convention.
    """
    vis_start, vis_end = ctx.visual_token_range
    layer_module = ctx._layer_modules[layer_idx]
    attn = layer_module.self_attn

    batch, seq_len = hidden_states.shape[:2]
    head_dim = attn.head_dim

    # Actual current-depth query / actual latest-available keys.
    normed_actual = layer_module.input_layernorm(hidden_states)
    q_raw = attn.q_proj(normed_actual)
    k_actual = attn.k_proj(normed_actual)

    # Same-token full-shadow current-depth visual keys.
    normed_oracle = layer_module.input_layernorm(oracle_input)
    k_oracle = attn.k_proj(normed_oracle)

    if ctx._model_family == "qwen35":
        q_raw = q_raw.view(batch, seq_len, -1, head_dim * 2)
        q, _gate = torch.chunk(q_raw, 2, dim=-1)
    else:
        q = q_raw.view(batch, seq_len, -1, head_dim)

    k_actual = k_actual.view(batch, seq_len, -1, head_dim)
    k_oracle = k_oracle.view(batch, seq_len, -1, head_dim)

    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k_actual = attn.k_norm(k_actual)
        k_oracle = attn.k_norm(k_oracle)

    # Replace ONLY visual keys with same-token current-depth oracle keys.
    k_mixed = k_actual.clone()
    k_mixed[:, vis_start:vis_end] = k_oracle[:, vis_start:vis_end]

    q = q.transpose(1, 2)
    k_mixed = k_mixed.transpose(1, 2)

    # Apply the exact current-layer positional encoding.
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if ctx._model_family == "qwen25vl":
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                apply_multimodal_rotary_pos_emb,
            )
            mrope_section = attn.config.rope_parameters["mrope_section"]
            q, k_mixed = apply_multimodal_rotary_pos_emb(
                q, k_mixed, cos, sin, mrope_section
            )
        else:
            q, k_mixed = _apply_rotary_pos_emb(
                q, k_mixed, cos, sin
            )

    num_kv_groups = attn.num_key_value_groups
    if num_kv_groups > 1:
        k_mixed = k_mixed.repeat_interleave(
            num_kv_groups,
            dim=1,
        )

    # We only need the final text/query row used by Q-VER.
    q_last = q[:, :, -1:, :]
    logits = torch.matmul(
        q_last.float(),
        k_mixed.float().transpose(2, 3),
    ) * attn.scaling

    if attention_mask is not None:
        mask = attention_mask.float()
        if mask.ndim == 4:
            mask = mask[:, :, -1:, :]
        elif mask.ndim == 3:
            mask = mask[:, -1:, :].unsqueeze(1)
        logits = logits + mask

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    # Mean across heads; extract visual keys.
    scores = probs.mean(dim=1)[:, 0, vis_start:vis_end].float()
    return scores


def _routing_gap_bucket(gap: int) -> Optional[str]:
    """Gap bins aligned with the paper's long-gap analysis."""
    if 5 <= gap <= 8:
        return "5-8"
    if 9 <= gap <= 12:
        return "9-12"
    if gap >= 13:
        return "13+"
    return None


@torch.no_grad()
def _routing_agreement_log(
    ctx: RoutingContext,
    layer_idx: int,
    hidden_states: Tensor,
    oracle_input: Optional[Tensor],
    decision: RoutingDecision,
    attention_mask: Optional[Tensor],
    position_embeddings,
) -> None:
    """Compare stale routing with oracle-key routing on identical events.

    The real stale routing trajectory is never changed. We only compute a
    second, diagnostic Top-K using current-depth oracle VISUAL keys and the
    SAME actual current query.

    Logged metrics:
      - Top-K overlap over all visual tokens.
      - Oracle re-admission Recall@K / Precision@K on previously deferred
        tokens.
      - Spearman rho over the previously deferred candidate ranking.
      - The same recall / rho statistics by layer-gap bin.
    """
    if not bool(getattr(ctx, "diag_routing_agreement", False)):
        return
    if oracle_input is None:
        return
    if not ctx.router.is_decision_layer(layer_idx):
        return

    selected_mask = decision.selected_mask
    _qual_trace_log(
        ctx=ctx,
        layer_idx=layer_idx,
        selected_mask=selected_mask,
    )
    stale_scores = decision.scores.float()

    B, V = selected_mask.shape

    prev = ctx._diag_routing_prev_selected_mask
    last = ctx._diag_routing_last_active_layer

    # First routing boundary: there is no previously deferred set yet.
    # Initialize state but DO NOT log it, otherwise the analysis would be
    # trivially inflated by tokens that have never become stale.
    if (
        prev is None
        or last is None
        or prev.shape != selected_mask.shape
        or last.shape != selected_mask.shape
    ):
        ctx._diag_routing_prev_selected_mask = selected_mask.detach().clone()
        init_last = torch.full(
            (B, V),
            int(layer_idx),
            device=selected_mask.device,
            dtype=torch.long,
        )
        init_last[selected_mask] = int(layer_idx)
        ctx._diag_routing_last_active_layer = init_last.detach()
        return

    prev = prev.to(selected_mask.device)
    last = last.to(selected_mask.device)

    oracle_scores = _routing_agreement_oracle_visual_scores(
        ctx=ctx,
        layer_idx=layer_idx,
        hidden_states=hidden_states,
        oracle_input=oracle_input,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )

    out_path = str(ctx.diag_routing_agreement_out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = []

    for b in range(B):
        k = int(selected_mask[b].sum().item())
        k = min(max(k, 1), V)

        oracle_idx = oracle_scores[b].topk(
            k,
            dim=-1,
        ).indices
        oracle_mask = torch.zeros(
            V,
            dtype=torch.bool,
            device=selected_mask.device,
        )
        oracle_mask[oracle_idx] = True

        stale_mask = selected_mask[b]
        deferred_mask = ~prev[b]

        topk_hit = stale_mask & oracle_mask
        topk_overlap_count = int(topk_hit.sum().item())

        stale_readmit = stale_mask & deferred_mask
        oracle_readmit = oracle_mask & deferred_mask
        readmit_hit = stale_readmit & oracle_readmit

        oracle_readmit_count = int(oracle_readmit.sum().item())
        stale_readmit_count = int(stale_readmit.sum().item())
        readmit_hit_count = int(readmit_hit.sum().item())

        recall = (
            readmit_hit_count / oracle_readmit_count
            if oracle_readmit_count > 0
            else None
        )
        precision = (
            readmit_hit_count / stale_readmit_count
            if stale_readmit_count > 0
            else None
        )

        deferred_idx = deferred_mask.nonzero(as_tuple=True)[0]
        deferred_rho = _routing_agreement_spearman(
            stale_scores[b, deferred_idx],
            oracle_scores[b, deferred_idx],
        )

        layer_gap = int(layer_idx) - last[b]

        gap_stats = {}

        for gap_name, lo, hi in (
            ("5-8", 5, 8),
            ("9-12", 9, 12),
            ("13+", 13, None),
        ):
            if hi is None:
                gap_mask = deferred_mask & (layer_gap >= lo)
            else:
                gap_mask = (
                    deferred_mask
                    & (layer_gap >= lo)
                    & (layer_gap <= hi)
                )

            cand_idx = gap_mask.nonzero(as_tuple=True)[0]
            oracle_target = oracle_readmit & gap_mask
            stale_target = stale_readmit & gap_mask
            hit = oracle_target & stale_target

            oracle_n = int(oracle_target.sum().item())
            stale_n = int(stale_target.sum().item())
            hit_n = int(hit.sum().item())
            cand_n = int(cand_idx.numel())

            gap_recall = hit_n / oracle_n if oracle_n > 0 else None
            gap_precision = hit_n / stale_n if stale_n > 0 else None
            gap_rho = _routing_agreement_spearman(
                stale_scores[b, cand_idx],
                oracle_scores[b, cand_idx],
            )

            gap_stats[gap_name] = {
                "candidate_n": cand_n,
                "oracle_readmit_n": oracle_n,
                "stale_readmit_n": stale_n,
                "hit_n": hit_n,
                "recall": gap_recall,
                "precision": gap_precision,
                "spearman": gap_rho,
            }

        rows.append({
            "sample_id": int(ctx._diag_sample_id),
            "layer": int(layer_idx),
            "batch": int(b),
            "num_visual": int(V),
            "k": int(k),
            "num_deferred": int(deferred_idx.numel()),
            "topk_overlap_count": topk_overlap_count,
            "topk_overlap": topk_overlap_count / float(k),
            "oracle_readmit_n": oracle_readmit_count,
            "stale_readmit_n": stale_readmit_count,
            "readmit_hit_n": readmit_hit_count,
            "readmit_recall": recall,
            "readmit_precision": precision,
            "deferred_spearman": deferred_rho,
            "gap_stats": gap_stats,
        })

    if rows:
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # Update actual-trajectory state AFTER measuring the current decision.
    new_last = last.clone()
    new_last[selected_mask] = int(layer_idx)
    ctx._diag_routing_last_active_layer = new_last.detach()
    ctx._diag_routing_prev_selected_mask = selected_mask.detach().clone()


# =====================================================================
# Hidden-state alignment diagnostic helpers
# =====================================================================


def _alignment_metric_pair(x: Tensor, ref: Tensor) -> tuple[Tensor, Tensor]:
    """Per-token cosine similarity and normalized L2 distance."""
    xf = x.float()
    rf = ref.float()
    cos = torch.nn.functional.cosine_similarity(xf, rf, dim=-1, eps=1e-8)
    nl2 = (xf - rf).norm(dim=-1) / rf.norm(dim=-1).clamp_min(1e-8)
    return cos, nl2


def _alignment_prepare_event(
    ctx: RoutingContext,
    hidden_states: Tensor,
    selected_mask: Tensor,
    visual_token_range: tuple[int, int],
    layer_idx: int,
    oracle_input: Optional[Tensor],
):
    """Capture stale and full-depth states for newly re-admitted tokens."""
    if not getattr(ctx, "diag_alignment", False) or oracle_input is None:
        return None

    vis_start, vis_end = visual_token_range
    h_vis = hidden_states[:, vis_start:vis_end, :]
    oracle_vis = oracle_input[:, vis_start:vis_end, :]
    B, V, _ = h_vis.shape

    prev = ctx._diag_alignment_prev_selected_mask
    last = ctx._diag_alignment_last_active_layer
    if (
        prev is None
        or prev.shape != selected_mask.shape
        or last is None
        or last.shape != selected_mask.shape
    ):
        ctx._diag_alignment_prev_selected_mask = selected_mask.detach().clone()
        ctx._diag_alignment_last_active_layer = torch.full(
            (B, V), int(layer_idx), device=selected_mask.device, dtype=torch.long
        )
        return None

    prev = prev.to(device=selected_mask.device)
    last = last.to(device=selected_mask.device)
    readmit_mask = selected_mask & (~prev)
    layer_gap = int(layer_idx) - last

    event = None
    if readmit_mask.any():
        entries = []
        for b in range(B):
            idx = readmit_mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            entries.append({
                "batch": int(b),
                "idx": idx.detach().clone(),
                "gap": layer_gap[b, idx].detach().clone(),
                "stale": h_vis[b, idx].detach().clone(),
                "oracle": oracle_vis[b, idx].detach().clone(),
            })
        if entries:
            event = {"layer": int(layer_idx), "entries": entries}

    new_last = last.clone()
    new_last[selected_mask] = int(layer_idx)
    ctx._diag_alignment_last_active_layer = new_last.detach()
    ctx._diag_alignment_prev_selected_mask = selected_mask.detach().clone()
    return event


def _alignment_log_event(
    ctx: RoutingContext,
    event,
    hidden_states_after: Tensor,
    visual_token_range: tuple[int, int],
) -> None:
    """Log alignment metrics and optionally dump hidden vectors.

    Scalar metrics are always written when REROUTE_ALIGNMENT_DIAG=1.
    PCA/UMAP vectors are additionally written only when
    REROUTE_ALIGNMENT_VECTOR_DIR is non-empty.

    For vector dumping, only events whose hidden state actually changed
    (stale != corrected) are saved. This excludes re-admitted tokens for
    which LTC was skipped because the required reference snapshot was no
    longer available (e.g., long-gap events under snapshot_keep=8).
    """
    if event is None or not getattr(ctx, "diag_alignment", False):
        return

    vis_start, vis_end = visual_token_range
    after_vis = hidden_states_after[:, vis_start:vis_end, :]

    out_path = ctx.diag_alignment_out_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    method = str(getattr(ctx, "diag_alignment_method", "auto"))
    if method == "auto":
        method = str(getattr(ctx, "correction_mode", "none"))

    vector_dir = str(getattr(ctx, "diag_alignment_vector_dir", "")).strip()
    vector_max = int(getattr(ctx, "diag_alignment_vector_max", 0))

    rows = []
    vector_records = []

    for entry in event["entries"]:
        b = entry["batch"]
        idx = entry["idx"]
        stale = entry["stale"]
        oracle = entry["oracle"]
        after = after_vis[b, idx].detach()

        stale_cos, stale_nl2 = _alignment_metric_pair(stale, oracle)
        after_cos, after_nl2 = _alignment_metric_pair(after, oracle)
        stale_norm = stale.float().norm(dim=-1)
        after_norm = after.float().norm(dim=-1)
        oracle_norm = oracle.float().norm(dim=-1)

        # Diagnostic-only: record whether LTC actually changed each re-admitted
        # token. This lets the gap analysis report correction coverage directly
        # (important for distinguishing cache eviction from approximation error).
        changed_mask = ~torch.isclose(
            stale,
            after,
            atol=1e-6,
            rtol=1e-5,
        ).all(dim=-1)

        for j in range(idx.numel()):
            rows.append({
                "sample_id": int(ctx._diag_sample_id),
                "method": method,
                "layer": int(event["layer"]),
                "batch": int(b),
                "visual_token": int(idx[j].item()),
                "layer_gap": int(entry["gap"][j].item()),
                "stale_cos": float(stale_cos[j].item()),
                "stale_norm_l2": float(stale_nl2[j].item()),
                "corrected_cos": float(after_cos[j].item()),
                "corrected_norm_l2": float(after_nl2[j].item()),
                "stale_norm": float(stale_norm[j].item()),
                "corrected_norm": float(after_norm[j].item()),
                "oracle_norm": float(oracle_norm[j].item()),
                "state_changed": bool(changed_mask[j].item()),
                "snapshot_policy": str(
                    getattr(ctx, "correction_snapshot_policy", "fifo")
                ),
            })

            # ----------------------------------------------------------
            # Optional PCA/UMAP vector dump.
            # ----------------------------------------------------------
            if (
                changed_mask is not None
                and ctx._diag_alignment_vector_count < vector_max
                and bool(changed_mask[j].item())
            ):
                vector_records.append({
                    "sample_id": int(ctx._diag_sample_id),
                    "method": method,
                    "layer": int(event["layer"]),
                    "batch": int(b),
                    "visual_token": int(idx[j].item()),
                    "layer_gap": int(entry["gap"][j].item()),
                    # fp16 CPU tensors keep diagnostic storage small.
                    "stale": stale[j].detach().to(
                        device="cpu", dtype=torch.float16
                    ),
                    "corrected": after[j].detach().to(
                        device="cpu", dtype=torch.float16
                    ),
                    "oracle": oracle[j].detach().to(
                        device="cpu", dtype=torch.float16
                    ),
                })
                ctx._diag_alignment_vector_count += 1

    # Existing scalar JSONL output.
    if rows:
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        ctx._diag_alignment_num_events += len(rows)

    # Save one shard per sample x routing layer. This avoids repeatedly
    # rewriting one large .pt file and makes layer-wise PCA easy to filter.
    if vector_records:
        os.makedirs(vector_dir, exist_ok=True)
        shard_path = os.path.join(
            vector_dir,
            (
                f"sample_{int(ctx._diag_sample_id):06d}"
                f"_layer_{int(event['layer']):02d}.pt"
            ),
        )
        torch.save(
            {
                "sample_id": int(ctx._diag_sample_id),
                "layer": int(event["layer"]),
                "method": method,
                "num_records": len(vector_records),
                "records": vector_records,
            },
            shard_path,
        )



# =====================================================================
# Local / global correction helpers for stale re-admitted tokens
# =====================================================================


def _refresh_correction_env(ctx: RoutingContext) -> None:
    """Refresh correction hyperparameters from environment variables.

    Defaults are deliberately conservative and training-free.

    Main modes:
      local            : h <- (1-w) h + w local_state
      local_transition : h <- h + gamma_p * local_depth_transition
    """
    ctx.correction_mode = os.environ.get("REROUTE_CORRECTION_MODE", "none")
    ctx.correction_topm = int(os.environ.get("REROUTE_CORRECTION_TOPM", "8"))
    ctx.correction_spatial_lambda = float(
        os.environ.get("REROUTE_CORRECTION_SPATIAL_LAMBDA", "0.15")
    )
    ctx.correction_tau_l = float(os.environ.get("REROUTE_CORRECTION_TAU_L", "0.07"))
    ctx.correction_anchor_mode = os.environ.get(
        "REROUTE_CORRECTION_ANCHOR_MODE",
        "semantic_spatial",
    ).lower()
    ctx.correction_tau_g = float(os.environ.get("REROUTE_CORRECTION_TAU_G", "0.07"))
    ctx.correction_max_anchors = int(
        os.environ.get("REROUTE_CORRECTION_MAX_ANCHORS", "3")
    )
    ctx.correction_min_layer_gap = int(
        os.environ.get("REROUTE_CORRECTION_MIN_LAYER_GAP", "1")
    )

    # LTC historical-state cache policy.
    #
    # Default is the original FIFO cache. If SNAPSHOT_KEEP is not supplied,
    # preserve the previous code's effective default:
    #   max(correction_max_anchors + 4, 8).
    ctx.correction_snapshot_policy = os.environ.get(
        "REROUTE_CORRECTION_SNAPSHOT_POLICY",
        "fifo",
    ).strip().lower()

    if ctx.correction_snapshot_policy not in ("fifo", "live", "full"):
        raise ValueError(
            "Unknown REROUTE_CORRECTION_SNAPSHOT_POLICY="
            f"{ctx.correction_snapshot_policy!r}. "
            "Expected one of: fifo, live, full"
        )

    snapshot_keep_env = os.environ.get("REROUTE_CORRECTION_SNAPSHOT_KEEP")
    if snapshot_keep_env is None:
        ctx.correction_snapshot_keep = max(
            int(ctx.correction_max_anchors) + 4,
            8,
        )
    else:
        ctx.correction_snapshot_keep = max(int(snapshot_keep_env), 1)

    snapshot_debug = os.environ.get(
        "REROUTE_CORRECTION_SNAPSHOT_DEBUG",
        "0",
    )
    ctx.correction_snapshot_debug = snapshot_debug.lower() not in (
        "0", "false", "no", "off"
    )

    mode = ctx.correction_mode
    default_lw = {
        "none": 0.0,
        "local": 0.2,
        "global": 0.0,
        "local_global": 0.1,
        "local_transition": 0.0,
    }.get(mode, 0.0)
    default_gw = {
        "none": 0.0,
        "local": 0.0,
        "global": 0.2,
        "local_global": 0.2,
        "local_transition": 0.0,
    }.get(mode, 0.0)

    ctx.correction_local_weight = float(
        os.environ.get("REROUTE_CORRECTION_LOCAL_WEIGHT", str(default_lw))
    )
    ctx.correction_global_weight = float(
        os.environ.get("REROUTE_CORRECTION_GLOBAL_WEIGHT", str(default_gw))
    )

    # For local_transition, gamma_max is the maximum residual-transition
    # strength. If REROUTE_CORRECTION_GAMMA_MAX is not provided, fall back to
    # REROUTE_CORRECTION_LOCAL_WEIGHT when it is set, so existing sweep scripts
    # can be reused.
    if "REROUTE_CORRECTION_GAMMA_MAX" in os.environ:
        gamma_default = os.environ["REROUTE_CORRECTION_GAMMA_MAX"]
    elif "REROUTE_CORRECTION_LOCAL_WEIGHT" in os.environ:
        gamma_default = os.environ["REROUTE_CORRECTION_LOCAL_WEIGHT"]
    else:
        gamma_default = "0.8" if mode in ("query_local_transition", "qver") else "0.3"
    ctx.correction_gamma_max = float(gamma_default)

    norm_flag = os.environ.get("REROUTE_CORRECTION_NORM_PRESERVE", "1")
    ctx.correction_norm_preserve = norm_flag.lower() not in ("0", "false", "no", "off")

    ctx.correction_norm_target = os.environ.get(
        "REROUTE_CORRECTION_NORM_TARGET",
        "cached",
    ).strip().lower()

    # Backward compatibility: disabling norm preservation always means no
    # post-correction rescaling, regardless of the selected target mode.
    if not ctx.correction_norm_preserve:
        ctx.correction_norm_target = "none"

    valid_norm_targets = {
        "cached",
        "current_anchor",
        "transition_ratio",
        "none",
    }
    if ctx.correction_norm_target not in valid_norm_targets:
        raise ValueError(
            "Unknown REROUTE_CORRECTION_NORM_TARGET="
            f"{ctx.correction_norm_target!r}. "
            "Expected one of: cached, current_anchor, transition_ratio, none"
        )

    # Query-relevant local transition correction.  When mode is
    # query_local_transition/qver, default to eta=0.20.  Plain
    # local_transition keeps eta=0.0, exactly recovering the existing LTC.
    eta_default = "0.2" if mode in ("query_local_transition", "qver") else "0.0"
    ctx.correction_query_eta = float(
        os.environ.get("REROUTE_CORRECTION_QUERY_ETA", eta_default)
    )
    ctx.correction_query_norm = os.environ.get(
        "REROUTE_CORRECTION_QUERY_NORM", "minmax"
    ).lower()


def _correction_enabled(ctx: RoutingContext) -> bool:
    return ctx.correction_mode in (
        "local", "global", "local_global",
        "local_transition", "query_local_transition", "qver",
    )


def _safe_l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _normalize_query_relevance(scores: Optional[Tensor], mode: str = "minmax") -> Optional[Tensor]:
    """Normalize per-token query relevance for query-relevant LTC.

    Router scores can be attention probabilities or hidden-state similarities,
    depending on the router.  This helper maps them to a stable scale before
    adding them to the neighbor score.  Non-finite values are ignored.
    """
    if scores is None:
        return None

    s = scores.float()
    finite = torch.isfinite(s)
    if not finite.any():
        return torch.zeros_like(s)

    s_safe = torch.where(finite, s, torch.zeros_like(s))
    mode = (mode or "none").lower()

    if mode == "none":
        return s_safe

    if mode == "zscore":
        count = finite.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = s_safe.sum(dim=-1, keepdim=True) / count
        var = (((s_safe - mean) ** 2) * finite.float()).sum(dim=-1, keepdim=True) / count
        return torch.where(finite, (s_safe - mean) / var.sqrt().clamp_min(1e-6), torch.zeros_like(s))

    # Default: min-max normalize to [0, 1].
    pos_inf = torch.full_like(s, float("inf"))
    neg_inf = torch.full_like(s, float("-inf"))
    s_min = torch.where(finite, s, pos_inf).amin(dim=-1, keepdim=True)
    s_max = torch.where(finite, s, neg_inf).amax(dim=-1, keepdim=True)
    denom = (s_max - s_min).clamp_min(1e-6)
    out = (s - s_min) / denom
    return torch.where(finite, out, torch.zeros_like(s))


def _visual_token_coords(num_visual: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return normalized 2D coordinates for visual-token indices.

    For square grids such as LLaVA-1.5 24x24, coordinates are arranged on a
    square. For non-square visual sequences, fall back to a 1D coordinate line.
    """
    side = int(math.sqrt(num_visual))
    if side * side == num_visual:
        ys = torch.arange(side, device=device, dtype=dtype)
        xs = torch.arange(side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        denom = max(side - 1, 1)
        return coords / float(denom)

    pos = torch.arange(num_visual, device=device, dtype=dtype)
    denom = max(num_visual - 1, 1)
    return torch.stack([pos / float(denom), torch.zeros_like(pos)], dim=-1)


def _make_stage_anchor(
    hidden_states: Tensor,
    selected_mask: Tensor,
    prev_selected_mask: Optional[Tensor],
    visual_token_range: tuple[int, int],
) -> Tensor:
    """Build a global visual anchor from current-depth selected tokens.

    Prefer tokens that are selected in both the current and previous decision,
    because they are more likely to be fresh/current-depth. If that set is
    empty, fall back to all selected visual tokens.
    """
    vis_start, vis_end = visual_token_range
    h_vis = hidden_states[:, vis_start:vis_end, :]
    B, V, D = h_vis.shape

    if prev_selected_mask is not None and prev_selected_mask.shape == selected_mask.shape:
        source_mask = selected_mask & prev_selected_mask
    else:
        source_mask = selected_mask

    anchors = []
    for b in range(B):
        mask_b = source_mask[b]
        if not mask_b.any():
            mask_b = selected_mask[b]
        if not mask_b.any():
            anchors.append(h_vis[b].mean(dim=0))
        else:
            anchors.append(h_vis[b, mask_b].mean(dim=0))
    return torch.stack(anchors, dim=0).detach()


def _append_depth_anchor(ctx: RoutingContext, anchor: Tensor) -> None:
    ctx._correction_depth_anchors.append(anchor)
    max_anchors = max(int(ctx.correction_max_anchors), 1)
    if len(ctx._correction_depth_anchors) > max_anchors:
        ctx._correction_depth_anchors = ctx._correction_depth_anchors[-max_anchors:]


def _compute_local_correction_for_batch(
    h_vis_b: Tensor,
    readmit_mask_b: Tensor,
    source_mask_b: Tensor,
    coords: Tensor,
    topm: int,
    spatial_lambda: float,
    tau: float,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Compute local correction vectors for one batch element.

    Returns:
        readmit_idx: (R,) visual indices to correct
        local_vecs:  (R,D) local correction vectors
    """
    readmit_idx = readmit_mask_b.nonzero(as_tuple=True)[0]
    if readmit_idx.numel() == 0:
        return None, None

    h_norm = _safe_l2_normalize(h_vis_b.float(), dim=-1)
    local_vecs = []
    valid_idx = []

    for p_idx in readmit_idx.tolist():
        cand_mask = source_mask_b.clone()
        cand_mask[p_idx] = False
        cand_idx = cand_mask.nonzero(as_tuple=True)[0]
        if cand_idx.numel() == 0:
            continue

        sim = torch.matmul(h_norm[cand_idx], h_norm[p_idx])
        dist = torch.norm(coords[cand_idx] - coords[p_idx], dim=-1)
        score = sim - float(spatial_lambda) * dist

        k = min(int(topm), cand_idx.numel())
        top_score, top_pos = torch.topk(score, k=k, largest=True)
        top_idx = cand_idx[top_pos]
        weight = torch.softmax(top_score / max(float(tau), 1e-6), dim=0).to(h_vis_b.dtype)
        local = (weight.unsqueeze(-1) * h_vis_b[top_idx]).sum(dim=0)
        local_vecs.append(local)
        valid_idx.append(p_idx)

    if not local_vecs:
        return None, None

    return (
        torch.tensor(valid_idx, device=h_vis_b.device, dtype=torch.long),
        torch.stack(local_vecs, dim=0),
    )


def _compute_local_transition_correction_for_batch(
    h_vis_b: Tensor,
    readmit_mask_b: Tensor,
    source_mask_b: Tensor,
    coords: Tensor,
    layer_gap_b: Tensor,
    last_active_b: Tensor,
    layer_snapshots: dict[int, Tensor],
    batch_idx: int,
    topm: int,
    spatial_lambda: float,
    tau: float,
    gamma_max: float,
    norm_preserve: bool,
    norm_target: str = "cached",
    anchor_mode: str = "semantic_spatial",
    query_relevance_b: Optional[Tensor] = None,
    query_eta: float = 0.0,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Identity-preserving local transition correction for one batch.

    For a re-admitted token p whose last active layer is ell_p, estimate the
    missing update from nearby fresh tokens:
        h_p <- h_p + gamma_p * sum_q alpha_pq * (h_q^ell - h_q^ell_p)

    This transfers local depth-transition residuals instead of directly mixing
    neighboring current-depth states, preserving the identity of p.

    ``norm_target`` changes only the post-correction magnitude scaling; it does
    not alter routing, re-admission, anchor selection, or transition estimation.
    The default ``cached`` exactly preserves the previous implementation.
    """
    readmit_idx = readmit_mask_b.nonzero(as_tuple=True)[0]
    if readmit_idx.numel() == 0:
        return None, None

    corrected_vecs = []
    valid_idx = []

    for p_idx in readmit_idx.tolist():
        ref_layer = int(last_active_b[p_idx].item())
        ref_snapshot = layer_snapshots.get(ref_layer)
        if ref_snapshot is None:
            continue

        # h_p^{ell_p}. In compact Reroute, the current state of a re-admitted
        # token is typically the stale state; using the stored reference keeps
        # the notation explicit and avoids accidental drift.
        p_ref = ref_snapshot[batch_idx, p_idx].to(
            device=h_vis_b.device,
            dtype=h_vis_b.dtype,
        )

        cand_mask = source_mask_b.clone()
        cand_mask[p_idx] = False
        cand_idx = cand_mask.nonzero(as_tuple=True)[0]
        if cand_idx.numel() == 0:
            continue

        q_ref = ref_snapshot[batch_idx, cand_idx].to(
            device=h_vis_b.device,
            dtype=h_vis_b.dtype,
        )
        p_ref_norm = _safe_l2_normalize(p_ref.float().unsqueeze(0), dim=-1)[0]
        q_ref_norm = _safe_l2_normalize(q_ref.float(), dim=-1)
        sim = torch.matmul(q_ref_norm, p_ref_norm)
        dist = torch.norm(coords[cand_idx] - coords[p_idx], dim=-1)

        k = min(int(topm), cand_idx.numel())
        mode = str(anchor_mode).lower()

        # ------------------------------------------------------------
        # Random fresh anchors
        #
        # The candidate pool remains exactly the same fresh-source set.
        # Only anchor selection is randomized. Uniform weights keep this
        # as a pure random-anchor control.
        # ------------------------------------------------------------
        if mode == "random":
            perm = torch.randperm(
                cand_idx.numel(),
                device=cand_idx.device,
            )[:k]
            top_idx = cand_idx[perm]

            weight = torch.full(
                (k,),
                1.0 / float(k),
                device=h_vis_b.device,
                dtype=h_vis_b.dtype,
            )

        # ------------------------------------------------------------
        # Spatial-only anchors
        #
        # Smaller normalized grid distance is better.
        # ------------------------------------------------------------
        elif mode == "spatial":
            score = -dist

            top_score, top_pos = torch.topk(
                score,
                k=k,
                largest=True,
            )
            top_idx = cand_idx[top_pos]

            weight = torch.softmax(
                top_score / max(float(tau), 1e-6),
                dim=0,
            ).to(h_vis_b.dtype)

        # ------------------------------------------------------------
        # Semantic-only anchors
        #
        # Cosine similarity between p and candidate q at p's reference
        # depth.
        # ------------------------------------------------------------
        elif mode == "semantic":
            score = sim

            top_score, top_pos = torch.topk(
                score,
                k=k,
                largest=True,
            )
            top_idx = cand_idx[top_pos]

            weight = torch.softmax(
                top_score / max(float(tau), 1e-6),
                dim=0,
            ).to(h_vis_b.dtype)

        # ------------------------------------------------------------
        # Semantic + Spatial — final Q-VER + LTC
        # ------------------------------------------------------------
        elif mode in ("semantic_spatial", "semantic+spatial"):
            score = sim - float(spatial_lambda) * dist

            if query_relevance_b is not None and float(query_eta) != 0.0:
                score = score + float(query_eta) * query_relevance_b[cand_idx].to(
                    device=score.device,
                    dtype=score.dtype,
                )

            top_score, top_pos = torch.topk(
                score,
                k=k,
                largest=True,
            )
            top_idx = cand_idx[top_pos]

            weight = torch.softmax(
                top_score / max(float(tau), 1e-6),
                dim=0,
            ).to(h_vis_b.dtype)

        else:
            raise ValueError(
                f"Unknown REROUTE_CORRECTION_ANCHOR_MODE={anchor_mode!r}. "
                "Expected one of: random, spatial, semantic, semantic_spatial"
            )

        top_ref = ref_snapshot[batch_idx, top_idx].to(
            device=h_vis_b.device,
            dtype=h_vis_b.dtype,
        )
        transition = h_vis_b[top_idx] - top_ref
        local_delta = (weight.unsqueeze(-1) * transition).sum(dim=0)

        gamma = float(gamma_max)
        base = h_vis_b[p_idx]
        corrected = base + gamma * local_delta.to(dtype=base.dtype)

        # ------------------------------------------------------------
        # Post-LTC norm scaling.
        #
        # cached:
        #   Preserve the re-admitted token's own cached/base norm.
        #
        # current_anchor:
        #   Use the SAME LTC anchors and weights selected above, and set
        #   the target norm to their weighted CURRENT-layer norm:
        #
        #       n_p^cur = sum_q w_pq ||h_q[ell_s]||_2
        #
        # transition_ratio:
        #   Preserve p's cached norm while transferring the anchors'
        #   weighted multiplicative norm change from the reference layer.
        #
        # none:
        #   Leave the corrected vector unscaled.
        # ------------------------------------------------------------
        target_mode = str(norm_target).strip().lower()

        if norm_preserve and target_mode != "none":
            corr_norm = corrected.float().norm().clamp_min(1e-6)

            if target_mode == "cached":
                target_norm = base.float().norm().clamp_min(1e-6)

            elif target_mode == "current_anchor":
                current_anchor_norms = (
                    h_vis_b[top_idx]
                    .float()
                    .norm(dim=-1)
                    .clamp_min(1e-6)
                )
                target_norm = (
                    weight.float() * current_anchor_norms
                ).sum().clamp_min(1e-6)

            elif target_mode == "transition_ratio":
                base_norm = base.float().norm().clamp_min(1e-6)
                current_anchor_norms = (
                    h_vis_b[top_idx]
                    .float()
                    .norm(dim=-1)
                    .clamp_min(1e-6)
                )
                reference_anchor_norms = (
                    top_ref
                    .float()
                    .norm(dim=-1)
                    .clamp_min(1e-6)
                )
                weighted_ratio = (
                    weight.float()
                    * (current_anchor_norms / reference_anchor_norms)
                ).sum().clamp_min(1e-6)
                target_norm = base_norm * weighted_ratio

            else:
                raise ValueError(
                    f"Unknown LTC norm target: {norm_target!r}. "
                    "Expected one of: cached, current_anchor, "
                    "transition_ratio, none"
                )

            corrected = corrected * (
                target_norm / corr_norm
            ).to(dtype=corrected.dtype)

        corrected_vecs.append(corrected)
        valid_idx.append(p_idx)

    if not corrected_vecs:
        return None, None

    return (
        torch.tensor(valid_idx, device=h_vis_b.device, dtype=torch.long),
        torch.stack(corrected_vecs, dim=0),
    )


def _update_correction_snapshot_stats(ctx: RoutingContext) -> None:
    """Update lightweight LTC snapshot-cache statistics."""
    snapshots = ctx._correction_layer_snapshots
    current_count = len(snapshots)
    current_bytes = sum(
        int(t.numel()) * int(t.element_size())
        for t in snapshots.values()
    )

    ctx._correction_snapshot_current_count = current_count
    ctx._correction_snapshot_current_bytes = current_bytes
    ctx._correction_snapshot_peak_count = max(
        int(ctx._correction_snapshot_peak_count),
        current_count,
    )
    ctx._correction_snapshot_peak_bytes = max(
        int(ctx._correction_snapshot_peak_bytes),
        current_bytes,
    )


def _store_correction_snapshot_and_last_active(
    ctx: RoutingContext,
    layer_idx: int,
    h_vis: Tensor,
    selected_mask: Tensor,
) -> None:
    """Store LTC reference state and update per-token last-active depths.

    Cache policies
    --------------
    fifo:
        Original behavior. Keep only the most recent N full visual snapshots
        (N=8 by default).

    live:
        Keep the current snapshot plus only historical snapshots that are
        still referenced by currently deferred tokens through
        ``_correction_last_active_layer``. A long-deferred token therefore
        keeps its required reference snapshot alive regardless of gap length,
        while snapshots that can no longer be queried by any deferred token
        are immediately released.

        Importantly, each retained entry is still a *full* visual snapshot.
        Thus LTC anchor selection and transition computation are numerically
        equivalent to an untruncated/full-history cache for every live
        reference depth; only provably dead depths are evicted.

    full:
        Keep all snapshots. Useful as a diagnostic control (e.g. the
        32-snapshot/full-history experiment).
    """
    layer_key = int(layer_idx)
    B, V, _ = h_vis.shape

    # --------------------------------------------------------------
    # 1) Initialize/update last-active depth first.
    #
    # Selected tokens are current at this depth. Deferred tokens keep
    # their previous last-active depth, which is exactly the reference
    # snapshot LTC will need if they are re-admitted later.
    # --------------------------------------------------------------
    if (
        ctx._correction_last_active_layer is None
        or ctx._correction_last_active_layer.shape != selected_mask.shape
    ):
        ctx._correction_last_active_layer = torch.full(
            (B, V),
            layer_key,
            device=selected_mask.device,
            dtype=torch.long,
        )

    last = ctx._correction_last_active_layer.clone()
    last[selected_mask] = layer_key

    # --------------------------------------------------------------
    # 2) Apply cache policy.
    # --------------------------------------------------------------
    policy = str(
        getattr(ctx, "correction_snapshot_policy", "fifo")
    ).lower()

    if policy == "live":
        # Determine which *old* reference depths will remain reachable
        # after the last-active update above.
        #
        # Keep layer_key as well, because currently selected tokens can
        # become deferred at the next routing decision and will then need
        # the current-depth snapshot as their reference.
        required_layers = {layer_key}
        deferred_mask = ~selected_mask

        if deferred_mask.any():
            live_ref_layers = torch.unique(last[deferred_mask])
            required_layers.update(
                int(x)
                for x in live_ref_layers.detach().cpu().tolist()
            )

        # Evict dead snapshots BEFORE cloning the current one. This reduces
        # transient peak memory as well as steady-state memory.
        for key in list(ctx._correction_layer_snapshots.keys()):
            if int(key) not in required_layers:
                del ctx._correction_layer_snapshots[key]

        # The current full visual snapshot is required as a possible future
        # reference for tokens that are active now.
        ctx._correction_layer_snapshots[layer_key] = h_vis.detach().clone()

    elif policy == "fifo":
        # Original behavior: store current snapshot, then trim oldest entries.
        ctx._correction_layer_snapshots[layer_key] = h_vis.detach().clone()

        max_keep = max(
            int(getattr(ctx, "correction_snapshot_keep", 8)),
            1,
        )
        if len(ctx._correction_layer_snapshots) > max_keep:
            for key in sorted(ctx._correction_layer_snapshots.keys())[:-max_keep]:
                del ctx._correction_layer_snapshots[key]

    elif policy == "full":
        # Diagnostic/full-history control: never evict.
        ctx._correction_layer_snapshots[layer_key] = h_vis.detach().clone()

    else:
        # _refresh_correction_env validates this, but keep a local guard for
        # callers that construct/mutate RoutingContext manually.
        raise ValueError(
            f"Unknown correction_snapshot_policy={policy!r}. "
            "Expected one of: fifo, live, full"
        )

    # Publish updated routing/reference state only after cache maintenance.
    ctx._correction_last_active_layer = last.detach()
    ctx._correction_prev_selected_mask = selected_mask.detach().clone()

    _update_correction_snapshot_stats(ctx)

    if bool(getattr(ctx, "correction_snapshot_debug", False)):
        keys = sorted(int(k) for k in ctx._correction_layer_snapshots.keys())
        mib = float(ctx._correction_snapshot_current_bytes) / (1024.0 ** 2)
        peak_mib = float(ctx._correction_snapshot_peak_bytes) / (1024.0 ** 2)
        print(
            "[LTC-cache] "
            f"policy={policy} layer={layer_key} "
            f"kept={keys} count={len(keys)} "
            f"memory={mib:.2f}MiB peak={peak_mib:.2f}MiB",
            flush=True,
        )


def _compute_global_correction_for_batch(
    h_vis_b: Tensor,
    readmit_idx: Tensor,
    anchors_b: Tensor,
    tau: float,
) -> Optional[Tensor]:
    """Compute global depth-anchor correction for one batch element."""
    if readmit_idx is None or readmit_idx.numel() == 0 or anchors_b.numel() == 0:
        return None

    p = h_vis_b[readmit_idx]
    p_norm = _safe_l2_normalize(p.float(), dim=-1)
    a_norm = _safe_l2_normalize(anchors_b.float(), dim=-1)
    score = torch.matmul(p_norm, a_norm.transpose(0, 1))
    weight = torch.softmax(score / max(float(tau), 1e-6), dim=-1).to(h_vis_b.dtype)
    return torch.matmul(weight, anchors_b.to(h_vis_b.dtype))


def _apply_readmitted_correction(
    ctx: RoutingContext,
    hidden_states: Tensor,
    selected_mask: Tensor,
    visual_token_range: tuple[int, int],
    layer_idx: int,
    scores: Optional[Tensor] = None,
) -> Tensor:
    """Apply correction only to re-admitted visual tokens.

    Routing decisions and active-token budget are unchanged. Correction modes:
      - local/global/local_global: previous state-mixing variants.
      - local_transition: identity-preserving local depth-transition residual.
    """
    vis_start, vis_end = visual_token_range
    h_vis = hidden_states[:, vis_start:vis_end, :]
    B, V, D = h_vis.shape

    # If correction is off, keep lightweight tracking disabled.
    if not _correction_enabled(ctx):
        ctx._correction_prev_selected_mask = selected_mask.detach().clone()
        return hidden_states

    prev_mask = ctx._correction_prev_selected_mask

    # Always create/update the current global anchor at every decision layer,
    # so later re-admissions can attend to early/middle/current summaries.
    current_anchor = _make_stage_anchor(
        hidden_states=hidden_states,
        selected_mask=selected_mask,
        prev_selected_mask=prev_mask,
        visual_token_range=visual_token_range,
    )
    _append_depth_anchor(ctx, current_anchor)

    # First routing decision: all visual tokens have been active up to this
    # point, so initialize their reference layer and snapshot.
    if prev_mask is None or prev_mask.shape != selected_mask.shape:
        _store_correction_snapshot_and_last_active(
            ctx=ctx,
            layer_idx=layer_idx,
            h_vis=h_vis,
            selected_mask=selected_mask,
        )
        return hidden_states

    raw_readmit_mask = selected_mask & (~prev_mask)

    last_active = ctx._correction_last_active_layer
    if last_active is None or last_active.shape != selected_mask.shape:
        last_active = torch.full(
            selected_mask.shape,
            int(layer_idx),
            device=selected_mask.device,
            dtype=torch.long,
        )
        ctx._correction_last_active_layer = last_active

    layer_gap = int(layer_idx) - last_active
    min_gap = max(int(ctx.correction_min_layer_gap), 0)
    readmit_mask = raw_readmit_mask & (layer_gap >= min_gap)
    ctx._correction_num_gap_filtered += int((raw_readmit_mask & (~readmit_mask)).sum().item())

    if not readmit_mask.any():
        _store_correction_snapshot_and_last_active(
            ctx=ctx,
            layer_idx=layer_idx,
            h_vis=h_vis,
            selected_mask=selected_mask,
        )
        return hidden_states

    coords = _visual_token_coords(V, h_vis.device, torch.float32)

    # Fresh source tokens: selected now and selected in the previous decision.
    # If a batch has no such tokens, fall back to selected non-re-admitted tokens.
    source_mask = selected_mask & prev_mask
    fallback_mask = selected_mask & (~raw_readmit_mask)
    for b in range(source_mask.shape[0]):
        if not source_mask[b].any():
            source_mask[b] = fallback_mask[b]

    mode = ctx.correction_mode
    new_hidden = hidden_states.clone()
    new_h_vis = new_hidden[:, vis_start:vis_end, :]
    num_corrected = 0

    query_relevance = None
    if float(ctx.correction_query_eta) != 0.0 and scores is not None:
        query_relevance = _normalize_query_relevance(
            scores.to(device=h_vis.device),
            mode=ctx.correction_query_norm,
        )

    # ------------------------------------------------------------------
    # Final method: Local Transition Correction
    # ------------------------------------------------------------------
    if mode in ("local_transition", "query_local_transition", "qver"):
        for b in range(B):
            idx, corrected = _compute_local_transition_correction_for_batch(
                h_vis_b=h_vis[b],
                readmit_mask_b=readmit_mask[b],
                source_mask_b=source_mask[b],
                coords=coords,
                layer_gap_b=layer_gap[b],
                last_active_b=last_active[b],
                layer_snapshots=ctx._correction_layer_snapshots,
                batch_idx=b,
                topm=ctx.correction_topm,
                spatial_lambda=ctx.correction_spatial_lambda,
                tau=ctx.correction_tau_l,
                gamma_max=ctx.correction_gamma_max,
                norm_preserve=ctx.correction_norm_preserve,
                norm_target=ctx.correction_norm_target,
                anchor_mode=ctx.correction_anchor_mode,
                query_relevance_b=(query_relevance[b] if query_relevance is not None else None),
                query_eta=ctx.correction_query_eta,
            )
            if idx is None:
                continue
            new_h_vis[b, idx] = corrected.to(dtype=new_h_vis.dtype)
            num_corrected += int(idx.numel())

        new_hidden[:, vis_start:vis_end, :] = new_h_vis
        ctx._correction_num_corrected += num_corrected
        _store_correction_snapshot_and_last_active(
            ctx=ctx,
            layer_idx=layer_idx,
            h_vis=new_h_vis,
            selected_mask=selected_mask,
        )
        return new_hidden

    # ------------------------------------------------------------------
    # Previous ablation modes: Local-State / Global / Local+Global
    # ------------------------------------------------------------------
    local_w = float(ctx.correction_local_weight)
    global_w = float(ctx.correction_global_weight)
    if mode == "local":
        global_w = 0.0
    elif mode == "global":
        local_w = 0.0
    elif mode == "local_global":
        pass

    # Clamp to keep a residual path to the original hidden state.
    total_w = max(local_w + global_w, 0.0)
    if total_w >= 1.0:
        scale = 0.95 / max(total_w, 1e-6)
        local_w *= scale
        global_w *= scale
    base_w = 1.0 - local_w - global_w

    anchors = torch.stack(ctx._correction_depth_anchors, dim=1)  # (B,M,D)

    for b in range(B):
        readmit_idx = readmit_mask[b].nonzero(as_tuple=True)[0]
        if readmit_idx.numel() == 0:
            continue

        local_idx = None
        local_vec = None
        if local_w > 0.0:
            local_idx, local_vec = _compute_local_correction_for_batch(
                h_vis_b=h_vis[b],
                readmit_mask_b=readmit_mask[b],
                source_mask_b=source_mask[b],
                coords=coords,
                topm=ctx.correction_topm,
                spatial_lambda=ctx.correction_spatial_lambda,
                tau=ctx.correction_tau_l,
            )

        if local_w > 0.0 and local_idx is not None:
            idx = local_idx
        elif local_w > 0.0 and local_idx is None and global_w <= 0.0:
            continue
        else:
            idx = readmit_idx

        orig = h_vis[b, idx]
        corrected = base_w * orig

        if local_w > 0.0:
            if local_idx is None:
                local_for_idx = torch.zeros_like(orig)
            elif local_idx.numel() == idx.numel() and torch.equal(local_idx, idx):
                local_for_idx = local_vec
            else:
                local_map = {int(k): v for k, v in zip(local_idx.tolist(), local_vec)}
                local_for_idx = torch.stack(
                    [local_map.get(int(k), torch.zeros_like(orig[0])) for k in idx.tolist()],
                    dim=0,
                )
            corrected = corrected + local_w * local_for_idx.to(dtype=orig.dtype)

        if global_w > 0.0:
            global_vec = _compute_global_correction_for_batch(
                h_vis_b=h_vis[b],
                readmit_idx=idx,
                anchors_b=anchors[b],
                tau=ctx.correction_tau_g,
            )
            if global_vec is not None:
                corrected = corrected + global_w * global_vec.to(dtype=orig.dtype)

        new_h_vis[b, idx] = corrected.to(dtype=new_h_vis.dtype)
        num_corrected += int(idx.numel())

    new_hidden[:, vis_start:vis_end, :] = new_h_vis
    ctx._correction_num_corrected += num_corrected
    _store_correction_snapshot_and_last_active(
        ctx=ctx,
        layer_idx=layer_idx,
        h_vis=new_h_vis,
        selected_mask=selected_mask,
    )
    return new_hidden

def _forward_compact_route(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """Apply routing via compact gather → K×K attention → scatter back.

    This is the core compact dispatch path:
      1. Gather selected + text tokens into compact sequence (length M+K)
      2. Run K×K attention on compact sequence (efficiency gain)
      3. Run FFN on compact sequence
      4. Scatter results back to full sequence
      5. Unselected tokens keep their input (residual bypass)

    Unlike mask-based routing (L×L attention with column masking), this
    actually reduces attention FLOPs from O(L²) to O((M+K)²).

    Unlike physical_delete, the full sequence is preserved — unselected
    tokens can be re-selected at later routing layers.
    """
    vis_start, vis_end = ctx.visual_token_range
    batch = hidden_states.shape[0]
    device = hidden_states.device
    seq_len = hidden_states.shape[1]

    layer_module = ctx._layer_modules[layer_idx]

    # 0. Full-token current-depth reference BEFORE this layer.
    #    Alignment mode uses it read-only and does not replace real states.
    oracle_input = None
    if _shadow_reference_enabled(ctx):
        oracle_input = _oracle_get_shadow_input(ctx, hidden_states)

    # 1. Compute routing decision
    decision = ctx.router.compute_scores(
        layer_idx=layer_idx,
        attn_weights=ctx.attn_weights_cache,
        hidden_states=hidden_states,
        visual_token_range=ctx.visual_token_range,
    )
    ctx.routing_log[layer_idx] = decision

    # 1.4 Diagnostic-only paired routing comparison:
    #     Q_current x K_stale  vs.  Q_current x K_current-depth-oracle.
    #     This NEVER changes the real routing decision.
    _routing_agreement_log(
        ctx=ctx,
        layer_idx=layer_idx,
        hidden_states=hidden_states,
        oracle_input=oracle_input,
        decision=decision,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )

    # 1.5 Capture newly re-admitted token states before correction.
    selected_mask = decision.selected_mask  # (batch, num_visual)
    _qual_trace_log(
        ctx=ctx,
        layer_idx=layer_idx,
        selected_mask=selected_mask,
    )
    alignment_event = _alignment_prepare_event(
        ctx=ctx,
        hidden_states=hidden_states,
        selected_mask=selected_mask,
        visual_token_range=ctx.visual_token_range,
        layer_idx=layer_idx,
        oracle_input=oracle_input,
    )

    # 1.6 Oracle intervention only in explicit oracle modes.
    if _oracle_enabled(ctx) and oracle_input is not None:
        hidden_states = _oracle_replace_readmitted_tokens(
            ctx=ctx,
            hidden_states=hidden_states,
            selected_mask=selected_mask,
            visual_token_range=ctx.visual_token_range,
            oracle_input=oracle_input,
        )
    else:
        ctx._oracle_prev_selected_mask = selected_mask.detach().clone()

    # 1.7 Deployable correction still runs normally during alignment logging.
    if not _oracle_enabled(ctx) and _correction_enabled(ctx):
        hidden_states = _apply_readmitted_correction(
            ctx=ctx,
            hidden_states=hidden_states,
            selected_mask=selected_mask,
            visual_token_range=ctx.visual_token_range,
            layer_idx=layer_idx,
            scores=decision.scores,
        )

    _alignment_log_event(
        ctx=ctx,
        event=alignment_event,
        hidden_states_after=hidden_states,
        visual_token_range=ctx.visual_token_range,
    )

    # 2. Build compact indices: all text + selected visual tokens
    unselected_mask = ~selected_mask

    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    unsel_relative = unselected_mask[0].nonzero(as_tuple=True)[0]
    unsel_abs = unsel_relative + vis_start
    keep_mask[unsel_abs] = False
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]  # sorted ascending

    # ─── P-MOD-EQUIVALENT KV-SAVE SEMANTICS ───────────────────────────
    # Gather the kept subset BEFORE calling the layer, so the layer's
    # internal k_proj/v_proj run on K tokens and past_key_values.update
    # writes only K K,V per routed layer. This is what makes the paper
    # §4.3 KV-cache savings real (matches the original residual-skip semantics).
    # ───────────────────────────────────────────────────────────────────

    # 3. Gather compact hidden + position_ids + position_embeddings.
    compact_hidden = hidden_states[:, kept_indices].contiguous()
    compact_pos_ids = None
    if position_ids is not None:
        compact_pos_ids = position_ids[:, kept_indices].contiguous()

    compact_pe = None
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 4:
            # Qwen MROPE: (3, batch, seq, dim) — kept for future extension.
            compact_pe = (cos[:, :, kept_indices].contiguous(),
                          sin[:, :, kept_indices].contiguous())
        else:
            # LLaMA / standard rotary: (batch, seq, dim)
            compact_pe = (cos[:, kept_indices].contiguous(),
                          sin[:, kept_indices].contiguous())

    # 4. Run the decoder layer on compact (B, K, d).
    #    The layer's internal self_attn does k_proj/v_proj(compact_hidden)
    #    → past_key_values.update writes K K,V (true KV save).
    #    Attention is K×K via sdpa's causal handling.
    compact_out = original_forward(
        compact_hidden,
        attention_mask=None,
        position_ids=compact_pos_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=compact_pe,
        **kwargs,
    )

    # 5. Scatter compact output back to full L; deferred = input (residual).
    if isinstance(compact_out, tuple):
        h_compact = compact_out[0]
        output = hidden_states.clone()
        output[:, kept_indices] = h_compact

        # Keep the diagnostic full-token reference synchronized for the next
        # routed layer. This is expensive by design and used only when
        # REROUTE_ORACLE_MODE != none.
        if oracle_input is not None:
            _oracle_advance_shadow(
                original_forward=original_forward,
                ctx=ctx,
                oracle_input=oracle_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return (output,) + compact_out[1:]

    output = hidden_states.clone()
    output[:, kept_indices] = compact_out

    if oracle_input is not None:
        _oracle_advance_shadow(
            original_forward=original_forward,
            ctx=ctx,
            oracle_input=oracle_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    return output


def _find_visual_token_range_llava(
    model,
    input_ids: Tensor,
    pixel_values: Optional[Tensor] = None,
) -> tuple[int, int]:
    """Find visual token positions for LLaVA-1.5.

    In LLaVA-1.5 with HF transformers, image tokens are marked by a
    special token (typically image_token_index = 32000). After the model
    merges image features, the visual tokens replace these placeholders.

    For LLaVA-1.5 with 336px input and CLIP-ViT-L/14:
    - 576 visual tokens (24x24 patches)
    """
    if pixel_values is None:
        return (0, 0)

    image_token_id = model.config.image_token_index
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(image_positions) == 0:
        return (0, 0)

    vis_start = image_positions[0].item()
    num_patches = pixel_values.shape[-1] // model.config.vision_config.patch_size
    num_visual = num_patches * num_patches
    vis_end = vis_start + num_visual

    return (vis_start, vis_end)


def _find_visual_token_range_qwen25vl(
    model,
    input_ids: Tensor,
    **kwargs,
) -> tuple[int, int]:
    """Find visual token positions for Qwen2.5-VL.

    Qwen2.5-VL uses <|vision_start|> and <|vision_end|> special tokens
    to mark the visual region in the input sequence.
    """
    config = model.config
    vision_start_id = config.vision_start_token_id
    vision_end_id = config.vision_end_token_id

    ids = input_ids[0]
    start_positions = (ids == vision_start_id).nonzero(as_tuple=True)[0]
    end_positions = (ids == vision_end_id).nonzero(as_tuple=True)[0]

    if len(start_positions) == 0 or len(end_positions) == 0:
        return (0, 0)

    vis_start = start_positions[0].item() + 1
    vis_end = end_positions[0].item()

    return (vis_start, vis_end)


# =====================================================================
# Main patching API
# =====================================================================


def patch_model_for_routing(
    model,
    router: TokenRouter,
    dispatcher: TokenDispatcher,
    action: str,
    model_family: str = "llava",
) -> RoutingContext:
    """Patch a HuggingFace VLM for visual token routing.

    Args:
        model: HF model (LlavaForConditionalGeneration is the primary
            target; qwen2.5-vl / qwen3-vl hooks are kept for future
            extension but not exercised in the release configs).
        router: scoring strategy (FastVRouter or PDropRouter).
        dispatcher: dispatch strategy (kept for API compat).
        action: "physical_delete" | "compact_route" | "compact_route_stagewise".
        model_family: "llava" (default), "qwen25vl", or "qwen3vl".

    Returns:
        RoutingContext that holds shared state and routing logs.
        The context's visual_token_range is set automatically by the
        pre-forward hook on each prefill.
    """
    if model_family in ("llava", "qwen25vl", "qwen3vl"):
        layers = model.model.language_model.layers
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    
    _patch_nuwa_stage1_get_image_features(model, router)

    ctx = RoutingContext(
        visual_token_range=(0, 0),
        router=router,
        dispatcher=dispatcher,
        action=action,
    )
    ctx._model_family = model_family
    ctx._layer_modules = {i: layer for i, layer in enumerate(layers)}

    for i, layer in enumerate(layers):
        original_forward = layer.forward
        layer.forward = _make_patched_forward(original_forward, i, ctx)

    # Register a pre-forward hook on the outer model to auto-detect
    # visual token positions from input_ids before each forward pass.
    # This makes routing work automatically during lmms-eval without
    # manually setting visual_token_range per sample.
    find_range = get_visual_token_finder(model_family)

    def _pre_forward_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is None:
            return
        # --------------------------------------------------------------
        # Nuwa Stage-1 shortens visual placeholders before LLM:
        #   576 image tokens -> stage1_tokens, e.g. 336 for avg192.
        #
        # HF LLaVA expects the number of <image> placeholder tokens in
        # input_ids to match image_features.shape[1]. Since our patched
        # get_image_features() returns only stage1_tokens features, we must
        # also shorten input_ids / attention_mask accordingly.
        # --------------------------------------------------------------
        try:
            from .nuwa_hf import NuwaRouter

            if (
                isinstance(ctx.router, NuwaRouter)
                and kwargs.get("pixel_values") is not None
                and input_ids is not None
                and input_ids.dim() == 2
                and input_ids.shape[1] > 1
            ):
                image_token_id = module.config.image_token_index
                image_mask = input_ids == image_token_id

                if image_mask.any():
                    B, S = input_ids.shape
                    k = int(ctx.router.stage1_tokens)

                    keep_masks = []
                    for b in range(B):
                        keep_b = torch.ones(S, device=input_ids.device, dtype=torch.bool)
                        img_pos = image_mask[b].nonzero(as_tuple=True)[0]

                        # Keep only first k image placeholders.
                        # Text tokens are kept unchanged.
                        if img_pos.numel() > k:
                            keep_b[img_pos[k:]] = False

                        keep_masks.append(keep_b)

                    # Usually B=1 in lmms-eval/profiler. This also works
                    # when all batch elements have the same shortened length.
                    def _filter_seq_tensor(x):
                        if (
                            x is None
                            or not torch.is_tensor(x)
                            or x.dim() < 2
                            or x.shape[0] != B
                            or x.shape[1] != S
                        ):
                            return x
                        return torch.stack(
                            [x[b, keep_masks[b]] for b in range(B)],
                            dim=0,
                        )

                    if "input_ids" in kwargs:
                        kwargs["input_ids"] = _filter_seq_tensor(kwargs["input_ids"])
                        input_ids = kwargs["input_ids"]

                    if "attention_mask" in kwargs:
                        kwargs["attention_mask"] = _filter_seq_tensor(kwargs["attention_mask"])

                    if "position_ids" in kwargs:
                        kwargs["position_ids"] = _filter_seq_tensor(kwargs["position_ids"])

                    if "labels" in kwargs:
                        kwargs["labels"] = _filter_seq_tensor(kwargs["labels"])

        except Exception as e:
            warnings.warn(
                f"[Nuwa] failed to truncate image placeholders: {e}",
                stacklevel=2,
            )

        # Detect visual token range
        vis_range = find_range(module, input_ids,
                               pixel_values=kwargs.get("pixel_values"))

        # Nuwa Stage-1 changes the number of visual tokens before the LLM:
        # 576 -> stage1_tokens, e.g., 336 for avg192.
        try:
            from .nuwa_hf import NuwaRouter
            if isinstance(ctx.router, NuwaRouter) and vis_range[1] > vis_range[0]:
                vis_start = vis_range[0]
                vis_range = (vis_start, vis_start + int(ctx.router.stage1_tokens))
        except Exception:
            pass

        if vis_range[1] > vis_range[0]:
            # New prefill with visual tokens — unconditional reset.
            # (After physical_delete, ctx.visual_token_range is the
            # compacted range which could coincidentally match a new
            # sample's true range, causing stale router decisions.)
            ctx.visual_token_range = vis_range
            ctx.router.reset()
            ctx.attn_weights_cache = None
            ctx.routing_log.clear()
            ctx._compacted = False
            ctx._compact_from_layer = None
            ctx._compact_position_embeddings = None
            ctx._compact_position_ids = None
            # Stagewise reroute: clear stage state for fresh prefill.
            ctx._stage_compact_kept_indices = None
            ctx._stage_deferred_indices = None
            ctx._stage_deferred_hidden = None
            ctx._stage_full_seq_len = 0
            # Oracle diagnostic state for representation staleness.
            ctx.oracle_mode = os.environ.get("REROUTE_ORACLE_MODE", "none")
            ctx._oracle_shadow_hidden = None
            ctx._oracle_prev_selected_mask = None
            ctx._oracle_num_replaced = 0
            ctx._oracle_num_layers = 0
            # Hidden-state alignment diagnostic reset.
            ctx.diag_alignment = (
                os.environ.get("REROUTE_ALIGNMENT_DIAG", "0") == "1"
            )
            ctx.diag_alignment_out_path = os.environ.get(
                "REROUTE_ALIGNMENT_OUT", ctx.diag_alignment_out_path
            )
            ctx.diag_alignment_method = os.environ.get(
                "REROUTE_ALIGNMENT_METHOD", ctx.diag_alignment_method
            )
            ctx.diag_alignment_vector_dir = os.environ.get(
                "REROUTE_ALIGNMENT_VECTOR_DIR",
                ctx.diag_alignment_vector_dir,
            )
            ctx.diag_alignment_vector_max = int(
                os.environ.get(
                    "REROUTE_ALIGNMENT_VECTOR_MAX",
                    str(ctx.diag_alignment_vector_max),
                )
            )
            ctx._diag_alignment_prev_selected_mask = None
            ctx._diag_alignment_last_active_layer = None
            ctx._diag_alignment_num_events = 0

            # Routing-agreement diagnostic reset.
            ctx.diag_routing_agreement = (
                os.environ.get("REROUTE_ROUTING_AGREEMENT_DIAG", "0") == "1"
            )
            ctx.diag_routing_agreement_out_path = os.environ.get(
                "REROUTE_ROUTING_AGREEMENT_OUT",
                ctx.diag_routing_agreement_out_path,
            )
            ctx._diag_routing_prev_selected_mask = None
            ctx._diag_routing_last_active_layer = None

            # Local/global correction state for representation staleness.
            _refresh_correction_env(ctx)
            ctx._correction_prev_selected_mask = None
            ctx._correction_depth_anchors.clear()
            ctx._correction_layer_snapshots.clear()
            ctx._correction_last_active_layer = None
            ctx._correction_num_corrected = 0
            ctx._correction_num_gap_filtered = 0
            ctx._correction_snapshot_current_count = 0
            ctx._correction_snapshot_peak_count = 0
            ctx._correction_snapshot_current_bytes = 0
            ctx._correction_snapshot_peak_bytes = 0
            # Diagnostic: all-layer relevance shift reset.
            ctx.diag_all_layer_react = (
                os.environ.get("REROUTE_DIAG_ALL_LAYER_REACT", "0") == "1"
            )
            ctx.diag_topk = int(os.environ.get("REROUTE_DIAG_TOPK", str(ctx.diag_topk)))
            ctx.diag_ref_layer = int(
                os.environ.get("REROUTE_DIAG_REF_LAYER", str(ctx.diag_ref_layer))
            )
            ctx.diag_out_path = os.environ.get("REROUTE_DIAG_OUT", ctx.diag_out_path)

            # Diagnostic: selected visual-token trajectory reset.
            ctx.diag_token_trace = (
                os.environ.get("REROUTE_DIAG_TOKEN_TRACE", "0") == "1"
            )
            ctx.diag_token_indices = os.environ.get(
                "REROUTE_DIAG_TOKEN_INDICES",
                ctx.diag_token_indices,
            )
            ctx.diag_token_topk = int(
                os.environ.get("REROUTE_DIAG_TOKEN_TOPK", str(ctx.diag_token_topk))
            )
            ctx.diag_token_out_path = os.environ.get(
                "REROUTE_DIAG_TOKEN_OUT",
                ctx.diag_token_out_path,
            )
            ctx.diag_token_sample_filter = int(
                os.environ.get(
                    "REROUTE_DIAG_TOKEN_SAMPLE_ID",
                    str(ctx.diag_token_sample_filter),
                )
            )
            ctx.diag_token_map_path = os.environ.get(
                "REROUTE_DIAG_TOKEN_MAP",
                ctx.diag_token_map_path,
            )
            if ctx._diag_token_map_cache is None and ctx.diag_token_map_path:
                with open(ctx.diag_token_map_path, "r", encoding="utf-8") as f:
                    ctx._diag_token_map_cache = json.load(f)

            # Keep one shared sample counter for all diagnostics.
            if (
                ctx.diag_all_layer_react
                or ctx.diag_token_trace
                or ctx.diag_alignment
                or ctx.diag_routing_agreement
            ):
                ctx._diag_sample_id += 1

            if ctx.diag_alignment:
                if not ctx._diag_alignment_file_initialized:
                    out_dir = os.path.dirname(ctx.diag_alignment_out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                    with open(ctx.diag_alignment_out_path, "w", encoding="utf-8") as f:
                        pass
                    ctx._diag_alignment_file_initialized = True

            if ctx.diag_routing_agreement:
                if not ctx._diag_routing_file_initialized:
                    out_dir = os.path.dirname(ctx.diag_routing_agreement_out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                    with open(
                        ctx.diag_routing_agreement_out_path,
                        "w",
                        encoding="utf-8",
                    ) as f:
                        pass
                    ctx._diag_routing_file_initialized = True

            if ctx.diag_all_layer_react:
                if not ctx._diag_file_initialized:
                    out_dir = os.path.dirname(ctx.diag_out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)

                    # Truncate only once per run.
                    with open(ctx.diag_out_path, "w", encoding="utf-8") as f:
                        pass

                    ctx._diag_file_initialized = True

                ctx._diag_prev_layer_mask = None
                ctx._diag_ref_layer_mask = None
                ctx._diag_ref_layer_actual = None

            if ctx.diag_token_trace:
                if not ctx._diag_token_file_initialized:
                    out_dir = os.path.dirname(ctx.diag_token_out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)

                    # Truncate only once per run.
                    with open(ctx.diag_token_out_path, "w", encoding="utf-8") as f:
                        pass

                    ctx._diag_token_file_initialized = True

    handle = model.register_forward_pre_hook(
        _pre_forward_hook, with_kwargs=True,
    )
    model._routing_pre_hook_handle = handle

    return ctx


def unpatch_model(model, model_family: str = "llava") -> None:
    """Remove routing patches, restoring original forward methods."""
    if model_family in ("llava", "qwen25vl", "qwen3vl", "qwen35"):
        layers = model.model.language_model.layers
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    for layer in layers:
        if hasattr(layer.forward, '__wrapped__'):
            layer.forward = layer.forward.__wrapped__
        else:
            layer.forward = type(layer).forward.__get__(layer, type(layer))

    # Remove the pre-forward hook we registered (stored as _routing_pre_hook)
    if hasattr(model, '_routing_pre_hook_handle'):
        model._routing_pre_hook_handle.remove()
        del model._routing_pre_hook_handle


def get_visual_token_finder(model_family: str):
    """Return the appropriate visual token range finder for a model family."""
    if model_family == "llava":
        return _find_visual_token_range_llava
    elif model_family in ("qwen25vl", "qwen3vl", "qwen35"):
        # Qwen-family models share the vision_start/end token paradigm
        return _find_visual_token_range_qwen25vl
    else:
        raise ValueError(f"Unknown model family: {model_family}")


# ─────────────────────────────────────────────────────────────────────
# (extension hooks for future model families)
# ─────────────────────────────────────────────────────────────────────

def _forward_compact_route_stagewise(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """Routing-layer entry under stagewise: scatter prior stage back to
    full L if needed, compute new routing, gather compact, run layer,
    return compact (no scatter back — passes to next in-stage layer)."""
    # 1. If we're entering from a prior stage, reconstruct full L from
    #    (deferred snapshot) ∪ (compact updates carried in hidden_states).
    #    Only the deferred (L-K) positions were cached; the K kept rows
    #    are overwritten by the incoming compact hidden_states.
    entered_from_prior_stage = ctx._stage_compact_kept_indices is not None
    if entered_from_prior_stage:
        kept_prev = ctx._stage_compact_kept_indices       # (K,)
        deferred_prev = ctx._stage_deferred_indices       # (L-K,)
        B = hidden_states.shape[0]
        d = hidden_states.shape[-1]
        L = ctx._stage_full_seq_len
        full_hidden = hidden_states.new_empty((B, L, d))
        full_hidden[:, deferred_prev] = ctx._stage_deferred_hidden
        full_hidden[:, kept_prev]     = hidden_states
        hidden_states = full_hidden
        # position_embeddings / position_ids passed in from the outer
        # model are already full-L (rotary_emb is invariant across
        # layers) — no cache needed.

    # 1b. Re-capture attention weights at THIS layer on the full-L hidden.
    #
    #     When we entered via the stagewise early intercept (from a prior
    #     stage), Case 2's `_capture_attention_weights` was bypassed.
    #     PDrop's `compute_scores` reads `ctx.attn_weights_cache`, so
    #     without this re-capture the second+ decision layers (e.g. L=6,
    #     L=13, L=20 for drop_layers=[3,6,13,20]) would all score with
    #     stale L=3 attention and effectively collapse to FastV behavior.
    #
    #     For the first decision (kept_prev is None, e.g. L=3) Case 2 has
    #     already captured, so we skip to avoid double-work.
    if entered_from_prior_stage and layer_idx in _get_scoring_layers(ctx.router):
        try:
            _capture_attention_weights(
                layer_idx, ctx, hidden_states,
                None, position_embeddings,
            )
        except torch.cuda.OutOfMemoryError:
            import warnings
            warnings.warn(
                f"[OOM] stagewise re-capture skipped at decision layer "
                f"{layer_idx} (seq_len={hidden_states.shape[1]}). "
                f"Routing falls back to prior-layer scores.",
                stacklevel=2,
            )
            torch.cuda.empty_cache()

    # 2. Compute routing decision on full-L hidden_states.
    decision = ctx.router.compute_scores(
        layer_idx=layer_idx,
        attn_weights=ctx.attn_weights_cache,
        hidden_states=hidden_states,
        visual_token_range=ctx.visual_token_range,
    )
    ctx.routing_log[layer_idx] = decision

    vis_start, vis_end = ctx.visual_token_range
    seq_len = hidden_states.shape[1]
    device = hidden_states.device

    selected_mask = decision.selected_mask
    _qual_trace_log(
        ctx=ctx,
        layer_idx=layer_idx,
        selected_mask=selected_mask,
    )
    unselected_mask = ~selected_mask

    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    unsel_relative = unselected_mask[0].nonzero(as_tuple=True)[0]
    unsel_abs = unsel_relative + vis_start
    keep_mask[unsel_abs] = False
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]
    deferred_indices = (~keep_mask).nonzero(as_tuple=True)[0]

    # 3. Snapshot ONLY the deferred (L-K) positions for next stage —
    #    the K kept positions will be overwritten by the in-stage
    #    compact output at the next decision layer.
    ctx._stage_compact_kept_indices = kept_indices
    ctx._stage_deferred_indices = deferred_indices
    ctx._stage_deferred_hidden = hidden_states[:, deferred_indices].clone()
    ctx._stage_full_seq_len = seq_len

    # 4. Gather compact + run layer + return compact.
    compact_hidden = hidden_states[:, kept_indices].contiguous()
    compact_pos_ids = (
        position_ids[:, kept_indices].contiguous()
        if position_ids is not None else None
    )
    compact_pe = None
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 4:
            compact_pe = (cos[:, :, kept_indices].contiguous(),
                          sin[:, :, kept_indices].contiguous())
        else:
            compact_pe = (cos[:, kept_indices].contiguous(),
                          sin[:, kept_indices].contiguous())

    return original_forward(
        compact_hidden,
        attention_mask=None,
        position_ids=compact_pos_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=compact_pe,
        **kwargs,
    )


def _forward_compact_in_stage(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """In-stage non-routing layer under stagewise: input is compact
    (B, K, d) carried over from upstream routing layer. Gather PE /
    pos_ids to compact and run the layer on compact, return compact."""
    kept_indices = ctx._stage_compact_kept_indices

    compact_pos_ids = (
        position_ids[:, kept_indices].contiguous()
        if position_ids is not None else None
    )
    compact_pe = None
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 4:
            compact_pe = (cos[:, :, kept_indices].contiguous(),
                          sin[:, :, kept_indices].contiguous())
        else:
            compact_pe = (cos[:, kept_indices].contiguous(),
                          sin[:, kept_indices].contiguous())

    return original_forward(
        hidden_states,
        attention_mask=None,
        position_ids=compact_pos_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=compact_pe,
        **kwargs,
    )
