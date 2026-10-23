"""Training-free CatchUp routing for LLaVA-1.5.

At configured decoder stages, score query-conditioned visual relevance and
process only active tokens via compact gather/decoder/scatter.
Deferred visual tokens retain representations for later re-admission.
Local Transition Correction aligns re-admitted states using fresh anchors.
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
    # Local Transition Correction on newly re-admitted tokens.
    # 'none' is retained for Q-VER without LTC efficiency comparisons.
    correction_mode: str = "none"
    correction_topm: int = 8
    correction_spatial_lambda: float = 0.15
    correction_tau_l: float = 0.07
    correction_anchor_mode: str = "semantic_spatial"
    correction_min_layer_gap: int = 1
    correction_gamma_max: float = 0.8
    correction_norm_preserve: bool = True
    correction_norm_target: str = "cached"
    correction_query_eta: float = 0.0
    correction_query_norm: str = "minmax"

    # Need-based reference-state retention: keep only live reference depths.
    correction_snapshot_policy: str = "live"
    correction_snapshot_debug: bool = False

    _correction_prev_selected_mask: Optional[Tensor] = None
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


def _make_patched_forward(original_forward, layer_idx: int, ctx: RoutingContext):
    """Apply CatchUp's compact reversible routing to one decoder layer."""

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
        vis_start, vis_end = ctx.visual_token_range
        seq_len = hidden_states.shape[1]
        is_prefill = seq_len > 1 and vis_end > vis_start and seq_len >= vis_end

        if is_prefill and ctx.router.should_route(layer_idx):
            if ctx.router.is_decision_layer(layer_idx):
                try:
                    _capture_attention_weights(
                        layer_idx, ctx, hidden_states,
                        attention_mask, position_embeddings,
                    )
                except torch.cuda.OutOfMemoryError:
                    warnings.warn(
                        f"Query-attention capture OOM at layer {layer_idx}; "
                        "continuing without routing this layer.",
                        stacklevel=2,
                    )
                    torch.cuda.empty_cache()
                    ctx.attn_weights_cache = None

            if ctx.attn_weights_cache is not None:
                if ctx.action != "compact_route":
                    raise ValueError("Only compact_route is supported in CatchUp.")
                return _forward_compact_route(
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

    q = q_raw.view(batch, seq_len, -1, head_dim)
    k = k.view(batch, seq_len, -1, head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    # LLaVA-1.5 uses standard LLaMA rotary embeddings.
    if position_embeddings is not None:
        cos, sin = position_embeddings
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


# =====================================================================
# Local / global correction helpers for stale re-admitted tokens
# =====================================================================


def _refresh_correction_env(ctx: RoutingContext) -> None:
    """Load the final Q-VER/LTC settings; reject unsupported research modes."""
    mode = os.environ.get("REROUTE_CORRECTION_MODE", "none").strip().lower()
    if mode not in ("qver", "none"):
        raise ValueError(
            f"Unsupported REROUTE_CORRECTION_MODE={mode!r}. "
            "Only 'qver' (LTC) and 'none' (without LTC) are supported."
        )
    ctx.correction_mode = mode
    ctx.correction_topm = int(os.environ.get("REROUTE_CORRECTION_TOPM", "8"))
    ctx.correction_spatial_lambda = float(
        os.environ.get("REROUTE_CORRECTION_SPATIAL_LAMBDA", "0.15")
    )
    ctx.correction_tau_l = float(os.environ.get("REROUTE_CORRECTION_TAU_L", "0.07"))
    ctx.correction_anchor_mode = os.environ.get(
        "REROUTE_CORRECTION_ANCHOR_MODE", "semantic_spatial"
    ).lower()
    ctx.correction_min_layer_gap = int(
        os.environ.get("REROUTE_CORRECTION_MIN_LAYER_GAP", "1")
    )
    ctx.correction_gamma_max = float(
        os.environ.get("REROUTE_CORRECTION_GAMMA_MAX", "0.8")
    )

    # Only need-based live cache retention is supported.
    policy = os.environ.get("REROUTE_CORRECTION_SNAPSHOT_POLICY", "live").strip().lower()
    if policy != "live":
        raise ValueError(
            f"Unsupported REROUTE_CORRECTION_SNAPSHOT_POLICY={policy!r}. "
            "Only 'live' is supported."
        )
    ctx.correction_snapshot_policy = "live"
    snapshot_debug = os.environ.get("REROUTE_CORRECTION_SNAPSHOT_DEBUG", "0")
    ctx.correction_snapshot_debug = snapshot_debug.lower() not in (
        "0", "false", "no", "off"
    )

    # Preserve the previous norm-rescaling behavior and parameter defaults.
    norm_flag = os.environ.get("REROUTE_CORRECTION_NORM_PRESERVE", "1")
    ctx.correction_norm_preserve = norm_flag.lower() not in (
        "0", "false", "no", "off"
    )
    ctx.correction_norm_target = os.environ.get(
        "REROUTE_CORRECTION_NORM_TARGET", "cached"
    ).strip().lower()
    if not ctx.correction_norm_preserve:
        ctx.correction_norm_target = "none"
    if ctx.correction_norm_target not in (
        "cached", "current_anchor", "transition_ratio", "none"
    ):
        raise ValueError(
            "Unsupported REROUTE_CORRECTION_NORM_TARGET="
            f"{ctx.correction_norm_target!r}"
        )

    # Keep the existing default and allow the release script to set eta=0.
    ctx.correction_query_eta = float(
        os.environ.get("REROUTE_CORRECTION_QUERY_ETA", "0.2" if mode == "qver" else "0.0")
    )
    ctx.correction_query_norm = os.environ.get(
        "REROUTE_CORRECTION_QUERY_NORM", "minmax"
    ).lower()


def _correction_enabled(ctx: RoutingContext) -> bool:
    return ctx.correction_mode == "qver"


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
    """Keep only reference depths needed by currently deferred visual tokens.

    The current full visual snapshot is retained for possible future deferral.
    Old full snapshots referenced by deferred tokens are retained; other
    snapshots are released before cloning the current state.
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
    # Need-based live retention. Preserve the original live branch exactly.
    required_layers = {layer_key}
    deferred_mask = ~selected_mask
    if deferred_mask.any():
        live_ref_layers = torch.unique(last[deferred_mask])
        required_layers.update(
            int(x) for x in live_ref_layers.detach().cpu().tolist()
        )

    # Release dead depths before cloning the current full visual state.
    for key in list(ctx._correction_layer_snapshots.keys()):
        if int(key) not in required_layers:
            del ctx._correction_layer_snapshots[key]
    ctx._correction_layer_snapshots[layer_key] = h_vis.detach().clone()

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
            f"policy=live layer={layer_key} "
            f"kept={keys} count={len(keys)} "
            f"memory={mib:.2f}MiB peak={peak_mib:.2f}MiB",
            flush=True,
        )


def _apply_readmitted_correction(
    ctx: RoutingContext,
    hidden_states: Tensor,
    selected_mask: Tensor,
    visual_token_range: tuple[int, int],
    layer_idx: int,
    scores: Optional[Tensor] = None,
) -> Tensor:
    """Apply correction only to re-admitted visual tokens.

    Routing decisions and active-token budget are unchanged.
    Only identity-preserving local depth-transition residuals are applied.
    """
    vis_start, vis_end = visual_token_range
    h_vis = hidden_states[:, vis_start:vis_end, :]
    B, V, D = h_vis.shape

    # If correction is off, keep lightweight tracking disabled.
    if not _correction_enabled(ctx):
        ctx._correction_prev_selected_mask = selected_mask.detach().clone()
        return hidden_states

    prev_mask = ctx._correction_prev_selected_mask

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
    if mode == "qver":
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

    decision = ctx.router.compute_scores(
        layer_idx=layer_idx,
        attn_weights=ctx.attn_weights_cache,
        hidden_states=hidden_states,
        visual_token_range=ctx.visual_token_range,
    )
    ctx.routing_log[layer_idx] = decision
    selected_mask = decision.selected_mask

    # Restore stale re-admitted representations before compact execution.
    if _correction_enabled(ctx):
        hidden_states = _apply_readmitted_correction(
            ctx=ctx,
            hidden_states=hidden_states,
            selected_mask=selected_mask,
            visual_token_range=ctx.visual_token_range,
            layer_idx=layer_idx,
            scores=decision.scores,
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

        return (output,) + compact_out[1:]

    output = hidden_states.clone()
    output[:, kept_indices] = compact_out

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
            target).
        router: QVERouter for reversible CatchUp selection.
        dispatcher: dispatch strategy (kept for API compat).
        action: "physical_delete" | "compact_route" | "compact_route_stagewise".
        model_family: "llava" only.

    Returns:
        RoutingContext that holds shared state and routing logs.
        The context's visual_token_range is set automatically by the
        pre-forward hook on each prefill.
    """
    if model_family != "llava":
        raise ValueError(f"Unsupported model family: {model_family}")
    layers = model.model.language_model.layers
    
    # Q-VER uses the original LLaVA visual feature projection.

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
        # Detect visual token range
        vis_range = find_range(module, input_ids,
                               pixel_values=kwargs.get("pixel_values"))

        if vis_range[1] > vis_range[0]:
            # New prefill with visual tokens — unconditional reset.
            # (After physical_delete, ctx.visual_token_range is the
            # compacted range which could coincidentally match a new
            # sample's true range, causing stale router decisions.)
            ctx.visual_token_range = vis_range
            ctx.router.reset()
            ctx.attn_weights_cache = None
            ctx.routing_log.clear()
            # Reinitialize LTC references for each independent image prefill.
            _refresh_correction_env(ctx)
            ctx._correction_prev_selected_mask = None
            ctx._correction_layer_snapshots.clear()
            ctx._correction_last_active_layer = None
            ctx._correction_num_corrected = 0
            ctx._correction_num_gap_filtered = 0
            ctx._correction_snapshot_current_count = 0
            ctx._correction_snapshot_peak_count = 0
            ctx._correction_snapshot_current_bytes = 0
            ctx._correction_snapshot_peak_bytes = 0
    handle = model.register_forward_pre_hook(
        _pre_forward_hook, with_kwargs=True,
    )
    model._routing_pre_hook_handle = handle

    return ctx


def unpatch_model(model, model_family: str = "llava") -> None:
    """Remove routing patches, restoring original forward methods."""
    if model_family != "llava":
        raise ValueError(f"Unsupported model family: {model_family}")
    layers = model.model.language_model.layers

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
    if model_family != "llava":
        raise ValueError(f"Unsupported model family: {model_family}")
    return _find_visual_token_range_llava
