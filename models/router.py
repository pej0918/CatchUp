"""Token routing scorers for visual token selection.

QVERouter implements reversible visual token selection.

The router scores over visual tokens only; text / system tokens are
always kept active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


def _aggregate_query_attention(
    attn_avg: Tensor,            # (batch, S, S) — mean across heads
    vis_start: int,
    vis_end: int,
) -> Tensor:
    """Last-text-token attention over the visual region."""
    return attn_avg[:, -1, vis_start:vis_end].float()



@dataclass
class RoutingDecision:
    """Output of a router's score-and-select step.

    Attributes:
        selected_mask: (batch, num_visual) bool — True for tokens to attend.
        scores: (batch, num_visual) float — raw importance scores.
        selected_indices: (batch, K) long — indices into the visual token
            range (relative, 0-based). Sorted ascending.
    """
    selected_mask: Tensor
    scores: Tensor
    selected_indices: Tensor


class TokenRouter(ABC):
    """Base interface for visual-token routing scorers."""

    @abstractmethod
    def should_route(self, layer_idx: int) -> bool:
        """Return True if routing should be applied at this layer."""
        ...

    @abstractmethod
    def is_decision_layer(self, layer_idx: int) -> bool:
        """Return True if a new routing decision is made at this layer.

        For mask-based routing, should_route() determines where masks are
        applied (every layer after the first decision). For physical delete,
        only decision layers need to compact — subsequent layers just see
        the shorter sequence.
        """
        ...

    @abstractmethod
    def compute_scores(
        self,
        layer_idx: int,
        attn_weights: Tensor,
        hidden_states: Tensor,
        visual_token_range: tuple[int, int],
        **kwargs,
    ) -> RoutingDecision:
        """Score visual tokens and select top-K.

        Args:
            layer_idx: current decoder layer index.
            attn_weights: (batch, num_heads, seq_len, seq_len) from this layer.
            hidden_states: (batch, seq_len, hidden_dim) input to this layer.
            visual_token_range: (start, end) absolute positions of visual
                tokens in the sequence.

        Returns:
            RoutingDecision with mask, scores, and sorted selected indices.
        """
        ...


class QVERouter(TokenRouter):
    """Query-guided reversible visual-token routing for CatchUp.

    Scores all original visual tokens at each routing stage, then selects
    the current Top-K. Unselected tokens remain eligible for re-admission.
    """

    def __init__(
        self,
        drop_layers: list[int] | None = None,
        keep_ratios: list[float] | None = None,
        monotonic: bool = False,
    ) -> None:
        # Defaults for a 32-layer model (LLaMA-2-7B style)
        self.drop_layers = drop_layers or [7, 15, 23]
        self.keep_ratios = keep_ratios or [0.75, 0.5, 0.25]
        self.monotonic = monotonic
        assert len(self.drop_layers) == len(self.keep_ratios)

        # Track current stage and cached decision per stage
        self._stage_decisions: dict[int, RoutingDecision] = {}
        self._original_num_visual: int | None = None

    # ------------------------------------------------------------------
    def should_route(self, layer_idx: int) -> bool:
        return layer_idx in self.drop_layers or any(
            layer_idx > dl for dl in self.drop_layers
        )

    def is_decision_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.drop_layers

    def _current_stage(self, layer_idx: int) -> int | None:
        """Return the most recent drop-layer stage for this layer_idx."""
        stage = None
        for i, dl in enumerate(self.drop_layers):
            if layer_idx >= dl:
                stage = i
        return stage

    # ------------------------------------------------------------------
    def compute_scores(
        self,
        layer_idx: int,
        attn_weights: Tensor,
        hidden_states: Tensor,
        visual_token_range: tuple[int, int],
        **kwargs,
    ) -> RoutingDecision:
        stage = self._current_stage(layer_idx)
        if stage is None:
            # Before any drop layer — keep all visual tokens
            vis_start, vis_end = visual_token_range
            num_visual = vis_end - vis_start
            batch = hidden_states.shape[0]
            return RoutingDecision(
                selected_mask=torch.ones(batch, num_visual, dtype=torch.bool,
                                         device=hidden_states.device),
                scores=torch.ones(batch, num_visual, device=hidden_states.device),
                selected_indices=torch.arange(num_visual, device=hidden_states.device)
                    .unsqueeze(0).expand(batch, -1),
            )

        # Return cached decision if we already scored this stage
        if stage in self._stage_decisions:
            return self._stage_decisions[stage]

        vis_start, vis_end = visual_token_range
        num_visual = vis_end - vis_start

        if self._original_num_visual is None:
            self._original_num_visual = num_visual

        k = max(1, int(self._original_num_visual * self.keep_ratios[stage]))

        # attn_weights: (batch, heads, seq, seq) → mean across heads.
        attn_avg = attn_weights.mean(dim=1)

        scores = _aggregate_query_attention(
            attn_avg,
            vis_start=vis_start,
            vis_end=vis_end,
        )

        # Enforce monotonicity: tokens dropped at previous stages must
        # remain dropped. Required for delete (can't bring back removed
        # tokens). Disabled for residual_skip with re-entry.
        if self.monotonic and stage > 0:
            prev_stage = stage - 1
            if prev_stage in self._stage_decisions:
                prev_mask = self._stage_decisions[prev_stage].selected_mask
                # Only apply mask-based enforcement if shapes match.
                # For physical delete, shapes differ (sequence was compacted)
                # and monotonicity is implicit — deleted tokens are gone.
                if prev_mask.shape[1] == scores.shape[1]:
                    scores = scores.masked_fill(~prev_mask, float('-inf'))

        actual_k = min(k, num_visual)
        topk = scores.topk(actual_k, dim=-1)
        selected_indices = topk.indices.sort(dim=-1).values

        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, selected_indices, True)

        decision = RoutingDecision(
            selected_mask=mask,
            scores=scores,
            selected_indices=selected_indices,
        )
        self._stage_decisions[stage] = decision
        return decision

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all cached decisions (call between samples)."""
        self._stage_decisions.clear()
        self._original_num_visual = None
