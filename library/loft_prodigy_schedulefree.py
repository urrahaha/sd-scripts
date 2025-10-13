# LoFT-aware Prodigy+ScheduleFree optimizer integration
# This subclasses ProdigyPlusScheduleFree to apply a lightweight LoFT-style
# second-moment projection adjustment for LoFT parameters (U/V factors).
#
# Notes
# - First-moment calibration is not applicable in Schedule-Free mode (no m_t),
#   so we only adjust the denominator used for adaptive scaling.
# - This is an approximation that scales the per-rank denominator by the peer
#   Gram diagonal (sqrt(diag(V^T V)) for U, sqrt(diag(U^T U)) for V).
# - Works with linear and Conv2d (1x1 and kxk) modules by broadcasting.

from __future__ import annotations

import torch

try:
    from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
except Exception as e:  # pragma: no cover
    raise ImportError(
        "LoFTProdigyScheduleFree requires the 'prodigy-plus-schedule-free' package.\n"
        "Install from https://github.com/LoganBooker/prodigy-plus-schedule-free or pip if available.\n"
        f"Import error: {e}"
    )


class LoFTProdigyScheduleFree(ProdigyPlusScheduleFree):
    """Prodigy+ScheduleFree with LoFT denominator adjustment for LoFT params.

    Expects LoFT modules to tag their tensors:
      - param._loft_role in {"U", "V"}
      - param._loft_peer -> peer weight tensor (V for U, U for V)
    These are set by networks/loft.py.
    """

    @torch.no_grad()
    def initialise_state(self, p, group):
        state = super().initialise_state(p, group)
        # No extra state required beyond peer references stored on the parameter.
        return state

    def _loft_scale_from_peer(self, role: str, peer_w: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute per-rank scale = sqrt(diag(peer^T peer) + eps) as a length-r vector.
        peer_w can be Linear or Conv2d weights.
        """
        eps = 1e-8
        # Flatten to 2D [r, N] if role is U (peer is V^T with [r, in, ...])
        # or [M, r] if role is V (peer is U with [out, r, ...]) and reduce along the non-r dimension to get diag.
        if role == "U":
            # peer is V^T with shape [r, in] or [r, in, kH, kW]
            v2d = peer_w.flatten(1) if peer_w.dim() > 2 else peer_w
            # diag(V^T V) -> sum over columns
            diag = (v2d * v2d).sum(dim=1)
        else:
            # role == "V": peer is U with shape [out, r] or [out, r, 1, 1]
            u2d = peer_w.flatten(1) if peer_w.dim() > 2 else peer_w
            # u2d shape [out, r]; diag(U^T U) -> sum over rows
            diag = (u2d * u2d).sum(dim=0)
        return torch.sqrt(diag.to(device=device, dtype=torch.float32).clamp_min(eps)).to(dtype=dtype)

    def _apply_loft_denominator_adjustment(self, p: torch.Tensor, denom: torch.Tensor):
        role = getattr(p, "_loft_role", None)
        peer = getattr(p, "_loft_peer", None)
        if role not in ("U", "V") or peer is None:
            return denom

        # Compute per-rank scale and broadcast to denom shape
        scale_r = self._loft_scale_from_peer(role, peer, denom.device, denom.dtype)  # [r]
        if p.dim() == 2:
            # Linear: U [out, r] or V [r, in]
            if role == "U":  # scale along r (dim=1)
                scale = scale_r.view(1, -1)
            else:  # role == "V": scale along r (dim=0)
                scale = scale_r.view(-1, 1)
        else:
            # Conv2d cases: U [out, r, 1, 1], V [r, in, kH, kW]
            if role == "U":
                # Broadcast over out, h, w dims
                shape = [1, -1] + [1] * (p.dim() - 2)
                scale = scale_r.view(*shape)
            else:  # V
                # Broadcast over in,kH,kW dims (rank axis is dim=0)
                shape = [-1] + [1] * (p.dim() - 1)
                scale = scale_r.view(*shape)

        return denom * scale

    @torch.no_grad()
    def update_second_moment(self, state, group, grad, beta2, w, return_denom=True, denom_before_update=False):
        # Let the base optimizer build the denominator first
        denom = super().update_second_moment(state, group, grad, beta2, w, return_denom=return_denom, denom_before_update=denom_before_update)
        if return_denom and denom is not None:
            p = w  # 'w' is the parameter tensor cast to float in caller; use the original param ref via state
            # Attempt to recover original parameter from state dict (PyTorch keeps state keyed by original param tensor)
            # We can locate the parameter by matching shapes; however, we get a more direct path by passing the original
            # parameter in callers. Since that's invasive, rely on tags attached to the parameter we have access to.
            if hasattr(p, "_loft_role"):
                denom = self._apply_loft_denominator_adjustment(p, denom)
        return denom
