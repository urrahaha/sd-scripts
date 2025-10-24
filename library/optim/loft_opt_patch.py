import types
import os
from typing import Dict, Any, Optional

import torch


def _to_2d(w: torch.Tensor) -> torch.Tensor:
    return w.flatten(1) if w.dim() > 2 else w


def _compute_transport_C(role: str, param: torch.nn.Parameter, peer: torch.nn.Parameter, prev_peer2d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute transport matrix C for LoFT state calibration.

    role == 'U': C_V^k = (V_{k-1}^T V_k) (V_k^T V_k)^{-1}
    role == 'V': C_U^k = (U_{k-1}^T U_k) (U_k^T U_k)^{-1}
    """
    peer2d = _to_2d(peer.data).float()
    # Build Gram and cross terms in r x r space
    # For U: peer is V^T [r, d], so V_k = peer2d.T [d, r]
    # For V: peer is U [m, r] already; treat similarly via transposes
    # Using Vt representation consistently
    gram = peer2d @ peer2d.t()  # [r, r]
    r = gram.shape[0]
    gram = gram + eps * torch.eye(r, device=gram.device, dtype=gram.dtype)
    cross = prev_peer2d @ peer2d.t()  # [r, r]
    C = cross @ torch.linalg.inv(gram)
    return C


def patch_optimizer_for_loft(
    optimizer: torch.optim.Optimizer,
    eps: float = 1e-6,
    enabled: bool = True,
    flag_path: Optional[str] = None,
) -> torch.optim.Optimizer:
    """Monkey‑patch optimizer to approximate LoFT Blocks 3–4 for LoFT‑tagged params.

    - Recalibrate first moment: m^{k-1} <- m^{k-1} C
    - Recalibrate second moment: v^{k-1} <- v^{k-1} (C ⊙ C)  (approximation along r-dim)
    - Keep optimizer type unchanged; compatible with ProdigyPlus ScheduleFree.
    """
    if getattr(optimizer, "_loft_patched", False):
        return optimizer

    # Discover LoFT parameters from param_groups
    loft_metas: Dict[int, Dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            role = getattr(p, "_loft_role", None)
            peer = getattr(p, "_loft_peer", None)
            if role in {"U", "V"} and isinstance(peer, torch.nn.Parameter):
                peer2d = _to_2d(peer.data).float().detach().clone()
                self2d = _to_2d(p.data).float().detach().clone()
                loft_metas[id(p)] = {
                    "param": p,
                    "role": role,
                    "peer": peer,
                    "prev_peer2d": peer2d,
                    "prev_self2d": self2d,
                }

    if not loft_metas:
        return optimizer

    optimizer._loft_patched = True
    optimizer._loft_eps = float(eps)
    optimizer._loft_metas = loft_metas
    optimizer._loft_state_to_param: Dict[int, torch.nn.Parameter] = {}
    optimizer._loft_transport_enabled = bool(enabled)
    optimizer._loft_transport_flag_path = flag_path

    # Map from state object id to parameter; refreshed each step
    def _refresh_state_map(self: torch.optim.Optimizer):
        mapping: Dict[int, torch.nn.Parameter] = {}
        for meta in self._loft_metas.values():
            p = meta["param"]
            if p in self.state:
                mapping[id(self.state[p])] = p
        self._loft_state_to_param = mapping

    # Wrap first/second moment updates if they exist on the optimizer
    if hasattr(optimizer, "update_first_moment") and hasattr(optimizer, "update_second_moment"):
        orig_update_first = optimizer.update_first_moment
        orig_update_second = optimizer.update_second_moment

        def loft_update_first_moment(self, state, group, grad, beta1):  # type: ignore[no-redef]
            if not getattr(self, "_loft_transport_enabled", True):
                return orig_update_first(state, group, grad, beta1)
            p = self._loft_state_to_param.get(id(state))
            if p is not None:
                meta = self._loft_metas.get(id(p))
                if meta is not None and p.requires_grad:
                    role = meta["role"]
                    peer = meta["peer"]
                    prev_peer2d = meta["prev_peer2d"]
                    try:
                        C = _compute_transport_C(role, p, peer, prev_peer2d, eps=self._loft_eps)
                        if "exp_avg" in state:
                            ea = state["exp_avg"]
                            if ea.dim() > 2:
                                ea2d = ea.flatten(1).float()  # [m, r] or [r, d]
                            else:
                                ea2d = ea.float()
                            if role == "U":
                                ea2d = ea2d @ C
                            else:  # V
                                ea2d = C @ ea2d
                            ea.copy_(ea2d.to(dtype=ea.dtype).view_as(ea))
                    except Exception:
                        pass  # be robust: fall back silently if anything goes wrong
            # Call original bound method (already bound to self)
            return orig_update_first(state, group, grad, beta1)

        def loft_update_second_moment(self, state, group, grad, beta2, w, return_denom=True, denom_before_update=False):  # type: ignore[no-redef]
            if not getattr(self, "_loft_transport_enabled", True):
                return orig_update_second(state, group, grad, beta2, w, return_denom, denom_before_update)
            p = self._loft_state_to_param.get(id(state))
            if p is not None:
                meta = self._loft_metas.get(id(p))
                if meta is not None and p.requires_grad:
                    role = meta["role"]
                    peer = meta["peer"]
                    prev_peer2d = meta["prev_peer2d"]
                    try:
                        C = _compute_transport_C(role, p, peer, prev_peer2d, eps=self._loft_eps)
                        Csq = C.pow(2)
                        if "exp_avg_sq" in state:
                            v = state["exp_avg_sq"]
                            if v.dim() > 2:
                                v2d = v.flatten(1).float()
                            else:
                                v2d = v.float()
                            if role == "U":
                                v2d = v2d @ Csq
                            else:  # V
                                v2d = Csq @ v2d
                            v.copy_(v2d.to(dtype=v.dtype).view_as(v))
                        # Factored case: approximate by transforming the r-dimension accumulator
                        elif "exp_avg_sq_metadata" in state:
                            try:
                                row_var, col_var = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
                                if role == "U":
                                    # columns correspond to r-dim
                                    col2d = col_var.flatten(1).float()  # [1, r] -> [1, r]
                                    col2d = col2d @ Csq
                                    col_var.copy_(col2d.to(dtype=col_var.dtype).view_as(col_var))
                                else:
                                    row2d = row_var.flatten(0).float().view(row_var.shape[0], -1)
                                    row2d = Csq @ row2d
                                    row_var.copy_(row2d.to(dtype=row_var.dtype).view_as(row_var))
                            except Exception:
                                pass
                    except Exception:
                        pass
            # Call original bound method (already bound to self)
            return orig_update_second(state, group, grad, beta2, w, return_denom, denom_before_update)

        optimizer.update_first_moment = types.MethodType(loft_update_first_moment, optimizer)
        optimizer.update_second_moment = types.MethodType(loft_update_second_moment, optimizer)

    # Wrap step to refresh state map and update prev snapshots
    orig_step = optimizer.step

    def loft_step(self, closure=None):  # type: ignore[no-redef]
        # Dynamic toggle: env var or flag file
        try:
            v = os.environ.get("LOFT_STATE_TRANSPORT")
            if v is not None:
                self._loft_transport_enabled = (str(v).lower() not in {"0", "false", "off"})
            fp = getattr(self, "_loft_transport_flag_path", None)
            if fp:
                if os.path.exists(fp):
                    with open(fp, "r") as f:
                        s = f.read().strip()
                    self._loft_transport_enabled = (s.lower() not in {"0", "false", "off"})
        except Exception:
            pass
        _refresh_state_map(self)
        loss = orig_step(closure)
        # update prev snapshots after parameters are updated
        for meta in self._loft_metas.values():
            p = meta["param"]
            peer = meta["peer"]
            meta["prev_self2d"] = _to_2d(p.data).float().detach().clone()
            meta["prev_peer2d"] = _to_2d(peer.data).float().detach().clone()
        return loss

    optimizer.step = types.MethodType(loft_step, optimizer)

    return optimizer
