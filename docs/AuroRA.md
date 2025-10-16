# AuroRA: Adaptive Nonlinear LoRA for Stable Diffusion

- **[summary]** **A**ctivate Yo**ur** L**o**w-**R**ank **A**daptation (AuroRA) augments standard LoRA with an Adaptive Nonlinear Layer (ANL) between `lora_down` and `lora_up`.
- **[key idea]** Apply a fixed nonlinearity with a small learnable channel-mixing and optional learnable spline augmentation:
  - Forward: `z = lora_down(x)` → `z1 = tanh(z) + β · (w_s · s(z))` → `z2 = H(z1)` → `y = lora_up(tanh(z2))`.
  - Offline merge: compute `A_tilde = tanh(H @ (tanh(A) + β·(w_s·s(A))))`, then merge `ΔW = B @ A_tilde`.

## Why AuroRA over Standard LoRA

- **[expressivity]** A channel-mixing `H` (1×1 conv or linear) and nonlinearities can represent richer transforms than plain rank‑r linear adapters.
- **[parameter‑efficiency]** Only `r×r` mixing and `r×K` spline weights added (K small, e.g. 4). Often yields better fit at the same or lower rank.
- **[static merge]** Like LoRA, AuroRA supports static/offline merge; the ANL is evaluated once to bake the delta into base weights.

## Technical details

- **[modules]** Implemented in `sd-scripts/networks/aurora.py` (`AuroRAModule`, `AuroRAInfModule`).
- **[ANL]**
  - `H`: `Linear(r→r)` for Linear layers, `Conv2d(r→r, 1×1)` for Conv2d.
  - Nonlinearities: `tanh` pre/post `H`. Optional spline augmentation replaces/augments the first tanh.
- **[spline]** Small set of basis functions per element with trainable mixing per rank channel:
  - Basis: `φ_m(z) = tanh(a·(z − c_m))`, m=1..K.
  - Combine: `(w_s · s(z)) = Σ_m φ_m(z) · w_s[:, m]`.
  - Gate: `β` (per-module scalar) turns spline on/off; defaults to 0 (disabled).

## GUI usage (Kohya GUI)

- **[select]** In `LoRA` tab set `LoRA type = AuroRA`.
- **[core params]** Use `network_dim` and `network_alpha` as usual. Block dims/LR weighting work as with LoRA/NLoRA.
- **[spline controls]** In Advanced → Weights:
  - `AuroRA spline gate (beta)`: default 0 (disabled). Try 0.05–0.2 to enable.
  - `AuroRA spline scale`: default 1.5 (leave empty to use default).
  - `AuroRA spline centers`: default `-1.0,-0.5,0.5,1.0` (leave empty for default). Comma-separated list.

## Recommended settings

- **[rank]** Start with `network_dim = 4` (r=4) for SD1.5; tune 2–8 based on capacity.
- **[alpha]** Set `network_alpha = rank` (common LoRA practice), or slightly larger (1.0×–2.0× rank).
- **[spline]**
  - `β (spline gate)`: 0.05–0.2 initial. Keep small to avoid overfitting.
  - `scale a`: 1.5 (default). If you observe saturation, try 1.0–2.0.
  - `centers c_m`: `[-1.0, -0.5, 0.5, 1.0]`. If activations are mostly within [-1,1], this works well.
- **[dropouts]** `rank_dropout` and `module_dropout` work normally. Consider `rank_dropout=0.05–0.2` if combining multiple LoRAs.

## Training (CLI)

Example (abbreviated) using `train_network.py` with AuroRA via `network_args`:

```bash
python sd-scripts/train_network.py \
  --network_module networks.aurora \
  --network_dim 4 --network_alpha 4 \
  --network_args spline_gate=0.1 spline_scale=1.5 spline_centers=-1.0,-0.5,0.5,1.0 \
  ... # dataset/optimizer/scheduler/etc
```

- You can omit `spline_*` to use defaults (gate=0 disables spline).

## Offline merge

- **[standard]** `sd-scripts/networks/merge_lora.py` merges SD models; `sd-scripts/networks/sdxl_merge_lora.py` merges SDXL.
- **[aurora-aware]** If the checkpoint has `anl_H` (and optionally `spline_*`), the scripts compute `A_tilde` with the ANL before merging.

## Tips

- **[stability]** The gating `β` defaults to 0 to preserve standard LoRA behavior at init; increase gradually.
- **[compute]** Overhead is modest: an extra r×r mixing and small basis eval. Typically negligible vs U‑Net cost.
- **[compatibility]** Works with existing LoRA training options (block dims/LR weights, conv LoRA, etc.).

## File references

- `sd-scripts/networks/aurora.py`
- `sd-scripts/networks/merge_lora.py`
- `sd-scripts/networks/sdxl_merge_lora.py`
- GUI wiring in `kohya_gui/lora_gui.py`
