# THREE-VIZ MISSION STATUS (single source of truth — read this first in any session)

Last updated: 2026-10-25. Owner: Hermes (orchestration + checking). Maker: Copilot Opus 5.5 (tmux `vlaviz:opus`).

## Mission

Three INDEPENDENT visualization pages, each explaining one model's internals with REAL
local weights and REAL BEHAVIOR dataset input frames. No downloads. No cross-model
fabricated mappings. No toy data. No popups/overlays — in-scene walkthrough style only.

| Page | Model | Weights (local) | Status |
|---|---|---|---|
| `/vla` | k005b dual-head VLA | `outputs/k005b-task0-v3/checkpoints` (step_150000) | DONE + verified |
| `/pi0` | pi0.5 b1k finetune | `~/workspace/model/openpi_comet/pi05-b1kpt12-cs32` (+`pi05-b1kpt50-cs32`; base ref: HF cache `lerobot/pi0_base`) | IN PROGRESS |
| `/groot` | GR00T N1.7-3B (incl. sim_behavior_r1_pro embodiment id=23, latest=global_step196000) | `~/workspace/model/nvidia/GR00T-N1.7-3B` | DONE (self-tested, awaiting Hermes acceptance) |

Input frames for ALL pages: local BEHAVIOR dataset
`~/workspace/behavior_challenge_data/dataset/{behavior-1k-knack, 2026-challenge-demos}`.

## Verified facts (do not re-litigate)

- k005b action head: K-step Euler decode (`sample_action_chunk`), K=10 default, K=1 = old
  single-shot; checkpoint `action_decode` stamp gates K (unstamped lineage = K 1 only).
  PR #638 merged; 155 tests green; bf16 launch bug fixed (20b39c2).
- GR00T N1.7 (from config.json + weight shapes): DiT 32 layers, 32 heads, head_dim 48,
  ada_norm, interleave_self_attention=true, action_horizon 40, action dim 132, backbone
  embedding 2048 (Cosmos-Reason2-2B), Euler 4 steps (draw 4, not 10), embodiment
  sim_behavior_r1_pro id=23. 6 minor details listed in
  `docs/design/` GR00T understanding doc §4 — all 6 verified (local sources + shallow Isaac-GR00T
  clone; cloning code is allowed, weights are never downloaded).
- pi0: state and action dims are INDEPENDENT projections (61-dim state -> state token;
  23/32-dim action -> action tokens). Never ask "how do the dims map" — per-embodiment
  projection configs are the answer (precedent: official GR00T BEHAVIOR baseline).

## Remaining steps (in order)

1. ~~`/groot`: verify the 6 details -> exporter -> page -> self-test -> commit.~~ DONE:
   `tools/export_groot.py` -> `public/groot-*.json`; `src/groot/*`; G0-G9 self-test: 0 console
   errors, arrow audit empty (292 edges); `next build` OK; homepage card added.
2. `/pi0`: exporter (pi05 weights, BEHAVIOR frames) -> page -> self-test -> commit.
3. Hermes acceptance per page: value spot-check vs weights, step-through screenshots,
   no orphan arrows, no popups. Fail -> back to Opus.
4. Homepage: three cards -> `/vla` `/pi0` `/groot`. Final report to Frank.
5. SEPARATE TRACK (awaits Frank's go): restart k005b 10-step training from step_158500
   (`--flow-steps 1 --fresh-optimizer`); when step_160000 stamps, re-export `/vla`
   activations so its Euler animation shows REAL 10-step values.

## Standing rules (Frank)

- Entropy control law: fix problems NOW, however small; never defer cleanup.
- Real data only; toy must be labeled; never animate a mechanism the checkpoint doesn't have.
- Design-first gate: understanding doc -> design doc -> Frank review -> implement.
- Do not bounce micro-decisions to Frank (code-vs-weights download distinction, etc.).
- Every arrow has a visible source and target block; data visibly walks THROUGH the
  layer stacks ("portals").
