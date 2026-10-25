# THREE-VIZ MISSION STATUS (single source of truth — read this first in any session)

Last updated: 2026-10-25 (/pi0 done). Owner: Hermes (orchestration + checking). Maker: Copilot Opus 5.5 (tmux `vlaviz:opus`).

## Mission

Three INDEPENDENT visualization pages, each explaining one model's internals with REAL
local weights and REAL BEHAVIOR dataset input frames. No downloads. No cross-model
fabricated mappings. No toy data. No popups/overlays — in-scene walkthrough style only.

| Page | Model | Weights (local) | Status |
|---|---|---|---|
| `/vla` | k005b dual-head VLA | `outputs/k005b-task0-v3/checkpoints` (step_150000) | REWORK DONE (a)(b)(c) + single-column layer-pair stack (28 pairs: trunk Li beside expert Ei, pair headers, tower titles beside the column; old 14+14 wrap deleted), self-tested: 84-shot sweep audit empty, build OK, /pi0 261 + /groot 292 edges clean; awaiting Hermes re-acceptance |
| `/pi0` | pi0.5 b1k finetune | `~/workspace/model/openpi_comet/pi05-b1kpt12-cs32` (+`pi05-b1kpt50-cs32`) | ENGINE INSTANCE (111ec10): same 3D engine as /vla, single-column 18 layer pairs, MQA joint attention, gated adaRMSNorm, 10-step Euler; self-test: audit noIn/noOut empty (1494 edges; trunk single-layer 650, RMSNorm 1514, D7 1495), 0 console errors, build OK; awaiting Hermes re-acceptance |
| `/groot` | GR00T N1.7-3B (incl. sim_behavior_r1_pro embodiment id=23, latest=global_step196000) | `~/workspace/model/nvidia/GR00T-N1.7-3B` | ENGINE INSTANCE (walls in src/groot deleted): same 3D engine as /vla, single column trunk 20 (16 LM + 4 vlsa) beside DiT 32 (cross text/self/cross image/self), VL-token bus to cross K/V, AdaLayerNorm, 4-step Euler t 0->1; all chapters have groot text; self-test: audit noIn/noOut empty (1809 edges; trunk single-layer 720, D7 1810), 0 console errors, build OK; slot 23 untrained -> action has no semantics (stated in D7); awaiting Hermes re-acceptance |

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
  pi0.5 (the implemented /pi0 model) has no state token: state is discretized into the
  text prefix; time enters via adaRMSNorm; action chunk 32x32 -> first 23 dims = R1 Pro.

## Remaining steps (in order)

1. ~~`/groot`: verify the 6 details -> exporter -> page -> self-test -> commit.~~ DONE:
   `tools/export_groot.py` -> `public/groot-*.json`; `src/groot/*`; G0-G9 self-test: 0 console
   errors, arrow audit empty (292 edges); `next build` OK; homepage card added.
2. ~~`/pi0`: exporter (pi05 weights, BEHAVIOR frames) -> page -> self-test -> commit.~~ DONE:
   `tools/export_pi0.py` (+ `tools/ref_pi0_jax.py`, official JAX agreement 2.6e-5) ->
   `public/pi0-*.json`; `src/pi0/*`; P0-P8 self-test: 0 console errors, arrow audit empty
   (261 edges); `next build` OK; homepage card added.
3. ~~Hermes acceptance per page~~ DONE 2026-10-25: `/groot` and `/pi0` both ACCEPTED
   (independent re-verification: arrow audits noIn/noOut empty, 0 console errors, no
   overlays/popups, weight spot-checks exact vs local checkpoints). `/vla` verified earlier.
4. ~~Homepage: three cards~~ DONE (all three cards + real screenshots served 200).
   (The earlier "mission complete" verdict is withdrawn — /vla was reopened by Frank's
   post-acceptance directives; see Current reasoning.)
5. SEPARATE TRACK (training): 10-step flow restart RAN overnight from step_150000
   (--flow-steps 1 regression mode, fresh optimizer), reached step 158500 healthy
   (action 0.539); **Frank PAUSED it at 10:25 Oct 25** — do not relaunch without his go.
   Resume point when he does: checkpoint step_150000 (158500 was not a save point).
   When step_160000 stamps, re-export `/vla` activations so its Euler animation shows
   REAL 10-step values.

## Current reasoning (rewritten each sentinel run)

2026-10-25 21:28 — (a)(b)(c) are now COMMITTED (baac081: trunk visual language, one
representative head ×16, label/tag dedupe, fit-to-view labels, occluder fade, framing
margin). Opus is in final polish: tower titles moved beside their columns (trunk left,
experts right) to stop overlapping embedding blocks, then the closing sequence is already
underway — full 84-shot sweep (w*.png via /tmp/dshots/list.txt) → `next build` →
/pi0+`/groot` regression audits → docs + commit. Its plan matches the required sequence
exactly; no steering. 5 files still modified mid-iteration (VlaModelLayout, Arrow,
SectionLabels, Walkthrough00, Walkthrough11) — normal while the sweep runs. After Opus
stops: Hermes re-acceptance of /vla (arrow audit noIn/noOut empty, 0 console errors,
per-step screenshots vs directive (b)) — that is the only remaining unowned step; do it
in the next sentinel run once Opus's closing commit lands. Still gated on Frank:
training restart (paused 10:25, do not relaunch; resume from step_150000); step_160000
stamp → real 10-step activation re-export replaces the pre-stamp preview data.

## Standing rules (Frank)

- Entropy control law: fix problems NOW, however small; never defer cleanup.
- Real data only; toy must be labeled; never animate a mechanism the checkpoint doesn't have.
- Design-first gate: understanding doc -> design doc -> Frank review -> implement.
- Do not bounce micro-decisions to Frank (code-vs-weights download distinction, etc.).
- Every arrow has a visible source and target block; data visibly walks THROUGH the
  layer stacks ("portals").
