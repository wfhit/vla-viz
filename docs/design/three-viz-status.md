# THREE-VIZ MISSION STATUS (single source of truth — read this first in any session)

Last updated: 2026-10-26 02:49 sentinel (all pages re-accepted, waiting on Frank's training go). Owner: Hermes (orchestration + checking). Maker: Copilot Opus 5.5 (tmux `vlaviz:opus`). Dev server: localhost:3002.

## Mission

Three INDEPENDENT visualization pages, each explaining one model's internals with REAL
local weights and REAL BEHAVIOR dataset input frames. No downloads. No cross-model
fabricated mappings. No toy data. No popups/overlays — in-scene walkthrough style only.

| Page | Model | Weights (local) | Status |
|---|---|---|---|
| `/vla` | k005b dual-head VLA | `outputs/k005b-task0-v3/checkpoints` (step_150000) | REWORK DONE (a)(b)(c) + single-column layer-pair stack (28 pairs: trunk Li beside expert Ei, pair headers, tower titles beside the column; old 14+14 wrap deleted). RE-ACCEPTED 2026-10-25 late: 21-phase sweep audit noIn/noOut empty (2332 edges; trunk 1024, RMSNorm 2352, D7 2333), 0 JS errors, no overlays; k005b-*.json untouched since prior acceptance |
| `/pi0` | pi0.5 b1k finetune | `~/workspace/model/openpi_comet/pi05-b1kpt12-cs32` (+`pi05-b1kpt50-cs32`) | ENGINE INSTANCE (111ec10): same 3D engine as /vla, single-column 18 layer pairs, MQA joint attention, gated adaRMSNorm, 10-step Euler. RE-ACCEPTED 2026-10-25 late: 17-phase sweep audit noIn/noOut empty (1494 edges; trunk 650, RMSNorm 1514, D7 1495), 0 JS errors, no overlays; weight spot-checks EXACT vs zarr checkpoint (L0.q_w permuted [h,dim,in], L0.gate_w [F,in], both max|diff|=0) |
| `/groot` | GR00T N1.7-3B (incl. sim_behavior_r1_pro embodiment id=23, latest=global_step196000) | `~/workspace/model/nvidia/GR00T-N1.7-3B` | ENGINE INSTANCE (e3ff769, walls in src/groot deleted): same 3D engine as /vla, single column trunk 20 (16 LM + 4 vlsa) beside DiT 32 (cross text/self/cross image/self), VL-token bus to cross K/V, AdaLayerNorm, 4-step Euler t 0->1; all chapters have groot text. RE-ACCEPTED 2026-10-25 late: 17-phase sweep audit noIn/noOut empty (1809 edges; trunk single-layer 720, RMSNorm 1817, D7 1810), 0 JS errors, no overlays; weight spot-checks EXACT vs safetensors (L0.ln1_w full, L0.q_w strided, E0.q_w head0 slice, E0.q_b — all max|diff|=0); slot 23 untrained -> action has no semantics (stated in D7) |

Input frames for ALL pages: local BEHAVIOR dataset
`~/workspace/behavior_challenge_data/dataset/{behavior-1k-knack, 2026-challenge-demos}`.

Path continuity (Frank's top rule, plan §3.6): DONE on all 3 pages. Traced object per chapter =
last token, one channel; W03 RMSNorm, W04/W05 attention (q,k,s_j,p_j,o[0]), W06 o_proj+residual
(closes back to W03's x[c]), D7 Euler step 1 (x -> input chain -> every layer -> v -> x_next =
board 1). Ledgers in commentary + ASCII 3D tags on the traced cells, same numbers.
`tools/check_trace.py public/*-trace.json` = [ok] x3 (also cross-checks fm.json boards).
3D chain now complete in all three demo chapters: W03 x[c] -> mean/rsqrt readouts (beside the
traced column) -> g[c] -> y[c]; W04 q/k/s/p/o; D7 pass 1 h0 -> E{l}[c] per layer -> norm -> v ->
x/x_next. Block-name tags that overlap on screen keep only the brightest/nearest one.
Regression: per-chapter shots on all 3 pages, 600-step sweep audits noIn/noOut empty.

## Verified facts (do not re-litigate)

- k005b action head: K-step Euler decode (`sample_action_chunk`), K=10 default, K=1 = old
  single-shot; checkpoint `action_decode` stamp gates K (unstamped lineage = K 1 only).
  PR #638 merged (K-step decode); bf16 boundary fix 20b39c2 merged as PR #639
  (2026-10-25 16:12 CST); docs entropy pass merged as PR #640; 155 tests green.
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

2026-10-26 06:54 — NO CHANGE since 05:53; the reasoning stands. Facts checked this
run: HEAD still e3ff769, working tree clean except this doc; all three pages 200 on
port 3002; Opus idle with no pending decisions; no training process alive and no
training log to tail — training still PAUSED by Frank (10:25 Oct 25), resume point
step_150000, cc10step session idle. The one thing that matters is unchanged: it is
Frank's gate, not ours.
Correct order from here: (1) Frank says go -> resume from step_150000 in cc10step;
(2) at step_160000 stamped -> re-export /vla activations -> /vla Euler shows real
10-step values -> light re-audit of /vla only; (3) until then every sentinel run =
facts + this reasoning, silent. Known accepted cosmetic item (groot D7 strip labels)
stays as-is.

## Standing rules (Frank)

- Entropy control law: fix problems NOW, however small; never defer cleanup.
- Real data only; toy must be labeled; never animate a mechanism the checkpoint doesn't have.
- Design-first gate: understanding doc -> design doc -> Frank review -> implement.
- Do not bounce micro-decisions to Frank (code-vs-weights download distinction, etc.).
- Every arrow has a visible source and target block; data visibly walks THROUGH the
  layer stacks ("portals").
