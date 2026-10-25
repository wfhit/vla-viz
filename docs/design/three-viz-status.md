# THREE-VIZ MISSION STATUS (single source of truth — read this first in any session)

Last updated: 2026-10-25 (/pi0 done). Owner: Hermes (orchestration + checking). Maker: Copilot Opus 5.5 (tmux `vlaviz:opus`).

## Mission

Three INDEPENDENT visualization pages, each explaining one model's internals with REAL
local weights and REAL BEHAVIOR dataset input frames. No downloads. No cross-model
fabricated mappings. No toy data. No popups/overlays — in-scene walkthrough style only.

| Page | Model | Weights (local) | Status |
|---|---|---|---|
| `/vla` | k005b dual-head VLA | `outputs/k005b-task0-v3/checkpoints` (step_150000) | REWORK DONE (a)(b)(c), self-tested; awaiting Hermes re-acceptance (see Current reasoning) |
| `/pi0` | pi0.5 b1k finetune | `~/workspace/model/openpi_comet/pi05-b1kpt12-cs32` (+`pi05-b1kpt50-cs32`) | DONE + ACCEPTED (Hermes, 2026-10-25: auditEdges 261 edges noIn/noOut empty on P5/P7 re-run independently, 0 console errors, no overlays, 2 weight spot-checks exact vs openpi checkpoint) |
| `/groot` | GR00T N1.7-3B (incl. sim_behavior_r1_pro embodiment id=23, latest=global_step196000) | `~/workspace/model/nvidia/GR00T-N1.7-3B` | DONE + ACCEPTED (Hermes, 2026-10-25) |

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

2026-10-25 20:26 — MISSION REOPENED on /vla: the earlier /vla acceptance is SUPERSEDED.
Frank issued three directives after acceptance, in order: (a) D-chapters must reuse the
page's established visual language (one per-layer view: AdaRMSNorm→Q/K/V→attn→o_proj→⊕→
AdaRMSNorm→SwiGLU→⊕, 28 isomorphic layers; t-bus + Euler strip are annotations, not a new
style); (b) layout: each step's workset centered, camera close (fill 1.05–1.7), attention
matrices/head blocks anchored inside their layer blocks, labels pinned to blocks — sweep
all 21 steps with per-step screenshots; (c) collapse repetition: same-type block drawn
once + "×16/×32 heads" multiplicity text, at most one same-name label per screen pinned
to its block, priority = finish the 28-layer structure main line (layer→coupling→output)
before decorative detail. Opus is currently implementing (c) exactly (VlaModelLayout
z-slot collapse, hide-extra-heads, label dedupe, multiplicity labels) — no steering
needed. ~20 files uncommitted (+1209/−664 incl. Walkthrough11 649 lines); k005b-acts.json
re-export on old weights is sanctioned (Frank ruling: pre-stamp preview with warning).
2026-10-25 update — (a)(b)(c) implemented and self-tested: one representative head / kv
group per layer (others share its z slot, hidden; text and label say "x16 heads"); same-name
labels and block tags deduped (nearest the focus wins); pinned labels dropped when the
label plus its text do not fit entirely in view (no giant close-up fragments); blocks in
front of the framed working set fade; frameCubes adds a 1.2 margin and lifts the subject
above the caption bar. Sweep of 84 shots (21 chapters × steps 0/2/4/6): arrow audit
noIn/noOut empty on all, 0 console errors; `next build` OK; /pi0 (261 edges) and /groot
(292 edges) regression audits empty.
Sequence when Opus stops: self-test (0 console errors, arrow audit noIn/noOut empty,
per-step screenshots vs directive (b)) → Hermes re-acceptance of /vla → commit. Only then
is the mission complete again. Still gated on Frank: training restart (paused 10:25, do
not relaunch); step_160000 stamp → real 10-step activation re-export replaces the
pre-stamp preview data.

## Standing rules (Frank)

- Entropy control law: fix problems NOW, however small; never defer cleanup.
- Real data only; toy must be labeled; never animate a mechanism the checkpoint doesn't have.
- Design-first gate: understanding doc -> design doc -> Frank review -> implement.
- Do not bounce micro-decisions to Frank (code-vs-weights download distinction, etc.).
- Every arrow has a visible source and target block; data visibly walks THROUGH the
  layer stacks ("portals").
