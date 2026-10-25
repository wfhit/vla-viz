
# VLA 可视化

3D interactive visualizations of vision-language-action models, built from real weights and real
activations. Forked from Brendan Bycroft's llm-viz (MIT, see [`LICENSE`](LICENSE)); third-party
assets (fonts) retain their original licenses. Internal use only.

Pages:
* `/vla` — k005b VLA (Qwen3-VL-2B trunk + DiT action expert, 10-step flow matching).
* `/pi0`, `/groot` — planned, see `docs/design/three-viz-plan.md`.

### k005b VLA Visualization

The 3D view renders the real k005b fused VLA checkpoint (Qwen3-VL-2B trunk, 28 layers, C=2048,
GQA 16Q/8KV × 128, SwiGLU FF 6144, plus the action expert, knack head and done head).
Every displayed number is real: weights are read from the checkpoint, and activations are captured
with forward hooks while running one real sample. Large tensors show a real sub-window (the true
indices and full-tensor stats are shown on hover and in the model card).

The data files `public/k005b-weights.json`, `public/k005b-acts.json` and `public/k005b-check.json`
are produced by the exporter (CPU only, ~20s):

```
python tools/export_k005b.py
```

`k005b-check.json` holds reference values computed in Python, used to verify the values read in the browser.

## Running Locally

1. Install dependencies: `npm install`
1. Start the dev server: `npm run dev`
