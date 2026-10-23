
# Brendan Bycroft's Home Page & Projects

This repository contains my (Brendan's) homepage, as well as a number of non-trivial projects.

They are kept in a single repository for ease of deployment, as well as sharing a bunch of js utils
which are otherwise a pain to share around.

**License:** MIT (see [`LICENSE`](LICENSE)), except the personal homepage (`src/homepage/`, `public/me.jpeg`)
and third-party assets (fonts, RISC-V tests, etc.), which retain their original rights/licenses.

## Projects

The main projects are:
* k005b VLA Visualization: 3D interactive model of the real k005b VLA (Qwen3-VL-2B trunk + action expert) running inference.
* [WIP] CPU Simulation: A 2D digital schematic editor with full a execution model, showcasing a simple
  RISC-V based CPU

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

### CPU Simulation (WIP; not exposed yet!)

This project runs 2D schematic digital circuits, with a fully fledged editor. The intent is to
add a number of walkthroughs, showing things such as:
  * how a simple RISC-V CPU is constructed
  * the constituent parts down to gate level: instruction decode, ALU, add, etc
  * higher level CPU ideas, like various levels of pipelining, caching, etc

## Running Locally

1. Install dependencies: `yarn`
1. Start the dev server: `yarn dev`
