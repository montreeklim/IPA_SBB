# Conjecture

Small Sage-based experiments for the following geometric operation.

Start with a simplex inscribed in the unit sphere, choose a point `P` on a facet, project it radially to

```text
Q = P / ||P||
```

and replace that facet by the smaller facets obtained by joining `Q` to the facet vertices.

In this repo:

- 2D means a triangle in the unit circle, with edges replaced by broken edges.
- 3D means a tetrahedron in the unit sphere, with triangular faces replaced by three triangles.

The main mathematical question is whether this preserves convexity.

## Current focus

The computational picture is:

- in 2D, the construction appears always convex;
- in 3D, convexity can fail;
- the relevant 3D one-face criterion seems to be whether the orthogonal projection of `Q` back to the face plane lands inside the original face.

The code is for examples, inspection, and visualization.

## Requirements

- `sage`
- a Unix-like environment if you want automatic opening via `xdg-open`

The script imports `sage.all`, so the normal entry point is:

```bash
sage conjecture.py -- [flags]
```

Plain `python conjecture.py` only works if that interpreter already has Sage available as a Python module.

## Quick Start

Default 3D run:

```bash
sage conjecture.py -- --seed 42
```
(this creates a nonconvex case)

This writes `output-<timestamp>.html` and opens it with `xdg-open`.

To suppress auto-open:

```bash
sage conjecture.py -- --seed 42 --no-open
```

To choose the output path:

```bash
sage conjecture.py -- --output my-example.html --no-open
```

## Modes

### Default 3D mode

Without `--2d` and without `--continue`, the script:

- starts from a regular tetrahedron on the unit sphere;
- samples one point on each of the four original faces;
- projects each sampled point radially;
- replaces every original face once.

### 2D mode

With `--2d`, the script switches to the triangle-in-circle analogue:

- start from a regular triangle on the unit circle;
- sample one point on each original edge;
- project each sampled point radially to the circle;
- replace each edge by two edges through the projected point.

### Continue mode

With `--continue X`, the script does iterative refinement instead:

- start from the raw triangle or tetrahedron;
- perform exactly `X` refinements;
- at each step choose uniformly from the full current boundary;
- refine the chosen current facet;
- add the new child facets back to the same boundary pool.

This is interpretation B of “continue”: `X` is the total number of refinements from the raw simplex, not “extra refinements after the default batch”.

## Display flags

### `--1-face`

`--1-face` is display-only.

It does not change the constructed object. It only highlights one refinement/facet in the rendering.

In 3D, `--1-face` also draws:

- the affine plane containing the displayed parent face;
- the circle cut out by that plane on the sphere;
- a normal segment indicating the associated halfspace direction.

`--1-dface` is just an alias for `--1-face`.

## Other flags

- `--seed N`: reproducible random choices.
- `--output PATH`: output HTML path. If `PATH` has no `.html` suffix, one is added.
- `--no-open`: do not call `xdg-open`.
- `--degenerate`: in default 3D all-face mode, force one sampled point to lie on an edge.

Notes:

- `--degenerate` is 3D-only.
- `--degenerate` cannot be combined with `--continue`.
- `--near-edge` still appears in `--help`, but the current code rejects it because `--1-face` no longer changes the construction.

## Examples

Default 3D:

```bash
sage conjecture.py -- --seed 42
```

2D:

```bash
sage conjecture.py -- --2d --seed 42
```

3D, display one face only:

```bash
sage conjecture.py -- --seed 42 --1-face
```

2D, display one edge only:

```bash
sage conjecture.py -- --2d --seed 42 --1-face
```

3D iterative refinement:

```bash
sage conjecture.py -- --continue 10 --seed 42
```

2D iterative refinement:

```bash
sage conjecture.py -- --2d --continue 10 --seed 42
```

Write without opening:

```bash
sage conjecture.py -- --continue 10 --seed 42 --no-open --output out.html
```

## Output

The script prints a short textual report to stdout and writes an HTML file.

Rendering currently splits as follows:

- 2D scenes are exported as an HTML page containing an embedded image.
- 3D scenes are exported through Sage's HTML viewer inside a thin wrapper page.

In `--continue` mode, the HTML can also include a right-hand legend listing the added points by refinement step.

## File layout

- [conjecture.py](/home/coniglio/repos/IPA_SBB/conjecture/conjecture.py): main script
- [AGENT.md](/home/coniglio/repos/IPA_SBB/conjecture/AGENT.md): working mathematical and code notes

## Notes on the code

The old multi-file setup was merged into one script. The current top-level dispatch is in `build_scene()`, which chooses between:

- default 2D all-edges mode,
- default 3D all-faces mode,
- iterative 2D mode,
- iterative 3D mode.

There are still some older helper builders in the file, but they are not on the main CLI path.
