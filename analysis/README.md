# Layer-1 analysis

Run from the `twolayer-torch` directory:

```bash
python -m analysis.analyse
```

Or point at explicit paths:

```bash
python -m analysis.analyse \
  --checkpoints analysis/checkpoints \
  --output analysis/output \
  --data-root natural-movies \
  --device cpu
```

## Inputs

`analysis/checkpoints/` should contain:

- `A.npy` — required, shape `(M, 1, N)` complex64.
- `whitening_patch20.npz` — required for pixel-space figures and
  reconstruction.
- `layer1_energy.npy` or `layer1_last.pt` — optional, used for the
  training-energy plot.

Raw movie chunks are read from `natural-movies/vid075-chunks/` by default.
Use `--n-chunks` to limit how many are loaded, or `--data-root` to point
elsewhere. If no chunks are found, the raw-data sections are skipped.


