# SINDy for d–q current dynamics

Prepare a data, fit sparse differential equations for `id` and `iq`, and replay the currents with fixed-step RK4.

## Run in MATLAB

Use MATLAB R2021a or later. Open the repository folder in MATLAB. Signal Processing Toolbox is required when `smoothCurrents` is enabled.

1. Run `prepare_data.m` and select the data CSV. File 1 creates `D` in the MATLAB workspace and saves `combined_data.mat` and `combined_data.csv` under `results/<CSV name>/`.
2. Run `identify_sindy.m` in the same workspace. File 2 reads `D` and saves the model and replay outputs in the folder created by File 1.

```matlab
prepare_controldesk_data
identify_sindy
```

To run File 2 in a new MATLAB session, load the saved `D` into the workspace first.

```matlab
load('results/my_recording/combined_data.mat', 'D');
identify_sindy
```

## Required workspace variable

File 2 requires a scalar struct named **`D`** in the MATLAB workspace. The saved copy is in `combined_data.mat`. File 2 uses `resultFolder` from File 1 when available and otherwise writes to `results/`.

| Field | Contents |
| --- | --- |
| `filtered` | N rows with `[id, iq, Vd, Vq, we]` in the first five columns. Currents are in A, voltages in the logged scale, and electrical speed in rad/s. |
| `t` | N timestamps in seconds on the combined timeline. |
| `segment` | N numeric segment IDs. Change the ID at each file boundary or discontinuity. |
| `Ts` | Positive sample interval in seconds, uniform within each segment. |

Optional fields include `raw` for raw-current plots, `tFile` for file-relative training windows, `joinRows` for boundaries within legacy segment IDs, and `validForIdentification` to mark usable samples. For older datasets without a validity mask, `sgFrame` specifies the Savitzky–Golay frame length. `filterBeforeSelection` indicates that filtering preceded segment selection.

## Candidate library

`sindy_library.m` defines the shared library used for training and replay. Its columns are `1`, `Vd`, `Vq`, `id`, `iq`, `we`, `id*we`, `iq*we`, `id*iq`, `iq^2`, `iq^3`, and `we*iq^2`.

- `libraryType = 'broad'` is the default and gives each equation all library columns.
- `libraryType = 'physics'` gives the d-axis equation `Vd`, `id`, and `iq*we`, and the q-axis equation `Vq`, `iq`, `we`, and `id*we`. Both equations also receive `id*iq`, `iq^2`, `iq^3`, and `we*iq^2`, as in the original physics library.
- `includeId = false` identifies the q-axis equation and supplies measured `id` during replay.

Edit the settings at the top of File 2. With column normalization enabled, `lambdaRel` controls pruning relative to the RMS derivative, and `ridge` controls regularization. With `normalizeColumns = false`, `lambda` thresholds coefficients in physical units.

After File 2 runs, the workspace contains `SindyModel` and `SindyReplay`. To evaluate the model on measured rows `Y` with the same five input columns, use the following commands.

```matlab
[Theta, terms] = sindy_library(Y);
assert(isequal(terms, SindyModel.terms));
derivatives = Theta * SindyModel.Xi;
```

`Xi` has one row per term and two columns ordered as `did/dt` and `diq/dt`. With `includeId = false`, only the second column represents a fitted equation.

## Training and replay

`trainingPeriods` selects the segment IDs used in the fit. `fitTimeWindows` takes an N-by-2 array of inclusive time intervals in seconds. `fitTimeBasis = 'file'` uses `D.tFile` for those intervals. Replay covers all segments so that held-out segments can be scored as well.

Replay restarts from the measured currents at each window (`replayWindowSec = 0.02` by default). Increase `replaySubsteps` to reduce the integration step. A replay window stops when a modeled current exceeds `maxCurrent` or becomes nonfinite. Check `FailedWindows` alongside MSE, which uses the finite replay samples.

Each run saves a model MAT file, a coefficient CSV, and replay MAT and summary CSV files. `writeReplayCSV = true` also writes the sample-level replay. `replayEnabled = false` skips integration. `saveTrainingCopy = true` saves a separate copy of `D`.
