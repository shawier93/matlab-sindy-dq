# SINDy for d–q current dynamics

Prepare a ControlDesk recording, fit sparse differential equations for `id` and `iq`, and replay the currents with fixed-step RK4.

## Run in MATLAB

Use MATLAB R2021a or later. Open the repository folder in MATLAB. Signal Processing Toolbox is required when `smoothCurrents` is enabled.

1. Run `prepare_controldesk_data.m` and select the ControlDesk CSV. File 1 saves `combined_data.mat` and `combined_data.csv` under `results/<CSV name>/`.
2. Run File 2 using `dataFile` and `resultFolder` from File 1.

```matlab
prepare_controldesk_data
[model, replay] = identify_sindy(dataFile, OutputFolder=resultFolder);
```

For an existing MAT dataset, pass its path directly. With no file argument, File 2 reads `combined_data.mat` from the current folder. Use `OutputFolder` to keep separate runs.

```matlab
[model, replay] = identify_sindy("path/to/combined_data.mat", ...
    OutputFolder="results/validation", IncludeId=false, ...
    LibraryType="physics", TrainingPeriods=[1 2], Plot=false);
```

## Dataset format

The MAT file must contain a scalar struct named `D` with these fields.

| Field | Contents |
| --- | --- |
| `filtered` | N rows with `[id, iq, Vd, Vq, we]` in the first five columns. Currents are in A, voltages in the logged scale, and electrical speed in rad/s. |
| `t` | N timestamps in seconds on the combined timeline. |
| `segment` | N numeric segment IDs. Change the ID at each file boundary or discontinuity. |
| `Ts` | Positive sample interval in seconds, uniform within each segment. |

Optional fields include `raw` for raw-current plots, `tFile` for file-relative training windows, `joinRows` for boundaries within legacy segment IDs, and `validForIdentification` to mark usable samples. For older datasets without a validity mask, `sgFrame` specifies the Savitzky–Golay frame length. `filterBeforeSelection` indicates that filtering preceded segment selection.

## Candidate library

`sindy_library.m` defines the shared library used for training and replay. Its columns are `1`, `Vd`, `Vq`, `id`, `iq`, `we`, `id*we`, `iq*we`, `id*iq`, `iq^2`, `iq^3`, and `we*iq^2`.

- `LibraryType="broad"` is the default and gives each equation all library columns.
- `LibraryType="physics"` gives the d-axis equation `Vd`, `id`, and `iq*we`, and the q-axis equation `Vq`, `iq`, `we`, and `id*we`. Both equations also receive `id*iq`, `iq^2`, `iq^3`, and `we*iq^2`, as in the original physics library.
- `IncludeId=false` identifies the q-axis equation and supplies measured `id` during replay.

The sparse fit selects coefficients from these candidates. With column normalization enabled, `LambdaRel` controls pruning relative to the RMS derivative, and `Ridge` controls regularization. With `NormalizeColumns=false`, `Lambda` thresholds coefficients in physical units.

To evaluate a saved model on measured rows `Y` with the same five input columns:

```matlab
[Theta, terms] = sindy_library(Y);
assert(isequal(terms, model.terms));
derivatives = Theta * model.Xi;
```

`Xi` has one row per term and two columns ordered as `did/dt` and `diq/dt`. With `IncludeId=false`, only the second column represents a fitted equation.

## Training and replay

`TrainingPeriods` selects the segment IDs used in the fit. `FitTimeWindows` takes an N-by-2 array of inclusive time intervals in seconds. `FitTimeBasis="file"` uses `D.tFile` for those intervals. Replay covers all segments so that held-out segments can be scored as well.

Replay restarts from the measured currents at each window (`ReplayWindowSec=0.02` by default). Increase `ReplaySubsteps` to reduce the integration step. A replay window stops when a modeled current exceeds `MaxCurrent` or becomes nonfinite. Check `FailedWindows` alongside MSE, which uses the finite replay samples.

Each run saves a model MAT file, a coefficient CSV, and replay MAT and summary CSV files. `WriteReplayCSV=true` also writes the sample-level replay. `Plot=false` disables figures, and `ReplayEnabled=false` skips integration. `SaveTrainingCopy=true` saves a separate copy of `D`.
