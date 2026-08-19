# Training results

Every RMSE and MAE quoted in the project README traces to one of the three JSON
files in this directory. Each row is one architecture on one C-MAPSS sub-dataset,
evaluated on held-out test engines.

| File | Models | Produced by |
|---|---|---|
| `training_results.json` | LSTM, CNN, Transformer, Ensemble — all four datasets | the base training path, now `scripts/train.py --models lstm cnn transformer --regimes config` |
| `advanced_training_results.json` | EnhancedLSTM-Weighted, EnhancedLSTM-Asymmetric, TwoStage — all four datasets | the advanced training path, now `scripts/train.py --models enhanced-lstm-weighted enhanced-lstm-asymmetric twostage --regimes auto` |
| `improved_training_results.json` | ImprovedLSTM, GRU — FD001 only | `scripts/train.py --models improved-lstm gru --datasets FD001 --regimes config` |

`scripts/train.py` writes new runs to `models/training_runs/` (git-ignored) so
these published records are never overwritten by an experiment.

## Reading the numbers

**Evaluation protocol.** RMSE and MAE are computed over *every sliding window* of
every test trajectory, not over one prediction per engine. The standard C-MAPSS
benchmark scores a single final-window prediction per engine, so these figures are
**not directly comparable to published C-MAPSS leaderboard results** — most of the
extra windows sit early in a trajectory where the target is pinned at the 125-cycle
cap. See `docs/architecture.md`.

**Excluded engines.** Engines whose test trajectory is shorter than the 30-cycle
window produce no sequences and contribute nothing: 6 engines in FD002, 11 in
FD004, none in FD001 or FD003.

**`test_cmapss` is `null` in two files.** The scripts that produced
`advanced_training_results.json` and `improved_training_results.json` called
`cmapss_score(targets, preds)`, reversing the `(y_pred, y_true)` argument order and
inverting the asymmetry the metric exists to express. The stored scores were
therefore meaningless and have been removed rather than published. RMSE and MAE are
symmetric and unaffected. `scripts/train.py` calls the metric correctly.

**`regimes_mode`.** Newer rows record whether operating-regime clustering was
applied to every dataset (`config`) or only to the genuinely multi-condition
FD002/FD004 (`auto`). The published rows predate this field; the table above states
which setting each used.

## Checkpoints

Model checkpoints (`checkpoints/`) and fitted preprocessors (`preprocessors/`) are
**not** committed — they are large binaries that would dominate the clone. The
`checkpoint` field records where a run wrote its best checkpoint, relative to the
repository root. Regenerate them with `scripts/train.py`.
