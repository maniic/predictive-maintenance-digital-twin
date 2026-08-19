# Full results

Every model, every dataset. The headline table in the [README](../README.md)
shows only the winner per dataset; this is the complete picture, including the
models that lost and why that is interesting.

All figures trace to a committed JSON file in [`models/`](../models/) — see
[models/README.md](../models/README.md) for which run produced which row.

**Before comparing to a paper:** these metrics are computed over every sliding
window of every test trajectory, not one prediction per engine as the standard
C-MAPSS benchmark specifies. They are internally consistent and valid for
comparing architectures against each other on this data; they are not
leaderboard figures. Full explanation in
[architecture.md § Evaluation protocol](architecture.md#evaluation-protocol).

## FD001 — one operating condition, one fault mode

| Model | RMSE | MAE |
|-------|------|-----|
| **LSTM** | **13.48** | 9.86 |
| TwoStage | 14.01 | 10.17 |
| Ensemble | 14.14 | 10.81 |
| EnhancedLSTM-Weighted | 14.65 | 11.00 |
| Transformer | 14.66 | 10.96 |
| EnhancedLSTM-Asymmetric | 15.41 | 11.12 |
| ImprovedLSTM | 15.81 | 12.58 |
| GRU | 16.05 | 13.81 |
| CNN | 17.63 | 13.80 |

The easiest dataset, and the one where the plain bidirectional LSTM wins.
Nothing here needs the extra machinery: with a single flight condition and a
single fault mode, degradation is close to monotonic and a well-regularized
recurrent model captures it. The attention variants (ImprovedLSTM, GRU) added
capacity without added signal and landed mid-table — which is why they are not
in the README's headline set.

## FD002 — six operating conditions, one fault mode

| Model | RMSE | MAE |
|-------|------|-----|
| **EnhancedLSTM-Asymmetric** | **16.77** | 13.76 |
| EnhancedLSTM-Weighted | 16.94 | 13.71 |
| LSTM | 17.49 | 14.01 |
| TwoStage | 17.50 | 13.48 |
| Ensemble | 19.80 | 17.02 |
| CNN | 20.29 | 15.83 |
| Transformer | 39.36 | 36.33 |

The Transformer's collapse is the most informative result in the project. At
39.4 RMSE it is more than twice as bad as the winner, and it is the *only*
dataset where it fails this way. Six flight conditions mean the same raw sensor
value carries different information depending on regime; the recurrent models
absorb that through their state, and the regime feature helps, but a
self-attention encoder with only sinusoidal positional information has no
mechanism to condition on it. Capacity was not the problem — this is an
inductive-bias mismatch.

The ensemble also drops below its own members here, because it averages the
failing Transformer into the result. Inverse-validation-RMSE weighting reduces
that contribution but does not eliminate it.

## FD003 — one operating condition, two fault modes

| Model | RMSE | MAE |
|-------|------|-----|
| **TwoStage** | **11.71** | 7.53 |
| EnhancedLSTM-Asymmetric | 12.00 | 8.17 |
| LSTM | 12.23 | 8.37 |
| EnhancedLSTM-Weighted | 13.38 | 8.86 |
| Ensemble | 13.66 | 10.13 |
| CNN | 16.82 | 12.05 |
| Transformer | 19.47 | 14.60 |

The best single result in the project. Two fault modes (HPC and fan degradation)
mean two different degradation signatures in one dataset, and the Two-Stage
model's explicit health classification is exactly the right structure: it can
route a window to a specialized regression head instead of forcing one head to
model both signatures at once.

## FD004 — six operating conditions, two fault modes

| Model | RMSE | MAE |
|-------|------|-----|
| **EnhancedLSTM-Asymmetric** | **14.75** | 9.33 |
| LSTM | 14.87 | 9.83 |
| Ensemble | 16.02 | 10.14 |
| TwoStage | 16.68 | 9.81 |
| EnhancedLSTM-Weighted | 17.19 | 12.96 |
| CNN | 17.45 | 11.23 |
| Transformer | 19.23 | 11.67 |

The hardest dataset — six conditions and two fault modes at once — and the
margin at the top is thin: 14.75 against 14.87 is within noise of a rerun. The
honest reading is that the asymmetric loss and the plain LSTM are tied here, and
the asymmetric variant's real advantage is the *shape* of its errors rather than
their magnitude: it is trained to prefer under-predicting remaining life, which
is the cheaper mistake.

## What the comparison shows

- **No architecture wins everywhere.** LSTM takes FD001, Two-Stage takes FD003,
  the asymmetric EnhancedLSTM takes both six-condition datasets.
- **Structure beats capacity.** The two wins by non-LSTM models come from
  models whose structure matches the dataset's difficulty — health-state routing
  for multiple fault modes, asymmetric loss for multiple regimes — not from
  models with more parameters.
- **The ensemble is never the best.** It is second or third almost everywhere,
  which is what a robustness-oriented average should look like, and it is
  actively hurt on FD002 by including a model that fails there.
- **Difficulty ordering is as designed.** FD003 (11.71) < FD001 (13.48) <
  FD004 (14.75) < FD002 (16.77). FD003 beats FD001 despite having two fault
  modes because its trajectories are longer, giving the models more history.

## Reproducing a row

```bash
pip install -e .
python scripts/train.py --models twostage --datasets FD003
```

Hyperparameters are carried over verbatim from the scripts that produced these
results, so a re-run trains the same configuration. Exact figures will differ
slightly — training is seeded for the data split but not for CUDA/CPU kernel
non-determinism.
