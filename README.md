# Predictive Maintenance Digital Twin

A deep learning system for predicting Remaining Useful Life (RUL) of turbofan jet engines using the NASA C-MAPSS dataset. Features multiple neural network architectures, ensemble methods, and an interactive web dashboard.

## Results

### Model Performance on C-MAPSS Dataset

Best results per dataset (RMSE in cycles):

| Dataset | Best Model | RMSE | MAE | Operating Conditions | Fault Modes |
|---------|------------|------|-----|---------------------|-------------|
| FD001 | LSTM | **13.48** | 9.86 | 1 | 1 (HPC) |
| FD002 | EnhancedLSTM | **16.77** | 13.76 | 6 | 1 (HPC) |
| FD003 | TwoStage | **11.71** | 7.53 | 1 | 2 (HPC+Fan) |
| FD004 | LSTM | **14.87** | 9.83 | 6 | 2 (HPC+Fan) |

### All Model Results

<details>
<summary>FD001 Results</summary>

| Model | RMSE | MAE |
|-------|------|-----|
| LSTM | 13.48 | 9.86 |
| TwoStage | 14.01 | 10.17 |
| Ensemble | 14.14 | 10.81 |
| Transformer | 14.66 | 10.96 |
| EnhancedLSTM-Weighted | 14.65 | 11.00 |
| EnhancedLSTM-Asymmetric | 15.41 | 11.12 |
| CNN | 17.63 | 13.80 |

</details>

<details>
<summary>FD002 Results</summary>

| Model | RMSE | MAE |
|-------|------|-----|
| EnhancedLSTM-Asymmetric | 16.77 | 13.76 |
| EnhancedLSTM-Weighted | 16.94 | 13.71 |
| LSTM | 17.49 | 14.01 |
| TwoStage | 17.50 | 13.48 |
| Ensemble | 19.80 | 17.02 |
| CNN | 20.29 | 15.83 |
| Transformer | 39.36 | 36.33 |

</details>

<details>
<summary>FD003 Results</summary>

| Model | RMSE | MAE |
|-------|------|-----|
| TwoStage | 11.71 | 7.53 |
| EnhancedLSTM-Asymmetric | 12.00 | 8.17 |
| LSTM | 12.23 | 8.37 |
| EnhancedLSTM-Weighted | 13.38 | 8.86 |
| Ensemble | 13.66 | 10.13 |
| CNN | 16.82 | 12.05 |
| Transformer | 19.47 | 14.60 |

</details>

<details>
<summary>FD004 Results</summary>

| Model | RMSE | MAE |
|-------|------|-----|
| EnhancedLSTM-Asymmetric | 14.75 | 9.33 |
| LSTM | 14.87 | 9.83 |
| Ensemble | 16.02 | 10.14 |
| TwoStage | 16.68 | 9.81 |
| EnhancedLSTM-Weighted | 17.19 | 12.96 |
| CNN | 17.45 | 11.23 |
| Transformer | 19.23 | 11.67 |

</details>

## Features

- **7 Deep Learning Models**: LSTM, CNN, Transformer, Enhanced LSTM variants, Two-Stage predictor, Ensemble
- **MLflow Integration**: Experiment tracking, model versioning, metric logging
- **Interactive Web Dashboard**: Real-time predictions, simulation, model comparison
- **Comprehensive Evaluation**: RMSE, MAE, C-MAPSS scoring function

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for web dashboard)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/predictive-maintenance-digital-twin.git
cd predictive-maintenance-digital-twin

# Create Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Download Dataset

Download the NASA C-MAPSS dataset and place files in `data/raw/`:

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
├── test_FD002.txt
├── RUL_FD002.txt
├── train_FD003.txt
├── test_FD003.txt
├── RUL_FD003.txt
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
```

Dataset available from: [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

### Train Models

```bash
# Train all base models on FD001
python scripts/train_all.py --dataset FD001

# Train a specific model
python scripts/train.py --model lstm --dataset FD001
```

### Run Web Dashboard

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)


## Models

### LSTM (Bidirectional)
- 2-layer bidirectional LSTM
- Hidden size: 128
- Dropout: 0.3
- Best for single operating condition datasets

### Temporal CNN
- 4 convolutional blocks with increasing channels
- Kernel size: 3 with dilation
- Batch normalization + ReLU

### Transformer
- 4-layer encoder with 4 attention heads
- Positional encoding
- Model dimension: 64

### Enhanced LSTM
- Attention mechanism
- Residual connections
- Weighted or asymmetric loss functions

### Two-Stage Predictor
- Health indicator extraction (autoencoder)
- RUL regression from health features
- Best for multi-fault mode datasets

### Ensemble
- Weighted average of LSTM, CNN, Transformer
- Uncertainty quantification
- Generally most robust

## Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  sequence_length: 30
  stride: 1
  val_ratio: 0.2

preprocessing:
  normalization: minmax
  rul_cap: 125

training:
  batch_size: 64
  learning_rate: 0.001
  max_epochs: 100
  early_stopping_patience: 15
```

## C-MAPSS Dataset

The Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset contains run-to-failure data from turbofan engines:

| Dataset | Engines (Train) | Engines (Test) | Operating Conditions | Fault Modes |
|---------|-----------------|----------------|---------------------|-------------|
| FD001 | 100 | 100 | 1 | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC degradation) |
| FD003 | 100 | 100 | 1 | 2 (HPC + Fan degradation) |
| FD004 | 249 | 248 | 6 | 2 (HPC + Fan degradation) |

**Sensors (21 total):**
- Temperature measurements
- Pressure measurements
- Fan/core speeds
- Bypass ratio, bleed enthalpy, coolant bleed
- And more

## Evaluation Metrics

- **RMSE**: Root Mean Square Error (cycles)
- **MAE**: Mean Absolute Error (cycles)
- **C-MAPSS Score**: Asymmetric scoring function that penalizes late predictions more heavily

```
Score = Σ exp(-d/13) - 1  if d < 0 (early prediction)
        Σ exp(d/10) - 1   if d ≥ 0 (late prediction)
```

## Acknowledgments

- NASA Prognostics Center of Excellence for the C-MAPSS dataset
