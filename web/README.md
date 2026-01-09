# Digital Twin Dashboard

Next.js web application for the Predictive Maintenance Digital Twin system. Provides real-time RUL (Remaining Useful Life) predictions for turbofan engines using deep learning models.

## Tech Stack

- **Next.js 14** - React framework with App Router
- **Tailwind CSS** - Utility-first styling
- **Plotly.js** - Interactive charts and visualizations
- **Python Backend** - ML model inference via subprocess

## Features

- **Prediction** - Run RUL predictions on C-MAPSS dataset engines with multiple models
- **Simulation** - Real-time engine degradation simulation with live RUL updates
- **Comparison** - Compare model performance across datasets (RMSE, MAE, C-MAPSS Score)

## Setup

### Prerequisites

- Node.js 18+
- Python 3.10+ with project dependencies installed

### Install Dependencies

```bash
cd web
npm install
```

### Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
web/
├── app/
│   ├── api/              # API routes (Python bridge)
│   │   ├── comparison/   # Model comparison data
│   │   ├── engines/      # Available engines list
│   │   └── predict/      # RUL prediction endpoint
│   ├── comparison/       # Model comparison page
│   ├── prediction/       # Prediction interface
│   ├── simulation/       # Live simulation demo
│   ├── globals.css       # Global styles + CSS variables
│   ├── layout.jsx        # Root layout
│   └── page.jsx          # Home page
├── public/               # Static assets
├── package.json
├── tailwind.config.js
└── README.md
```

## API Endpoints

All API routes spawn Python subprocesses for ML inference:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Run RUL prediction for an engine |
| `/api/engines` | GET | List available engines per dataset |
| `/api/comparison` | GET | Get model comparison metrics |

### Prediction Request

```json
{
  "dataset": "FD001",
  "engine": 1,
  "model": "ensemble"
}
```

### Prediction Response

```json
{
  "dataset": "FD001",
  "engine": 1,
  "model": "ensemble",
  "predicted_rul": 45.2,
  "confidence_interval": [38.1, 52.3],
  "model_predictions": {
    "lstm": 43.5,
    "cnn": 47.8,
    "transformer": 44.3
  }
}
```

## Running with Python Backend

The web app requires the Python ML backend to be set up:

1. From the project root, set up the Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. Ensure trained models exist in `models/` directory

3. Start the Next.js dev server from `web/` directory:
   ```bash
   npm run dev
   ```

## Design System

### CSS Variables

```css
--bg-primary: #050505
--bg-secondary: #0a0a0a
--bg-panel: #0f0f0f
--accent: #00ffaa
--accent-red: #ff4444
--accent-amber: #ffaa00
--text-primary: #fafafa
--text-secondary: #666
--border: #1a1a1a
```

### Utility Classes

- `.panel` - Card container with border
- `.btn-primary` / `.btn-secondary` - Button styles
- `.metric-value` / `.metric-label` - Data display
- `.nav-link` - Navigation links
- `.status-dot` - Status indicators
- `.skeleton` - Loading placeholder

## Responsive Design

- Mobile-first approach
- Collapsible sidebars on mobile (< 768px)
- Responsive grids for metrics and cards
- Horizontally scrollable tables on small screens
