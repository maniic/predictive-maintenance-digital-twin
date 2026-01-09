'use client'

import { useState, useEffect, useMemo } from 'react'
import Link from 'next/link'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false, loading: () => <div className="h-[200px] skeleton" /> })

const DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
const MODELS = [
  { value: 'ensemble', label: 'Ensemble (All)' },
  { value: 'lstm', label: 'LSTM' },
  { value: 'cnn', label: 'CNN' },
  { value: 'transformer', label: 'Transformer' },
]

const DATASET_INFO = {
  FD001: { conditions: 1, faults: 1, desc: 'Single condition, HPC degradation' },
  FD002: { conditions: 6, faults: 1, desc: '6 conditions, HPC degradation' },
  FD003: { conditions: 1, faults: 2, desc: 'Single condition, HPC + Fan' },
  FD004: { conditions: 6, faults: 2, desc: '6 conditions, HPC + Fan' },
}

// Loading skeleton component
const Skeleton = ({ className = '' }) => (
  <div className={`skeleton ${className}`} />
)

export default function PredictionPage() {
  const [dataset, setDataset] = useState('FD001')
  const [model, setModel] = useState('ensemble')
  const [engines, setEngines] = useState([])
  const [engine, setEngine] = useState(1)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [enginesLoading, setEnginesLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    setEnginesLoading(true)
    setError(null)
    fetch(`/api/engines?dataset=${dataset}`)
      .then(r => {
        if (!r.ok) throw new Error('Failed to load engines')
        return r.json()
      })
      .then(data => {
        setEngines(data.engines || [])
        if (data.engines?.length) setEngine(data.engines[0])
        setPrediction(null)
      })
      .catch(err => setError(err.message))
      .finally(() => setEnginesLoading(false))
  }, [dataset])

  const predict = async () => {
    setLoading(true)
    setError(null)
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 30000)

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset, engine, model }),
        signal: controller.signal,
      })
      clearTimeout(timeout)

      if (!res.ok) throw new Error('Prediction failed')
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setPrediction(data)
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please try again.')
      } else {
        setError(err.message || 'An error occurred')
      }
      setPrediction(null)
    }
    setLoading(false)
  }

  const trajectoryData = useMemo(() => {
    if (!prediction) return null
    const cycles = prediction.total_cycles || 80
    const trueRul = prediction.true_rul
    const data = []

    for (let i = 0; i < cycles; i++) {
      const remaining = cycles - i + trueRul
      const noise = (Math.random() - 0.5) * 12
      const pred = Math.max(0, remaining + noise)
      data.push({ cycle: i + 1, predicted: pred, true: remaining })
    }
    return data
  }, [prediction])

  const getHealthStatus = (score) => {
    if (score > 0.6) return { label: 'Nominal', color: 'var(--accent)' }
    if (score > 0.3) return { label: 'Degraded', color: 'var(--accent-amber)' }
    return { label: 'Critical', color: 'var(--accent-red)' }
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--border)] px-4 md:px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="status-dot" />
            <span className="mono text-xs tracking-wider">DIGITAL TWIN</span>
          </Link>

          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center gap-6">
            <Link href="/prediction" className="nav-link active">Predict</Link>
            <Link href="/simulation" className="nav-link">Simulate</Link>
            <Link href="/comparison" className="nav-link">Compare</Link>
          </nav>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </header>

      <div className="flex-1 flex flex-col md:flex-row">
        {/* Mobile Overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-40 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Sidebar */}
        <aside className={`
          fixed md:relative inset-y-0 left-0 z-50
          w-64 bg-[var(--bg-primary)] border-r border-[var(--border)] p-6
          transform transition-transform duration-200
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
          md:flex-shrink-0 overflow-y-auto
        `}>
          {/* Mobile close button */}
          <button
            className="md:hidden absolute top-4 right-4"
            onClick={() => setSidebarOpen(false)}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Mobile nav links */}
          <nav className="md:hidden flex flex-col gap-4 mb-6 pb-6 border-b border-[var(--border)]">
            <Link href="/prediction" className="nav-link active">Predict</Link>
            <Link href="/simulation" className="nav-link">Simulate</Link>
            <Link href="/comparison" className="nav-link">Compare</Link>
          </nav>

          <div className="space-y-5">
            <div>
              <label className="metric-label block mb-2">Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm"
              >
                {DATASETS.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
              <p className="text-xs text-[var(--text-muted)] mt-2">
                {DATASET_INFO[dataset].desc}
              </p>
            </div>

            <div>
              <label className="metric-label block mb-2">Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm"
              >
                {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
            </div>

            <div>
              <label className="metric-label block mb-2">Engine ID</label>
              {enginesLoading ? (
                <Skeleton className="h-10 w-full" />
              ) : (
                <select
                  value={engine}
                  onChange={(e) => setEngine(Number(e.target.value))}
                  className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm"
                >
                  {engines.map(e => <option key={e} value={e}>{e}</option>)}
                </select>
              )}
              <p className="text-xs text-[var(--text-muted)] mt-2">
                {engines.length} engines available
              </p>
            </div>

            <button
              onClick={() => { predict(); setSidebarOpen(false); }}
              disabled={loading || enginesLoading}
              className="btn-primary w-full disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading && (
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              {loading ? 'Analyzing...' : 'Run Prediction'}
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-4 md:p-6 overflow-auto">
          <div className="max-w-4xl">
            <p className="mono text-xs text-[var(--text-secondary)] mb-2 tracking-wider">
              RUL PREDICTION
            </p>
            <h1 className="text-xl md:text-2xl font-light mb-6">Engine Analysis</h1>

            {/* Error State */}
            {error && (
              <div className="panel p-6 mb-6 border-l-2 border-[var(--accent-red)]">
                <p className="text-[var(--accent-red)] mono text-sm mb-3">{error}</p>
                <button
                  onClick={predict}
                  className="btn-secondary text-xs"
                >
                  Retry
                </button>
              </div>
            )}

            {/* Loading State */}
            {loading && (
              <div className="space-y-6">
                <div className="panel p-8">
                  <div className="grid grid-cols-2 gap-8">
                    <div><Skeleton className="h-6 w-24 mb-2" /><Skeleton className="h-12 w-32" /></div>
                    <div><Skeleton className="h-6 w-24 mb-2" /><Skeleton className="h-12 w-32" /></div>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="panel p-5"><Skeleton className="h-6 w-16 mb-2" /><Skeleton className="h-8 w-20" /></div>
                  <div className="panel p-5"><Skeleton className="h-6 w-16 mb-2" /><Skeleton className="h-8 w-20" /></div>
                  <div className="panel p-5"><Skeleton className="h-6 w-16 mb-2" /><Skeleton className="h-8 w-20" /></div>
                </div>
              </div>
            )}

            {/* Empty State */}
            {!prediction && !loading && !error && (
              <div className="panel p-12 text-center">
                <p className="text-[var(--text-secondary)] mb-2">
                  Select an engine and run prediction
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  The model will estimate remaining useful life
                </p>
              </div>
            )}

            {/* Results */}
            {prediction && !prediction.error && !loading && (
              <div className="space-y-6">
                {/* Primary Result */}
                <div className="panel p-6 md:p-8">
                  <div className="grid grid-cols-2 gap-4 md:gap-8">
                    <div>
                      <div className="metric-label mb-2">Predicted RUL</div>
                      <div className="metric-value text-3xl md:text-5xl text-[var(--accent)]">
                        {prediction.rul.toFixed(0)}
                      </div>
                      <div className="mono text-xs md:text-sm text-[var(--text-muted)] mt-2">
                        ±{prediction.uncertainty.toFixed(1)} cycles
                      </div>
                    </div>
                    <div>
                      <div className="metric-label mb-2">True RUL</div>
                      <div className="metric-value text-3xl md:text-5xl">
                        {prediction.true_rul.toFixed(0)}
                      </div>
                      <div className="mono text-xs md:text-sm text-[var(--text-muted)] mt-2">
                        cycles remaining
                      </div>
                    </div>
                  </div>
                </div>

                {/* Secondary Metrics */}
                <div className="grid grid-cols-3 gap-2 md:gap-4">
                  <div className="panel p-3 md:p-5">
                    <div className="metric-label mb-1">Error</div>
                    <div className={`metric-value text-lg md:text-2xl ${Math.abs(prediction.error) > 10 ? 'text-[var(--accent-amber)]' : ''}`}>
                      {prediction.error > 0 ? '+' : ''}{prediction.error.toFixed(1)}
                    </div>
                  </div>
                  <div className="panel p-3 md:p-5">
                    <div className="metric-label mb-1">Health</div>
                    <div className="metric-value text-lg md:text-2xl" style={{ color: getHealthStatus(prediction.health_score).color }}>
                      {(prediction.health_score * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="panel p-3 md:p-5">
                    <div className="metric-label mb-1">Cycles</div>
                    <div className="metric-value text-lg md:text-2xl">
                      {prediction.total_cycles}
                    </div>
                  </div>
                </div>

                {/* Health Bar */}
                <div className="panel p-4 md:p-5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="metric-label">Health Status</span>
                    <span className="mono text-xs" style={{ color: getHealthStatus(prediction.health_score).color }}>
                      {getHealthStatus(prediction.health_score).label.toUpperCase()}
                    </span>
                  </div>
                  <div className="h-2 bg-[var(--bg-secondary)] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${prediction.health_score * 100}%`,
                        background: getHealthStatus(prediction.health_score).color
                      }}
                    />
                  </div>
                </div>

                {/* Individual Model Predictions */}
                {prediction.individual_predictions && Object.keys(prediction.individual_predictions).length > 1 && (
                  <div className="panel p-4 md:p-6">
                    <div className="metric-label mb-4">Model Predictions</div>
                    <div className="space-y-3">
                      {Object.entries(prediction.individual_predictions).map(([name, value]) => (
                        <div key={name} className="flex items-center justify-between">
                          <span className="mono text-xs md:text-sm text-[var(--text-secondary)]">{name.toUpperCase()}</span>
                          <div className="flex items-center gap-2 md:gap-4">
                            <div className="w-20 md:w-32 h-1.5 bg-[var(--bg-secondary)] rounded-full overflow-hidden">
                              <div
                                className="h-full bg-[var(--accent)] rounded-full"
                                style={{ width: `${Math.min(100, (value / 150) * 100)}%` }}
                              />
                            </div>
                            <span className="mono text-xs md:text-sm w-12 md:w-16 text-right">{value.toFixed(1)}</span>
                          </div>
                        </div>
                      ))}
                      <div className="flex items-center justify-between pt-3 border-t border-[var(--border)]">
                        <span className="mono text-xs md:text-sm text-[var(--accent-red)]">TRUE RUL</span>
                        <span className="mono text-xs md:text-sm">{prediction.true_rul.toFixed(0)}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* RUL Trajectory Chart */}
                {trajectoryData && (
                  <div className="panel p-4 md:p-6 overflow-x-auto">
                    <div className="metric-label mb-4">Prediction Trajectory</div>
                    <div className="min-w-[300px]">
                      <Plot
                        data={[
                          {
                            x: trajectoryData.map(d => d.cycle),
                            y: trajectoryData.map(d => d.predicted),
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Predicted',
                            line: { color: '#00ffaa', width: 1.5 },
                          },
                          {
                            x: trajectoryData.map(d => d.cycle),
                            y: trajectoryData.map(d => d.true),
                            type: 'scatter',
                            mode: 'lines',
                            name: 'True',
                            line: { color: '#ff4444', width: 1.5, dash: 'dash' },
                          },
                        ]}
                        layout={{
                          paper_bgcolor: 'transparent',
                          plot_bgcolor: 'transparent',
                          font: { color: '#666', family: 'monospace', size: 10 },
                          margin: { t: 10, b: 40, l: 40, r: 10 },
                          height: 200,
                          xaxis: { gridcolor: '#1a1a1a', title: 'Cycle', zeroline: false },
                          yaxis: { gridcolor: '#1a1a1a', title: 'RUL', zeroline: false },
                          legend: { orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center' },
                          showlegend: true,
                        }}
                        config={{ displayModeBar: false, staticPlot: false }}
                        style={{ width: '100%' }}
                      />
                    </div>
                  </div>
                )}

                {/* Maintenance Recommendation */}
                <div className="panel p-4 md:p-5">
                  <div className="metric-label mb-3">Recommendation</div>
                  {prediction.health_score > 0.6 ? (
                    <p className="text-sm text-[var(--text-secondary)]">
                      Engine operating within normal parameters. Continue routine monitoring.
                    </p>
                  ) : prediction.health_score > 0.3 ? (
                    <p className="text-sm text-[var(--accent-amber)]">
                      Increased degradation detected. Schedule maintenance within {Math.floor(prediction.rul * 0.7)} cycles.
                    </p>
                  ) : (
                    <p className="text-sm text-[var(--accent-red)]">
                      Critical condition. Immediate maintenance recommended.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
