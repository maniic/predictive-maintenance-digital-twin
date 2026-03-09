'use client'

import { useState, useEffect } from 'react'
import Navigation from '../../components/Navigation'
import Sidebar from '../../components/Sidebar'
import MetricCard from '../../components/MetricCard'
import PlotlyChart from '../../components/PlotlyChart'
import { fetchEngines, fetchPrediction, fetchTrajectory } from '../../lib/api'

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

const Skeleton = ({ className = '' }) => <div className={`skeleton ${className}`} />

const Spinner = () => (
  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
  </svg>
)

export default function PredictionPage() {
  const [dataset, setDataset] = useState('FD001')
  const [model, setModel] = useState('ensemble')
  const [engines, setEngines] = useState([])
  const [engine, setEngine] = useState(1)
  const [prediction, setPrediction] = useState(null)
  const [trajectoryData, setTrajectoryData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [trajectoryLoading, setTrajectoryLoading] = useState(false)
  const [enginesLoading, setEnginesLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    setEnginesLoading(true)
    setError(null)
    fetchEngines(dataset)
      .then(data => {
        setEngines(data.engines || [])
        if (data.engines?.length) setEngine(data.engines[0])
        setPrediction(null)
        setTrajectoryData(null)
      })
      .catch(err => setError(err.message))
      .finally(() => setEnginesLoading(false))
  }, [dataset])

  const predict = async () => {
    setLoading(true)
    setError(null)
    setTrajectoryData(null)
    try {
      const data = await fetchPrediction(dataset, engine, model)
      setPrediction(data)

      // Fetch real trajectory in background
      setTrajectoryLoading(true)
      fetchTrajectory(dataset, engine)
        .then(traj => setTrajectoryData(traj.trajectory))
        .catch(() => {}) // silently fail, prediction still shows
        .finally(() => setTrajectoryLoading(false))
    } catch (err) {
      setError(err.message || 'An error occurred')
      setPrediction(null)
    }
    setLoading(false)
  }

  const getHealthStatus = (score) => {
    if (score > 0.6) return { label: 'Nominal', color: 'var(--green)', bg: 'var(--green-dim)' }
    if (score > 0.3) return { label: 'Degraded', color: 'var(--amber)', bg: 'var(--amber-dim)' }
    return { label: 'Critical', color: 'var(--red)', bg: 'var(--red-dim)' }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation
        activePage="/prediction"
        onMenuToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      <div className="flex-1 flex flex-col md:flex-row">
        <Sidebar
          activePage="/prediction"
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        >
          {/* Dataset */}
          <div>
            <label className="data-label block mb-2">Dataset</label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="w-full bg-[var(--bg-raised)] border border-[var(--border)] px-3 py-2 font-mono text-sm text-[var(--text-primary)] rounded-sm"
            >
              {DATASETS.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <p className="font-mono text-[0.6rem] text-[var(--text-faint)] mt-1.5">
              {DATASET_INFO[dataset].desc}
            </p>
          </div>

          {/* Model */}
          <div>
            <label className="data-label block mb-2">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full bg-[var(--bg-raised)] border border-[var(--border)] px-3 py-2 font-mono text-sm text-[var(--text-primary)] rounded-sm"
            >
              {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>

          {/* Engine */}
          <div>
            <label className="data-label block mb-2">Engine ID</label>
            {enginesLoading ? (
              <Skeleton className="h-[38px] w-full" />
            ) : (
              <select
                value={engine}
                onChange={(e) => setEngine(Number(e.target.value))}
                className="w-full bg-[var(--bg-raised)] border border-[var(--border)] px-3 py-2 font-mono text-sm text-[var(--text-primary)] rounded-sm"
              >
                {engines.map(e => <option key={e} value={e}>{e}</option>)}
              </select>
            )}
            <p className="font-mono text-[0.6rem] text-[var(--text-faint)] mt-1.5">
              {engines.length} engines available
            </p>
          </div>

          <button
            onClick={() => { predict(); setSidebarOpen(false) }}
            disabled={loading || enginesLoading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {loading && <Spinner />}
            {loading ? 'Analyzing...' : 'Run Prediction'}
          </button>
        </Sidebar>

        {/* Main */}
        <main className="flex-1 p-4 md:p-6 overflow-auto">
          <div className="max-w-4xl">
            <div className="mb-6">
              <div className="data-label mb-1">RUL Prediction</div>
              <h1 className="text-xl md:text-2xl font-300 text-[var(--text-bright)]">Engine Analysis</h1>
            </div>

            {/* Error */}
            {error && (
              <div className="card p-5 mb-5 border-l-2 border-l-[var(--red)]">
                <p className="font-mono text-sm text-[var(--red)] mb-3">{error}</p>
                <button onClick={predict} className="btn-secondary text-xs">Retry</button>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="space-y-4">
                <div className="card p-6">
                  <div className="grid grid-cols-2 gap-8">
                    <div><Skeleton className="h-5 w-20 mb-2" /><Skeleton className="h-10 w-28" /></div>
                    <div><Skeleton className="h-5 w-20 mb-2" /><Skeleton className="h-10 w-28" /></div>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  {[1,2,3].map(i => <div key={i} className="card p-4"><Skeleton className="h-4 w-14 mb-2" /><Skeleton className="h-7 w-16" /></div>)}
                </div>
              </div>
            )}

            {/* Empty */}
            {!prediction && !loading && !error && (
              <div className="card p-12 text-center">
                <div className="text-[var(--text-muted)] mb-1">Select an engine and run prediction</div>
                <div className="font-mono text-[0.65rem] text-[var(--text-faint)]">
                  Model will estimate remaining useful life with uncertainty bounds
                </div>
              </div>
            )}

            {/* Results */}
            {prediction && !prediction.error && !loading && (
              <div className="space-y-4 animate-fade-up">
                {/* Primary */}
                <div className="card p-6 md:p-8">
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <div className="data-label mb-2">Predicted RUL</div>
                      <div className="data-value text-3xl md:text-5xl text-[var(--amber)]">
                        {prediction.rul.toFixed(0)}
                      </div>
                      <div className="font-mono text-xs text-[var(--text-muted)] mt-1">
                        &plusmn;{prediction.uncertainty.toFixed(1)} cycles
                      </div>
                    </div>
                    <div>
                      <div className="data-label mb-2">True RUL</div>
                      <div className="data-value text-3xl md:text-5xl">
                        {prediction.true_rul.toFixed(0)}
                      </div>
                      <div className="font-mono text-xs text-[var(--text-muted)] mt-1">
                        cycles remaining
                      </div>
                    </div>
                  </div>
                </div>

                {/* Secondary metrics */}
                <div className="grid grid-cols-3 gap-3">
                  <MetricCard
                    label="Error"
                    value={`${prediction.error > 0 ? '+' : ''}${prediction.error.toFixed(1)}`}
                    highlight={Math.abs(prediction.error) > 10 ? 'text-[var(--amber)]' : ''}
                  />
                  <MetricCard
                    label="Health"
                    value={`${(prediction.health_score * 100).toFixed(0)}%`}
                    highlight={`text-[${getHealthStatus(prediction.health_score).color}]`}
                  />
                  <MetricCard label="Cycles" value={prediction.total_cycles} />
                </div>

                {/* Health bar */}
                <div className="card p-5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="data-label">Health Status</span>
                    <span
                      className="font-condensed text-[0.7rem] uppercase tracking-wider font-500 px-2 py-0.5 rounded-sm"
                      style={{
                        color: getHealthStatus(prediction.health_score).color,
                        background: getHealthStatus(prediction.health_score).bg,
                      }}
                    >
                      {getHealthStatus(prediction.health_score).label}
                    </span>
                  </div>
                  <div className="h-1.5 bg-[var(--bg-raised)] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${prediction.health_score * 100}%`,
                        background: getHealthStatus(prediction.health_score).color,
                        boxShadow: `0 0 8px ${getHealthStatus(prediction.health_score).color}40`,
                      }}
                    />
                  </div>
                </div>

                {/* Individual predictions */}
                {prediction.individual_predictions && Object.keys(prediction.individual_predictions).length > 1 && (
                  <div className="card p-5">
                    <div className="data-label mb-4">Model Breakdown</div>
                    <div className="space-y-2.5">
                      {Object.entries(prediction.individual_predictions).map(([name, value]) => (
                        <div key={name} className="flex items-center gap-3">
                          <span className="font-mono text-[0.65rem] text-[var(--text-muted)] w-20 md:w-24 uppercase">{name}</span>
                          <div className="flex-1 h-1 bg-[var(--bg-raised)] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[var(--amber)] rounded-full"
                              style={{ width: `${Math.min(100, (value / 150) * 100)}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs text-[var(--text-secondary)] w-12 text-right">{value.toFixed(1)}</span>
                        </div>
                      ))}
                      <div className="flex items-center gap-3 pt-2.5 border-t border-[var(--border)]">
                        <span className="font-mono text-[0.65rem] text-[var(--red)] w-20 md:w-24">TRUE</span>
                        <div className="flex-1" />
                        <span className="font-mono text-xs w-12 text-right">{prediction.true_rul.toFixed(0)}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Trajectory chart — real ML data */}
                {trajectoryLoading && (
                  <div className="card p-8 text-center">
                    <div className="flex items-center justify-center gap-3">
                      <Spinner />
                      <span className="text-[var(--text-secondary)] text-sm">Computing per-cycle trajectory...</span>
                    </div>
                  </div>
                )}
                {trajectoryData && trajectoryData.length > 1 && (
                  <div className="card p-5">
                    <div className="data-label mb-4">Prediction Trajectory</div>
                    <PlotlyChart
                      data={[
                        {
                          x: trajectoryData.map(d => d.cycle),
                          y: trajectoryData.map(d => d.predicted_rul + 1.96 * (d.uncertainty || 5)),
                          type: 'scatter', mode: 'lines', name: 'Upper CI',
                          line: { color: 'rgba(245, 158, 11, 0.2)', width: 0, shape: 'spline', smoothing: 1.3 },
                          showlegend: false,
                        },
                        {
                          x: trajectoryData.map(d => d.cycle),
                          y: trajectoryData.map(d => Math.max(0, d.predicted_rul - 1.96 * (d.uncertainty || 5))),
                          type: 'scatter', mode: 'lines', name: '95% CI',
                          fill: 'tonexty',
                          fillcolor: 'rgba(245, 158, 11, 0.07)',
                          line: { color: 'rgba(245, 158, 11, 0.2)', width: 0, shape: 'spline', smoothing: 1.3 },
                          showlegend: false,
                        },
                        {
                          x: trajectoryData.map(d => d.cycle),
                          y: trajectoryData.map(d => d.predicted_rul),
                          type: 'scatter', mode: 'lines', name: 'Predicted',
                          line: { color: '#f59e0b', width: 2, shape: 'spline', smoothing: 1.3 },
                        },
                        {
                          x: trajectoryData.map(d => d.cycle),
                          y: trajectoryData.map(d => d.true_rul),
                          type: 'scatter', mode: 'lines', name: 'True RUL',
                          line: { color: '#ef4444', width: 2, dash: 'dash', shape: 'spline', smoothing: 1.3 },
                        },
                      ]}
                      layout={{
                        height: 300,
                        margin: { t: 20, b: 50, l: 50, r: 20 },
                        xaxis: { title: { text: 'Cycle', font: { size: 10 } } },
                        yaxis: { title: { text: 'RUL (cycles)', font: { size: 10 } } },
                        legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', font: { size: 9 } },
                      }}
                    />
                  </div>
                )}

                {/* Recommendation */}
                <div className="card p-5">
                  <div className="data-label mb-2">Recommendation</div>
                  {prediction.health_score > 0.6 ? (
                    <p className="text-sm text-[var(--text-secondary)]">
                      Engine within normal parameters. Continue routine monitoring.
                    </p>
                  ) : prediction.health_score > 0.3 ? (
                    <p className="text-sm text-[var(--amber)]">
                      Elevated degradation. Schedule maintenance within {Math.floor(prediction.rul * 0.7)} cycles.
                    </p>
                  ) : (
                    <p className="text-sm text-[var(--red)]">
                      Critical condition — immediate maintenance recommended.
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
