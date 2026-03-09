'use client'

import { useState, useEffect, useRef } from 'react'
import Navigation from '../../components/Navigation'
import Sidebar from '../../components/Sidebar'
import PlotlyChart from '../../components/PlotlyChart'
import { fetchSimulation } from '../../lib/api'

const FAULT_MODES = [
  { value: 'hpc', label: 'HPC Degradation' },
  { value: 'fan', label: 'Fan Degradation' },
  { value: 'combined', label: 'Combined' },
]
const UPDATE_SPEEDS = [
  { value: 200, label: '0.2s (Fast)' },
  { value: 500, label: '0.5s' },
  { value: 1000, label: '1.0s' },
  { value: 2000, label: '2.0s (Slow)' },
]

export default function SimulationPage() {
  // Config
  const [initialRul, setInitialRul] = useState(150)
  const [degradationRate, setDegradationRate] = useState(1.0)
  const [faultMode, setFaultMode] = useState('hpc')
  const [updateSpeed, setUpdateSpeed] = useState(500)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Simulation state
  const [trajectory, setTrajectory] = useState(null)
  const [playbackIndex, setPlaybackIndex] = useState(0)
  const [running, setRunning] = useState(false)
  const [simLoading, setSimLoading] = useState(false)
  const [error, setError] = useState(null)

  const intervalRef = useRef(null)

  // Current point from trajectory
  const currentPoint = trajectory?.[playbackIndex] || null
  const isComplete = trajectory && playbackIndex >= trajectory.length - 1

  const effectiveLife = Math.floor(initialRul / degradationRate)

  const runSimulation = async () => {
    setSimLoading(true)
    setError(null)
    setRunning(false)
    setPlaybackIndex(0)
    setTrajectory(null)

    try {
      const data = await fetchSimulation(initialRul, degradationRate, faultMode)
      setTrajectory(data.trajectory)
    } catch (err) {
      setError(err.message)
    } finally {
      setSimLoading(false)
    }
  }

  const start = () => {
    if (!trajectory) {
      runSimulation()
      return
    }
    if (isComplete) return
    setRunning(true)
  }
  const pause = () => setRunning(false)

  const reset = () => {
    setRunning(false)
    setPlaybackIndex(0)
    setTrajectory(null)
  }

  const skipCycles = (n) => {
    if (!trajectory) return
    setPlaybackIndex(i => Math.min(i + n, trajectory.length - 1))
  }

  // Playback interval
  useEffect(() => {
    if (running && trajectory && !isComplete) {
      intervalRef.current = setInterval(() => {
        setPlaybackIndex(i => {
          const next = i + 1
          if (next >= trajectory.length - 1) {
            setRunning(false)
            return trajectory.length - 1
          }
          return next
        })
      }, updateSpeed)
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
    return () => clearInterval(intervalRef.current)
  }, [running, trajectory, isComplete, updateSpeed])

  // Auto-play after trajectory loads
  useEffect(() => {
    if (trajectory && trajectory.length > 0 && !running) {
      setRunning(true)
    }
  }, [trajectory])

  // Visible data (up to current playback point)
  const visibleData = trajectory?.slice(0, playbackIndex + 1) || []

  const getStatusClass = () => {
    if (!currentPoint) return ''
    if (currentPoint.true_rul < 10) return 'critical'
    if (currentPoint.true_rul < 25) return 'warning'
    return ''
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation
        activePage="/simulation"
        statusClass={getStatusClass()}
        onMenuToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      <div className="flex-1 flex relative">
        <Sidebar
          activePage="/simulation"
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        >
          <div>
            <label className="metric-label block mb-2">Initial RUL</label>
            <input
              type="range"
              min="50"
              max="300"
              step="10"
              value={initialRul}
              onChange={(e) => setInitialRul(Number(e.target.value))}
              disabled={running || simLoading}
              className="w-full accent-[var(--accent)] disabled:opacity-50"
            />
            <div className="flex justify-between text-xs text-[var(--text-muted)] mt-1">
              <span>50</span>
              <span className="text-[var(--text-secondary)]">{initialRul} cycles</span>
              <span>300</span>
            </div>
          </div>

          <div>
            <label className="metric-label block mb-2">Degradation Rate</label>
            <select
              value={degradationRate}
              onChange={(e) => setDegradationRate(Number(e.target.value))}
              disabled={running || simLoading}
              className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm disabled:opacity-50"
            >
              <option value={0.5}>0.5x (Slow)</option>
              <option value={1.0}>1.0x (Normal)</option>
              <option value={1.5}>1.5x (Fast)</option>
              <option value={2.0}>2.0x (Very Fast)</option>
            </select>
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Effective lifespan: ~{effectiveLife} cycles
            </p>
          </div>

          <div>
            <label className="metric-label block mb-2">Fault Mode</label>
            <select
              value={faultMode}
              onChange={(e) => setFaultMode(e.target.value)}
              disabled={running || simLoading}
              className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm disabled:opacity-50"
            >
              {FAULT_MODES.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}
            </select>
          </div>

          <div>
            <label className="metric-label block mb-2">Playback Speed</label>
            <select
              value={updateSpeed}
              onChange={(e) => setUpdateSpeed(Number(e.target.value))}
              className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm"
            >
              {UPDATE_SPEEDS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </div>

          <div className="pt-4 border-t border-[var(--border)] space-y-2">
            {simLoading ? (
              <button disabled className="btn-primary w-full opacity-50 flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Computing...
              </button>
            ) : !running ? (
              <button onClick={start} className="btn-primary w-full" disabled={isComplete && trajectory}>
                {!trajectory ? 'Run Simulation' : isComplete ? 'Complete' : 'Resume'}
              </button>
            ) : (
              <button onClick={pause} className="btn-secondary w-full">
                Pause
              </button>
            )}
            <div className="grid grid-cols-2 gap-2">
              <button onClick={reset} className="btn-secondary w-full" disabled={simLoading}>
                Reset
              </button>
              <button
                onClick={() => skipCycles(10)}
                disabled={running || !trajectory || isComplete || simLoading}
                className="btn-secondary w-full disabled:opacity-50"
              >
                +10 Cycles
              </button>
            </div>
          </div>
        </Sidebar>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-auto">
          <div className="max-w-4xl">
            <p className="mono text-xs text-[var(--text-secondary)] mb-2 tracking-wider">
              {trajectory ? 'ML-BACKED SIMULATION' : 'DEGRADATION SIMULATION'}
            </p>
            <h1 className="text-2xl font-light mb-6">Engine Degradation</h1>

            {error && (
              <div className="panel p-4 mb-6 border-l-2 border-[var(--accent-red)]">
                <p className="mono text-sm text-[var(--accent-red)]">{error}</p>
              </div>
            )}

            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[var(--border)] mb-6">
              <div className="bg-[var(--bg-primary)] p-4 md:p-5">
                <div className="metric-label mb-1">Cycle</div>
                <div className="metric-value text-xl md:text-2xl">
                  {currentPoint?.cycle || 0}
                </div>
              </div>
              <div className="bg-[var(--bg-primary)] p-4 md:p-5">
                <div className="metric-label mb-1">Predicted RUL</div>
                <div className={`metric-value text-xl md:text-2xl ${currentPoint && currentPoint.true_rul < 25 ? 'text-[var(--accent-red)]' : 'text-[var(--accent)]'}`}>
                  {currentPoint?.predicted_rul?.toFixed(1) ?? '—'}
                </div>
                {currentPoint?.uncertainty && (
                  <div className="text-xs text-[var(--text-muted)]">&plusmn;{currentPoint.uncertainty.toFixed(1)}</div>
                )}
              </div>
              <div className="bg-[var(--bg-primary)] p-4 md:p-5">
                <div className="metric-label mb-1">True RUL</div>
                <div className="metric-value text-xl md:text-2xl">
                  {currentPoint?.true_rul ?? '—'}
                </div>
              </div>
              <div className="bg-[var(--bg-primary)] p-4 md:p-5">
                <div className="metric-label mb-1">Health</div>
                <div className="metric-value text-xl md:text-2xl">
                  {currentPoint ? `${(currentPoint.health_score * 100).toFixed(0)}%` : '—'}
                </div>
              </div>
            </div>

            {/* Alerts */}
            {currentPoint && currentPoint.true_rul > 0 && currentPoint.true_rul < 10 && (
              <div className="panel p-4 mb-6 border-l-2 border-[var(--accent-red)]">
                <p className="mono text-sm text-[var(--accent-red)]">
                  CRITICAL: RUL below 10 cycles! Immediate maintenance required!
                </p>
              </div>
            )}
            {currentPoint && currentPoint.true_rul >= 10 && currentPoint.true_rul < 25 && (
              <div className="panel p-4 mb-6 border-l-2 border-[var(--accent-amber)]">
                <p className="mono text-sm text-[var(--accent-amber)]">
                  WARNING: RUL below 25 cycles. Schedule maintenance soon.
                </p>
              </div>
            )}
            {currentPoint && currentPoint.true_rul <= 0 && (
              <div className="panel p-4 mb-6 border-l-2 border-[var(--accent)]">
                <p className="mono text-sm text-[var(--accent)]">
                  Simulation complete — Engine reached end of life.
                </p>
              </div>
            )}

            {!trajectory && !simLoading && !error && (
              <div className="panel p-12 text-center">
                <p className="text-[var(--text-secondary)] mb-2">
                  Configure parameters and run simulation
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  Uses trained ML models for real-time RUL prediction
                </p>
              </div>
            )}

            {simLoading && (
              <div className="panel p-12 text-center">
                <div className="flex items-center justify-center gap-3 mb-3">
                  <svg className="animate-spin h-5 w-5 text-[var(--accent)]" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <p className="text-[var(--text-secondary)]">
                    Running simulation with ML models...
                  </p>
                </div>
                <p className="text-xs text-[var(--text-muted)]">
                  Computing {effectiveLife} cycles of degradation
                </p>
              </div>
            )}

            {/* RUL Chart */}
            {visibleData.length > 1 && (
              <div className="panel p-6 mb-6">
                <div className="metric-label mb-4">RUL Prediction Over Time</div>
                <PlotlyChart
                  data={[
                    {
                      x: visibleData.map(d => d.cycle),
                      y: visibleData.map((d, i) => d.predicted_rul + 1.96 * (d.uncertainty || 5)),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Upper CI',
                      line: { color: 'rgba(0, 255, 170, 0.3)', width: 0, shape: 'spline', smoothing: 1.3 },
                      showlegend: false,
                    },
                    {
                      x: visibleData.map(d => d.cycle),
                      y: visibleData.map((d, i) => Math.max(0, d.predicted_rul - 1.96 * (d.uncertainty || 5))),
                      type: 'scatter',
                      mode: 'lines',
                      name: '95% CI',
                      fill: 'tonexty',
                      fillcolor: 'rgba(0, 255, 170, 0.1)',
                      line: { color: 'rgba(0, 255, 170, 0.3)', width: 0, shape: 'spline', smoothing: 1.3 },
                      showlegend: false,
                    },
                    {
                      x: visibleData.map(d => d.cycle),
                      y: visibleData.map(d => d.predicted_rul),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Predicted',
                      line: { color: '#00ffaa', width: 2, shape: 'spline', smoothing: 1.3 },
                    },
                    {
                      x: visibleData.map(d => d.cycle),
                      y: visibleData.map(d => d.true_rul),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'True',
                      line: { color: '#ff4444', width: 2, dash: 'dash', shape: 'spline', smoothing: 1.3 },
                    },
                  ]}
                  layout={{
                    margin: { t: 20, b: 50, l: 50, r: 20 },
                    height: 280,
                    xaxis: { title: 'Cycle' },
                    yaxis: { title: 'RUL (cycles)' },
                    legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' },
                    transition: { duration: 300, easing: 'cubic-in-out' },
                  }}
                />
              </div>
            )}

            {/* Health Chart */}
            {visibleData.length > 1 && (
              <div className="panel p-6">
                <div className="metric-label mb-4">Health Score Over Time</div>
                <PlotlyChart
                  data={[{
                    x: visibleData.map(d => d.cycle),
                    y: visibleData.map(d => d.health_score * 100),
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0, 255, 170, 0.15)',
                    line: { color: '#00ffaa', width: 2, shape: 'spline', smoothing: 1.3 },
                  }]}
                  layout={{
                    margin: { t: 20, b: 50, l: 50, r: 20 },
                    height: 200,
                    xaxis: { title: 'Cycle' },
                    yaxis: { title: 'Health (%)', range: [0, 100] },
                    transition: { duration: 300, easing: 'cubic-in-out' },
                  }}
                />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
