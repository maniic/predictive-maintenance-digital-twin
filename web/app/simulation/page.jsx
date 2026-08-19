'use client'

import { useState, useEffect, useRef } from 'react'
import Navigation from '../../components/Navigation'
import Sidebar from '../../components/Sidebar'
import PlotlyChart from '../../components/PlotlyChart'
import DemoNotice from '../../components/DemoNotice'
import { DEMO_MODE, fetchSimulation } from '../../lib/api'

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

const Spinner = () => (
  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
  </svg>
)

export default function SimulationPage() {
  const [initialRul, setInitialRul] = useState(150)
  const [degradationRate, setDegradationRate] = useState(1.0)
  const [faultMode, setFaultMode] = useState('hpc')
  const [updateSpeed, setUpdateSpeed] = useState(500)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const [trajectory, setTrajectory] = useState(null)
  const [playbackIndex, setPlaybackIndex] = useState(0)
  const [running, setRunning] = useState(false)
  const [simLoading, setSimLoading] = useState(false)
  const [error, setError] = useState(null)

  const intervalRef = useRef(null)

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
    if (!trajectory) { runSimulation(); return }
    if (isComplete) return
    setRunning(true)
  }
  const pause = () => setRunning(false)
  const reset = () => { setRunning(false); setPlaybackIndex(0); setTrajectory(null) }
  const skipCycles = (n) => {
    if (!trajectory) return
    setPlaybackIndex(i => Math.min(i + n, trajectory.length - 1))
  }

  useEffect(() => {
    if (running && trajectory && !isComplete) {
      intervalRef.current = setInterval(() => {
        setPlaybackIndex(i => {
          const next = i + 1
          if (next >= trajectory.length - 1) { setRunning(false); return trajectory.length - 1 }
          return next
        })
      }, updateSpeed)
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
    return () => clearInterval(intervalRef.current)
  }, [running, trajectory, isComplete, updateSpeed])

  useEffect(() => {
    if (trajectory && trajectory.length > 0 && !running) setRunning(true)
  }, [trajectory])

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
            <label className="data-label block mb-2">Initial RUL</label>
            <input
              type="range" min="50" max="300" step="10"
              value={initialRul}
              onChange={(e) => setInitialRul(Number(e.target.value))}
              disabled={running || simLoading}
              className="w-full disabled:opacity-40"
            />
            <div className="flex justify-between font-mono text-[0.6rem] text-[var(--text-faint)] mt-1">
              <span>50</span>
              <span className="text-[var(--text-secondary)]">{initialRul} cycles</span>
              <span>300</span>
            </div>
          </div>

          <div>
            <label className="data-label block mb-2">Degradation Rate</label>
            <select
              value={degradationRate}
              onChange={(e) => setDegradationRate(Number(e.target.value))}
              disabled={running || simLoading}
              className="w-full bg-[var(--bg-raised)] border border-[var(--border)] px-3 py-2 font-mono text-sm text-[var(--text-primary)] rounded-sm disabled:opacity-40"
            >
              <option value={0.5}>0.5x (Slow)</option>
              <option value={1.0}>1.0x (Normal)</option>
              <option value={1.5}>1.5x (Fast)</option>
              <option value={2.0}>2.0x (Very Fast)</option>
            </select>
            <p className="font-mono text-[0.6rem] text-[var(--text-faint)] mt-1">
              ~{effectiveLife} effective cycles
            </p>
          </div>

          <div>
            <label className="data-label block mb-2">Fault Mode</label>
            <select
              value={faultMode}
              onChange={(e) => setFaultMode(e.target.value)}
              disabled={running || simLoading}
              className="w-full bg-[var(--bg-raised)] border border-[var(--border)] px-3 py-2 font-mono text-sm text-[var(--text-primary)] rounded-sm disabled:opacity-40"
            >
              {FAULT_MODES.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}
            </select>
          </div>

          <div>
            <label className="data-label block mb-2">Playback Speed</label>
            <select
              value={updateSpeed}
              onChange={(e) => setUpdateSpeed(Number(e.target.value))}
              className="w-full bg-[var(--bg-raised)] border border-[var(--border)] px-3 py-2 font-mono text-sm text-[var(--text-primary)] rounded-sm"
            >
              {UPDATE_SPEEDS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </div>

          <div className="pt-4 border-t border-[var(--border)] space-y-2">
            {simLoading ? (
              <button disabled className="btn-primary w-full opacity-50 flex items-center justify-center gap-2">
                <Spinner />
                Computing...
              </button>
            ) : !running ? (
              <button onClick={start} className="btn-primary w-full" disabled={isComplete && trajectory}>
                {!trajectory ? 'Run Simulation' : isComplete ? 'Complete' : 'Resume'}
              </button>
            ) : (
              <button onClick={pause} className="btn-secondary w-full">Pause</button>
            )}
            <div className="grid grid-cols-2 gap-2">
              <button onClick={reset} className="btn-secondary w-full" disabled={simLoading}>Reset</button>
              <button
                onClick={() => skipCycles(10)}
                disabled={running || !trajectory || isComplete || simLoading}
                className="btn-secondary w-full disabled:opacity-40"
              >
                +10 Cycles
              </button>
            </div>
          </div>
        </Sidebar>

        {/* Main */}
        <main className="flex-1 p-4 md:p-6 overflow-auto">
          <div className="max-w-4xl">
            <div className="mb-6">
              <div className="data-label mb-1">
                {DEMO_MODE
                  ? 'Degradation Simulation (Demo)'
                  : trajectory
                    ? 'ML-Backed Simulation'
                    : 'Degradation Simulation'}
              </div>
              <h1 className="text-xl md:text-2xl font-300 text-[var(--text-bright)]">Engine Degradation</h1>
            </div>

            <DemoNotice>
              This degradation curve is computed in your browser, not by the
              digital-twin simulator or a trained model.
            </DemoNotice>

            {error && (
              <div className="card p-4 mb-5 border-l-2 border-l-[var(--red)]">
                <p className="font-mono text-sm text-[var(--red)]">{error}</p>
              </div>
            )}

            {/* Live Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-px border border-[var(--border)] mb-5">
              {[
                { label: 'Cycle', value: currentPoint?.cycle || 0 },
                {
                  label: 'Predicted RUL',
                  value: currentPoint?.predicted_rul?.toFixed(1) ?? '—',
                  color: currentPoint && currentPoint.true_rul < 25 ? 'var(--red)' : 'var(--amber)',
                  sub: currentPoint?.uncertainty ? `±${currentPoint.uncertainty.toFixed(1)}` : null,
                },
                { label: 'True RUL', value: currentPoint?.true_rul ?? '—' },
                { label: 'Health', value: currentPoint ? `${(currentPoint.health_score * 100).toFixed(0)}%` : '—' },
              ].map((m) => (
                <div key={m.label} className="bg-[var(--bg-surface)] p-4">
                  <div className="data-label mb-1">{m.label}</div>
                  <div className="data-value text-xl md:text-2xl" style={m.color ? { color: m.color } : {}}>
                    {m.value}
                  </div>
                  {m.sub && <div className="font-mono text-[0.6rem] text-[var(--text-faint)]">{m.sub}</div>}
                </div>
              ))}
            </div>

            {/* Alerts */}
            {currentPoint && currentPoint.true_rul > 0 && currentPoint.true_rul < 10 && (
              <div className="card p-4 mb-5 border-l-2 border-l-[var(--red)] bg-[var(--red-dim)]">
                <p className="font-condensed text-sm uppercase tracking-wider text-[var(--red)]">
                  Critical — RUL below 10 cycles. Immediate maintenance required.
                </p>
              </div>
            )}
            {currentPoint && currentPoint.true_rul >= 10 && currentPoint.true_rul < 25 && (
              <div className="card p-4 mb-5 border-l-2 border-l-[var(--amber)] bg-[var(--amber-dim)]">
                <p className="font-condensed text-sm uppercase tracking-wider text-[var(--amber)]">
                  Warning — RUL below 25 cycles. Schedule maintenance.
                </p>
              </div>
            )}
            {currentPoint && currentPoint.true_rul <= 0 && (
              <div className="card p-4 mb-5 border-l-2 border-l-[var(--green)] bg-[var(--green-dim)]">
                <p className="font-condensed text-sm uppercase tracking-wider text-[var(--green)]">
                  Simulation complete — engine reached end of life.
                </p>
              </div>
            )}

            {/* Empty / Loading */}
            {!trajectory && !simLoading && !error && (
              <div className="card p-12 text-center">
                <div className="text-[var(--text-muted)] mb-1">Configure parameters and run simulation</div>
                <div className="font-mono text-[0.65rem] text-[var(--text-faint)]">
                  {DEMO_MODE
                    ? 'Hosted demo: the degradation curve runs in your browser, not through a trained model'
                    : 'Uses trained ML models for real-time RUL prediction'}
                </div>
              </div>
            )}

            {simLoading && (
              <div className="card p-12 text-center">
                <div className="flex items-center justify-center gap-3 mb-2">
                  <Spinner />
                  <span className="text-[var(--text-secondary)]">Running simulation...</span>
                </div>
                <div className="font-mono text-[0.65rem] text-[var(--text-faint)]">
                  Computing {effectiveLife} cycles of degradation
                </div>
              </div>
            )}

            {/* RUL Chart */}
            {visibleData.length > 1 && (
              <div className="card p-5 mb-4">
                <div className="data-label mb-4">RUL Over Time</div>
                <PlotlyChart
                  data={[
                    {
                      x: visibleData.map(d => d.cycle),
                      y: visibleData.map(d => d.predicted_rul + 1.96 * (d.uncertainty || 5)),
                      type: 'scatter', mode: 'lines', name: 'Upper CI',
                      line: { color: 'rgba(245, 158, 11, 0.2)', width: 0, shape: 'spline', smoothing: 1.3 },
                      showlegend: false,
                    },
                    {
                      x: visibleData.map(d => d.cycle),
                      y: visibleData.map(d => Math.max(0, d.predicted_rul - 1.96 * (d.uncertainty || 5))),
                      type: 'scatter', mode: 'lines', name: '95% CI',
                      fill: 'tonexty',
                      fillcolor: 'rgba(245, 158, 11, 0.07)',
                      line: { color: 'rgba(245, 158, 11, 0.2)', width: 0, shape: 'spline', smoothing: 1.3 },
                      showlegend: false,
                    },
                    {
                      x: visibleData.map(d => d.cycle),
                      y: visibleData.map(d => d.predicted_rul),
                      type: 'scatter', mode: 'lines', name: 'Predicted',
                      line: { color: '#f59e0b', width: 2, shape: 'spline', smoothing: 1.3 },
                    },
                    {
                      x: visibleData.map(d => d.cycle),
                      y: visibleData.map(d => d.true_rul),
                      type: 'scatter', mode: 'lines', name: 'True',
                      line: { color: '#ef4444', width: 2, dash: 'dash', shape: 'spline', smoothing: 1.3 },
                    },
                  ]}
                  layout={{
                    margin: { t: 20, b: 50, l: 50, r: 20 },
                    height: 280,
                    xaxis: { title: { text: 'Cycle', font: { size: 10 } } },
                    yaxis: { title: { text: 'RUL (cycles)', font: { size: 10 } } },
                    legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', font: { size: 9 } },
                  }}
                />
              </div>
            )}

            {/* Health Chart */}
            {visibleData.length > 1 && (
              <div className="card p-5">
                <div className="data-label mb-4">Health Score</div>
                <PlotlyChart
                  data={[{
                    x: visibleData.map(d => d.cycle),
                    y: visibleData.map(d => d.health_score * 100),
                    type: 'scatter', mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(245, 158, 11, 0.08)',
                    line: { color: '#f59e0b', width: 2, shape: 'spline', smoothing: 1.3 },
                  }]}
                  layout={{
                    margin: { t: 20, b: 50, l: 50, r: 20 },
                    height: 200,
                    xaxis: { title: { text: 'Cycle', font: { size: 10 } } },
                    yaxis: { title: { text: 'Health (%)', font: { size: 10 } }, range: [0, 100] },
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
