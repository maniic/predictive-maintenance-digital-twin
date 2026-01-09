'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
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
  const [dataset, setDataset] = useState('FD001')
  const [initialRul, setInitialRul] = useState(150)
  const [degradationRate, setDegradationRate] = useState(1.0)
  const [faultMode, setFaultMode] = useState('hpc')
  const [updateSpeed, setUpdateSpeed] = useState(500)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // State
  const [running, setRunning] = useState(false)
  const [cycle, setCycle] = useState(0)
  const [history, setHistory] = useState({
    cycles: [],
    predictedRuls: [],
    trueRuls: [],
    health: [],
    uncertainties: [],
  })

  const intervalRef = useRef(null)

  // Computed values
  const effectiveLife = Math.floor(initialRul / degradationRate)
  const trueRul = Math.max(0, effectiveLife - cycle)
  const noise = (Math.random() - 0.5) * 8
  const predictedRul = Math.max(0, trueRul + noise)
  const uncertainty = 3 + Math.random() * 4
  const healthScore = Math.min(1, Math.max(0, trueRul / 125))

  const start = () => setRunning(true)
  const pause = () => setRunning(false)

  const reset = () => {
    setRunning(false)
    setCycle(0)
    setHistory({
      cycles: [],
      predictedRuls: [],
      trueRuls: [],
      health: [],
      uncertainties: [],
    })
  }

  const skipCycles = (n) => {
    for (let i = 0; i < n && effectiveLife - (cycle + i) > 0; i++) {
      const c = cycle + i + 1
      const tr = Math.max(0, effectiveLife - c)
      const pr = Math.max(0, tr + (Math.random() - 0.5) * 8)
      setHistory(h => ({
        cycles: [...h.cycles, c],
        predictedRuls: [...h.predictedRuls, pr],
        trueRuls: [...h.trueRuls, tr],
        health: [...h.health, Math.min(100, Math.max(0, tr / 125 * 100))],
        uncertainties: [...h.uncertainties, 3 + Math.random() * 4],
      }))
    }
    setCycle(c => Math.min(c + n, effectiveLife))
  }

  useEffect(() => {
    if (running && trueRul > 0) {
      intervalRef.current = setInterval(() => {
        setCycle(c => {
          const newCycle = c + 1
          const tr = Math.max(0, effectiveLife - newCycle)
          const pr = Math.max(0, tr + (Math.random() - 0.5) * 8)

          setHistory(h => ({
            cycles: [...h.cycles, newCycle],
            predictedRuls: [...h.predictedRuls, pr],
            trueRuls: [...h.trueRuls, tr],
            health: [...h.health, Math.min(100, Math.max(0, tr / 125 * 100))],
            uncertainties: [...h.uncertainties, 3 + Math.random() * 4],
          }))

          return newCycle
        })
      }, updateSpeed)
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (trueRul <= 0 && running) setRunning(false)
    }
    return () => clearInterval(intervalRef.current)
  }, [running, trueRul, updateSpeed, effectiveLife])

  const getStatusClass = () => {
    if (trueRul < 10) return 'critical'
    if (trueRul < 25) return 'warning'
    return ''
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--border)] px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="md:hidden p-2 -ml-2 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18" />
              </svg>
            </button>
            <Link href="/" className="flex items-center gap-2">
              <div className={`status-dot ${getStatusClass()}`} />
              <span className="mono text-xs tracking-wider">DIGITAL TWIN</span>
            </Link>
          </div>
          <nav className="hidden sm:flex items-center gap-6">
            <Link href="/prediction" className="nav-link">Predict</Link>
            <Link href="/simulation" className="nav-link active">Simulate</Link>
            <Link href="/comparison" className="nav-link">Compare</Link>
          </nav>
        </div>
      </header>

      <div className="flex-1 flex relative">
        {/* Mobile sidebar overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-40 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Sidebar */}
        <aside className={`
          fixed md:relative inset-y-0 left-0 z-50
          w-72 bg-[var(--bg-primary)] border-r border-[var(--border)] p-6 flex-shrink-0
          transform transition-transform duration-200
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
        `}>
          <div className="space-y-5">
            <div>
              <label className="metric-label block mb-2">Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                disabled={running}
                className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm disabled:opacity-50"
              >
                {DATASETS.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>

            <div>
              <label className="metric-label block mb-2">Initial RUL</label>
              <input
                type="range"
                min="50"
                max="300"
                step="10"
                value={initialRul}
                onChange={(e) => setInitialRul(Number(e.target.value))}
                disabled={running}
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
                disabled={running}
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
                disabled={running}
                className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm disabled:opacity-50"
              >
                {FAULT_MODES.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
            </div>

            <div>
              <label className="metric-label block mb-2">Update Speed</label>
              <select
                value={updateSpeed}
                onChange={(e) => setUpdateSpeed(Number(e.target.value))}
                className="w-full bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm"
              >
                {UPDATE_SPEEDS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </div>

            <div className="pt-4 border-t border-[var(--border)] space-y-2">
              {!running ? (
                <button onClick={start} className="btn-primary w-full" disabled={trueRul <= 0}>
                  {trueRul <= 0 ? 'Complete' : 'Start Simulation'}
                </button>
              ) : (
                <button onClick={pause} className="btn-secondary w-full">
                  Pause
                </button>
              )}
              <div className="grid grid-cols-2 gap-2">
                <button onClick={reset} className="btn-secondary w-full">
                  Reset
                </button>
                <button
                  onClick={() => skipCycles(10)}
                  disabled={running || trueRul <= 0}
                  className="btn-secondary w-full disabled:opacity-50"
                >
                  +10 Cycles
                </button>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-auto">
          <div className="max-w-4xl">
            <p className="mono text-xs text-[var(--text-secondary)] mb-2 tracking-wider">
              LIVE SIMULATION
            </p>
            <h1 className="text-2xl font-light mb-6">Engine Degradation</h1>

            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[var(--border)] mb-6">
              <div className="bg-[var(--bg-primary)] p-4 md:p-5">
                <div className="metric-label mb-1">Cycle</div>
                <div className="metric-value text-xl md:text-2xl">{cycle}</div>
              </div>
              <div className="bg-[var(--bg-primary)] p-4 md:p-5">
                <div className="metric-label mb-1">Predicted RUL</div>
                <div className={`metric-value text-xl md:text-2xl ${trueRul < 25 ? 'text-[var(--accent-red)]' : 'text-[var(--accent)]'}`}>
                  {predictedRul.toFixed(1)}
                </div>
                <div className="text-xs text-[var(--text-muted)]">±{uncertainty.toFixed(1)}</div>
              </div>
              <div className="bg-[var(--bg-primary)] p-4 md:p-5">
                <div className="metric-label mb-1">True RUL</div>
                <div className="metric-value text-xl md:text-2xl">{trueRul}</div>
              </div>
              <div className="bg-[var(--bg-primary)] p-4 md:p-5">
                <div className="metric-label mb-1">Health</div>
                <div className="metric-value text-xl md:text-2xl">{(healthScore * 100).toFixed(0)}%</div>
              </div>
            </div>

            {/* Alerts */}
            {trueRul > 0 && trueRul < 10 && (
              <div className="panel p-4 mb-6 border-l-2 border-[var(--accent-red)]">
                <p className="mono text-sm text-[var(--accent-red)]">
                  CRITICAL: RUL below 10 cycles! Immediate maintenance required!
                </p>
              </div>
            )}
            {trueRul >= 10 && trueRul < 25 && (
              <div className="panel p-4 mb-6 border-l-2 border-[var(--accent-amber)]">
                <p className="mono text-sm text-[var(--accent-amber)]">
                  WARNING: RUL below 25 cycles. Schedule maintenance soon.
                </p>
              </div>
            )}
            {trueRul >= 25 && trueRul < 50 && (
              <div className="panel p-4 mb-6 border-l-2 border-[var(--text-muted)]">
                <p className="mono text-sm text-[var(--text-secondary)]">
                  NOTICE: RUL below 50 cycles. Plan maintenance.
                </p>
              </div>
            )}
            {trueRul <= 0 && (
              <div className="panel p-4 mb-6 border-l-2 border-[var(--accent)]">
                <p className="mono text-sm text-[var(--accent)]">
                  Simulation complete — Engine reached end of life.
                </p>
              </div>
            )}

            {/* RUL Chart */}
            {history.cycles.length > 1 && (
              <div className="panel p-6 mb-6">
                <div className="metric-label mb-4">RUL Prediction Over Time</div>
                <Plot
                  data={[
                    // Confidence band upper
                    {
                      x: history.cycles,
                      y: history.predictedRuls.map((r, i) => r + 1.96 * history.uncertainties[i]),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Upper CI',
                      line: { color: 'rgba(0, 255, 170, 0.3)', width: 0, shape: 'spline', smoothing: 1.3 },
                      showlegend: false,
                    },
                    // Confidence band lower with fill
                    {
                      x: history.cycles,
                      y: history.predictedRuls.map((r, i) => Math.max(0, r - 1.96 * history.uncertainties[i])),
                      type: 'scatter',
                      mode: 'lines',
                      name: '95% CI',
                      fill: 'tonexty',
                      fillcolor: 'rgba(0, 255, 170, 0.1)',
                      line: { color: 'rgba(0, 255, 170, 0.3)', width: 0, shape: 'spline', smoothing: 1.3 },
                      showlegend: false,
                    },
                    // Predicted RUL
                    {
                      x: history.cycles,
                      y: history.predictedRuls,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Predicted',
                      line: { color: '#00ffaa', width: 2, shape: 'spline', smoothing: 1.3 },
                    },
                    // True RUL
                    {
                      x: history.cycles,
                      y: history.trueRuls,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'True',
                      line: { color: '#ff4444', width: 2, dash: 'dash', shape: 'spline', smoothing: 1.3 },
                    },
                  ]}
                  layout={{
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#666', family: 'Geist Mono', size: 11 },
                    margin: { t: 20, b: 50, l: 50, r: 20 },
                    height: 280,
                    xaxis: { gridcolor: '#1a1a1a', title: 'Cycle' },
                    yaxis: { gridcolor: '#1a1a1a', title: 'RUL (cycles)' },
                    legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' },
                    transition: { duration: 300, easing: 'cubic-in-out' },
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>
            )}

            {/* Health Chart */}
            {history.cycles.length > 1 && (
              <div className="panel p-6">
                <div className="metric-label mb-4">Health Score Over Time</div>
                <Plot
                  data={[{
                    x: history.cycles,
                    y: history.health,
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0, 255, 170, 0.15)',
                    line: { color: '#00ffaa', width: 2, shape: 'spline', smoothing: 1.3 },
                  }]}
                  layout={{
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#666', family: 'Geist Mono', size: 11 },
                    margin: { t: 20, b: 50, l: 50, r: 20 },
                    height: 200,
                    xaxis: { gridcolor: '#1a1a1a', title: 'Cycle' },
                    yaxis: { gridcolor: '#1a1a1a', title: 'Health (%)', range: [0, 100] },
                    transition: { duration: 300, easing: 'cubic-in-out' },
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
