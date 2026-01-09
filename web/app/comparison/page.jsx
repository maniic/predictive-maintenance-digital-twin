'use client'

import { useState, useEffect, useMemo } from 'react'
import Link from 'next/link'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const METRICS = [
  { value: 'test_rmse', label: 'RMSE' },
  { value: 'test_mae', label: 'MAE' },
  { value: 'test_cmapss', label: 'Score' },
]

const Skeleton = ({ className = '' }) => (
  <div className={`skeleton ${className}`} />
)

export default function ComparisonPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState('all')
  const [metric, setMetric] = useState('test_rmse')
  const [sortBy, setSortBy] = useState('test_rmse')
  const [sortAsc, setSortAsc] = useState(true)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 30000)
      const res = await fetch('/api/comparison', { signal: controller.signal })
      clearTimeout(timeout)
      if (!res.ok) throw new Error('Failed to fetch comparison data')
      const d = await res.json()
      setData(d)
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please try again.')
      } else {
        setError(err.message || 'Failed to load data')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const datasets = useMemo(() => {
    if (!data?.results) return []
    return ['all', ...new Set(data.results.map(r => r.dataset))]
  }, [data])

  const filteredResults = useMemo(() => {
    if (!data?.results) return []
    let results = selectedDataset === 'all'
      ? data.results
      : data.results.filter(r => r.dataset === selectedDataset)

    results = [...results].sort((a, b) => {
      const aVal = a[sortBy] || 0
      const bVal = b[sortBy] || 0
      return sortAsc ? aVal - bVal : bVal - aVal
    })

    return results
  }, [data, selectedDataset, sortBy, sortAsc])

  const chartData = useMemo(() => {
    if (!data?.results) return []

    const byDataset = {}
    data.results.forEach(r => {
      if (!byDataset[r.dataset]) byDataset[r.dataset] = {}
      byDataset[r.dataset][r.model] = r[metric]
    })

    const models = [...new Set(data.results.map(r => r.model))]
    const datasets = Object.keys(byDataset)

    if (selectedDataset !== 'all') {
      return models.map(model => ({
        x: [model.toUpperCase()],
        y: [byDataset[selectedDataset]?.[model] || 0],
        type: 'bar',
        name: model,
        marker: { color: '#00ffaa' },
      }))
    }

    return models.map((model, i) => ({
      x: datasets,
      y: datasets.map(ds => byDataset[ds]?.[model] || 0),
      type: 'bar',
      name: model.toUpperCase(),
      marker: { color: ['#00ffaa', '#ff6b35', '#4dabf7', '#be4bdb', '#fcc419'][i % 5] },
    }))
  }, [data, metric, selectedDataset])

  const handleSort = (col) => {
    if (sortBy === col) {
      setSortAsc(!sortAsc)
    } else {
      setSortBy(col)
      setSortAsc(true)
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--border)] px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="status-dot" />
            <span className="mono text-xs tracking-wider">DIGITAL TWIN</span>
          </Link>
          <nav className="flex items-center gap-6">
            <Link href="/prediction" className="nav-link">Predict</Link>
            <Link href="/simulation" className="nav-link">Simulate</Link>
            <Link href="/comparison" className="nav-link active">Compare</Link>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <p className="mono text-xs text-[var(--text-secondary)] mb-2 tracking-wider">
            MODEL COMPARISON
          </p>
          <h1 className="text-2xl font-light mb-6">Performance Analysis</h1>

          {error ? (
            <div className="panel p-8 text-center">
              <p className="text-[var(--accent-red)] mb-4">{error}</p>
              <button onClick={fetchData} className="btn-primary">
                Retry
              </button>
            </div>
          ) : loading ? (
            <div className="space-y-6">
              <div className="flex gap-4">
                <Skeleton className="h-16 w-40" />
                <Skeleton className="h-16 w-40" />
              </div>
              <Skeleton className="h-64 w-full" />
              <Skeleton className="h-96 w-full" />
            </div>
          ) : (
            <div className="space-y-6">
              {/* Filters */}
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="flex-1 sm:flex-initial">
                  <label className="metric-label block mb-2">Dataset</label>
                  <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    className="w-full sm:w-auto bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm"
                  >
                    {datasets.map(d => (
                      <option key={d} value={d}>{d === 'all' ? 'All Datasets' : d}</option>
                    ))}
                  </select>
                </div>
                <div className="flex-1 sm:flex-initial">
                  <label className="metric-label block mb-2">Metric</label>
                  <select
                    value={metric}
                    onChange={(e) => setMetric(e.target.value)}
                    className="w-full sm:w-auto bg-[var(--bg-panel)] border border-[var(--border)] px-3 py-2 mono text-sm"
                  >
                    {METRICS.map(m => (
                      <option key={m.value} value={m.value}>{m.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Chart */}
              <div className="panel p-6">
                <div className="metric-label mb-4">
                  {METRICS.find(m => m.value === metric)?.label} by Model
                </div>
                <Plot
                  data={chartData}
                  layout={{
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#666', family: 'monospace', size: 10 },
                    margin: { t: 10, b: 60, l: 50, r: 10 },
                    height: 250,
                    barmode: 'group',
                    xaxis: { gridcolor: '#1a1a1a', zeroline: false },
                    yaxis: { gridcolor: '#1a1a1a', title: METRICS.find(m => m.value === metric)?.label, zeroline: false },
                    legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
                    showlegend: selectedDataset === 'all',
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>

              {/* Table */}
              <div className="panel overflow-x-auto">
                <table className="w-full min-w-[500px]">
                  <thead>
                    <tr className="border-b border-[var(--border)]">
                      <th className="text-left p-4 metric-label">Model</th>
                      <th className="text-left p-4 metric-label">Dataset</th>
                      <th
                        className="text-right p-4 metric-label cursor-pointer hover:text-[var(--text-primary)]"
                        onClick={() => handleSort('test_rmse')}
                      >
                        RMSE {sortBy === 'test_rmse' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th
                        className="text-right p-4 metric-label cursor-pointer hover:text-[var(--text-primary)]"
                        onClick={() => handleSort('test_mae')}
                      >
                        MAE {sortBy === 'test_mae' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th
                        className="text-right p-4 metric-label cursor-pointer hover:text-[var(--text-primary)]"
                        onClick={() => handleSort('test_cmapss')}
                      >
                        Score {sortBy === 'test_cmapss' && (sortAsc ? '↑' : '↓')}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="mono text-sm">
                    {filteredResults.slice(0, 15).map((r, i) => (
                      <tr key={i} className="border-b border-[var(--border)] hover:bg-[var(--bg-secondary)]">
                        <td className="p-4">{r.model}</td>
                        <td className="p-4 text-[var(--text-secondary)]">{r.dataset}</td>
                        <td className="p-4 text-right">{r.test_rmse?.toFixed(2)}</td>
                        <td className="p-4 text-right">{r.test_mae?.toFixed(2)}</td>
                        <td className="p-4 text-right text-[var(--text-secondary)]">
                          {r.test_cmapss?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Summary */}
              <div className="text-xs text-[var(--text-muted)]">
                Showing {Math.min(15, filteredResults.length)} of {filteredResults.length} results
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
