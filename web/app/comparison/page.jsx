'use client'

import { useState, useEffect, useMemo } from 'react'
import Navigation from '../../components/Navigation'
import PlotlyChart from '../../components/PlotlyChart'
import { fetchComparison } from '../../lib/api'

const METRICS = [
  { value: 'test_rmse', label: 'RMSE' },
  { value: 'test_mae', label: 'MAE' },
  { value: 'test_cmapss', label: 'Score' },
]

const BAR_COLORS = ['#f59e0b', '#3b82f6', '#10b981', '#ef4444', '#a855f7', '#06b6d4', '#ec4899', '#84cc16']

const Skeleton = ({ className = '' }) => <div className={`skeleton ${className}`} />

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
      const d = await fetchComparison()
      setData(d)
    } catch (err) {
      setError(err.message || 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  const datasets = useMemo(() => {
    if (!data?.results) return []
    return ['all', ...new Set(data.results.map(r => r.dataset))]
  }, [data])

  const models = useMemo(() => {
    if (!data?.results) return []
    return [...new Set(data.results.map(r => r.model))]
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
    const dsets = Object.keys(byDataset)

    if (selectedDataset !== 'all') {
      return models.map((model, i) => ({
        x: [model.toUpperCase()],
        y: [byDataset[selectedDataset]?.[model] || 0],
        type: 'bar',
        name: model,
        marker: { color: BAR_COLORS[i % BAR_COLORS.length] },
      }))
    }
    return models.map((model, i) => ({
      x: dsets,
      y: dsets.map(ds => byDataset[ds]?.[model] || 0),
      type: 'bar',
      name: model.toUpperCase(),
      marker: { color: BAR_COLORS[i % BAR_COLORS.length] },
    }))
  }, [data, metric, selectedDataset, models])

  // Heatmap data for "all" view
  const heatmapData = useMemo(() => {
    if (!data?.results || selectedDataset !== 'all') return null
    const byDataset = {}
    data.results.forEach(r => {
      if (!byDataset[r.dataset]) byDataset[r.dataset] = {}
      byDataset[r.dataset][r.model] = r[metric]
    })
    const dsets = [...new Set(data.results.map(r => r.dataset))].sort()
    const z = models.map(model =>
      dsets.map(ds => byDataset[ds]?.[model] ?? null)
    )
    return {
      x: dsets,
      y: models.map(m => m.toUpperCase()),
      z,
    }
  }, [data, metric, selectedDataset, models])

  // Radar data for single dataset view
  const radarData = useMemo(() => {
    if (!data?.results || selectedDataset === 'all') return null
    const dsResults = data.results.filter(r => r.dataset === selectedDataset)
    if (dsResults.length < 2) return null

    const metricKeys = ['test_rmse', 'test_mae', 'test_cmapss']
    const metricLabels = ['RMSE', 'MAE', 'Score']

    // Normalize each metric to 0-1 (inverted so lower=better=outer)
    const maxVals = metricKeys.map(k => Math.max(...dsResults.map(r => r[k] || 0)))
    const minVals = metricKeys.map(k => Math.min(...dsResults.map(r => r[k] || 0)))

    return dsResults.map((r, i) => {
      const vals = metricKeys.map((k, mi) => {
        const range = maxVals[mi] - minVals[mi]
        if (range === 0) return 0.5
        // Invert: lower metric = higher score (closer to 1)
        return 1 - (r[k] - minVals[mi]) / range
      })
      return {
        type: 'scatterpolar',
        r: [...vals, vals[0]], // close the polygon
        theta: [...metricLabels, metricLabels[0]],
        fill: 'toself',
        fillcolor: BAR_COLORS[i % BAR_COLORS.length] + '15',
        name: r.model.toUpperCase(),
        line: { color: BAR_COLORS[i % BAR_COLORS.length], width: 2 },
      }
    })
  }, [data, selectedDataset, models])

  const handleSort = (col) => {
    if (sortBy === col) setSortAsc(!sortAsc)
    else { setSortBy(col); setSortAsc(true) }
  }

  const SortIcon = ({ col }) => {
    if (sortBy !== col) return null
    return <span className="ml-1 text-[var(--amber)]">{sortAsc ? '\u2191' : '\u2193'}</span>
  }

  const summaryText = data?.results
    ? `${models.length} models across ${[...new Set(data.results.map(r => r.dataset))].length} datasets`
    : ''

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation activePage="/comparison" />

      <main className="flex-1">
        <div className="max-w-5xl mx-auto px-4 md:px-6 py-6 md:py-8">
          <div className="mb-6">
            <div className="data-label mb-1">Model Comparison</div>
            <h1 className="text-xl md:text-2xl font-300 text-[var(--text-bright)]">Performance Analysis</h1>
            {summaryText && (
              <p className="font-mono text-[0.65rem] text-[var(--text-faint)] mt-1">{summaryText}</p>
            )}
          </div>

          {error ? (
            <div className="card p-8 text-center">
              <p className="text-[var(--red)] mb-4">{error}</p>
              <button onClick={fetchData} className="btn-primary">Retry</button>
            </div>
          ) : loading ? (
            <div className="space-y-5">
              <div className="flex gap-4">
                <Skeleton className="h-14 w-36" />
                <Skeleton className="h-14 w-36" />
              </div>
              <Skeleton className="h-64 w-full" />
              <Skeleton className="h-80 w-full" />
            </div>
          ) : (
            <div className="space-y-5 animate-fade-up">
              {/* Filters */}
              <div className="flex flex-col sm:flex-row gap-4">
                <div>
                  <label className="data-label block mb-2">Dataset</label>
                  <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    className="w-full sm:w-auto bg-[var(--bg-raised)] border border-[var(--border)] px-3 py-2 font-mono text-sm text-[var(--text-primary)] rounded-sm"
                  >
                    {datasets.map(d => (
                      <option key={d} value={d}>{d === 'all' ? 'All Datasets' : d}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="data-label block mb-2">Metric</label>
                  <select
                    value={metric}
                    onChange={(e) => setMetric(e.target.value)}
                    className="w-full sm:w-auto bg-[var(--bg-raised)] border border-[var(--border)] px-3 py-2 font-mono text-sm text-[var(--text-primary)] rounded-sm"
                  >
                    {METRICS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                  </select>
                </div>
              </div>

              {/* Bar Chart */}
              <div className="card p-5">
                <div className="data-label mb-4">
                  {METRICS.find(m => m.value === metric)?.label} by Model
                </div>
                <PlotlyChart
                  data={chartData}
                  layout={{
                    margin: { t: 10, b: 60, l: 50, r: 10 },
                    height: 260,
                    barmode: 'group',
                    yaxis: { title: { text: METRICS.find(m => m.value === metric)?.label, font: { size: 10 } } },
                    legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center', font: { size: 9 } },
                    showlegend: selectedDataset === 'all',
                  }}
                />
              </div>

              {/* Heatmap — all datasets view */}
              {heatmapData && (
                <div className="card p-5">
                  <div className="data-label mb-4">
                    {METRICS.find(m => m.value === metric)?.label} Heatmap
                  </div>
                  <PlotlyChart
                    data={[{
                      x: heatmapData.x,
                      y: heatmapData.y,
                      z: heatmapData.z,
                      type: 'heatmap',
                      colorscale: 'Viridis',
                      reversescale: true,
                      hoverongaps: false,
                      colorbar: {
                        title: { text: METRICS.find(m => m.value === metric)?.label, font: { size: 10, color: '#94a3b8' } },
                        tickfont: { size: 9, color: '#94a3b8' },
                      },
                    }]}
                    layout={{
                      margin: { t: 10, b: 50, l: 100, r: 10 },
                      height: Math.max(200, models.length * 35 + 80),
                      xaxis: { tickfont: { size: 10 } },
                      yaxis: { tickfont: { size: 10 }, autorange: 'reversed' },
                    }}
                  />
                </div>
              )}

              {/* Radar chart — single dataset view */}
              {radarData && (
                <div className="card p-5">
                  <div className="data-label mb-4">Model Comparison — {selectedDataset}</div>
                  <PlotlyChart
                    data={radarData}
                    layout={{
                      margin: { t: 30, b: 30, l: 60, r: 60 },
                      height: 320,
                      polar: {
                        bgcolor: 'transparent',
                        radialaxis: {
                          visible: true,
                          range: [0, 1],
                          tickfont: { size: 8, color: '#475569' },
                          gridcolor: 'rgba(30, 41, 59, 0.8)',
                        },
                        angularaxis: {
                          tickfont: { size: 10, color: '#94a3b8' },
                          gridcolor: 'rgba(30, 41, 59, 0.8)',
                        },
                      },
                      legend: { orientation: 'h', y: -0.1, x: 0.5, xanchor: 'center', font: { size: 9 } },
                      showlegend: true,
                    }}
                  />
                  <p className="font-mono text-[0.6rem] text-[var(--text-faint)] mt-2">
                    Normalized 0-1 (outer = better). Lower RMSE/MAE/Score maps to higher radar value.
                  </p>
                </div>
              )}

              {/* Table */}
              <div className="card overflow-x-auto">
                <table className="w-full min-w-[600px]">
                  <thead>
                    <tr className="border-b border-[var(--border)]">
                      <th className="text-left px-4 py-3 data-label">Model</th>
                      <th className="text-left px-4 py-3 data-label">Dataset</th>
                      <th
                        className="text-right px-4 py-3 data-label cursor-pointer hover:text-[var(--text-secondary)] transition-colors"
                        onClick={() => handleSort('test_rmse')}
                      >
                        RMSE<SortIcon col="test_rmse" />
                      </th>
                      <th
                        className="text-right px-4 py-3 data-label cursor-pointer hover:text-[var(--text-secondary)] transition-colors"
                        onClick={() => handleSort('test_mae')}
                      >
                        MAE<SortIcon col="test_mae" />
                      </th>
                      <th
                        className="text-right px-4 py-3 data-label cursor-pointer hover:text-[var(--text-secondary)] transition-colors"
                        onClick={() => handleSort('test_cmapss')}
                      >
                        Score<SortIcon col="test_cmapss" />
                      </th>
                      <th
                        className="text-right px-4 py-3 data-label cursor-pointer hover:text-[var(--text-secondary)] transition-colors"
                        onClick={() => handleSort('epochs')}
                      >
                        Epochs<SortIcon col="epochs" />
                      </th>
                      <th
                        className="text-right px-4 py-3 data-label cursor-pointer hover:text-[var(--text-secondary)] transition-colors"
                        onClick={() => handleSort('val_rmse')}
                      >
                        Val RMSE<SortIcon col="val_rmse" />
                      </th>
                    </tr>
                  </thead>
                  <tbody className="font-mono text-sm">
                    {filteredResults.map((r, i) => (
                      <tr
                        key={i}
                        className="border-b border-[var(--border)] hover:bg-[var(--bg-raised)]/50 transition-colors"
                      >
                        <td className="px-4 py-3 text-[var(--text-bright)]">{r.model}</td>
                        <td className="px-4 py-3 text-[var(--text-muted)]">{r.dataset}</td>
                        <td className="px-4 py-3 text-right">{r.test_rmse?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right">{r.test_mae?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-[var(--text-muted)]">
                          {r.test_cmapss?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className="px-4 py-3 text-right text-[var(--text-muted)]">
                          {r.epochs ?? '—'}
                        </td>
                        <td className="px-4 py-3 text-right text-[var(--text-muted)]">
                          {r.val_rmse?.toFixed(2) ?? '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="font-mono text-[0.65rem] text-[var(--text-faint)]">
                {filteredResults.length} results
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
