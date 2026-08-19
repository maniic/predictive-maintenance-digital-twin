'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { fetchComparison } from '../lib/api'

const Skeleton = ({ className = '' }) => <div className={`skeleton ${className}`} />

export default function Home() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetchComparison()
      .then((data) => {
        if (!data?.results?.length) return
        const results = data.results
        const models = new Set(results.map((r) => r.model))
        const datasets = new Set(results.map((r) => r.dataset))
        // Best RMSE across every model and dataset, matching the README badge.
        const rmses = results.map((r) => r.test_rmse).filter((v) => typeof v === 'number')
        const bestRmse = rmses.length ? Math.min(...rmses) : null
        setStats({
          modelCount: models.size,
          bestRmse: bestRmse ? bestRmse.toFixed(1) : '—',
          datasetCount: datasets.size,
        })
      })
      .catch(() => {})
  }, [])

  const statsData = [
    {
      value: stats?.modelCount ?? '9',
      label: 'DL Models',
      sub: 'LSTM / CNN / Transformer / attention variants',
    },
    { value: stats?.bestRmse ?? '—', label: 'Best RMSE', sub: 'Cycles error (best dataset)' },
    { value: '21', label: 'Sensors', sub: 'Temp, pressure, speed' },
    { value: stats?.datasetCount ?? '4', label: 'Datasets', sub: 'FD001 — FD004' },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top bar */}
      <header className="border-b border-[var(--border)] bg-[var(--bg-surface)]/80 backdrop-blur-sm">
        <div className="flex items-center justify-between px-4 md:px-6 h-12">
          <div className="flex items-center gap-2.5">
            <div className="indicator" />
            <span className="font-condensed text-[0.7rem] font-600 uppercase tracking-[0.15em] text-[var(--text-secondary)]">
              RUL Monitor
            </span>
          </div>
          <nav className="flex items-center gap-6">
            <Link href="/prediction" className="nav-link">
              Prediction
            </Link>
            <Link href="/simulation" className="nav-link">
              Simulation
            </Link>
            <Link href="/comparison" className="nav-link">
              Comparison
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <main className="flex-1 flex items-center justify-center px-6">
        <div className="max-w-4xl w-full py-16 md:py-24">
          {/* System header */}
          <div className="animate-fade-up">
            <div className="flex items-center gap-3 mb-6">
              <div className="h-px flex-1 bg-gradient-to-r from-[var(--amber)]/40 to-transparent" />
              <span className="font-condensed text-[0.65rem] uppercase tracking-[0.2em] text-[var(--amber)]">
                Predictive Maintenance System
              </span>
              <div className="h-px flex-1 bg-gradient-to-l from-[var(--amber)]/40 to-transparent" />
            </div>
          </div>

          <div className="text-center animate-fade-up-1">
            <h1 className="text-3xl md:text-5xl lg:text-6xl font-300 tracking-tight mb-4 text-[var(--text-bright)]">
              Remaining Useful Life
            </h1>
            <h2 className="text-lg md:text-xl font-300 text-[var(--text-muted)] mb-8">
              Turbofan Engine Prognostics — NASA C-MAPSS
            </h2>
            <p className="text-sm text-[var(--text-secondary)] max-w-xl mx-auto leading-relaxed mb-10">
              Every flight wears an engine down. This system reads 21 onboard sensors and estimates
              how many flights remain before maintenance is due — deep learning ensembles (LSTM,
              CNN, Transformer) with calibrated uncertainty, trained on NASA&apos;s run-to-failure
              C-MAPSS data.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap justify-center gap-3 mb-16 animate-fade-up-2">
            <Link href="/prediction" className="btn-primary">
              Run Prediction
            </Link>
            <Link href="/simulation" className="btn-secondary">
              Degradation Sim
            </Link>
            <Link href="/comparison" className="btn-secondary">
              Model Comparison
            </Link>
          </div>

          {/* Stats strip */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px border border-[var(--border)] animate-fade-up-3">
            {statsData.map((stat) => (
              <div
                key={stat.label}
                className="bg-[var(--bg-surface)] p-5 md:p-6 group hover:bg-[var(--bg-raised)] transition-colors"
              >
                {stats === null && stat.label !== 'Sensors' ? (
                  <>
                    <Skeleton className="h-8 w-12 mb-2" />
                    <div className="data-label mb-1">{stat.label}</div>
                    <div className="font-mono text-[0.6rem] text-[var(--text-faint)]">
                      {stat.sub}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="data-value text-2xl md:text-3xl mb-1 group-hover:text-[var(--amber)] transition-colors">
                      {stat.value}
                    </div>
                    <div className="data-label mb-1">{stat.label}</div>
                    <div className="font-mono text-[0.6rem] text-[var(--text-faint)]">
                      {stat.sub}
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>

          {/* Module cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-6">
            {[
              {
                href: '/prediction',
                tag: 'PRED',
                title: 'Prediction',
                desc: 'Ensemble RUL prediction with confidence intervals from multiple deep learning architectures.',
              },
              {
                href: '/simulation',
                tag: 'SIM',
                title: 'Simulation',
                desc: 'Real-time engine degradation playback with live ML-backed RUL tracking and health alerts.',
              },
              {
                href: '/comparison',
                tag: 'COMP',
                title: 'Comparison',
                desc: 'Cross-model performance analysis with RMSE, MAE, and asymmetric C-MAPSS scoring.',
              },
            ].map((card) => (
              <Link
                key={card.href}
                href={card.href}
                className="card p-5 hover:border-[var(--border-light)] transition-all group"
              >
                <div className="flex items-center gap-2.5 mb-3">
                  <span className="font-mono text-[0.6rem] px-1.5 py-0.5 bg-[var(--amber-dim)] text-[var(--amber)] rounded-sm">
                    {card.tag}
                  </span>
                  <span className="font-condensed text-xs uppercase tracking-wider text-[var(--text-secondary)] group-hover:text-[var(--text-bright)] transition-colors">
                    {card.title}
                  </span>
                </div>
                <p className="text-[0.8rem] text-[var(--text-muted)] leading-relaxed group-hover:text-[var(--text-secondary)] transition-colors">
                  {card.desc}
                </p>
              </Link>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border)] px-6 py-3">
        <div className="flex items-center justify-between font-mono text-[0.6rem] text-[var(--text-faint)]">
          <span>NASA C-MAPSS / PyTorch Lightning / MLflow</span>
          <span>Next.js + Plotly.js</span>
        </div>
      </footer>
    </div>
  )
}
