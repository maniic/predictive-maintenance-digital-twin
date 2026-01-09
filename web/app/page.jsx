import Link from 'next/link'

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--border)] px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="status-dot pulse-slow" />
            <span className="mono text-xs tracking-wider">DIGITAL TWIN</span>
          </div>
          <nav className="flex items-center gap-6">
            <Link href="/prediction" className="nav-link">Predict</Link>
            <Link href="/simulation" className="nav-link">Simulate</Link>
            <Link href="/comparison" className="nav-link">Compare</Link>
          </nav>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex items-center">
        <div className="max-w-6xl mx-auto px-6 py-16 w-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            {/* Left Column - Text */}
            <div>
              <p className="mono text-xs text-[var(--accent)] mb-4 tracking-wider animate-fade-in">
                PREDICTIVE MAINTENANCE
              </p>
              <h1 className="text-4xl lg:text-5xl font-light mb-6 tracking-tight leading-tight animate-slide-up">
                Remaining Useful Life
                <br />
                <span className="text-[var(--text-secondary)]">Prediction System</span>
              </h1>
              <p className="text-[var(--text-secondary)] mb-8 leading-relaxed max-w-lg animate-slide-up-delay">
                Deep learning models for turbofan engine prognostics using the NASA C-MAPSS dataset.
                Real-time RUL prediction with uncertainty quantification and ensemble methods.
              </p>
              <div className="flex flex-wrap gap-3 animate-slide-up-delay-2">
                <Link href="/prediction" className="btn-primary">
                  Run Prediction
                </Link>
                <Link href="/simulation" className="btn-secondary">
                  Live Demo
                </Link>
                <Link href="/comparison" className="btn-secondary">
                  View Models
                </Link>
              </div>
            </div>

            {/* Right Column - Visual */}
            <div className="hidden lg:block">
              <div className="relative">
                {/* Background glow */}
                <div className="absolute inset-0 bg-gradient-radial from-[var(--accent)]/5 to-transparent blur-3xl" />

                {/* Engine visualization placeholder */}
                <div className="relative p-8">
                  <div className="aspect-square flex items-center justify-center">
                    <div className="relative">
                      {/* Animated rings */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-48 h-48 border border-[var(--border)] rounded-full animate-spin-slow opacity-30" />
                      </div>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-36 h-36 border border-[var(--border)] rounded-full animate-spin-slow-reverse opacity-40" />
                      </div>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-24 h-24 border border-[var(--accent)]/30 rounded-full animate-pulse-slow" />
                      </div>

                      {/* Center content */}
                      <div className="relative w-64 h-64 flex flex-col items-center justify-center text-center">
                        <div className="metric-label mb-2">System Status</div>
                        <div className="metric-value text-4xl text-[var(--accent)] mb-1">ONLINE</div>
                        <div className="text-xs text-[var(--text-muted)]">7 models ready</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[var(--border)] mt-16 animate-fade-in-delay">
            {[
              { value: '7', label: 'Deep Learning Models', detail: 'LSTM, CNN, Transformer, Ensemble' },
              { value: '11.7', label: 'Best RMSE Score', detail: 'Cycles prediction error' },
              { value: '21', label: 'Engine Sensors', detail: 'Temperature, pressure, speed' },
              { value: '4', label: 'C-MAPSS Datasets', detail: 'FD001 - FD004' },
            ].map((stat) => (
              <div key={stat.label} className="bg-[var(--bg-primary)] p-6 hover:bg-[var(--bg-secondary)] transition-colors group">
                <div className="metric-value text-3xl mb-1 group-hover:text-[var(--accent)] transition-colors">{stat.value}</div>
                <div className="metric-label mb-2">{stat.label}</div>
                <div className="text-xs text-[var(--text-muted)]">{stat.detail}</div>
              </div>
            ))}
          </div>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-12">
            <Link href="/prediction" className="panel p-6 hover:border-[var(--accent)]/30 transition-all group">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded bg-[var(--accent)]/10 flex items-center justify-center text-[var(--accent)] text-sm">
                  P
                </div>
                <div className="metric-label group-hover:text-[var(--text-primary)] transition-colors">Prediction</div>
              </div>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                Analyze engine sensor data and get RUL predictions with confidence intervals from multiple deep learning models.
              </p>
            </Link>

            <Link href="/simulation" className="panel p-6 hover:border-[var(--accent)]/30 transition-all group">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded bg-[var(--accent)]/10 flex items-center justify-center text-[var(--accent)] text-sm">
                  S
                </div>
                <div className="metric-label group-hover:text-[var(--text-primary)] transition-colors">Simulation</div>
              </div>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                Watch real-time engine degradation simulation with live RUL updates, health scores, and maintenance alerts.
              </p>
            </Link>

            <Link href="/comparison" className="panel p-6 hover:border-[var(--accent)]/30 transition-all group">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded bg-[var(--accent)]/10 flex items-center justify-center text-[var(--accent)] text-sm">
                  C
                </div>
                <div className="metric-label group-hover:text-[var(--text-primary)] transition-colors">Comparison</div>
              </div>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                Compare model performance across datasets with RMSE, MAE, and C-MAPSS scores. Filter and sort results.
              </p>
            </Link>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border)] px-6 py-4">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 mono text-xs text-[var(--text-muted)]">
          <div className="flex items-center gap-4">
            <span>NASA C-MAPSS Dataset</span>
            <span className="text-[var(--border)]">|</span>
            <span>PyTorch Lightning</span>
          </div>
          <div className="flex items-center gap-4">
            <span>MLflow Tracking</span>
            <span className="text-[var(--border)]">|</span>
            <span>Next.js + Plotly</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
