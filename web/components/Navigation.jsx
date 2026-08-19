'use client'

import Link from 'next/link'

const NAV_ITEMS = [
  { href: '/prediction', label: 'Prediction' },
  { href: '/simulation', label: 'Simulation' },
  { href: '/comparison', label: 'Comparison' },
]

export default function Navigation({ activePage, statusClass = '', onMenuToggle }) {
  return (
    <header className="border-b border-[var(--border)] bg-[var(--bg-surface)]/80 backdrop-blur-sm sticky top-0 z-30">
      <div className="flex items-center justify-between px-4 md:px-6 h-12">
        {/* Left: hamburger + logo */}
        <div className="flex items-center gap-3">
          {onMenuToggle && (
            <button
              onClick={onMenuToggle}
              className="md:hidden p-1.5 -ml-1.5 text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18" />
              </svg>
            </button>
          )}
          <Link href="/" className="flex items-center gap-2.5">
            <div className={`indicator ${statusClass}`} />
            <span className="font-condensed text-[0.7rem] font-600 uppercase tracking-[0.15em] text-[var(--text-secondary)]">
              RUL Monitor
            </span>
          </Link>
          <span className="hidden sm:inline font-mono text-[0.6rem] text-[var(--text-faint)] ml-1">
            v2.1
          </span>
          {process.env.NEXT_PUBLIC_DEMO_MODE === '1' && (
            <span
              className="font-mono text-[0.6rem] uppercase tracking-wider px-1.5 py-0.5 rounded-sm border border-[var(--amber)] text-[var(--amber)] ml-1"
              title="Hosted demo: real C-MAPSS ground truth, illustrative predictions calibrated to the reported test error. No live model."
            >
              demo
            </span>
          )}
        </div>

        {/* Center: nav links */}
        <nav className="hidden md:flex items-center gap-8">
          {NAV_ITEMS.map(item => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-link ${activePage === item.href ? 'active' : ''}`}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Right: system info */}
        <div className="flex items-center gap-4">
          <span className="hidden lg:inline font-mono text-[0.6rem] text-[var(--text-faint)]">
            C-MAPSS / FD001–FD004
          </span>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--green)]" />
            <span className="font-mono text-[0.6rem] text-[var(--text-muted)]">SYS OK</span>
          </div>
        </div>
      </div>
    </header>
  )
}

export { NAV_ITEMS }
