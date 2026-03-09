'use client'

import Link from 'next/link'

const NAV_ITEMS = [
  { href: '/prediction', label: 'Predict' },
  { href: '/simulation', label: 'Simulate' },
  { href: '/comparison', label: 'Compare' },
]

export default function Navigation({ activePage, statusClass = '', onMenuToggle }) {
  return (
    <header className="border-b border-[var(--border)] px-4 md:px-6 py-4">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-4">
          {onMenuToggle && (
            <button
              onClick={onMenuToggle}
              className="md:hidden p-2 -ml-2 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18" />
              </svg>
            </button>
          )}
          <Link href="/" className="flex items-center gap-2">
            <div className={`status-dot ${statusClass}`} />
            <span className="mono text-xs tracking-wider">DIGITAL TWIN</span>
          </Link>
        </div>

        <nav className="hidden md:flex items-center gap-6">
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

        {/* Mobile menu button (fallback when no sidebar hamburger) */}
        {!onMenuToggle && (
          <button
            className="md:hidden p-2"
            onClick={() => {}}
            id="mobile-nav-toggle"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        )}
      </div>
    </header>
  )
}

export { NAV_ITEMS }
