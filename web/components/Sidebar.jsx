'use client'

import Link from 'next/link'
import { NAV_ITEMS } from './Navigation'

export default function Sidebar({ activePage, isOpen, onClose, children }) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`
        fixed md:relative inset-y-0 left-0 z-50
        w-[260px] bg-[var(--bg-surface)] border-r border-[var(--border)]
        transform transition-transform duration-200
        ${isOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
        md:flex-shrink-0 overflow-y-auto
      `}
      >
        {/* Mobile header */}
        <div className="md:hidden flex items-center justify-between p-4 border-b border-[var(--border)]">
          <span className="font-condensed text-xs uppercase tracking-wider text-[var(--text-muted)]">
            Controls
          </span>
          <button
            onClick={onClose}
            className="p-1 text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Mobile nav */}
        <nav className="md:hidden flex flex-col gap-1 p-3 border-b border-[var(--border)]">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`px-3 py-2 rounded text-sm ${
                activePage === item.href
                  ? 'text-[var(--amber)] bg-[var(--amber-dim)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-raised)]'
              } transition-colors`}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Controls */}
        <div className="p-5 space-y-5">{children}</div>
      </aside>
    </>
  )
}
