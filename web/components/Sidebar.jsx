'use client'

import Link from 'next/link'
import { NAV_ITEMS } from './Navigation'

export default function Sidebar({ activePage, isOpen, onClose, children }) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={onClose}
        />
      )}

      <aside className={`
        fixed md:relative inset-y-0 left-0 z-50
        w-64 bg-[var(--bg-primary)] border-r border-[var(--border)] p-6
        transform transition-transform duration-200
        ${isOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
        md:flex-shrink-0 overflow-y-auto
      `}>
        {/* Mobile close button */}
        <button
          className="md:hidden absolute top-4 right-4"
          onClick={onClose}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Mobile nav links */}
        <nav className="md:hidden flex flex-col gap-4 mb-6 pb-6 border-b border-[var(--border)]">
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

        <div className="space-y-5">
          {children}
        </div>
      </aside>
    </>
  )
}
