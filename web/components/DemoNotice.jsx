import { DEMO_MODE } from '../lib/api'

/**
 * Banner shown only on the hosted (static) build.
 *
 * The deployed dashboard has no Python backend and no trained checkpoints, so
 * its predictions are illustrative values calibrated to each dataset's reported
 * test error — the engine data and ground-truth RUL are real C-MAPSS. Saying so
 * plainly is the point of this component.
 */
export default function DemoNotice({ children }) {
  if (!DEMO_MODE) return null

  return (
    <div className="card p-3 mb-5 border-l-2 border-l-[var(--amber)]">
      <p className="font-mono text-[0.65rem] leading-relaxed text-[var(--text-muted)]">
        <span className="text-[var(--amber)]">DEMO </span>
        {children} Run it against the real models locally — see the{' '}
        <a
          href="https://github.com/maniic/predictive-maintenance-digital-twin#reproduce-these-results"
          className="text-[var(--amber)] underline underline-offset-2"
          target="_blank"
          rel="noreferrer"
        >
          README
        </a>
        .
      </p>
    </div>
  )
}
