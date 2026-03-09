export default function MetricCard({ label, value, unit, highlight }) {
  return (
    <div className="panel p-3 md:p-5">
      <div className="metric-label mb-1">{label}</div>
      <div className={`metric-value text-lg md:text-2xl ${highlight || ''}`}>
        {value}
      </div>
      {unit && <div className="text-xs text-[var(--text-muted)]">{unit}</div>}
    </div>
  )
}
