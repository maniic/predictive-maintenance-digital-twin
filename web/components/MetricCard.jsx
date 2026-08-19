export default function MetricCard({ label, value, unit, highlight }) {
  return (
    <div className="card p-4">
      <div className="data-label mb-1.5">{label}</div>
      <div className={`data-value text-lg md:text-xl ${highlight || ''}`}>{value}</div>
      {unit && (
        <div className="font-mono text-[0.65rem] text-[var(--text-muted)] mt-0.5">{unit}</div>
      )}
    </div>
  )
}
