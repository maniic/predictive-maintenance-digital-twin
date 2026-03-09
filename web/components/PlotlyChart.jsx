'use client'

import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => <div className="h-[200px] skeleton" />,
})

export default function PlotlyChart({ data, layout, config, className = '' }) {
  const defaultLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#666', family: 'Geist Mono, monospace', size: 11 },
    margin: { t: 10, b: 40, l: 40, r: 10 },
    height: 200,
    ...layout,
    xaxis: { gridcolor: '#1a1a1a', zeroline: false, ...layout?.xaxis },
    yaxis: { gridcolor: '#1a1a1a', zeroline: false, ...layout?.yaxis },
  }

  const defaultConfig = {
    displayModeBar: false,
    staticPlot: false,
    ...config,
  }

  return (
    <div className={className}>
      <Plot
        data={data}
        layout={defaultLayout}
        config={defaultConfig}
        style={{ width: '100%' }}
      />
    </div>
  )
}
