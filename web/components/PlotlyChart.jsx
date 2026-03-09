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
    font: { color: '#94a3b8', family: 'IBM Plex Mono, monospace', size: 10 },
    margin: { t: 10, b: 40, l: 45, r: 10 },
    height: 220,
    ...layout,
    xaxis: {
      gridcolor: 'rgba(30, 41, 59, 0.8)',
      zeroline: false,
      tickfont: { size: 9 },
      ...layout?.xaxis,
    },
    yaxis: {
      gridcolor: 'rgba(30, 41, 59, 0.8)',
      zeroline: false,
      tickfont: { size: 9 },
      ...layout?.yaxis,
    },
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
