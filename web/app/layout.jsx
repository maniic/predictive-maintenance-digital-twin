import './globals.css'

export const metadata = {
  title: 'Digital Twin — Predictive Maintenance',
  description: 'RUL Prediction for Turbofan Engines',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  )
}
