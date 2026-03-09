import './globals.css'

export const metadata = {
  title: 'RUL Monitor — Predictive Maintenance Digital Twin',
  description: 'Remaining Useful Life prediction for turbofan engines using deep learning',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased relative">
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  )
}
