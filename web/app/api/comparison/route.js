import { spawn } from 'child_process'
import path from 'path'

export async function GET() {
  const projectRoot = path.resolve(process.cwd(), '..')

  return new Promise((resolve) => {
    const py = spawn('python', [
      'src/api/predict.py',
      '--action', 'comparison',
    ], { cwd: projectRoot })

    let stdout = ''
    let stderr = ''

    py.stdout.on('data', (data) => { stdout += data })
    py.stderr.on('data', (data) => { stderr += data })

    py.on('close', (code) => {
      if (code !== 0) {
        resolve(Response.json({ error: stderr || 'Failed to get comparison' }, { status: 500 }))
      } else {
        try {
          const result = JSON.parse(stdout)
          resolve(Response.json(result))
        } catch {
          resolve(Response.json({ error: 'Invalid response' }, { status: 500 }))
        }
      }
    })
  })
}
