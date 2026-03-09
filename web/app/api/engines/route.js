import { spawn } from 'child_process'
import path from 'path'

const PROCESS_TIMEOUT_MS = 30000

export async function GET(request) {
  const { searchParams } = new URL(request.url)
  const dataset = searchParams.get('dataset') || 'FD001'

  const projectRoot = path.resolve(process.cwd(), '..')

  return new Promise((resolve) => {
    const py = spawn('python', [
      'src/api/predict.py',
      '--action', 'engines',
      '--dataset', dataset,
    ], { cwd: projectRoot })

    let stdout = ''
    let stderr = ''

    const timer = setTimeout(() => {
      py.kill('SIGTERM')
      resolve(Response.json({ error: 'Request timed out' }, { status: 504 }))
    }, PROCESS_TIMEOUT_MS)

    py.stdout.on('data', (data) => { stdout += data })
    py.stderr.on('data', (data) => { stderr += data })

    py.on('close', (code) => {
      clearTimeout(timer)
      if (code !== 0) {
        resolve(Response.json({ error: 'Failed to get engines' }, { status: 500 }))
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
