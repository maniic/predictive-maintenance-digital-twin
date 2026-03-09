import { spawn } from 'child_process'
import path from 'path'

const PROCESS_TIMEOUT_MS = 30000

export async function POST(request) {
  const { dataset = 'FD001', engine = 1, model = 'ensemble' } = await request.json()

  const projectRoot = path.resolve(process.cwd(), '..')

  return new Promise((resolve) => {
    const py = spawn('python', [
      'src/api/predict.py',
      '--action', 'predict',
      '--dataset', dataset,
      '--engine', String(engine),
      '--model', model,
    ], { cwd: projectRoot })

    let stdout = ''
    let stderr = ''

    const timer = setTimeout(() => {
      py.kill('SIGTERM')
      resolve(Response.json({ error: 'Prediction timed out' }, { status: 504 }))
    }, PROCESS_TIMEOUT_MS)

    py.stdout.on('data', (data) => { stdout += data })
    py.stderr.on('data', (data) => { stderr += data })

    py.on('close', (code) => {
      clearTimeout(timer)
      if (code !== 0) {
        resolve(Response.json({ error: 'Prediction failed' }, { status: 500 }))
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
