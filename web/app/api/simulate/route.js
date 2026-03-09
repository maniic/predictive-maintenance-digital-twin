import { spawn } from 'child_process'
import path from 'path'

const PROCESS_TIMEOUT_MS = 60000

export async function POST(request) {
  const { initialRul = 150, degradationRate = 1.0, faultMode = 'hpc' } = await request.json()

  const projectRoot = path.resolve(process.cwd(), '..')

  return new Promise((resolve) => {
    const py = spawn('python', [
      'src/api/predict.py',
      '--action', 'simulate',
      '--initial_rul', String(initialRul),
      '--rate', String(degradationRate),
      '--mode', faultMode,
    ], { cwd: projectRoot })

    let stdout = ''
    let stderr = ''

    const timer = setTimeout(() => {
      py.kill('SIGTERM')
      resolve(Response.json({ error: 'Simulation timed out' }, { status: 504 }))
    }, PROCESS_TIMEOUT_MS)

    py.stdout.on('data', (data) => { stdout += data })
    py.stderr.on('data', (data) => { stderr += data })

    py.on('close', (code) => {
      clearTimeout(timer)
      if (code !== 0) {
        resolve(Response.json({ error: 'Simulation failed' }, { status: 500 }))
      } else {
        try {
          const result = JSON.parse(stdout)
          resolve(Response.json(result))
        } catch {
          resolve(Response.json({ error: 'Invalid response from simulation' }, { status: 500 }))
        }
      }
    })
  })
}
