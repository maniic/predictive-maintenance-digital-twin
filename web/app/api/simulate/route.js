import { runPython } from '../../../lib/python'

const TIMEOUT_MS = 60000

/** GET /api/simulate?initial_rul=150&rate=1.0&mode=hpc */
export async function GET(request) {
  const { searchParams } = new URL(request.url)
  return run({
    initialRul: searchParams.get('initial_rul') || '150',
    degradationRate: searchParams.get('rate') || '1.0',
    faultMode: searchParams.get('mode') || 'hpc',
  })
}

export async function POST(request) {
  const { initialRul = 150, degradationRate = 1.0, faultMode = 'hpc' } = await request.json()
  return run({ initialRul, degradationRate, faultMode })
}

function run({ initialRul, degradationRate, faultMode }) {
  return runPython(
    [
      '--action',
      'simulate',
      '--initial_rul',
      String(initialRul),
      '--rate',
      String(degradationRate),
      '--mode',
      faultMode,
    ],
    { timeoutMs: TIMEOUT_MS, errorMessage: 'Simulation failed' },
  )
}
