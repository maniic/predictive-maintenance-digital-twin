import { runPython } from '../../../lib/python'

const TIMEOUT_MS = 30000

/**
 * GET /api/predict?dataset=FD001&engine=9&model=ensemble
 * GET /api/predict?action=trajectory&dataset=FD001&engine=9
 *
 * GET is what the dashboard client sends; POST is kept for scripted callers.
 */
export async function GET(request) {
  const { searchParams } = new URL(request.url)
  return run({
    action: searchParams.get('action') === 'trajectory' ? 'trajectory' : 'predict',
    dataset: searchParams.get('dataset') || 'FD001',
    engine: searchParams.get('engine') || '1',
    model: searchParams.get('model') || 'ensemble',
  })
}

export async function POST(request) {
  const {
    dataset = 'FD001',
    engine = 1,
    model = 'ensemble',
    action = 'predict',
  } = await request.json()
  return run({ action, dataset, engine, model })
}

function run({ action, dataset, engine, model }) {
  return runPython(
    ['--action', action, '--dataset', dataset, '--engine', String(engine), '--model', model],
    { timeoutMs: TIMEOUT_MS, errorMessage: 'Prediction failed' },
  )
}
