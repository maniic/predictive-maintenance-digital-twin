import { runPython } from '../../../lib/python'

/** GET /api/comparison — published training results from models/*.json */
export async function GET() {
  return runPython(['--action', 'comparison'], {
    errorMessage: 'Failed to load comparison data',
  })
}
