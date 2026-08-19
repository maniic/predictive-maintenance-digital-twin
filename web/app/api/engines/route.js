import { runPython } from '../../../lib/python'

/** GET /api/engines?dataset=FD001 */
export async function GET(request) {
  const { searchParams } = new URL(request.url)
  return runPython(['--action', 'engines', '--dataset', searchParams.get('dataset') || 'FD001'], {
    errorMessage: 'Failed to list engines',
  })
}
