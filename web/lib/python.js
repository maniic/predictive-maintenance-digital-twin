/**
 * Shared bridge to the Python inference CLI (src/api/predict.py).
 *
 * Every API route shells out through this helper so the interpreter choice,
 * timeout handling and error reporting stay in one place. The interpreter is
 * resolved as: $PYTHON, else the active virtualenv's python, else `python3`
 * (`python` frequently does not exist on Linux, which used to surface as an
 * opaque 500).
 */
import { spawn } from 'child_process'
import path from 'path'

export const PROJECT_ROOT = path.resolve(process.cwd(), '..')

function pythonBinary() {
  if (process.env.PYTHON) return process.env.PYTHON
  if (process.env.VIRTUAL_ENV) return path.join(process.env.VIRTUAL_ENV, 'bin', 'python')
  return 'python3'
}

/**
 * Run src/api/predict.py and parse its JSON output.
 *
 * @param {string[]} args    CLI arguments after the script path
 * @param {object}   options
 * @param {number}   options.timeoutMs
 * @param {string}   options.errorMessage  shown to the client on failure
 * @returns {Promise<Response>}
 */
export function runPython(args, { timeoutMs = 30000, errorMessage = 'Request failed' } = {}) {
  return new Promise((resolve) => {
    let py
    try {
      py = spawn(pythonBinary(), ['src/api/predict.py', ...args], { cwd: PROJECT_ROOT })
    } catch (err) {
      resolve(Response.json({ error: `Could not start Python: ${err.message}` }, { status: 500 }))
      return
    }

    let stdout = ''
    let stderr = ''
    let settled = false

    const finish = (body, status) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve(Response.json(body, status ? { status } : undefined))
    }

    const timer = setTimeout(() => {
      py.kill('SIGTERM')
      finish({ error: `${errorMessage}: timed out after ${timeoutMs / 1000}s` }, 504)
    }, timeoutMs)

    py.stdout.on('data', (chunk) => { stdout += chunk })
    py.stderr.on('data', (chunk) => { stderr += chunk })

    py.on('error', (err) => {
      // ENOENT here means the interpreter itself is missing — the most common
      // local-setup failure, so name it rather than returning a bare 500.
      const detail = err.code === 'ENOENT'
        ? `Python interpreter '${pythonBinary()}' not found. Activate the project venv or set PYTHON.`
        : err.message
      finish({ error: detail }, 500)
    })

    py.on('close', (code) => {
      if (code !== 0) {
        const detail = stderr.trim().split('\n').pop() || `exit code ${code}`
        finish({ error: `${errorMessage}: ${detail}` }, 500)
        return
      }
      try {
        finish(JSON.parse(stdout))
      } catch {
        finish({ error: `${errorMessage}: unparseable output from predict.py` }, 500)
      }
    })
  })
}
