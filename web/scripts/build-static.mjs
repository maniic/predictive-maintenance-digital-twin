/**
 * Static (demo-mode) build for hosting on GitHub Pages.
 *
 * Next.js cannot statically export dynamic route handlers, so the app/api
 * directory (which shells out to the Python backend) is moved aside for the
 * duration of the build and restored afterwards. The exported site runs
 * entirely on precomputed demo data (see ../scripts/export_demo_data.py).
 *
 * Usage: node scripts/build-static.mjs
 * Env:   NEXT_PUBLIC_BASE_PATH  e.g. /predictive-maintenance-digital-twin
 */
import { execSync } from 'node:child_process'
import { existsSync, renameSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const apiDir = path.join(webRoot, 'app', 'api')
const apiBackup = path.join(webRoot, '.api-routes-excluded')

if (existsSync(apiBackup)) renameSync(apiBackup, apiDir) // recover from a crashed build
renameSync(apiDir, apiBackup)
try {
  execSync('npx next build', {
    cwd: webRoot,
    stdio: 'inherit',
    env: { ...process.env, STATIC_EXPORT: '1', NEXT_PUBLIC_DEMO_MODE: '1' },
  })
} finally {
  renameSync(apiBackup, apiDir)
}
