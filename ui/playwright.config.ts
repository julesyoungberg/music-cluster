import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const uiDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(uiDir, '..');
const workspace = process.env.MUSIC_CLUSTER_E2E_WORKSPACE ?? path.join(repoRoot, '.e2e-workspace');
const python = process.env.PYTHON ?? path.join(repoRoot, '.venv', 'bin', 'python');

const API_PORT = 8123;
const UI_PORT = 4173;

/**
 * The browser suite drives the real application: the built SvelteKit bundle
 * talking to a real uvicorn process over HTTP, against a library that was
 * analysed and fitted for real. Nothing is mocked, so a break anywhere between
 * a click and the database shows up here.
 */
export default defineConfig({
  testDir: './e2e',
  // The API is one shared, stateful process, so the specs run in order.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : [['list']],
  timeout: 60_000,
  expect: { timeout: 10_000 },

  use: {
    baseURL: `http://127.0.0.1:${UI_PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure'
  },

  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // Environments that ship their own Chromium (some CI images, dev
        // containers) point at it here rather than downloading a second copy.
        launchOptions: process.env.PLAYWRIGHT_CHROMIUM_PATH
          ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH }
          : {}
      }
    }
  ],

  webServer: [
    {
      // Seeding runs as part of the command so it is guaranteed to finish
      // before uvicorn binds the port Playwright waits for.
      command:
        `${python} scripts/seed_e2e_library.py ${workspace} && ` +
        `${python} -m uvicorn music_cluster.api:app --host 127.0.0.1 --port ${API_PORT}`,
      cwd: repoRoot,
      url: `http://127.0.0.1:${API_PORT}/api/info`,
      reuseExistingServer: !process.env.CI,
      timeout: 300_000,
      stdout: 'pipe',
      stderr: 'pipe',
      env: {
        MUSIC_CLUSTER_DB: path.join(workspace, 'library.db'),
        MUSIC_CLUSTER_CONFIG: path.join(workspace, 'config.yaml'),
        MUSIC_CLUSTER_CACHE: path.join(workspace, 'cache')
      }
    },
    {
      command: `npm run build && npx vite preview --port ${UI_PORT} --strictPort`,
      cwd: uiDir,
      url: `http://127.0.0.1:${UI_PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        VITE_API_BASE: `http://127.0.0.1:${API_PORT}`
      }
    }
  ]
});
