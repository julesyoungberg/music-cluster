import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';
import {
  API_PORT,
  UI_PORT,
  apiBase,
  outputDir,
  repoRoot,
  uiDir,
  workspace
} from './snapshots/paths';

const run = path.join(workspace, 'run');
const python = process.env.PYTHON ?? path.join(repoRoot, '.venv', 'bin', 'python');
const quote = (value: string) => JSON.stringify(value);
// UI_SNAPSHOT_RESEED=1 synthesises the library again, e.g. after changing
// analysis code or interrupting the first seed.
const reseed = process.env.UI_SNAPSHOT_RESEED ? ' --reseed' : '';

/**
 * `npm run ui:snapshot` — capture every screen of the real app for a person or
 * an agent to look at. Not a test suite: nothing is compared to a baseline.
 * See ui/snapshots/README.md.
 */
export default defineConfig({
  testDir: './snapshots',
  outputDir: path.join(outputDir, '.playwright'),
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  timeout: 150_000,
  expect: { timeout: 15_000 },
  globalSetup: './snapshots/global-setup.ts',
  globalTeardown: './snapshots/global-teardown.ts',

  use: {
    baseURL: `http://127.0.0.1:${UI_PORT}`,
    viewport: { width: 1280, height: 800 },
    trace: 'off',
    screenshot: 'off',
    // A step that cannot find its element should fail fast and name the
    // locator, rather than hang until the whole flow times out.
    actionTimeout: 15_000,
    navigationTimeout: 30_000
  },

  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 800 },
        launchOptions: process.env.PLAYWRIGHT_CHROMIUM_PATH
          ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH }
          : {}
      }
    }
  ],

  webServer: [
    {
      // Every run starts from a fresh copy of the seeded library.
      command:
        `${quote(python)} scripts/prepare_ui_snapshot.py ${quote(workspace)}${reseed} && ` +
        `${quote(python)} -m uvicorn music_cluster.api:app --host 127.0.0.1 --port ${API_PORT}`,
      cwd: repoRoot,
      url: `${apiBase}/api/info`,
      reuseExistingServer: false,
      timeout: 300_000,
      stdout: 'ignore',
      stderr: 'pipe',
      env: {
        MUSIC_CLUSTER_DB: path.join(run, 'library.db'),
        MUSIC_CLUSTER_CONFIG: path.join(run, 'config.yaml'),
        MUSIC_CLUSTER_CACHE: path.join(run, 'cache')
      }
    },
    {
      // The dev server, not a build: it is up in seconds and always serves
      // the source as it is now, which is what an agent iterating needs.
      command: `npx vite dev --host 127.0.0.1 --port ${UI_PORT} --strictPort`,
      cwd: uiDir,
      url: `http://127.0.0.1:${UI_PORT}`,
      reuseExistingServer: false,
      timeout: 120_000,
      stdout: 'ignore',
      env: {
        VITE_API_BASE: apiBase
      }
    }
  ]
});
