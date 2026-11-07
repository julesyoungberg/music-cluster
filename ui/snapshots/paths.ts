import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

export const uiDir = path.resolve(here, '..');
export const repoRoot = path.resolve(uiDir, '..');
export const workspace =
  process.env.UI_SNAPSHOT_WORKSPACE ?? path.join(repoRoot, '.ui-snapshot-workspace');
export const outputDir = process.env.UI_SNAPSHOT_OUT ?? path.join(uiDir, 'ui-snapshots');
/** One JSON object per captured screen, appended as the run goes. */
export const screensLog = path.join(outputDir, 'screens.jsonl');

// Separate ports from the e2e suite, so a server one of them left running is
// never mistaken for the other's (they point at different databases).
export const API_PORT = 8124;
export const UI_PORT = 4174;
export const apiBase = `http://127.0.0.1:${API_PORT}`;
