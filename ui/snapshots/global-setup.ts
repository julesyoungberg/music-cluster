import fs from 'node:fs';
import { outputDir } from './paths';

/** Start every run with an empty output folder, so stale screens never mislead. */
export default function globalSetup() {
  fs.rmSync(outputDir, { recursive: true, force: true });
  fs.mkdirSync(outputDir, { recursive: true });
}
