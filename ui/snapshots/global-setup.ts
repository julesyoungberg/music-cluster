import fs from 'node:fs';
import { marker, outputDir, screensLog } from './paths';

/**
 * Start every run with an empty output folder, so stale screens never mislead.
 * Only a folder an earlier run created (it holds the marker, or the screens log
 * of a run that predates it) is cleared; any other non-empty folder is refused
 * rather than deleted.
 */
export default function globalSetup() {
  if (fs.existsSync(outputDir) && fs.readdirSync(outputDir).length > 0) {
    if (!fs.existsSync(marker) && !fs.existsSync(screensLog)) {
      throw new Error(
        `${outputDir} is not empty and was not made by ui:snapshot; ` +
          'point UI_SNAPSHOT_OUT at a new or empty folder.'
      );
    }
    fs.rmSync(outputDir, { recursive: true, force: true });
  }
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(marker, 'Output of npm run ui:snapshot; cleared on every run.\n');
}
