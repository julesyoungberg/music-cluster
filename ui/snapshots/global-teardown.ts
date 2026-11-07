import fs from 'node:fs';
import path from 'node:path';
import type { ScreenRecord } from './capture';
import { marker, outputDir, screensLog } from './paths';

/**
 * Turn screens.jsonl into report.md: the file to open first. It lists every
 * screen with its files, and puts anything that looks wrong at the top.
 */
export default function globalTeardown() {
  // Setup refused this folder; leave it exactly as it was.
  if (!fs.existsSync(marker)) return;
  if (!fs.existsSync(screensLog)) {
    fs.writeFileSync(
      path.join(outputDir, 'report.md'),
      '# UI snapshot\n\nNo screens were captured: the run stopped before any flow reached ' +
        'its first screen. Check the output of the run (or the CI job log) for why, usually ' +
        'seeding the library or starting a server.\n'
    );
    return;
  }

  const screens: ScreenRecord[] = fs
    .readFileSync(screensLog, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));

  const failures = screens.filter((s) => s.failure);
  const noisy = screens.filter((s) => !s.failure && !s.expectedProblems && s.problems.length > 0);

  const lines: string[] = [
    '# UI snapshot',
    '',
    `${screens.length} screens captured on ${new Date().toISOString()}. ` +
      `${failures.length} broken flows, ${noisy.length} screens with unexpected browser errors.`,
    '',
    'Each screen has a `.png` screenshot and an `.aria.yml` accessibility tree (the text and ' +
      'controls on the page). Nothing here is compared against a baseline: look at the ' +
      'screens you changed, and treat the sections below as leads.',
    ''
  ];

  if (failures.length > 0) {
    lines.push('## Broken flows', '');
    for (const screen of failures) {
      lines.push(`### ${screen.flow}`, '', `Page at failure: ${files(screen)}`, '');
      lines.push('```', screen.failure ?? '', '```', '');
      pushProblems(lines, screen);
    }
  }

  if (noisy.length > 0) {
    lines.push('## Unexpected browser errors', '');
    for (const screen of noisy) {
      lines.push(`### ${screen.name}`, '', `${screen.url} — ${files(screen)}`, '');
      pushProblems(lines, screen);
    }
  }

  lines.push('## Screens', '', '| Screen | What it shows | Errors | Files |', '|---|---|---|---|');
  for (const screen of screens) {
    if (!screen.screenshot) continue;
    const errors = screen.failure
      ? 'flow broke'
      : screen.problems.length === 0
        ? ''
        : `${screen.problems.length}${screen.expectedProblems ? ' (expected)' : ''}`;
    lines.push(
      `| ${screen.name} | ${(screen.note ?? screen.flow).replace(/\|/g, '\\|')} | ${errors} | ${files(screen)} |`
    );
  }
  lines.push('');

  fs.writeFileSync(path.join(outputDir, 'report.md'), lines.join('\n'));
  console.log(`\nUI snapshot: ${path.join(outputDir, 'report.md')}`);
}

function files(screen: ScreenRecord): string {
  const links = [];
  if (screen.screenshot) links.push(`[png](${screen.screenshot})`);
  if (screen.aria) links.push(`[aria](${screen.aria})`);
  return links.join(' ') || '—';
}

function pushProblems(lines: string[], screen: ScreenRecord) {
  for (const problem of screen.problems.slice(0, 20)) {
    lines.push(`- **${problem.kind}** ${problem.text.split('\n')[0]}`);
  }
  if (screen.problems.length > 20) lines.push(`- …and ${screen.problems.length - 20} more`);
  if (screen.problems.length > 0) lines.push('');
}
