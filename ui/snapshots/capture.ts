import { test as base, type Page, type TestInfo } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { outputDir, screensLog } from './paths';

/** Something the browser reported that a person would not see on screen. */
export interface Problem {
  kind: 'console' | 'pageerror' | 'request-failed' | 'http-error';
  text: string;
}

/** One line of screens.jsonl: everything captured for one screen. */
export interface ScreenRecord {
  name: string;
  note?: string;
  flow: string;
  url: string;
  screenshot: string;
  aria?: string;
  problems: Problem[];
  /** Problems the screen is meant to show, such as the API-down state. */
  expectedProblems: boolean;
  /** Set when the flow broke before or at this screen. */
  failure?: string;
}

export interface ShotOptions {
  /** One line on what the screen should show, for whoever reads the report. */
  note?: string;
  /** The errors on this screen are the point of it; do not flag them. */
  expectProblems?: boolean;
  /** Capture only the visible viewport rather than the full page. */
  viewportOnly?: boolean;
}

export type Shot = (name: string, options?: ShotOptions) => Promise<void>;

/**
 * Watch a page for console errors, uncaught exceptions and failed or erroring
 * requests. Problems collect until the next screen is captured, so each screen
 * is charged with what happened on the way to it.
 */
function watch(page: Page) {
  let pending: Problem[] = [];
  const push = (problem: Problem) => pending.push(problem);

  page.on('console', (message) => {
    if (message.type() === 'error' || message.type() === 'warning') {
      push({ kind: 'console', text: `${message.type()}: ${message.text()}` });
    }
  });
  page.on('pageerror', (error) => push({ kind: 'pageerror', text: error.stack ?? error.message }));
  page.on('requestfailed', (request) => {
    const reason = request.failure()?.errorText ?? 'failed';
    // Navigating away cancels in-flight requests; that is not a fault.
    if (reason.includes('ERR_ABORTED')) return;
    push({ kind: 'request-failed', text: `${request.method()} ${request.url()} — ${reason}` });
  });
  page.on('response', (response) => {
    if (response.status() >= 400) {
      push({
        kind: 'http-error',
        text: `${response.request().method()} ${response.url()} — ${response.status()}`
      });
    }
  });

  return {
    take(): Problem[] {
      const taken = pending;
      pending = [];
      return taken;
    }
  };
}

function record(entry: ScreenRecord) {
  fs.mkdirSync(outputDir, { recursive: true });
  fs.appendFileSync(screensLog, JSON.stringify(entry) + '\n');
}

function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

async function settle(page: Page) {
  // Let data load and transitions finish. Pages that poll never go idle, so
  // this is best effort rather than a wait that can fail the run.
  await page.waitForLoadState('networkidle', { timeout: 5_000 }).catch(() => {});
  await page.waitForTimeout(300);
}

async function capture(
  page: Page,
  info: TestInfo,
  problems: Problem[],
  name: string,
  options: ShotOptions & { failure?: string } = {}
) {
  await settle(page);
  const screenshot = `${name}.png`;
  await page.screenshot({
    path: path.join(outputDir, screenshot),
    fullPage: !options.viewportOnly,
    animations: 'disabled'
  });

  let aria: string | undefined;
  try {
    const tree = await page.locator('body').ariaSnapshot({ timeout: 5_000 });
    aria = `${name}.aria.yml`;
    fs.writeFileSync(path.join(outputDir, aria), tree + '\n');
  } catch {
    // A page that is mid-crash may not answer; the screenshot still helps.
  }

  record({
    name,
    note: options.note,
    flow: info.titlePath.slice(1).join(' › '),
    url: page.url(),
    screenshot,
    aria,
    problems,
    expectedProblems: !!options.expectProblems,
    failure: options.failure
  });
}

/**
 * `test` with a `shot` fixture: `await shot('sort-review', { note })` saves a
 * screenshot, the accessibility tree and any browser errors for this screen.
 * If a flow breaks, the page as it was is captured too, named after the test.
 */
export const test = base.extend<{ shot: Shot }>({
  shot: async ({ page }, use, info) => {
    const watcher = watch(page);
    const shoot: Shot = (name, options) => capture(page, info, watcher.take(), name, options);

    await use(shoot);

    if (info.status !== info.expectedStatus) {
      const message = (info.error?.message ?? 'flow failed')
        // eslint-disable-next-line no-control-regex
        .replace(/\u001b\[[0-9;]*m/g, '')
        .split('\n')
        .slice(0, 6)
        .join('\n');
      await capture(page, info, watcher.take(), `FAILED-${slug(info.title)}`, {
        failure: message,
        viewportOnly: true
      }).catch(() => {});
    } else {
      const leftover = watcher.take();
      if (leftover.length > 0) {
        record({
          name: `after-${slug(info.title)}`,
          note: 'Problems logged after the last screen of this flow was captured.',
          flow: info.titlePath.slice(1).join(' › '),
          url: page.url(),
          screenshot: '',
          problems: leftover,
          expectedProblems: false
        });
      }
    }
  }
});

export { expect } from '@playwright/test';
