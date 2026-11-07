import { expect, test, type Page } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * The journey the app exists for: point at a folder of new buys, review what
 * the sorter suggests, decide, and commit.
 *
 * This runs against the real API and a real fitted model, so it also checks
 * that the suggestions are right — a track synthesised from the Techno family
 * has to come back suggested as Techno.
 */

const uiDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(uiDir, '..', '..');
const workspace = process.env.MUSIC_CLUSTER_E2E_WORKSPACE ?? path.join(repoRoot, '.e2e-workspace');
const newMusic = path.join(workspace, 'music', 'NewMusic');

/** Start a sort session over the seeded folder of new music. */
async function startSession(page: Page, name: string, { autoAccept = false } = {}) {
  await page.goto('/sort');
  await page.getByRole('button', { name: /New sort/ }).click();

  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();

  await dialog.getByPlaceholder('/path/to/folder').fill(newMusic);
  await dialog.getByPlaceholder('/path/to/folder').press('Enter');
  await dialog.getByPlaceholder(/Defaults to the folder name/).fill(name);
  if (autoAccept) {
    await dialog.getByRole('checkbox').check();
  }

  await dialog.getByRole('button', { name: 'Start', exact: false }).last().click();

  // Analysis is a background task; the UI navigates to the review screen when
  // it finishes.
  await expect(page).toHaveURL(/\/sort\/\d+$/, { timeout: 120_000 });
  await expect(page.getByRole('heading', { name, level: 1 })).toBeVisible();
}

test.describe('sorting new music', () => {
  test('a session scores every incoming track against the groups', async ({ page }) => {
    await startSession(page, 'Playwright buys');

    await expect(page.getByText('6 to review')).toBeVisible();
    await expect(page.getByText('0 filed')).toBeVisible();
  });

  test('the top suggestion matches the sound the track was built from', async ({ page }) => {
    await startSession(page, 'Suggestion check');

    // Each queue entry shows the track name over its top suggestion. A track
    // synthesised from the Techno family must come back suggested as Techno.
    const queue = page.getByRole('complementary').getByRole('button');
    await expect(queue).toHaveCount(6);

    for (let index = 0; index < 6; index++) {
      const text = (await queue.nth(index).innerText()).replace(/\s+/g, ' ');
      const [, family] = text.match(/New (Deep House|Techno|Drum & Bass)/) ?? [];
      expect(family, `queue entry ${index} had no recognisable name: ${text}`).toBeTruthy();
      expect(text.replace(`New ${family}`, '')).toContain(family);
    }
  });

  test('a track can be filed into its suggested group', async ({ page }) => {
    await startSession(page, 'Filing one');

    await page.getByRole('button', { name: /^1 / }).first().click();

    await expect(page.getByText('5 to review')).toBeVisible();
    await expect(page.getByText('1 filed')).toBeVisible();
  });

  test('a track can be skipped', async ({ page }) => {
    await startSession(page, 'Skipping one');

    await page.getByRole('button', { name: 'Skip s' }).click();

    await expect(page.getByText('5 to review')).toBeVisible();
    await expect(page.getByRole('button', { name: /^Skipped/ })).toBeVisible();
  });

  test('keyboard shortcuts file and skip without touching the mouse', async ({ page }) => {
    await startSession(page, 'Keyboard');

    await page.keyboard.press('1');
    await expect(page.getByText('1 filed')).toBeVisible();

    await page.keyboard.press('s');
    await expect(page.getByText('4 to review')).toBeVisible();
  });

  test('accept confident files the easy tracks in one go', async ({ page }) => {
    await startSession(page, 'Accept confident');

    await page.getByRole('button', { name: /Accept confident/ }).click();

    await expect(page.getByText('0 to review')).toBeVisible();
    await expect(page.getByText('6 filed')).toBeVisible();
    await expect(page.getByText('Nothing left to review')).toBeVisible();
  });

  test('pre-accepting at creation time skips the queue', async ({ page }) => {
    await startSession(page, 'Pre-accepted', { autoAccept: true });

    await expect(page.getByText('6 filed')).toBeVisible();
  });

  test('committing shows the plan before anything is written', async ({ page }) => {
    await startSession(page, 'Commit preview');
    await page.getByRole('button', { name: /Accept confident/ }).click();
    await expect(page.getByText('6 filed')).toBeVisible();

    await page.getByRole('button', { name: /^Commit/ }).click();

    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    // The promise in the README: you see exactly what would be written first.
    await expect(dialog.getByText(/playlist|copy|move/i).first()).toBeVisible();
  });

  test('a session appears in the list and can be deleted', async ({ page }) => {
    await startSession(page, 'Disposable');

    await page.goto('/sort');
    const row = page
      .locator('div.flex.items-center')
      .filter({ has: page.getByRole('link', { name: 'Disposable' }) });
    await expect(row).toHaveCount(1);

    await row.getByRole('button', { name: 'Delete session' }).click();

    await expect(page.getByRole('link', { name: 'Disposable' })).toHaveCount(0);
  });
});

/**
 * The player's audio element never joins the DOM, so the tests keep a handle
 * on every media element the page starts, to read what is really playing.
 */
async function watchAudio(page: Page) {
  await page.addInitScript(() => {
    const started: HTMLMediaElement[] = [];
    (window as unknown as { __audio: HTMLMediaElement[] }).__audio = started;
    const play = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = function (this: HTMLMediaElement) {
      if (!started.includes(this)) started.push(this);
      return play.call(this);
    };
  });
}

/** Every media element the page has started, and where each one is now. */
function audio(page: Page) {
  return page.evaluate(() =>
    (window as unknown as { __audio: HTMLMediaElement[] }).__audio.map((element) => ({
      playing: !element.paused,
      // Paused by the user, rather than stopped by reaching the end.
      paused: element.paused && !element.ended,
      time: element.currentTime,
      src: element.src
    }))
  );
}

/** Wait for the one audible element to be past `seconds`, and return it. */
async function playingPast(page: Page, seconds = 0) {
  let found: Awaited<ReturnType<typeof audio>>[number] | undefined;
  await expect
    .poll(async () => {
      const playing = (await audio(page)).filter((a) => a.playing && a.time > seconds);
      found = playing[0];
      return playing.length;
    })
    .toBe(1);
  return found!;
}

test.describe('auditioning a track', () => {
  test('the review screen draws a waveform for the current track', async ({ page }) => {
    await startSession(page, 'Waveform');
    const waveform = page.getByRole('slider', { name: /^Waveform for/ });

    await expect(waveform).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });

    // Moving through the queue and straight back cancels and re-asks for
    // waveforms mid-flight; each one still has to arrive.
    for (const key of ['j', 'k', 'j', 'j', 'k', 'k']) await page.keyboard.press(key);
    await expect(waveform).toHaveAttribute('data-state', 'ready');
    await expect(page.getByText('1 / 6')).toBeVisible();
  });

  test('the play button and the space bar play and pause', async ({ page }) => {
    await watchAudio(page);
    await startSession(page, 'Play and pause');
    const button = page.getByRole('button', { name: /^(Play|Pause) New / });

    await button.click();
    await playingPast(page);
    await expect(button).toHaveAccessibleName(/^Pause /);

    await button.click();
    await expect.poll(async () => (await audio(page)).every((a) => a.paused)).toBe(true);

    // Space reaches both the waveform and the page when the waveform has
    // focus; it must still toggle exactly once. The waveform only handles keys
    // once it has drawn.
    const waveform = page.getByRole('slider', { name: /^Waveform for/ });
    await expect(waveform).toHaveAttribute('data-state', 'ready');
    await waveform.focus();
    await page.keyboard.press(' ');
    await playingPast(page);
    await page.keyboard.press(' ');
    await expect.poll(async () => (await audio(page)).every((a) => a.paused)).toBe(true);
  });

  test('clicking the waveform plays from there, and the next track replaces it', async ({
    page
  }) => {
    await watchAudio(page);
    await startSession(page, 'Seek and switch');
    const waveform = page.getByRole('slider', { name: /^Waveform for/ });
    await expect(waveform).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });

    // The seeded tracks are four seconds long: a quarter in is one second.
    const box = (await waveform.boundingBox())!;
    await page.mouse.click(box.x + box.width / 4, box.y + box.height / 2);
    const first = await playingPast(page, 0.75);

    await page.keyboard.press('j');
    await page.getByRole('button', { name: /^Play New / }).click();
    const second = await playingPast(page);

    expect(second.src).not.toBe(first.src);
  });

  test('the API serves audio the player can load', async ({ page, request }) => {
    await page.goto('/library');
    await expect(page.getByText('Deep House Artist').first()).toBeVisible();

    const response = await request.get(
      `${process.env.E2E_API_BASE ?? 'http://127.0.0.1:8123'}/api/tracks/1/audio`
    );

    expect(response.status()).toBe(200);
    expect(response.headers()['content-type']).toContain('audio/');
  });
});
