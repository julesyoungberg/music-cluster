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

test.describe('auditioning a track', () => {
  test('the review screen draws a waveform for the current track', async ({ page }) => {
    await startSession(page, 'Waveform');

    // The waveform is fetched from the API and drawn to a canvas; if the
    // request or the draw fails, the canvas never appears.
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 30_000 });
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
