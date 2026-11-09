import { expect, test } from '@playwright/test';
import { existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

/**
 * Gathering a group's tracks into a new folder, then undoing it. Copy mode, so
 * the seeded library other specs use is never moved.
 */

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const workspace = process.env.MUSIC_CLUSTER_E2E_WORKSPACE ?? path.join(repoRoot, '.e2e-workspace');
const crate = path.join(workspace, 'filed-crate');

test('copies a group into its folder and undoes it', async ({ page }) => {
  rmSync(crate, { recursive: true, force: true });
  mkdirSync(crate, { recursive: true });

  await page.goto('/groups');
  await page
    .getByRole('link', { name: /Techno/ })
    .first()
    .click();
  await expect(page.getByText('Techno Artist').first()).toBeVisible();

  // Point the group somewhere new so its tracks are outside its folder.
  const original = (await page.locator('p.font-mono').first().innerText()).trim();
  const setDestination = async (folder: string) => {
    await page.getByRole('button', { name: 'Choose folder' }).click();
    await page.getByRole('dialog').getByPlaceholder('/path/to/folder').fill(folder);
    await page.getByRole('dialog').getByRole('button', { name: 'Done' }).click();
    await page.getByRole('button', { name: /Save changes/ }).click();
    await expect(page.getByText('Group updated').last()).toBeVisible();
  };
  await setDestination(crate);

  await page.getByRole('button', { name: /File tracks into this folder/ }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('button', { name: /^Copy \d+ files$/ }).click();

  await expect(page.getByText('Filed', { exact: true })).toBeVisible();
  expect(readdirSync(crate).length).toBeGreaterThan(0);

  await page.getByRole('button', { name: /Undo filing/ }).click();
  await expect(page.getByText(/^Reversed \d+ operations$/)).toBeVisible();
  expect(existsSync(crate) ? readdirSync(crate) : []).toEqual([]);

  // Leave the shared library as the other specs expect it.
  await setDestination(original);
});
