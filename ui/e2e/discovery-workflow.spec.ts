import { expect, test } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * The second workflow: one big unsorted pile becomes candidate groups the user
 * names and keeps. The rule this suite is here to protect is that nothing
 * becomes a group until a human says so.
 */

const uiDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(uiDir, '..', '..');
const workspace = process.env.MUSIC_CLUSTER_E2E_WORKSPACE ?? path.join(repoRoot, '.e2e-workspace');
const pile = path.join(workspace, 'music', 'DJ');
const apiBase = process.env.E2E_API_BASE ?? 'http://127.0.0.1:8123';

test.describe('breaking down a pile', () => {
  test('a run proposes candidates without creating any groups', async ({ page }) => {
    const groupsBefore = await countGroups(page);

    await page.goto('/discover');
    await page.getByRole('button', { name: /New run/ }).click();

    const dialog = page.getByRole('dialog');
    await dialog.getByPlaceholder('/path/to/folder').fill(pile);
    await dialog.getByPlaceholder('/path/to/folder').press('Enter');
    await dialog.getByPlaceholder(/Defaults to the folder name/).fill('Playwright pile');
    await dialog.getByLabel('Method').selectOption('kmeans');
    await dialog.getByPlaceholder('Auto').fill('3');
    await dialog.getByRole('button', { name: 'Find groups' }).click();

    await expect(page).toHaveURL(/\/discover\/\d+$/, { timeout: 120_000 });
    await expect(page.getByRole('heading', { name: 'Playwright pile', level: 1 })).toBeVisible();

    // Three proposals, and not one new group.
    await expect(page.getByRole('button', { name: /Keep as group/ })).toHaveCount(3);
    expect(await countGroups(page)).toBe(groupsBefore);
  });

  test('keeping a candidate creates exactly one group', async ({ page }) => {
    const groupsBefore = await countGroups(page);

    await page.goto('/discover');
    await page
      .getByRole('link', { name: /Playwright pile/ })
      .first()
      .click();
    await page
      .getByRole('button', { name: /Keep as group/ })
      .first()
      .click();

    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await dialog.getByRole('textbox').first().fill('Kept by Playwright');
    await dialog.getByRole('button', { name: 'Create group' }).click();

    // The candidate is marked as kept, and the toast names the new group.
    await expect(page.getByText('Created Kept by Playwright')).toBeVisible();
    await expect(page.getByText('kept', { exact: true })).toBeVisible();

    await page.goto('/groups');
    await expect(page.getByText('Kept by Playwright')).toBeVisible();
    expect(await countGroups(page)).toBe(groupsBefore + 1);
  });
});

/**
 * Put the collection back the way the seed left it.
 *
 * The API is one shared process, so a group promoted here would otherwise
 * change the model every later spec sorts against.
 */
test.afterAll(async ({ request }) => {
  const info = await (await request.get(`${apiBase}/api/info`)).json();
  const collection = info.collections[0];
  if (!collection) return;

  const { groups } = await (
    await request.get(`${apiBase}/api/collections/${collection.id}/groups`)
  ).json();
  for (const group of groups) {
    if (group.name === 'Kept by Playwright') {
      await request.delete(`${apiBase}/api/groups/${group.id}`);
    }
  }

  const { runs } = await (await request.get(`${apiBase}/api/discovery`)).json();
  for (const run of runs) {
    await request.delete(`${apiBase}/api/discovery/${run.id}`);
  }

  await request.post(`${apiBase}/api/collections/${collection.id}/fit`);
});

/** How many groups the active collection currently has, read from the API. */
async function countGroups(page: import('@playwright/test').Page): Promise<number> {
  const info = await page.request.get(`${apiBase}/api/info`);
  const { collections } = await info.json();
  return collections[0]?.group_count ?? 0;
}
