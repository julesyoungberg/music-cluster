import { expect, test } from '@playwright/test';
import { existsSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const workspace = process.env.MUSIC_CLUSTER_E2E_WORKSPACE ?? path.join(repoRoot, '.e2e-workspace');
const exportDir = path.join(workspace, 'dj-export');

test('exports every group to a Rekordbox XML file', async ({ page }) => {
  rmSync(exportDir, { recursive: true, force: true });
  await page.goto('/groups');
  await page.getByRole('button', { name: 'Export to DJ app' }).click();

  const dialog = page.getByRole('dialog');
  await dialog.getByRole('radio', { name: /Rekordbox/ }).check();
  await dialog.getByPlaceholder('/path/to/folder').fill(exportDir);
  await dialog.getByRole('button', { name: 'Export', exact: true }).click();

  await expect(page.getByText('Exported', { exact: true })).toBeVisible();
  const xmlFiles = () =>
    existsSync(exportDir) ? readdirSync(exportDir).filter((name) => name.endsWith('.xml')) : [];
  await expect.poll(() => xmlFiles().length).toBe(1);
  const xml = readFileSync(path.join(exportDir, xmlFiles()[0]), 'utf-8');
  for (const group of ['Deep House', 'Techno']) {
    expect(xml).toContain(`Name="${group}" Type="1"`);
  }
});
