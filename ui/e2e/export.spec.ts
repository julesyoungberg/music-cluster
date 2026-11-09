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

test('writes Serato crates after a preview, then removes them', async ({ page }) => {
  const seratoDir = path.join(workspace, '_Serato_');
  const subcrates = path.join(seratoDir, 'Subcrates');
  rmSync(seratoDir, { recursive: true, force: true });
  const crates = () => (existsSync(subcrates) ? readdirSync(subcrates) : []);

  await page.goto('/groups');
  await page.getByRole('button', { name: 'Export to DJ app' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('radio', { name: /^Serato/ }).check();
  await dialog.getByPlaceholder('/path/to/folder').fill(seratoDir);

  await dialog.getByRole('button', { name: 'Preview crates' }).click();
  await expect(dialog.getByText('Will write 4 crates')).toBeVisible();
  expect(crates()).toEqual([]);

  await dialog.getByRole('button', { name: 'Write 4 crates' }).click();
  await expect(page.getByText('Crates written', { exact: true })).toBeVisible();
  expect(crates()).toHaveLength(4);

  await page.getByRole('button', { name: 'Export to DJ app' }).click();
  await dialog.getByRole('radio', { name: /^Serato/ }).check();
  await dialog.getByPlaceholder('/path/to/folder').fill(seratoDir);
  await dialog.getByRole('button', { name: 'Remove Cratewise crates' }).click();
  await dialog.getByRole('button', { name: 'Confirm removal' }).click();
  await expect(page.getByText('Crates removed', { exact: true })).toBeVisible();
  expect(crates()).toEqual([]);
});
