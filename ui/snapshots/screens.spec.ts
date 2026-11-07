import path from 'node:path';
import { expect, test } from './capture';
import { apiBase, workspace } from './paths';

/**
 * Every screen of the app, and the main states inside each flow, captured
 * against a freshly seeded library (three genre folders, fitted, plus a
 * folder of six new tracks and the genre folders as an unsorted pile).
 *
 * Steps wait for what the screen needs before capturing it, but nothing is
 * checked beyond that: the output is for looking at, not a pass/fail verdict.
 * When you add a screen or a state to the app, add a `shot` for it here.
 */

const music = path.join(workspace, 'seed', 'music');

test.describe('browsing', () => {
  test('overview', async ({ page, shot }) => {
    await page.goto('/');
    await expect(page.getByText(/tracks in the library/)).toBeVisible();
    await shot('overview', { note: 'Library totals, the fitted collection, the three ways in' });
  });

  test('groups', async ({ page, shot }) => {
    await page.goto('/groups');
    await expect(page.getByText('Techno', { exact: true }).first()).toBeVisible();
    await shot('groups', { note: 'The collection’s groups' });

    await page.getByRole('button', { name: 'Export to DJ app' }).click();
    await expect(page.getByRole('dialog')).toBeVisible();
    await shot('groups-export-dialog', { note: 'Choosing a DJ app to export the groups to' });
    await page.keyboard.press('Escape');

    await page
      .getByRole('link', { name: /Techno/ })
      .first()
      .click();
    await expect(page.getByText('Techno Artist').first()).toBeVisible();
    await shot('group-detail', { note: 'One group and its reference tracks' });

    await page.goto('/groups/check');
    await expect(page.getByText(/Deep House/).first()).toBeVisible();
    await shot('groups-overlap', { note: 'Overlap report between groups' });
  });

  test('library', async ({ page, shot }) => {
    await page.goto('/library');
    await expect(page.getByText('Deep House Artist').first()).toBeVisible();
    await shot('library', { note: 'Every analysed track with its tags' });

    await page.getByPlaceholder(/Search/i).fill('Drum & Bass Artist');
    await expect(page.getByText('Deep House Artist')).toHaveCount(0);
    await shot('library-search', { note: 'Search narrowed to one genre' });
  });

  test('settings', async ({ page, shot }) => {
    await page.goto('/settings');
    await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible();
    await shot('settings', { note: 'Current configuration' });
  });
});

test.describe('sorting new music', () => {
  test('review, commit and undo', async ({ page, shot }) => {
    await page.goto('/sort');
    await expect(page.getByRole('heading', { name: 'Sort', level: 1 })).toBeVisible();
    await shot('sort-empty', { note: 'No sessions yet' });

    await page.getByRole('button', { name: /New sort/ }).click();
    const dialog = page.getByRole('dialog');
    await dialog.getByPlaceholder('/path/to/folder').fill(path.join(music, 'NewMusic'));
    await dialog.getByPlaceholder('/path/to/folder').press('Enter');
    await dialog.getByPlaceholder(/Defaults to the folder name/).fill('Snapshot buys');
    await shot('sort-new-dialog', { note: 'Starting a session over a folder of new music' });

    await dialog.getByRole('button', { name: 'Start', exact: false }).last().click();
    await expect(page).toHaveURL(/\/sort\/\d+$/, { timeout: 120_000 });
    await expect(page.getByText('6 to review')).toBeVisible();
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 30_000 });
    await shot('sort-review', {
      note: 'Review queue: six tracks, each with its suggested group and a waveform'
    });

    await page.getByRole('button', { name: /Shortcuts/ }).click();
    await shot('sort-shortcuts', { note: 'Keyboard shortcuts help', viewportOnly: true });
    await page.keyboard.press('Escape');

    await page.keyboard.press('1');
    await expect(page.getByText('1 filed')).toBeVisible();
    await page.keyboard.press('s');
    await expect(page.getByText('4 to review')).toBeVisible();
    await shot('sort-one-filed-one-skipped', { note: 'After filing one track and skipping one' });

    await page.getByRole('button', { name: /Accept confident/ }).click();
    await expect(page.getByText('0 to review')).toBeVisible();
    await shot('sort-all-decided', { note: 'Queue empty after accepting confident suggestions' });

    await page.getByRole('button', { name: /^Commit/ }).click();
    const commit = page.getByRole('dialog', { name: 'Commit this session' });
    await expect(commit.getByRole('button', { name: 'Commit', exact: true })).toBeEnabled();
    await shot('sort-commit-preview', {
      note: 'Commit dialog as it opens: a dry run in the default mode',
      viewportOnly: true
    });

    await commit.getByRole('button', { name: 'Playlists only' }).click();
    await expect(commit.getByText(/\.m3u8?/).first()).toBeVisible();
    await shot('sort-commit-preview-playlists', {
      note: 'Dry run in playlist mode: the playlists that would be written',
      viewportOnly: true
    });

    await commit.getByRole('button', { name: 'Move files' }).click();
    await expect(commit.getByText(/Moving relocates the original files/)).toBeVisible();
    await shot('sort-commit-preview-move', {
      note: 'Dry run in move mode, with the warning that Undo can put files back',
      viewportOnly: true
    });

    // Playlists only: the snapshot workspace points playlist_dir at itself,
    // so committing writes nothing outside it.
    await commit.getByRole('button', { name: 'Playlists only' }).click();
    await expect(commit.getByText(/\.m3u8?/).first()).toBeVisible();
    await commit.getByRole('button', { name: 'Commit', exact: true }).click();
    await expect(page.getByText('Committed').first()).toBeVisible();
    await shot('sort-committed', { note: 'After committing: toast and the session state' });

    await page.getByRole('button', { name: /Reverse this session/ }).click();
    await expect(page.getByText(/Reversed/).first()).toBeVisible();
    await shot('sort-undone', { note: 'After Undo reversed the commit' });

    await page.goto('/sort');
    await expect(page.getByRole('link', { name: 'Snapshot buys' })).toBeVisible();
    await shot('sort-list', { note: 'Session list with one session' });
  });
});

test.describe('breaking down a pile', () => {
  test('discover candidates and keep one', async ({ page, shot }) => {
    await page.goto('/discover');
    await expect(page.getByRole('heading', { name: 'Discover', level: 1 })).toBeVisible();
    await shot('discover-empty', { note: 'No discovery runs yet' });

    await page.getByRole('button', { name: /New run/ }).click();
    const dialog = page.getByRole('dialog');
    await dialog.getByPlaceholder('/path/to/folder').fill(path.join(music, 'DJ'));
    await dialog.getByPlaceholder('/path/to/folder').press('Enter');
    await dialog.getByPlaceholder(/Defaults to the folder name/).fill('Snapshot pile');
    await dialog.getByLabel('Method').selectOption('kmeans');
    await dialog.getByPlaceholder('Auto').fill('3');
    await shot('discover-new-dialog', { note: 'Starting a run over an unsorted pile' });

    await dialog.getByRole('button', { name: 'Find groups' }).click();
    await expect(page).toHaveURL(/\/discover\/\d+$/, { timeout: 120_000 });
    await expect(page.getByRole('button', { name: /Keep as group/ })).toHaveCount(3);
    await shot('discover-run', { note: 'Three candidate groups, none kept yet' });

    await page
      .getByRole('button', { name: /Keep as group/ })
      .first()
      .click();
    await expect(page.getByRole('dialog')).toBeVisible();
    await shot('discover-keep-dialog', { note: 'Naming a candidate before it becomes a group' });

    await page.getByRole('dialog').getByRole('textbox').first().fill('Kept in snapshot');
    await page.getByRole('dialog').getByRole('button', { name: 'Create group' }).click();
    await expect(page.getByText('Created Kept in snapshot')).toBeVisible();
    await shot('discover-kept', { note: 'One candidate kept as a group' });

    await page.goto('/discover');
    await expect(page.getByRole('link', { name: /Snapshot pile/ })).toBeVisible();
    await shot('discover-list', { note: 'Run list with one run' });
  });
});

test.describe('failure states', () => {
  test('API unreachable', async ({ page, shot }) => {
    await page.route('**/api/**', (route) => route.abort('failed'));
    await page.goto('/');
    await expect(page.getByText(/Cannot reach the/i).first()).toBeVisible();
    await shot('api-unreachable', {
      note: 'Backend down: should say so and offer a retry, not show an empty library',
      expectProblems: true
    });
  });
});

test.describe('narrow window', () => {
  test.use({ viewport: { width: 760, height: 900 } });

  test('overview and review at a small window size', async ({ page, shot }) => {
    await page.goto('/');
    await expect(page.getByText(/tracks in the library/)).toBeVisible();
    await shot('narrow-overview', { note: 'Overview in a 760px-wide window' });

    // Reuses the sort flow's session. If that flow broke, skip rather than
    // report a second, misleading broken flow.
    const { sessions } = await (await page.request.get(`${apiBase}/api/sessions`)).json();
    const session = sessions.find((s: { name?: string }) => s.name === 'Snapshot buys');
    test.skip(!session, 'The sort flow did not create its session; see that flow instead.');
    await page.goto(`/sort/${session.id}`);
    await expect(page.getByRole('heading', { name: 'Snapshot buys', level: 1 })).toBeVisible();
    await shot('narrow-sort-review', { note: 'A sort session in a 760px-wide window' });
  });
});
