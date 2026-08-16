import { expect, test } from '@playwright/test';

/**
 * Read-only journeys through the app: the pages a user lands on before they
 * change anything. These run against the seeded library — three genre folders
 * of eight tracks, imported and fitted.
 */

test.describe('overview', () => {
  test('shows the library the API actually holds', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible();
    // The seeded library is 24 tracks; other specs may have analysed more by
    // the time this runs, so the invariant is that everything is analysed.
    const summary = await page.getByText(/tracks in the library/).innerText();
    const [, total, analysed] = summary.match(/(\d+) tracks in the library, (\d+) analysed/) ?? [];
    expect(Number(total)).toBeGreaterThanOrEqual(24);
    expect(analysed).toBe(total);
  });

  test('reports the fitted collection and its self-check accuracy', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('Groups', { exact: true }).first()).toBeVisible();
    await expect(page.getByText('Reference tracks')).toBeVisible();
    await expect(page.getByText('Self-check')).toBeVisible();
    // The synthetic folders are cleanly separable, so this is a high number,
    // and the collection is reported as fitted rather than as pending.
    await expect(page.getByText(/^(8\d|9\d|100)%$/)).toBeVisible();
    await expect(page.getByText('not fitted yet')).toHaveCount(0);
  });

  test('offers the three ways in', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByRole('link', { name: /Sort new music/ })).toBeVisible();
    await expect(page.getByRole('link', { name: /Break down a pile/ })).toBeVisible();
    await expect(page.getByRole('link', { name: /Manage groups/ })).toBeVisible();
  });
});

test.describe('navigation', () => {
  for (const [label, heading] of [
    ['Groups', 'Groups'],
    ['Sort', 'Sort'],
    ['Discover', 'Discover'],
    ['Library', 'Library'],
    ['Settings', 'Settings']
  ] as const) {
    test(`${label} is reachable from the header`, async ({ page }) => {
      await page.goto('/');
      await page.getByRole('navigation').getByRole('link', { name: label, exact: true }).click();

      await expect(page.getByRole('heading', { name: heading, level: 1 })).toBeVisible();
    });
  }

  test('marks the current page in the nav', async ({ page }) => {
    await page.goto('/library');

    await expect(
      page.getByRole('navigation').getByRole('link', { name: 'Library', exact: true })
    ).toHaveAttribute('aria-current', 'page');
  });
});

test.describe('groups', () => {
  test('lists every imported genre folder', async ({ page }) => {
    await page.goto('/groups');

    for (const name of ['Deep House', 'Techno', 'Drum & Bass']) {
      await expect(page.getByText(name, { exact: true }).first()).toBeVisible();
    }
  });

  test('shows the overlap report once a collection is fitted', async ({ page }) => {
    await page.goto('/groups');

    await page.getByRole('link', { name: 'Overlap report' }).click();

    await expect(page).toHaveURL(/\/groups\/check$/);
    await expect(page.getByText(/Deep House/).first()).toBeVisible();
  });

  test('a group detail lists its reference tracks', async ({ page }) => {
    await page.goto('/groups');
    await page
      .getByRole('link', { name: /Techno/ })
      .first()
      .click();

    await expect(page).toHaveURL(/\/groups\/\d+$/);
    await expect(page.getByText('Techno Artist').first()).toBeVisible();
  });
});

test.describe('library', () => {
  test('lists analysed tracks with their tags', async ({ page }) => {
    await page.goto('/library');

    await expect(page.getByText('Deep House Artist').first()).toBeVisible();
  });

  test('search narrows the list to matching tracks', async ({ page }) => {
    await page.goto('/library');
    await expect(page.getByText('Deep House Artist').first()).toBeVisible();

    await page.getByPlaceholder(/Search/i).fill('Drum & Bass Artist');

    await expect(page.getByText('Drum & Bass Artist').first()).toBeVisible();
    await expect(page.getByText('Deep House Artist')).toHaveCount(0);
  });
});

test.describe('settings', () => {
  test('renders the current configuration', async ({ page }) => {
    await page.goto('/settings');

    await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible();
    await expect(page.getByText(/confiden/i).first()).toBeVisible();
  });
});

test.describe('when the API is unreachable', () => {
  test('says so instead of showing an empty library', async ({ page }) => {
    // A blank page here is the difference between "you have no music" and
    // "the backend is not running", which are very different problems.
    await page.route('**/api/info', (route) => route.abort('failed'));
    await page.goto('/');

    await expect(page.getByText(/Cannot reach the/i).first()).toBeVisible();
    await expect(page.getByRole('button', { name: /Try again|Retry/i })).toBeVisible();
  });
});
