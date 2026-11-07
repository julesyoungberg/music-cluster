/**
 * Light/dark theme choice, shared by the layout (which applies it on load) and
 * the settings page (which changes it).
 *
 * Light ("Sleeve") and dark ("Booth") are equal themes, so with nothing stored
 * the app follows the operating system.
 */
export type Theme = 'light' | 'dark' | 'system';

export const THEME_KEY = 'music-cluster:theme';
export const DEFAULT_THEME: Theme = 'system';

export function parseTheme(value: string | null | undefined): Theme {
  return value === 'light' || value === 'dark' || value === 'system' ? value : DEFAULT_THEME;
}

export function isDark(theme: Theme, systemPrefersDark: boolean): boolean {
  return theme === 'dark' || (theme === 'system' && systemPrefersDark);
}

export function storedTheme(): Theme {
  try {
    return parseTheme(localStorage.getItem(THEME_KEY));
  } catch {
    return DEFAULT_THEME;
  }
}

/** Store the choice and switch the page to it. */
export function applyTheme(theme: Theme): void {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Storage can be unavailable (private windows); the theme still applies.
  }
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.classList.toggle('dark', isDark(theme, prefersDark));
}
