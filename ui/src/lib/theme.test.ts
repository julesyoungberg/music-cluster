import { describe, expect, it } from 'vitest';
import { DEFAULT_THEME, isDark, parseTheme } from './theme';

describe('theme', () => {
  it('starts dark when nothing valid is stored', () => {
    expect(DEFAULT_THEME).toBe('dark');
    expect(parseTheme(null)).toBe('dark');
    expect(parseTheme('neon')).toBe('dark');
  });

  it('keeps a stored choice', () => {
    expect(parseTheme('light')).toBe('light');
    expect(parseTheme('system')).toBe('system');
  });

  it('follows the system only when asked to', () => {
    expect(isDark('dark', false)).toBe(true);
    expect(isDark('light', true)).toBe(false);
    expect(isDark('system', true)).toBe(true);
    expect(isDark('system', false)).toBe(false);
  });
});
