import { describe, expect, it } from 'vitest';
import { DEFAULT_THEME, isDark, parseTheme } from './theme';

describe('theme', () => {
  it('follows the system when nothing valid is stored', () => {
    expect(DEFAULT_THEME).toBe('system');
    expect(parseTheme(null)).toBe('system');
    expect(parseTheme('neon')).toBe('system');
  });

  it('keeps a stored choice', () => {
    expect(parseTheme('light')).toBe('light');
    expect(parseTheme('dark')).toBe('dark');
  });

  it('follows the system only when asked to', () => {
    expect(isDark('dark', false)).toBe(true);
    expect(isDark('light', true)).toBe(false);
    expect(isDark('system', true)).toBe(true);
    expect(isDark('system', false)).toBe(false);
  });
});
