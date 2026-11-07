import { describe, expect, it } from 'vitest';
import { CRATE_COLORS, crateColor, crateColorName } from './crateColors';

describe('crate colours', () => {
  it('uses the colour a group was given', () => {
    expect(crateColorName({ id: 1, color: 'teal' })).toBe('teal');
    expect(crateColor({ id: 1, color: 'teal' })).toBe(CRATE_COLORS.teal);
  });

  it('falls back to a stable colour from the id', () => {
    expect(crateColorName({ id: 3 })).toBe(crateColorName({ id: 3, color: null }));
    expect(crateColorName({ id: 1 })).not.toBe(crateColorName({ id: 2 }));
  });

  it('ignores a colour it does not know', () => {
    expect(crateColorName({ id: 4, color: '#123456' })).toBe(crateColorName({ id: 4 }));
  });
});
