/**
 * Crate colours: each group gets a colour, the way DJs colour-code crates and
 * playlists, so a group reads the same at a glance on every screen (its card,
 * the picker, the sort scores and the queue).
 *
 * A group's own `color` (one of these names, set on its page) wins; otherwise
 * it gets one from its id, so colours are stable and need no setup. The values
 * are mid tones that stay legible as swatches and bars on both themes; they are
 * never used for text.
 */
export const CRATE_COLORS = {
  red: '#e5484d',
  orange: '#f76b15',
  amber: '#e2a336',
  lime: '#8db634',
  green: '#30a46c',
  teal: '#12a594',
  sky: '#3e9fd8',
  blue: '#3e63dd',
  violet: '#8e4ec6',
  pink: '#d6409f'
} as const;

export type CrateColorName = keyof typeof CRATE_COLORS;

export const CRATE_COLOR_NAMES = Object.keys(CRATE_COLORS) as CrateColorName[];

export function isCrateColor(value: string | null | undefined): value is CrateColorName {
  return !!value && Object.prototype.hasOwnProperty.call(CRATE_COLORS, value);
}

/** The colour name a group shows: its own choice, or one picked from its id. */
export function crateColorName(group: { id: number; color?: string | null }): CrateColorName {
  if (isCrateColor(group.color)) return group.color;
  const index = Math.abs(Math.trunc(group.id)) % CRATE_COLOR_NAMES.length;
  return CRATE_COLOR_NAMES[index];
}

/** The CSS colour for a group. */
export function crateColor(group: { id: number; color?: string | null }): string {
  return CRATE_COLORS[crateColorName(group)];
}
