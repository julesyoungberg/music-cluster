import { describe, expect, it } from 'vitest';
import {
  basename,
  cn,
  confidenceClasses,
  confidenceTone,
  formatDate,
  formatDuration,
  formatPercent,
  pluralize,
  trackLabel,
  trackOf
} from '$lib/utils';
import type { SortItem } from '$lib/types';

describe('formatDuration', () => {
  it('renders minutes and seconds with a padded seconds field', () => {
    expect(formatDuration(225)).toBe('3:45');
    expect(formatDuration(65)).toBe('1:05');
    expect(formatDuration(9)).toBe('0:09');
  });

  it('adds an hours field only once the track is an hour long', () => {
    expect(formatDuration(3599)).toBe('59:59');
    expect(formatDuration(3600)).toBe('1:00:00');
    expect(formatDuration(5025)).toBe('1:23:45');
  });

  it('shows placeholder dashes rather than NaN for a missing duration', () => {
    expect(formatDuration(undefined)).toBe('--:--');
    expect(formatDuration(null)).toBe('--:--');
    expect(formatDuration(0)).toBe('--:--');
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe('--:--');
  });
});

describe('trackOf', () => {
  it('swaps the sort item id for the track id it refers to', () => {
    const item = {
      id: 91,
      track_id: 7,
      filename: 'a.mp3',
      artist: 'Someone',
      title: 'A track'
    } as unknown as SortItem;

    const track = trackOf(item);

    // The item's own id must not leak through: playing 91 would play the wrong
    // track, or none at all.
    expect(track.id).toBe(7);
    expect(track.artist).toBe('Someone');
  });
});

describe('trackLabel', () => {
  it('prefers an explicit display name', () => {
    expect(trackLabel({ display: 'Shown', artist: 'A', title: 'B' })).toBe('Shown');
  });

  it('falls back through artist/title, title, then filename', () => {
    expect(trackLabel({ artist: 'A', title: 'B' })).toBe('A - B');
    expect(trackLabel({ title: 'B' })).toBe('B');
    expect(trackLabel({ filename: 'c.mp3' })).toBe('c.mp3');
    expect(trackLabel({})).toBe('Unknown track');
  });
});

describe('formatPercent', () => {
  it('scales a fraction and appends a percent sign', () => {
    expect(formatPercent(0.923)).toBe('92%');
    expect(formatPercent(0.923, 1)).toBe('92.3%');
    expect(formatPercent(1)).toBe('100%');
  });

  it('renders an em dash for a missing or non-finite value', () => {
    expect(formatPercent(null)).toBe('—');
    expect(formatPercent(undefined)).toBe('—');
    expect(formatPercent(Number.NaN)).toBe('—');
  });
});

describe('formatDate', () => {
  it('treats a naive SQLite timestamp as UTC', () => {
    // Without the UTC marker the string is read as local time, which shifts
    // every timestamp in the UI by the viewer's offset.
    const naive = formatDate('2024-05-01 12:00:00');
    const explicit = formatDate('2024-05-01T12:00:00Z');
    expect(naive).toBe(explicit);
  });

  it('returns the input unchanged when it is not a date', () => {
    expect(formatDate('not a date')).toBe('not a date');
    expect(formatDate(null)).toBe('—');
    expect(formatDate('')).toBe('—');
  });
});

describe('confidenceTone', () => {
  it('bands scores at 0.75 and 0.5', () => {
    expect(confidenceTone(0.9)).toBe('high');
    expect(confidenceTone(0.75)).toBe('high');
    expect(confidenceTone(0.74)).toBe('medium');
    expect(confidenceTone(0.5)).toBe('medium');
    expect(confidenceTone(0.49)).toBe('low');
    expect(confidenceTone(null)).toBe('low');
  });

  it('has a class for every band it can return', () => {
    for (const score of [0.9, 0.6, 0.1, null]) {
      expect(confidenceClasses[confidenceTone(score)]).toBeTruthy();
    }
  });
});

describe('basename', () => {
  it('takes the last segment of posix and windows paths alike', () => {
    expect(basename('/music/house/track.mp3')).toBe('track.mp3');
    expect(basename('C:\\Music\\House\\track.mp3')).toBe('track.mp3');
    expect(basename('track.mp3')).toBe('track.mp3');
  });

  it('is empty for no path at all', () => {
    expect(basename(null)).toBe('');
    expect(basename('')).toBe('');
  });
});

describe('pluralize', () => {
  it('uses the singular only for exactly one', () => {
    expect(pluralize(1, 'group')).toBe('1 group');
    expect(pluralize(0, 'group')).toBe('0 groups');
    expect(pluralize(2, 'group')).toBe('2 groups');
  });

  it('accepts an irregular plural', () => {
    expect(pluralize(2, 'entry', 'entries')).toBe('2 entries');
  });
});

describe('cn', () => {
  it('lets a later tailwind class win over an earlier conflicting one', () => {
    expect(cn('px-2', 'px-4')).toBe('px-4');
  });

  it('drops falsy entries', () => {
    const disabled = false as boolean;
    expect(cn('a', disabled && 'b', undefined, null, 'c')).toBe('a c');
  });
});
