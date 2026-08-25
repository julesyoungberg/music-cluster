import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { SortItem, Track } from '$lib/types';

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds?: number | null): string {
  if (!seconds || !Number.isFinite(seconds)) return '--:--';
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
    : `${minutes}:${String(secs).padStart(2, '0')}`;
}

/**
 * The Track behind a sort item.
 *
 * A sort item carries the track's metadata but its own `id`, so it cannot be
 * handed to anything that plays or draws audio without swapping the ID for the
 * track's. Doing that in one place keeps the two from being mixed up again.
 */
export function trackOf(item: SortItem): Track {
  return { ...item, id: item.track_id };
}

/** A display name. Takes metadata only, so sort items can be labelled too. */
export function trackLabel(track: Partial<Omit<Track, 'id'>>): string {
  if (track.display) return track.display;
  if (track.artist && track.title) return `${track.artist} - ${track.title}`;
  return track.title ?? track.filename ?? 'Unknown track';
}

export function formatPercent(value?: number | null, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDate(value?: string | null): string {
  if (!value) return '—';
  // SQLite hands back naive UTC strings; mark them so they render in local time.
  const iso = value.includes('T') ? value : `${value.replace(' ', 'T')}Z`;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

/** Colour band for a confidence score, shared by every confidence display. */
export function confidenceTone(confidence?: number | null): 'high' | 'medium' | 'low' {
  if (confidence === null || confidence === undefined) return 'low';
  if (confidence >= 0.75) return 'high';
  if (confidence >= 0.5) return 'medium';
  return 'low';
}

export const confidenceClasses: Record<'high' | 'medium' | 'low', string> = {
  high: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30',
  medium: 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30',
  low: 'bg-rose-500/15 text-rose-600 dark:text-rose-400 border-rose-500/30'
};

export function basename(path?: string | null): string {
  if (!path) return '';
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

/**
 * A sample's length, in the unit that makes it readable.
 *
 * `formatDuration` renders a 90 ms hat as "0:00", which is exactly the
 * information a one-shot library needs and does not get from minutes.
 */
export function formatSampleLength(seconds?: number | null): string {
  if (!seconds || !Number.isFinite(seconds)) return '—';
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 10) return `${seconds.toFixed(2)} s`;
  return formatDuration(seconds);
}
