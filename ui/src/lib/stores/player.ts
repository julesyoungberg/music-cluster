import { writable } from 'svelte/store';
import { api } from '$lib/services/api';
import type { Track } from '$lib/types';

/**
 * A single shared audio element.
 *
 * Auditioning is central to this app — you cannot decide which folder a track
 * belongs in without hearing it — but only one thing should ever be playing,
 * so playback lives in one place rather than per-component.
 */

interface PlayerState {
  track: Track | null;
  playing: boolean;
  currentTime: number;
  duration: number;
  loading: boolean;
}

const initial: PlayerState = {
  track: null,
  playing: false,
  currentTime: 0,
  duration: 0,
  loading: false
};

export const player = writable<PlayerState>(initial);

let audio: HTMLAudioElement | null = null;

function element(): HTMLAudioElement {
  if (audio) return audio;

  audio = new Audio();
  audio.preload = 'none';
  audio.addEventListener('timeupdate', () =>
    player.update((state) => ({ ...state, currentTime: audio!.currentTime }))
  );
  audio.addEventListener('durationchange', () =>
    player.update((state) => ({
      ...state,
      duration: Number.isFinite(audio!.duration) ? audio!.duration : 0
    }))
  );
  audio.addEventListener('playing', () =>
    player.update((state) => ({ ...state, playing: true, loading: false }))
  );
  audio.addEventListener('pause', () => player.update((state) => ({ ...state, playing: false })));
  audio.addEventListener('ended', () =>
    player.update((state) => ({ ...state, playing: false, currentTime: 0 }))
  );
  audio.addEventListener('error', () =>
    player.update((state) => ({ ...state, playing: false, loading: false }))
  );
  return audio;
}

/** Play a track, or toggle if it is already loaded. */
export async function playTrack(track: Track, startAt?: number): Promise<void> {
  const el = element();
  let current: PlayerState | undefined;
  player.subscribe((state) => (current = state))();

  if (current?.track?.id === track.id) {
    if (current.playing) {
      el.pause();
      return;
    }
    if (startAt !== undefined) el.currentTime = startAt;
    await el.play().catch(() => undefined);
    return;
  }

  player.update((state) => ({ ...state, track, loading: true, currentTime: 0, duration: 0 }));
  el.src = api.audioUrl(track.id);
  el.load();
  if (startAt !== undefined) el.currentTime = startAt;
  await el.play().catch(() => player.update((state) => ({ ...state, loading: false })));
}

export function togglePlay(): void {
  const el = element();
  if (el.paused) void el.play().catch(() => undefined);
  else el.pause();
}

export function stop(): void {
  const el = element();
  el.pause();
  el.currentTime = 0;
  player.set(initial);
}

export function seek(seconds: number): void {
  const el = element();
  if (Number.isFinite(el.duration)) {
    el.currentTime = Math.max(0, Math.min(seconds, el.duration));
  }
}

/** Jump to a fraction (0-1) of the track — used by the waveform scrubber. */
export function seekFraction(fraction: number): void {
  const el = element();
  if (Number.isFinite(el.duration)) seek(fraction * el.duration);
}

export function setVolume(volume: number): void {
  element().volume = Math.max(0, Math.min(1, volume));
}
