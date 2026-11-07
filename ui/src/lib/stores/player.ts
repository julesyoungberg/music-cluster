import { get, writable } from 'svelte/store';
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
  error: string;
}

const initial: PlayerState = {
  track: null,
  playing: false,
  currentTime: 0,
  duration: 0,
  loading: false,
  error: ''
};

export const player = writable<PlayerState>(initial);

let audio: HTMLAudioElement | null = null;

/**
 * Where the live audio element is kept across copies of this module.
 *
 * Normally there is one copy. But when the dev server hot-reloads this file (a
 * `git pull` while the app is open is enough) a fresh copy starts with no
 * element, makes its own, and the old one carries on playing where nothing can
 * reach it: two tracks at once, and a Stop button that stops only one. Each
 * new element retires whatever the previous copy left here.
 */
const LIVE_ELEMENT = '__musicClusterAudio';
type WithLiveElement = typeof globalThis & { [LIVE_ELEMENT]?: HTMLAudioElement };

function retire(element: HTMLAudioElement): void {
  element.pause();
  element.removeAttribute('src');
  element.load();
}

/**
 * Bumped by every new track. A `play()` that settles after the user has moved
 * on belongs to the old track and must not touch the new one's state.
 */
let generation = 0;

/**
 * Where playback should land once the track is ready.
 *
 * Seeking is only allowed after metadata has loaded, so a click on the waveform
 * of a track that is not playing yet has to be remembered rather than applied.
 * It can be a time in seconds or a fraction of a duration nobody knows yet.
 */
let pending: { seconds?: number; fraction?: number } | null = null;

let frame = 0;

/**
 * `timeupdate` only fires about four times a second, which makes a playhead
 * drawn from it visibly step along. Reading the element every frame while it
 * plays costs nothing and makes the waveform track the audio smoothly.
 */
function followPlayback(): void {
  cancelAnimationFrame(frame);
  const tick = () => {
    if (!audio || audio.paused) return;
    player.update((state) => ({ ...state, currentTime: audio!.currentTime }));
    frame = requestAnimationFrame(tick);
  };
  frame = requestAnimationFrame(tick);
}

function stopFollowing(): void {
  cancelAnimationFrame(frame);
  frame = 0;
}

function applyPending(): void {
  if (!audio || !pending) return;
  const duration = audio.duration;
  if (!Number.isFinite(duration) || duration <= 0) return;

  const target =
    pending.seconds !== undefined ? pending.seconds : (pending.fraction ?? 0) * duration;
  pending = null;
  audio.currentTime = Math.max(0, Math.min(target, Math.max(duration - 0.05, 0)));
  player.update((state) => ({ ...state, currentTime: audio!.currentTime }));
}

function element(): HTMLAudioElement {
  if (audio) return audio;

  const live = globalThis as WithLiveElement;
  const previous = live[LIVE_ELEMENT];
  if (previous) retire(previous);

  audio = new Audio();
  audio.preload = 'metadata';
  live[LIVE_ELEMENT] = audio;

  audio.addEventListener('timeupdate', () =>
    player.update((state) => ({ ...state, currentTime: audio!.currentTime }))
  );
  audio.addEventListener('loadedmetadata', () => {
    player.update((state) => ({
      ...state,
      duration: Number.isFinite(audio!.duration) ? audio!.duration : 0
    }));
    applyPending();
  });
  audio.addEventListener('durationchange', () =>
    player.update((state) => ({
      ...state,
      duration: Number.isFinite(audio!.duration) ? audio!.duration : 0
    }))
  );
  audio.addEventListener('playing', () => {
    player.update((state) => ({ ...state, playing: true, loading: false, error: '' }));
    followPlayback();
  });
  audio.addEventListener('pause', () => {
    stopFollowing();
    player.update((state) => ({ ...state, playing: false }));
  });
  audio.addEventListener('ended', () => {
    stopFollowing();
    player.update((state) => ({ ...state, playing: false, currentTime: 0 }));
  });
  audio.addEventListener('error', () => {
    stopFollowing();
    pending = null;
    player.update((state) => ({
      ...state,
      playing: false,
      loading: false,
      error: 'This track could not be played'
    }));
  });

  return audio;
}

/** Start the element, reporting a refusal only if it is still this track's turn. */
async function start(el: HTMLAudioElement, load: number): Promise<void> {
  try {
    await el.play();
  } catch {
    // Moving to another track rejects the old track's play(); that is not a
    // failure of the new one, which is still loading.
    if (load !== generation) return;
    player.update((state) => ({ ...state, playing: false, loading: false }));
  }
}

/**
 * Play a track, or pause it if it is the one playing.
 *
 * `startAt` seeks before playing (and never pauses); on a track that is not
 * loaded yet the seek is held until its metadata arrives.
 */
export async function playTrack(track: Track, startAt?: number): Promise<void> {
  const el = element();
  const current = get(player);

  if (current.track?.id === track.id) {
    if (startAt !== undefined) {
      seek(startAt);
      if (el.paused) await start(el, generation);
      return;
    }
    // The play button and the space bar both come here, and both read as
    // pause while the track is playing.
    if (!el.paused) {
      el.pause();
      return;
    }
    await start(el, generation);
    return;
  }

  generation += 1;
  const load = generation;
  pending = startAt === undefined ? null : { seconds: startAt };
  player.update((state) => ({
    ...state,
    track,
    loading: true,
    playing: false,
    currentTime: startAt ?? 0,
    duration: 0,
    error: ''
  }));

  el.src = api.audioUrl(track.id);
  el.load();
  await start(el, load);
}

export function togglePlay(): void {
  const el = element();
  if (el.paused) void start(el, generation);
  else el.pause();
}

export function stop(): void {
  const el = element();
  el.pause();
  el.currentTime = 0;
  stopFollowing();
  pending = null;
  player.set(initial);
}

export function seek(seconds: number): void {
  const el = element();
  if (!Number.isFinite(el.duration) || el.duration <= 0) {
    // Metadata has not arrived yet — land here once it does.
    pending = { seconds };
    player.update((state) => ({ ...state, currentTime: Math.max(0, seconds) }));
    return;
  }
  el.currentTime = Math.max(0, Math.min(seconds, Math.max(el.duration - 0.05, 0)));
  player.update((state) => ({ ...state, currentTime: el.currentTime }));
}

/** Jump to a fraction (0-1) of the track — used by the waveform scrubber. */
export function seekFraction(fraction: number): void {
  const el = element();
  const clamped = Math.max(0, Math.min(fraction, 1));
  if (!Number.isFinite(el.duration) || el.duration <= 0) {
    pending = { fraction: clamped };
    return;
  }
  seek(clamped * el.duration);
}

/** Seek relative to where playback is now, in seconds. */
export function nudge(seconds: number): void {
  const el = element();
  seek(el.currentTime + seconds);
}

export function setVolume(volume: number): void {
  element().volume = Math.max(0, Math.min(1, volume));
}

// Hot reloads replace this module; silence the copy being replaced straight
// away rather than when the new copy next plays.
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    stopFollowing();
    if (audio) retire(audio);
  });
}
