import { get } from 'svelte/store';
import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Track } from '$lib/types';

/**
 * jsdom has no media pipeline, so the store's single audio element is replaced
 * by a stand-in whose metadata, playback and errors the test fires by hand.
 */
class FakeAudio extends EventTarget {
  static last: FakeAudio;
  src = '';
  preload = '';
  volume = 1;
  currentTime = 0;
  duration = NaN;
  paused = true;
  loads = 0;
  /** When set, play() stays pending until 'playing', as a real element's does. */
  static holdPlay = false;
  private abandon: ((reason: unknown) => void) | null = null;

  constructor() {
    super();
    FakeAudio.last = this;
  }

  load() {
    // A real element rejects the play() it was still working on.
    this.abandon?.(new DOMException('The play() request was interrupted', 'AbortError'));
    this.abandon = null;
    this.loads += 1;
    this.duration = NaN;
    this.currentTime = 0;
  }

  play() {
    this.paused = false;
    if (!FakeAudio.holdPlay) return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      this.abandon = reject;
      this.addEventListener('playing', () => resolve(), { once: true });
    });
  }

  removeAttribute(name: string) {
    if (name === 'src') this.src = '';
  }

  /** Enough of the file has buffered for sound to start. */
  started() {
    this.dispatchEvent(new Event('playing'));
  }

  pause() {
    this.paused = true;
    this.dispatchEvent(new Event('pause'));
  }

  /** The browser has read the file's header. */
  metadata(duration: number) {
    this.duration = duration;
    this.dispatchEvent(new Event('loadedmetadata'));
  }
}

vi.stubGlobal('Audio', FakeAudio);
afterAll(() => {
  vi.unstubAllGlobals();
});

const { player, playTrack, seek, seekFraction, stop } = await import('$lib/stores/player');

const track = (id: number) => ({ id, filepath: `/m/${id}.wav`, filename: `${id}.wav` }) as Track;

describe('player', () => {
  beforeEach(() => {
    FakeAudio.holdPlay = false;
    stop();
  });

  it('loads the API audio stream for a new track', async () => {
    await playTrack(track(7));

    expect(FakeAudio.last.src).toMatch(/\/api\/tracks\/7\/audio$/);
    expect(get(player)).toMatchObject({ track: { id: 7 }, loading: true, playing: false });

    FakeAudio.last.started();

    expect(get(player)).toMatchObject({ loading: false, playing: true });
  });

  it('does not reload a track that is already loaded', async () => {
    await playTrack(track(7));
    const loads = FakeAudio.last.loads;

    await playTrack(track(7));

    expect(FakeAudio.last.loads).toBe(loads);
  });

  it('holds a seek made before metadata and lands on it once the track is ready', async () => {
    await playTrack(track(1), 42);
    expect(FakeAudio.last.currentTime).toBe(0);

    FakeAudio.last.metadata(120);

    expect(FakeAudio.last.currentTime).toBe(42);
    expect(get(player).currentTime).toBe(42);
  });

  it('turns a waveform click before metadata into a time once the duration is known', async () => {
    await playTrack(track(2));
    seekFraction(0.25);

    FakeAudio.last.metadata(200);

    expect(FakeAudio.last.currentTime).toBe(50);
  });

  it('keeps seeks inside the track', async () => {
    await playTrack(track(3));
    FakeAudio.last.metadata(10);

    seek(-5);
    expect(FakeAudio.last.currentTime).toBe(0);

    // Just short of the end, so seeking past it does not fire 'ended'.
    seek(99);
    expect(FakeAudio.last.currentTime).toBeCloseTo(9.95);
    expect(get(player).currentTime).toBeCloseTo(9.95);
  });

  it('reports a track that cannot be played and drops any held seek', async () => {
    await playTrack(track(4), 30);

    FakeAudio.last.dispatchEvent(new Event('error'));
    expect(get(player)).toMatchObject({ playing: false, error: 'This track could not be played' });

    FakeAudio.last.metadata(60);
    expect(FakeAudio.last.currentTime).toBe(0);
  });

  it('pauses the track that is playing when it is played again', async () => {
    // The play button and the space bar both call playTrack on the current
    // track; while it plays, that has to read as pause.
    await playTrack(track(5));
    FakeAudio.last.started();

    await playTrack(track(5));
    expect(FakeAudio.last.paused).toBe(true);
    expect(get(player).playing).toBe(false);

    await playTrack(track(5));
    expect(FakeAudio.last.paused).toBe(false);
  });

  it('seeks the playing track without pausing it', async () => {
    await playTrack(track(6));
    FakeAudio.last.metadata(100);
    FakeAudio.last.started();

    await playTrack(track(6), 30);

    expect(FakeAudio.last.paused).toBe(false);
    expect(FakeAudio.last.currentTime).toBe(30);
  });

  it('keeps the next track loading when switching abandons the previous one', async () => {
    FakeAudio.holdPlay = true;
    const first = playTrack(track(8));
    const second = playTrack(track(9));
    await first;

    // The first track's play() was rejected by the switch; that must not make
    // the second look loaded before a sound has come out.
    expect(get(player)).toMatchObject({ track: { id: 9 }, loading: true });

    FakeAudio.last.started();
    await second;
    expect(get(player)).toMatchObject({ track: { id: 9 }, loading: false, playing: true });
  });

  it('silences the element a replaced copy of the store was playing', async () => {
    // A dev-server hot reload evaluates this module again. The old copy's
    // element must not keep playing alongside the new one.
    await playTrack(track(10));
    FakeAudio.last.started();
    const old = FakeAudio.last;

    vi.resetModules();
    const reloaded = await import('$lib/stores/player');
    await reloaded.playTrack(track(11));

    expect(FakeAudio.last).not.toBe(old);
    expect(old.paused).toBe(true);
    expect(old.src).toBe('');
    expect(FakeAudio.last.src).toMatch(/\/api\/tracks\/11\/audio$/);
  });
});
