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

  constructor() {
    super();
    FakeAudio.last = this;
  }

  load() {
    this.loads += 1;
    this.duration = NaN;
    this.currentTime = 0;
  }

  play() {
    this.paused = false;
    this.dispatchEvent(new Event('playing'));
    return Promise.resolve();
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
    stop();
  });

  it('loads the API audio stream for a new track', async () => {
    await playTrack(track(7));

    expect(FakeAudio.last.src).toMatch(/\/api\/tracks\/7\/audio$/);
    expect(get(player)).toMatchObject({ track: { id: 7 }, playing: true });
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

    seek(99);
    expect(FakeAudio.last.currentTime).toBeLessThan(10);
  });

  it('reports a track that cannot be played and drops any held seek', async () => {
    await playTrack(track(4), 30);

    FakeAudio.last.dispatchEvent(new Event('error'));
    expect(get(player)).toMatchObject({ playing: false, error: 'This track could not be played' });

    FakeAudio.last.metadata(60);
    expect(FakeAudio.last.currentTime).toBe(0);
  });
});
