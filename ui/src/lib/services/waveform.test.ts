import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest';
import { api, ApiError } from '$lib/services/api';
import {
  cachedWaveform,
  forgetWaveform,
  loadWaveform,
  resampleEnvelope
} from '$lib/services/waveform';

describe('resampleEnvelope', () => {
  it('returns the input untouched when it already has the right length', () => {
    const peaks = [0.1, 0.4, 0.2];
    expect(resampleEnvelope(peaks, 3)).toBe(peaks);
  });

  it('is empty for no bars or no peaks', () => {
    expect(resampleEnvelope([0.5], 0)).toEqual([]);
    expect(resampleEnvelope([], 10)).toEqual([]);
  });

  it('keeps the peak of each bucket when downsampling', () => {
    // Averaging here would flatten transients until every waveform looks the
    // same; the loudest sample in a bucket is what makes a track recognisable.
    const peaks = [0.1, 0.9, 0.2, 0.3, 0.05, 0.8];
    expect(resampleEnvelope(peaks, 3)).toEqual([0.9, 0.3, 0.8]);
  });

  it('interpolates when stretching a short envelope over more bars', () => {
    const stretched = resampleEnvelope([0, 1], 3);

    expect(stretched).toHaveLength(3);
    expect(stretched[0]).toBeCloseTo(0);
    expect(stretched[1]).toBeCloseTo(0.5);
    expect(stretched[2]).toBeCloseTo(1);
  });

  it('never exceeds the range of the input', () => {
    const peaks = Array.from({ length: 500 }, (_, i) => Math.abs(Math.sin(i)));
    for (const bars of [1, 7, 240, 999]) {
      const out = resampleEnvelope(peaks, bars);
      expect(out).toHaveLength(bars);
      expect(Math.min(...out)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...out)).toBeLessThanOrEqual(1);
    }
  });
});

describe('loadWaveform', () => {
  let waveformSpy: MockInstance<typeof api.waveform>;

  beforeEach(() => {
    waveformSpy = vi.spyOn(api, 'waveform');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('fetches once and serves the rest from cache', async () => {
    waveformSpy.mockResolvedValue({ peaks: [0.2, 0.4], duration: 180, samples: 240 });

    const first = await loadWaveform(101, 240);
    const second = await loadWaveform(101, 240);

    expect(waveformSpy).toHaveBeenCalledTimes(1);
    expect(second).toEqual(first);
    expect(cachedWaveform(101, 240)).toEqual(first);
  });

  it('shares one request between concurrent callers', async () => {
    // Two components asking for the same track at once is the normal case in
    // the review screen, and decoding it twice is a visible stall.
    waveformSpy.mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(() => resolve({ peaks: [0.5], duration: 10, samples: 240 }), 5)
        )
    );

    const [a, b] = await Promise.all([loadWaveform(102, 240), loadWaveform(102, 240)]);

    expect(waveformSpy).toHaveBeenCalledTimes(1);
    expect(a).toEqual(b);
  });

  it('treats an empty waveform as a failure and does not cache it', async () => {
    waveformSpy.mockResolvedValue({ peaks: [], duration: 0, samples: 240 });

    await expect(loadWaveform(103, 240)).rejects.toBeInstanceOf(ApiError);
    expect(cachedWaveform(103, 240)).toBeUndefined();
  });

  it('caches per sample count, so a different resolution is fetched separately', async () => {
    waveformSpy.mockResolvedValue({ peaks: [0.1], duration: 5, samples: 240 });

    await loadWaveform(104, 240);
    await loadWaveform(104, 480);

    expect(waveformSpy).toHaveBeenCalledTimes(2);
  });

  it('re-fetches after the entry is forgotten', async () => {
    waveformSpy.mockResolvedValue({ peaks: [0.3], duration: 5, samples: 240 });

    await loadWaveform(105, 240);
    forgetWaveform(105, 240);
    await loadWaveform(105, 240);

    expect(waveformSpy).toHaveBeenCalledTimes(2);
    expect(cachedWaveform(105, 240)).toBeDefined();
  });

  it('retries after a failure rather than caching the error', async () => {
    waveformSpy
      .mockRejectedValueOnce(new ApiError('Could not read audio', 422))
      .mockResolvedValueOnce({ peaks: [0.7], duration: 30, samples: 240 });

    await expect(loadWaveform(106, 240)).rejects.toBeInstanceOf(ApiError);
    await expect(loadWaveform(106, 240)).resolves.toMatchObject({ peaks: [0.7] });
  });
});
