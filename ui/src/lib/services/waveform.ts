import { api, ApiError } from '$lib/services/api';

export interface WaveformData {
  peaks: number[];
  duration: number;
  samples: number;
}

/**
 * Waveforms, fetched once per track.
 *
 * Reviewing a session walks back and forth through the same tracks, and
 * re-decoding audio for a waveform the UI already had is the difference
 * between an instant redraw and a visible wait. The server caches too; this
 * keeps the request from being made at all.
 */
const CACHE_LIMIT = 300;

const cache = new Map<string, WaveformData>();

/**
 * Requests in flight, shared by everyone waiting on the same waveform.
 *
 * The request runs on its own controller rather than any one caller's signal:
 * otherwise the first caller to give up (a component re-rendering, the review
 * screen moving on and straight back) cancels it for everyone else, who then
 * wait on a cancelled request forever. It is cancelled only once every caller
 * has given up.
 */
interface Shared {
  promise: Promise<WaveformData>;
  controller: AbortController;
  waiting: number;
}
const inFlight = new Map<string, Shared>();

const keyFor = (trackId: number, samples: number) => `${trackId}:${samples}`;

export function cachedWaveform(trackId: number, samples: number): WaveformData | undefined {
  return cache.get(keyFor(trackId, samples));
}

function fetchShared(trackId: number, samples: number, key: string): Shared {
  const controller = new AbortController();
  const promise = api
    .waveform(trackId, samples, controller.signal)
    .then((result) => {
      const data: WaveformData = {
        peaks: Array.isArray(result?.peaks) ? result.peaks : [],
        duration: Number.isFinite(result?.duration) ? result.duration : 0,
        samples: result?.samples ?? samples
      };
      if (data.peaks.length === 0) throw new ApiError('Waveform was empty', 422);

      cache.set(key, data);
      while (cache.size > CACHE_LIMIT) {
        const oldest = cache.keys().next().value;
        if (oldest === undefined) break;
        cache.delete(oldest);
      }
      return data;
    })
    .finally(() => {
      if (inFlight.get(key) === shared) inFlight.delete(key);
    });
  const shared: Shared = { promise, controller, waiting: 0 };
  return shared;
}

const aborted = () => new DOMException('The waveform request was cancelled', 'AbortError');

export function loadWaveform(
  trackId: number,
  samples: number,
  signal?: AbortSignal
): Promise<WaveformData> {
  const key = keyFor(trackId, samples);

  const hit = cache.get(key);
  if (hit) return Promise.resolve(hit);
  if (signal?.aborted) return Promise.reject(aborted());

  // Two components asking for the same waveform share one request.
  let shared = inFlight.get(key);
  if (!shared) {
    shared = fetchShared(trackId, samples, key);
    inFlight.set(key, shared);
  }
  const request = shared;
  request.waiting += 1;

  if (!signal) return request.promise;

  return new Promise<WaveformData>((resolve, reject) => {
    const giveUp = () => {
      request.waiting -= 1;
      if (request.waiting === 0) {
        // Nobody wants it any more. Forget it first, so a caller arriving
        // later starts a fresh request instead of joining a cancelled one.
        if (inFlight.get(key) === request) inFlight.delete(key);
        request.controller.abort();
      }
      reject(aborted());
    };
    signal.addEventListener('abort', giveUp, { once: true });
    request.promise.then(
      (data) => {
        signal.removeEventListener('abort', giveUp);
        resolve(data);
      },
      (error) => {
        signal.removeEventListener('abort', giveUp);
        reject(error);
      }
    );
  });
}

/** Forget a track's waveform so the next request re-fetches it. */
export function forgetWaveform(trackId: number, samples: number): void {
  cache.delete(keyFor(trackId, samples));
}

/**
 * Fit an envelope to however many bars are on screen.
 *
 * The server is asked for one generous resolution per track, so resizing the
 * window — or drawing the same track at two sizes — never costs a request.
 * Buckets keep their peak rather than their average: a transient that survives
 * downsampling is what makes a waveform look like the track it belongs to.
 */
export function resampleEnvelope(peaks: number[], bars: number): number[] {
  if (bars <= 0 || peaks.length === 0) return [];
  if (peaks.length === bars) return peaks;

  const out = new Array<number>(bars);

  if (peaks.length < bars) {
    // Stretch: interpolate between the values we have.
    const scale = (peaks.length - 1) / Math.max(bars - 1, 1);
    for (let index = 0; index < bars; index++) {
      const position = index * scale;
      const left = Math.floor(position);
      const right = Math.min(left + 1, peaks.length - 1);
      const weight = position - left;
      out[index] = peaks[left] * (1 - weight) + peaks[right] * weight;
    }
    return out;
  }

  const width = peaks.length / bars;
  for (let index = 0; index < bars; index++) {
    const start = Math.floor(index * width);
    const end = Math.max(Math.floor((index + 1) * width), start + 1);
    let highest = 0;
    for (let at = start; at < end && at < peaks.length; at++) {
      if (peaks[at] > highest) highest = peaks[at];
    }
    out[index] = highest;
  }
  return out;
}
