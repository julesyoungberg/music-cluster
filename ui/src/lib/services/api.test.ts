import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_BASE, ApiError, api, pollTask } from '$lib/services/api';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' }
  });
}

describe('request', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends JSON bodies to paths under the API base', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: 1, name: 'Crates' }));

    await api.createCollection('Crates', 'my folders');

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE}/api/collections`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ name: 'Crates', description: 'my folders' });
    expect(init.headers['Content-Type']).toBe('application/json');
  });

  it('surfaces the server detail message on an error status', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: 'No collection with ID 9' }, 404));

    await expect(api.getCollection(9)).rejects.toMatchObject({
      name: 'ApiError',
      status: 404,
      message: 'No collection with ID 9'
    });
  });

  it('falls back to the status text when the error body is not JSON', async () => {
    fetchMock.mockResolvedValue(
      new Response('kaboom', { status: 500, statusText: 'Server Error' })
    );

    await expect(api.info()).rejects.toMatchObject({ status: 500, message: 'Server Error' });
  });

  it('explains that the API is unreachable rather than leaking a fetch error', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));

    // This is the message a user sees when they open the app without the
    // backend running, so it has to say what to do about it.
    await expect(api.info()).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining('Is it running on port 8000?')
    });
  });

  it('rethrows an abort so a superseded request is not reported as a failure', async () => {
    const aborted = new DOMException('The operation was aborted.', 'AbortError');
    fetchMock.mockRejectedValue(aborted);

    await expect(api.info()).rejects.toBe(aborted);
  });

  it('resolves to undefined for a 204 with no body', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await expect(api.deleteCollection(3)).resolves.toBeUndefined();
  });

  it('percent-encodes paths passed as query parameters', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ path: '/', directories: [] }));

    await api.browse('/Music/Deep House & Friends');

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${API_BASE}/api/browse?path=${encodeURIComponent('/Music/Deep House & Friends')}`
    );
  });
});

describe('audioUrl', () => {
  it('points at the streaming endpoint for a track', () => {
    expect(api.audioUrl(12)).toBe(`${API_BASE}/api/tracks/12/audio`);
  });
});

describe('pollTask', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reports progress until the task completes', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ id: 't', status: 'running', progress: 0.5 }))
      .mockResolvedValueOnce(jsonResponse({ id: 't', status: 'completed', progress: 1 }));

    const seen: string[] = [];
    const finished = await pollTask('t', (task) => seen.push(task.status), 0);

    expect(seen).toEqual(['running', 'completed']);
    expect(finished.status).toBe('completed');
  });

  it('throws the task error when the task fails', async () => {
    // A Response body can only be read once, so each poll gets a fresh one.
    fetchMock.mockImplementation(async () =>
      jsonResponse({ id: 't', status: 'failed', error: 'Could not decode audio' })
    );

    const failure = await pollTask('t', undefined, 0).catch((error) => error);

    expect(failure).toBeInstanceOf(ApiError);
    expect(failure.message).toBe('Could not decode audio');
  });
});
