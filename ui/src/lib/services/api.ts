import type {
  CheckMetrics,
  Collection,
  CommitPlan,
  DiscoveryCandidate,
  DiscoveryRun,
  Group,
  GroupQuality,
  OrganizeMode,
  SortItem,
  SortSession,
  TaskStatus,
  Track
} from '$lib/types';

export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://127.0.0.1:8000';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) }
    });
  } catch (error) {
    // A cancelled request is not a failure to report — the caller moved on.
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new ApiError(
      'Cannot reach the music-cluster API. Is it running on port 8000?',
      0
    );
  }

  if (response.status === 204) return undefined as T;

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* response had no JSON body */
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

const get = <T>(path: string) => request<T>(path);
const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });
const put = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'PUT', body: body === undefined ? undefined : JSON.stringify(body) });
const del = <T>(path: string) => request<T>(path, { method: 'DELETE' });

export const api = {
  info: () =>
    get<{
      database_path: string;
      track_count: number;
      analyzed_count: number;
      collections: Collection[];
      sessions: SortSession[];
      discovery_runs: DiscoveryRun[];
    }>('/api/info'),

  browse: (path?: string) =>
    get<{
      path: string;
      parent: string | null;
      directories: { name: string; path: string }[];
      audio_file_count: number;
    }>(`/api/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`),

  getConfig: () => get<Record<string, any>>('/api/config'),
  updateConfig: (values: Record<string, unknown>) =>
    put<Record<string, any>>('/api/config', { values }),

  // Tasks
  task: (id: string) => get<TaskStatus>(`/api/tasks/${id}`),

  // Collections
  listCollections: () => get<{ collections: Collection[] }>('/api/collections'),
  createCollection: (name: string, description?: string) =>
    post<Collection>('/api/collections', { name, description }),
  getCollection: (id: number) =>
    get<{ collection: Collection; groups: Group[]; model: Record<string, any> | null }>(
      `/api/collections/${id}`
    ),
  updateCollection: (id: number, patch: Partial<Collection>) =>
    put<Collection>(`/api/collections/${id}`, patch),
  deleteCollection: (id: number) => del<{ deleted: boolean }>(`/api/collections/${id}`),
  fitCollection: (id: number) => post<{ task_id: string }>(`/api/collections/${id}/fit`),
  checkCollection: (id: number) =>
    get<{ metrics: CheckMetrics; fitted_at: string; params: Record<string, unknown> }>(
      `/api/collections/${id}/check`
    ),
  exportGroups: (id: number, output: string, format = 'm3u8') =>
    post<{ playlists: string[] }>(
      `/api/collections/${id}/export?output=${encodeURIComponent(output)}&playlist_format=${format}`
    ),

  // Groups
  listGroups: (collectionId: number) =>
    get<{ groups: Group[] }>(`/api/collections/${collectionId}/groups`),
  createGroup: (collectionId: number, body: Partial<Group> & { name: string }) =>
    post<Group>(`/api/collections/${collectionId}/groups`, body),
  getGroup: (id: number, limit = 200, offset = 0) =>
    get<{ group: Group; size: number; members: Track[]; quality: GroupQuality | null }>(
      `/api/groups/${id}?limit=${limit}&offset=${offset}`
    ),
  updateGroup: (id: number, patch: Partial<Group>) => put<Group>(`/api/groups/${id}`, patch),
  deleteGroup: (id: number) => del<{ deleted: boolean }>(`/api/groups/${id}`),
  addGroupMembers: (id: number, body: { track_ids?: number[]; filepaths?: string[]; role?: string }) =>
    post<{ added: number; size: number }>(`/api/groups/${id}/members`, body),
  removeGroupMember: (groupId: number, trackId: number) =>
    del<{ removed: boolean }>(`/api/groups/${groupId}/members/${trackId}`),

  importFolder: (
    collectionId: number,
    body: { path: string; name?: string; recursive?: boolean; destination_path?: string }
  ) => post<{ task_id: string }>(`/api/collections/${collectionId}/import-folder`, body),
  importTree: (collectionId: number, body: { path: string; min_tracks?: number; recursive?: boolean }) =>
    post<{ task_id: string }>(`/api/collections/${collectionId}/import-tree`, body),
  importPlaylist: (collectionId: number, body: { path: string; name?: string; destination_path?: string }) =>
    post<{ task_id: string }>(`/api/collections/${collectionId}/import-playlist`, body),

  // Tracks
  listTracks: (params: { limit?: number; offset?: number; query?: string; unassignedIn?: number } = {}) => {
    const search = new URLSearchParams();
    search.set('limit', String(params.limit ?? 100));
    search.set('offset', String(params.offset ?? 0));
    if (params.query) search.set('query', params.query);
    if (params.unassignedIn) search.set('unassigned_in', String(params.unassignedIn));
    return get<{ tracks: Track[]; total: number }>(`/api/tracks?${search}`);
  },
  getTrack: (id: number) => get<Track & { groups: Group[]; exists: boolean }>(`/api/tracks/${id}`),
  waveform: (id: number, samples = 240, signal?: AbortSignal) =>
    request<{ peaks: number[]; duration: number; samples: number }>(
      `/api/tracks/${id}/waveform?samples=${samples}`,
      { signal }
    ),
  artwork: (id: number) =>
    get<{ artwork: string; mime_type: string } | undefined>(`/api/tracks/${id}/artwork`),
  audioUrl: (id: number) => `${API_BASE}/api/tracks/${id}/audio`,

  analyze: (paths: string[], recursive = true, update = false) =>
    post<{ task_id: string }>('/api/analyze', { paths, recursive, update }),

  // Sort sessions
  listSessions: (collectionId?: number) =>
    get<{ sessions: SortSession[] }>(
      `/api/sessions${collectionId ? `?collection_id=${collectionId}` : ''}`
    ),
  createSession: (body: {
    collection_id: number;
    paths: string[];
    name?: string;
    recursive?: boolean;
    auto_accept?: boolean;
  }) => post<{ task_id: string }>('/api/sessions', body),
  getSession: (id: number) =>
    get<{
      session: SortSession;
      groups: Group[];
      summary: Record<string, number>;
      total: number;
    }>(`/api/sessions/${id}`),
  deleteSession: (id: number) => del<{ deleted: boolean }>(`/api/sessions/${id}`),
  sessionItems: (id: number, params: { status?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set('status', params.status);
    search.set('limit', String(params.limit ?? 200));
    search.set('offset', String(params.offset ?? 0));
    return get<{ items: SortItem[]; summary: Record<string, number>; groups: Group[] }>(
      `/api/sessions/${id}/items?${search}`
    );
  },
  decideItem: (itemId: number, body: { group_id?: number | null; skip?: boolean }) =>
    put<SortItem>(`/api/items/${itemId}`, body),
  decideItems: (sessionId: number, body: { item_ids: number[]; group_id?: number | null; skip?: boolean }) =>
    put<{ updated: number; summary: Record<string, number> }>(`/api/sessions/${sessionId}/items`, body),
  acceptConfident: (sessionId: number) =>
    post<{ accepted: number; summary: Record<string, number> }>(
      `/api/sessions/${sessionId}/accept-confident`
    ),
  rescore: (sessionId: number) => post<{ task_id: string }>(`/api/sessions/${sessionId}/rescore`),
  commit: (
    sessionId: number,
    body: { mode?: OrganizeMode; playlist_dir?: string; on_conflict?: string; learn?: boolean; dry_run?: boolean }
  ) =>
    post<{ dry_run?: boolean; plan: CommitPlan; result?: any; learned?: number }>(
      `/api/sessions/${sessionId}/commit`,
      body
    ),
  undo: (sessionId: number, removePlaylists = false) =>
    post<{ undone: number; failures: unknown[] }>(
      `/api/sessions/${sessionId}/undo?remove_playlists=${removePlaylists}`
    ),

  // Discovery
  listDiscoveryRuns: () => get<{ runs: DiscoveryRun[] }>('/api/discovery'),
  discover: (body: {
    paths?: string[];
    track_ids?: number[];
    name?: string;
    algorithm?: string;
    granularity?: string;
    target_groups?: number;
    min_group_size?: number;
    use_llm?: boolean;
  }) => post<{ task_id: string }>('/api/discovery', body),
  getDiscoveryRun: (id: number) =>
    get<{ run: DiscoveryRun; candidates: DiscoveryCandidate[] }>(`/api/discovery/${id}`),
  deleteDiscoveryRun: (id: number) => del<{ deleted: boolean }>(`/api/discovery/${id}`),
  candidateMembers: (candidateId: number, limit = 100, offset = 0) =>
    get<{ total: number; tracks: Track[] }>(
      `/api/discovery/candidates/${candidateId}/members?limit=${limit}&offset=${offset}`
    ),
  promoteCandidate: (
    candidateId: number,
    body: {
      collection_id: number;
      name?: string;
      destination_path?: string;
      seeds_only?: boolean;
      target_group_id?: number;
    }
  ) => post<{ group: Group; added: number }>(`/api/discovery/candidates/${candidateId}/promote`, body),
  rejectCandidate: (candidateId: number) =>
    post<{ rejected: boolean }>(`/api/discovery/candidates/${candidateId}/reject`),

  similar: (seedTrackIds: number[], limit = 50) =>
    post<{ results: { track: Track; distance: number; relative_distance: number }[] }>(
      '/api/similar',
      { seed_track_ids: seedTrackIds, limit }
    )
};

/**
 * Poll a background task until it finishes.
 *
 * Analysis of a large folder can run for many minutes, so the UI kicks off a
 * task and follows it here rather than holding a request open.
 */
export async function pollTask(
  taskId: string,
  onProgress?: (task: TaskStatus) => void,
  intervalMs = 500
): Promise<TaskStatus> {
  for (;;) {
    const task = await api.task(taskId);
    onProgress?.(task);
    if (task.status !== 'running') {
      if (task.status === 'failed') throw new ApiError(task.error ?? 'Task failed', 500);
      return task;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
