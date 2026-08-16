/**
 * A track's database ID.
 *
 * Branded so it cannot be confused with the other IDs floating around the sort
 * flow — a sort item's ID is also a number, and passing one where a track was
 * meant loads a different track's audio and waveform entirely.
 */
declare const trackIdBrand: unique symbol;
export type TrackId = number & { readonly [trackIdBrand]: true };

export interface Track {
  id: TrackId;
  filepath: string;
  filename: string;
  display?: string;
  title?: string | null;
  artist?: string | null;
  album?: string | null;
  genre?: string | null;
  year?: number | null;
  tag_bpm?: number | null;
  tag_key?: string | null;
  duration?: number | null;
  role?: string;
}

export interface Collection {
  id: number;
  name: string;
  description?: string | null;
  settings?: Record<string, unknown>;
  group_count?: number;
  track_count?: number;
  fitted?: boolean;
  accuracy?: number | null;
  fitted_at?: string | null;
  updated_at?: string;
}

export interface Group {
  id: number;
  collection_id: number;
  name: string;
  description?: string | null;
  color?: string | null;
  destination_path?: string | null;
  source_kind?: string;
  source_path?: string | null;
  position?: number;
  size?: number;
}

export interface GroupQuality {
  group_id: number;
  name: string;
  size: number;
  correct: number;
  accuracy: number;
}

export interface ConfusablePair {
  from_group_id: number;
  from_name: string;
  to_group_id: number;
  to_name: string;
  count: number;
  rate: number;
}

export interface CheckMetrics {
  accuracy: number;
  n_tracks: number;
  n_groups: number;
  mean_confidence: number;
  per_group: GroupQuality[];
  group_names: string[];
  confusion: number[][];
  confusable_pairs: ConfusablePair[];
  embedding?: Record<string, unknown>;
  problems?: { group_id: number; name: string; issue: string }[];
}

export interface RankedGroup {
  group_id: number;
  group_name: string;
  distance: number;
  score: number;
}

/**
 * One incoming track awaiting a decision.
 *
 * `id` identifies the *item* in the review queue; `track_id` identifies the
 * audio. Anything that plays, draws or streams audio needs `track_id` — use
 * `trackOf()` to get a Track from an item rather than passing the item along.
 */
export interface SortItem extends Omit<Track, 'id'> {
  id: number;
  item_id?: number;
  session_id: number;
  track_id: TrackId;
  suggested_group_id: number | null;
  suggested_group_name?: string | null;
  final_group_id: number | null;
  final_group_name?: string | null;
  confidence: number | null;
  margin: number | null;
  distance: number | null;
  ranked: RankedGroup[];
  status: 'pending' | 'accepted' | 'skipped' | 'unmatched';
  applied: number;
}

export interface SortSession {
  id: number;
  collection_id: number;
  collection_name?: string;
  name: string | null;
  source_path: string | null;
  status: string;
  created_at: string;
  committed_at?: string | null;
  item_count?: number;
  pending_count?: number;
}

export interface CandidateStats {
  detected_bpm?: number;
  tag_bpm_median?: number | null;
  tag_bpm_range?: [number, number] | null;
  brightness_hz?: number;
  energy?: number;
  descriptors?: string[];
  top_genres?: { name: string; count: number }[];
  top_artists?: { name: string; count: number }[];
}

export interface DiscoveryCandidate {
  id: number;
  run_id: number;
  candidate_index: number;
  suggested_name: string;
  size: number;
  exemplars: number[];
  exemplar_tracks?: Track[];
  stats: CandidateStats;
  status: 'pending' | 'promoted' | 'rejected';
  promoted_group_id: number | null;
}

export interface DiscoveryRun {
  id: number;
  name: string | null;
  source_path: string | null;
  algorithm: string;
  params: Record<string, unknown>;
  metrics: Record<string, unknown>;
  status: string;
  created_at: string;
  candidate_count?: number;
}

export interface PlannedAction {
  track_id: number;
  group_id: number;
  group_name: string;
  action: string;
  source_path: string;
  dest_path: string | null;
  note: string | null;
}

export interface CommitPlan {
  mode: string;
  actions: PlannedAction[];
  playlists: { group_id: number; group_name: string; path: string; track_count: number }[];
  problems: Record<string, unknown>[];
  action_count: number;
  playlist_count: number;
}

export interface TaskStatus {
  id: string;
  kind: string;
  status: 'running' | 'completed' | 'failed';
  progress: number;
  current: number;
  total: number;
  message: string;
  result: unknown;
  error: string | null;
}

export type OrganizeMode = 'playlist' | 'copy' | 'move' | 'symlink';
