<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/services/api';
  import type { Track } from '$lib/types';
  import { Library, Search, ChevronLeft, ChevronRight, Loader2, Music, Play, Pause, SkipBack, SkipForward } from 'lucide-svelte';
  import TrackArtwork from '$lib/components/TrackArtwork.svelte';
  import Waveform from '$lib/components/Waveform.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import { useAudioPlayer } from '$lib/composables/useAudioPlayer';
  import { debounce } from '$lib/utils';
  import { addNotification } from '$lib/stores/notifications';

  let tracks: Track[] = [];
  let loading = true;
  let limit = 100;
  let offset = 0;
  let total = 0;
  let searchQuery = '';
  let waveformData: Map<number, { peaks: number[]; duration: number }> = new Map();
  let loadingWaveforms: Set<number> = new Set();

  // Use audio player composable
  const audioPlayer = useAudioPlayer(
    (time) => {
      // Time update handler - waveform will react to currentTime changes
    },
    () => {
      // Auto-play next track when current ends
      audioPlayer.playNextTrack(tracks);
    }
  );

  $: currentTrackId = audioPlayer.currentTrackId;
  $: playing = audioPlayer.playing;
  $: currentTime = audioPlayer.currentTime;
  $: duration = audioPlayer.duration;

  async function loadTracks() {
    loading = true;
    try {
      const result = await api.getTracks(limit, offset);
      tracks = result.tracks;
      total = result.total;
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : 'Failed to load tracks';
      addNotification('error', errorMsg);
      console.error('Failed to load tracks:', e);
    } finally {
      loading = false;
    }
  }

  async function performSearch() {
    if (!searchQuery.trim()) {
      offset = 0;
      await loadTracks();
      return;
    }
    loading = true;
    try {
      const result = await api.search(searchQuery, undefined, limit);
      tracks = result.tracks;
      total = result.total;
      offset = 0;
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : 'Failed to search tracks';
      addNotification('error', errorMsg);
      console.error('Failed to search:', e);
    } finally {
      loading = false;
    }
  }

  // Debounced search
  const debouncedSearch = debounce(performSearch, 300);

  // Trigger search when query changes
  $: if (searchQuery !== undefined) {
    if (searchQuery.trim()) {
      debouncedSearch();
    } else {
      offset = 0;
      loadTracks();
    }
  }

  async function loadWaveform(trackId: number) {
    if (waveformData.has(trackId) || loadingWaveforms.has(trackId)) return;
    
    loadingWaveforms.add(trackId);
    try {
      const data = await api.getTrackWaveform(trackId, 200);
      waveformData.set(trackId, { peaks: data.peaks, duration: data.duration });
    } catch (e) {
      console.error('Failed to load waveform:', e);
    } finally {
      loadingWaveforms.delete(trackId);
    }
  }

  function playTrack(trackId: number) {
    audioPlayer.playTrack(trackId);
    
    // Load waveform if not already loaded
    if (!waveformData.has(trackId)) {
      loadWaveform(trackId);
    }
  }

  function seekTo(time: number) {
    audioPlayer.seekTo(time);
  }

  onMount(() => {
    loadTracks();
  });
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <Library class="w-10 h-10" />
    Library
  </h1>

  <div class="mb-6">
    <div class="flex gap-2">
      <div class="flex-1 relative">
        <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
        <input
          type="text"
          bind:value={searchQuery}
          placeholder="Search tracks..."
          class="w-full pl-10 pr-4 py-2 border rounded-lg bg-background"
          aria-label="Search tracks"
        />
      </div>
    </div>
  </div>

  {#if loading}
    <LoadingState message="Loading tracks..." />
  {:else if tracks.length > 0}
    <div class="space-y-2">
      {#each tracks as track, index (track.id)}
        <div class="bg-card p-4 rounded-lg border hover:border-primary transition-colors {currentTrackId === track.id ? 'border-primary' : ''}">
          <div class="flex items-center gap-4">
            <TrackArtwork track={track} size={64} />
            <div class="flex-1 min-w-0">
              <p class="font-medium truncate">{track.filename}</p>
              <p class="text-sm text-muted-foreground truncate">{track.filepath}</p>
              {#if track.cluster}
                <p class="text-sm text-primary mt-1">
                  Cluster: {track.cluster.name || `Cluster ${track.cluster.index}`}
                </p>
              {/if}
              
              {#if currentTrackId === track.id && waveformData.has(track.id)}
                <div class="mt-2">
                  <Waveform
                    peaks={waveformData.get(track.id)!.peaks}
                    duration={waveformData.get(track.id)!.duration}
                    currentTime={currentTime}
                    height={30}
                    on:seek={(e: CustomEvent<number>) => seekTo(e.detail)}
                  />
                </div>
              {:else if currentTrackId === track.id}
                <div class="mt-2 h-[30px] flex items-center justify-center">
                  <Loader2 class="w-4 h-4 animate-spin text-muted-foreground" />
                </div>
              {/if}
            </div>
            <div class="flex items-center gap-2">
              {#if currentTrackId === track.id}
                <button
                  on:click={() => audioPlayer.playPreviousTrack(tracks)}
                  class="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors"
                  title="Previous track"
                  aria-label="Previous track"
                  disabled={index === 0}
                >
                  <SkipBack class="w-4 h-4" />
                </button>
              {/if}
              <button
                on:click={() => playTrack(track.id)}
                class="p-3 bg-primary text-primary-foreground rounded-full hover:opacity-90 transition-opacity flex items-center justify-center"
                title={currentTrackId === track.id && playing ? 'Pause' : 'Play'}
                aria-label={currentTrackId === track.id && playing ? 'Pause track' : 'Play track'}
              >
                {#if currentTrackId === track.id && playing}
                  <Pause class="w-5 h-5" />
                {:else}
                  <Play class="w-5 h-5" />
                {/if}
              </button>
              {#if currentTrackId === track.id}
                <button
                  on:click={() => audioPlayer.playNextTrack(tracks)}
                  class="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors"
                  title="Next track"
                  aria-label="Next track"
                  disabled={index === tracks.length - 1}
                >
                  <SkipForward class="w-4 h-4" />
                </button>
              {/if}
            </div>
          </div>
        </div>
      {/each}
    </div>

    <div class="mt-6 flex justify-between items-center">
      <button
        on:click={() => { offset = Math.max(0, offset - limit); loadTracks(); }}
        disabled={offset === 0}
        class="px-4 py-2 bg-secondary rounded-lg disabled:opacity-50 flex items-center gap-2 hover:opacity-90 transition-opacity"
      >
        <ChevronLeft class="w-4 h-4" />
        Previous
      </button>
      <span class="text-sm text-muted-foreground">
        Showing {offset + 1} - {Math.min(offset + limit, total)} of {total}
      </span>
      <button
        on:click={() => { offset += limit; loadTracks(); }}
        disabled={offset + limit >= total}
        class="px-4 py-2 bg-secondary rounded-lg disabled:opacity-50 flex items-center gap-2 hover:opacity-90 transition-opacity"
      >
        Next
        <ChevronRight class="w-4 h-4" />
      </button>
    </div>
  {:else}
    <div class="text-center py-12 text-muted-foreground flex flex-col items-center gap-2">
      <Music class="w-12 h-12 opacity-50" />
      <span>No tracks found</span>
    </div>
  {/if}
</div>
