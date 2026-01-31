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
  import { resourceManager } from '$lib/services/resourceManager';

  let tracks: Track[] = [];
  let loading = true;
  let limit = 100;
  let offset = 0;
  let total = 0;
  let searchQuery = '';
  let waveformData: Map<number, { peaks: number[]; duration: number }> = new Map();

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

  // Use stores directly (they're reactive)
  const { currentTrackId, playing, currentTime, duration } = audioPlayer;

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
    if (waveformData.has(trackId)) return;
    
    const data = await resourceManager.getWaveform(trackId, 200);
    if (data) {
      waveformData = new Map(waveformData); // Create new Map to trigger reactivity
      waveformData.set(trackId, { peaks: data.peaks, duration: data.duration });
    }
  }

  // Preload resources for all visible tracks
  async function preloadResources(trackList: Track[]) {
    const trackIds = trackList.map(t => t.id);
    
    // Preload artwork in background (batched, 10 concurrent)
    resourceManager.preloadArtwork(trackIds, 10);
    
    // Preload waveforms for all tracks (batched, 5 concurrent)
    // This ensures waveforms are visible for all tracks like SoundCloud/Rekordbox
    try {
      const waveformResults = await resourceManager.batchLoadWaveforms(trackIds, 200, 5);
      
      // Update waveform data (create new Map to trigger reactivity)
      const newWaveformData = new Map(waveformData);
      waveformResults.forEach((data, trackId) => {
        if (data && data.peaks && data.peaks.length > 0) {
          // Debug: check if peaks are valid
          const maxPeak = Math.max(...data.peaks);
          if (maxPeak > 0) {
            newWaveformData.set(trackId, { peaks: data.peaks, duration: data.duration });
          } else {
            console.warn(`Track ${trackId} has zero peaks`);
          }
        }
      });
      waveformData = newWaveformData;
    } catch (e) {
      console.error('Error preloading waveforms:', e);
    }
  }

  function playTrack(trackId: number) {
    audioPlayer.playTrack(trackId);
    
    // Load waveform if not already loaded (will use resource manager's deduplication)
    if (!waveformData.has(trackId)) {
      loadWaveform(trackId);
    }
  }

  function seekTo(time: number) {
    audioPlayer.seekTo(time);
  }

  // Preload resources when tracks change
  $: if (tracks.length > 0 && !loading) {
    // Trigger preload
    preloadResources(tracks);
  }
  
  // Make waveformData reactive by watching it
  $: waveformData; // This ensures UI updates when Map changes

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
        <div class="bg-card p-4 rounded-lg border hover:border-primary transition-colors {$currentTrackId === track.id ? 'border-primary' : ''}">
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
              
              <!-- Show waveform for all tracks (SoundCloud/Rekordbox style) -->
              <div class="mt-2">
                {#if waveformData.has(track.id)}
                  <Waveform
                    peaks={waveformData.get(track.id)!.peaks}
                    duration={waveformData.get(track.id)!.duration}
                    currentTime={$currentTrackId === track.id ? $currentTime : 0}
                    height={30}
                    on:seek={(e) => {
                      if ($currentTrackId === track.id) {
                        // If already playing, seek
                        seekTo(e.detail);
                      } else {
                        // If not playing, start playing from that position
                        playTrack(track.id);
                        // Small delay to ensure audio is loaded before seeking
                        setTimeout(() => seekTo(e.detail), 100);
                      }
                    }}
                  />
                {:else if resourceManager.isLoadingWaveform(track.id)}
                  <div class="h-[30px] flex items-center justify-center bg-secondary/50 rounded">
                    <Loader2 class="w-4 h-4 animate-spin text-muted-foreground" />
                  </div>
                {:else}
                  <div class="h-[30px] bg-secondary/30 rounded flex items-center justify-center">
                    <Music class="w-4 h-4 text-muted-foreground opacity-50" />
                  </div>
                {/if}
              </div>
            </div>
            <div class="flex items-center gap-2">
              <!-- Show skip buttons when any track is playing -->
              {#if $currentTrackId}
                <button
                  on:click={() => audioPlayer.playPreviousTrack(tracks)}
                  class="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors {$currentTrackId === track.id && index === 0 ? 'opacity-50' : ''}"
                  title="Previous track"
                  aria-label="Previous track"
                  disabled={$currentTrackId === track.id && index === 0}
                >
                  <SkipBack class="w-4 h-4" />
                </button>
              {/if}
              <button
                on:click={() => playTrack(track.id)}
                class="p-3 bg-primary text-primary-foreground rounded-full hover:opacity-90 transition-opacity flex items-center justify-center {$currentTrackId === track.id ? 'ring-2 ring-primary ring-offset-2' : ''}"
                title={$currentTrackId === track.id && $playing ? 'Pause' : 'Play'}
                aria-label={$currentTrackId === track.id && $playing ? 'Pause track' : 'Play track'}
              >
                {#if $currentTrackId === track.id && $playing}
                  <Pause class="w-5 h-5" />
                {:else}
                  <Play class="w-5 h-5" />
                {/if}
              </button>
              {#if $currentTrackId}
                <button
                  on:click={() => audioPlayer.playNextTrack(tracks)}
                  class="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors {$currentTrackId === track.id && index === tracks.length - 1 ? 'opacity-50' : ''}"
                  title="Next track"
                  aria-label="Next track"
                  disabled={$currentTrackId === track.id && index === tracks.length - 1}
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
