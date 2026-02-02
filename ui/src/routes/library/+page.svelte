<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { api } from '$lib/services/api';
  import type { Track } from '$lib/types';
  import { Library, Search, ChevronLeft, ChevronRight, Loader2, Music, Play, Pause, MoreVertical, Copy, FolderOpen } from 'lucide-svelte';
  import TrackArtwork from '$lib/components/TrackArtwork.svelte';
  import WaveformLoader from '$lib/components/WaveformLoader.svelte';
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
  let openMenuId: number | null = null;

  // Use audio player composable
  const audioPlayer = useAudioPlayer(
    (time) => {
      // Time update handler - waveform will react to currentTime changes
    },
    () => {
      // Auto-play next track when current ends
      audioPlayer.playNextTrack(tracks);
    },
    (error) => {
      // Error handler - notify user of playback errors
      addNotification('error', `Playback error: ${error}`);
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

  // Preload resources for visible tracks
  async function preloadResources(trackList: Track[]) {
    const trackIds = trackList.map(t => t.id);
    
    // Preload artwork in background (batched, 10 concurrent)
    resourceManager.preloadArtwork(trackIds, 10);
    
    // Preload waveforms for all visible tracks (batched, 5 concurrent)
    // WaveformLoader will handle individual loading, but preloading helps
    resourceManager.preloadWaveforms(trackIds, 200, 5);
  }

  function playTrack(trackId: number) {
    audioPlayer.playTrack(trackId);
    
    // Load waveform on-demand (will use resource manager's deduplication and cache)
    resourceManager.getWaveform(trackId, 200);
  }

  function seekTo(time: number) {
    // Ensure time is a number
    const seekTime = typeof time === 'number' ? time : Number(time);
    if (!isNaN(seekTime) && seekTime >= 0) {
      audioPlayer.seekTo(seekTime);
    } else {
      console.warn('Invalid seek time:', time);
    }
  }

  function toggleMenu(trackId: number) {
    openMenuId = openMenuId === trackId ? null : trackId;
  }

  function closeMenu() {
    openMenuId = null;
  }

  async function copyPath(filepath: string) {
    try {
      await navigator.clipboard.writeText(filepath);
      addNotification('success', 'Path copied to clipboard');
      closeMenu();
    } catch (e) {
      console.error('Failed to copy path:', e);
      addNotification('error', 'Failed to copy path');
    }
  }

  async function openInFinder(filepath: string) {
    try {
      // Use the Tauri API if available, otherwise try to open via file:// URL
      if (typeof window !== 'undefined' && (window as any).__TAURI__) {
        const { open } = await import('@tauri-apps/api/shell');
        await open(filepath);
      } else {
        // Fallback: try to open via file:// URL (may not work in all browsers)
        const fileUrl = `file://${filepath}`;
        window.open(fileUrl);
      }
      closeMenu();
    } catch (e) {
      console.error('Failed to open in Finder:', e);
      addNotification('error', 'Failed to open in Finder');
    }
  }

  // Close menu when clicking outside
  function handleClickOutside(event: MouseEvent) {
    const target = event.target as HTMLElement;
    if (!target.closest('.track-menu')) {
      closeMenu();
    }
  }

  onMount(() => {
    document.addEventListener('click', handleClickOutside);
  });

  onDestroy(() => {
    document.removeEventListener('click', handleClickOutside);
  });

  // Preload resources when tracks change
  $: if (tracks.length > 0 && !loading) {
    // Trigger preload
    preloadResources(tracks);
  }

  onMount(() => {
    loadTracks();
  });
  
  onDestroy(() => {
    // Clean up - resource manager will handle its own cache limits
    // But we can clear any component-specific state here if needed
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
              <div class="flex items-center gap-2">
                <p class="font-medium truncate flex-1">{track.filename}</p>
                <div class="relative track-menu">
                  <button
                    on:click|stopPropagation={() => toggleMenu(track.id)}
                    class="p-1 text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors"
                    title="Track options"
                    aria-label="Track options"
                  >
                    <MoreVertical class="w-4 h-4" />
                  </button>
                  {#if openMenuId === track.id}
                    <div class="absolute right-0 top-8 z-50 bg-popover border rounded-md shadow-lg min-w-[180px]">
                      <div class="p-1">
                        <button
                          on:click|stopPropagation={() => copyPath(track.filepath)}
                          class="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent rounded-sm transition-colors"
                        >
                          <Copy class="w-4 h-4" />
                          Copy path
                        </button>
                        {#if typeof window !== 'undefined' && (window.navigator.platform.includes('Mac') || window.navigator.userAgent.includes('Mac'))}
                          <button
                            on:click|stopPropagation={() => openInFinder(track.filepath)}
                            class="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent rounded-sm transition-colors"
                          >
                            <FolderOpen class="w-4 h-4" />
                            Open in Finder
                          </button>
                        {/if}
                      </div>
                    </div>
                  {/if}
                </div>
              </div>
              {#if track.cluster}
                <p class="text-sm text-primary mt-1">
                  Cluster: {track.cluster.name || `Cluster ${track.cluster.index}`}
                </p>
              {/if}
              
              <!-- Show waveform for tracks (auto-load all visible) -->
              <div class="mt-2">
                <WaveformLoader 
                  trackId={track.id}
                  currentTime={$currentTrackId === track.id ? $currentTime : 0}
                  autoLoad={true}
                  on:seek={(e) => {
                    // Get the seek time from event detail
                    // In Svelte, e.detail contains the dispatched value
                    const seekTime = e.detail;
                    console.log('[library page] Seek event received:', { detail: seekTime, type: typeof seekTime, isNumber: typeof seekTime === 'number' });
                    
                    // Validate it's a valid positive number
                    if (typeof seekTime === 'number' && !isNaN(seekTime) && isFinite(seekTime) && seekTime >= 0) {
                      if ($currentTrackId === track.id && $playing) {
                        // Already playing this track - just seek
                        seekTo(seekTime);
                      } else {
                        // Not playing or different track - start playing and seek
                        playTrack(track.id);
                        // seekTo will wait for audio to be ready, so we can call it immediately
                        // It will handle waiting for the audio to load
                        seekTo(seekTime);
                      }
                    } else {
                      console.warn('Invalid seek time from event:', seekTime, 'type:', typeof seekTime, 'isNaN:', isNaN(seekTime), 'isFinite:', isFinite(seekTime));
                    }
                  }}
                />
              </div>
            </div>
            <div class="flex items-center gap-2">
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
