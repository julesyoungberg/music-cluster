<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/services/api';
  import type { Cluster, Track } from '$lib/types';
  import { CircleDot, Music, ArrowLeft, Edit2, Save, X, Play, Pause, ChevronLeft, ChevronRight, Loader2, MoreVertical, Copy, FolderOpen } from 'lucide-svelte';
  import { goto } from '$app/navigation';
  import TrackArtwork from '$lib/components/TrackArtwork.svelte';
  import WaveformLoader from '$lib/components/WaveformLoader.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import { useAudioPlayer } from '$lib/composables/useAudioPlayer';
  import { addNotification } from '$lib/stores/notifications';
  import { resourceManager } from '$lib/services/resourceManager';

  let editingName = false;
  let editedName = '';
  let savingName = false;

  let cluster: Cluster | null = null;
  let loading = true;
  let error: string | null = null;
  let limit = 50;
  let offset = 0;
  let total = 0;
  let openMenuId: number | null = null;

  // Use audio player composable
  const audioPlayer = useAudioPlayer(
    undefined,
    () => {
      // Auto-play next track when current ends
      if (cluster?.tracks) {
        audioPlayer.playNextTrack(cluster.tracks);
      }
    },
    (error) => {
      // Error handler - notify user of playback errors
      addNotification('error', `Playback error: ${error}`);
    }
  );

  // Use stores directly (they're reactive)
  const { currentTrackId, playing, currentTime, duration } = audioPlayer;

  $: clusterId = $page.params.id ? parseInt($page.params.id) : 0;

  async function startEditName() {
    editingName = true;
    editedName = cluster?.name || '';
  }

  async function saveName() {
    if (!cluster) return;
    savingName = true;
    try {
      await api.renameCluster(cluster.id, editedName);
      cluster.name = editedName;
      editingName = false;
      addNotification('success', 'Cluster renamed successfully');
    } catch (e) {
      addNotification('error', e instanceof Error ? e.message : 'Failed to rename cluster');
    } finally {
      savingName = false;
    }
  }

  function cancelEdit() {
    editingName = false;
    editedName = '';
  }

  async function loadCluster() {
    loading = true;
    try {
      const result = await api.getCluster(clusterId, limit, offset);
      cluster = result;
      total = result.total || (result.tracks?.length || 0);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load cluster';
    } finally {
      loading = false;
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

  // Preload resources when cluster tracks change
  $: if (cluster?.tracks && cluster.tracks.length > 0 && !loading) {
    // Trigger preload
    preloadResources(cluster.tracks);
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
    loadCluster();
    document.addEventListener('click', handleClickOutside);
  });
  
  onDestroy(() => {
    // Clean up - resource manager will handle its own cache limits
    document.removeEventListener('click', handleClickOutside);
  });
</script>

<div class="container mx-auto p-8">
  {#if loading}
    <LoadingState message="Loading cluster..." />
  {:else if error}
    <ErrorState error={error} onRetry={loadCluster} />
  {:else if cluster}
    <div class="mb-8">
      <button
        on:click={() => goto('/clusters')}
        class="mb-4 flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft class="w-4 h-4" />
        Back to Clusters
      </button>
      <div class="flex items-center gap-3 mb-4">
        <div class="p-3 bg-primary/10 rounded-lg">
          <CircleDot class="w-8 h-8 text-primary" />
        </div>
        <div class="flex-1">
          {#if editingName}
            <div class="flex items-center gap-2">
              <input
                type="text"
                bind:value={editedName}
                class="text-4xl font-bold bg-background border rounded px-2 py-1 flex-1"
                on:keydown={(e) => e.key === 'Enter' && saveName()}
                on:keydown={(e) => e.key === 'Escape' && cancelEdit()}
                disabled={savingName}
              />
              <button
                on:click={saveName}
                disabled={savingName}
                class="p-2 text-green-600 hover:bg-green-500/10 rounded transition-colors disabled:opacity-50"
              >
                <Save class="w-5 h-5" />
              </button>
              <button
                on:click={cancelEdit}
                disabled={savingName}
                class="p-2 text-muted-foreground hover:bg-secondary rounded transition-colors"
              >
                <X class="w-5 h-5" />
              </button>
            </div>
          {:else}
            <div class="flex items-center gap-2">
              <h1 class="text-4xl font-bold">
                {cluster.name || `Cluster ${cluster.cluster_index}`}
              </h1>
              <button
                on:click={startEditName}
                class="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors"
                title="Rename cluster"
              >
                <Edit2 class="w-4 h-4" />
              </button>
            </div>
          {/if}
          <p class="text-muted-foreground flex items-center gap-2 mt-1">
            <Music class="w-4 h-4" />
            {total || cluster.size} tracks
          </p>
        </div>
      </div>
    </div>

    {#if cluster.tracks && cluster.tracks.length > 0}
      <div class="space-y-2">
        <h2 class="text-2xl font-semibold mb-4 flex items-center gap-2">
          <Music class="w-6 h-6" />
          Tracks
        </h2>
        {#each cluster.tracks as track, index (track.id)}
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
                      console.log('[cluster page] Seek event received:', { detail: seekTime, type: typeof seekTime, isNumber: typeof seekTime === 'number' });
                      
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

        {#if total > limit}
          <div class="mt-6 flex justify-between items-center">
            <button
              on:click={() => { offset = Math.max(0, offset - limit); loadCluster(); }}
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
              on:click={() => { offset += limit; loadCluster(); }}
              disabled={offset + limit >= total}
              class="px-4 py-2 bg-secondary rounded-lg disabled:opacity-50 flex items-center gap-2 hover:opacity-90 transition-opacity"
            >
              Next
              <ChevronRight class="w-4 h-4" />
            </button>
          </div>
        {/if}
      </div>
    {/if}
  {/if}
</div>
