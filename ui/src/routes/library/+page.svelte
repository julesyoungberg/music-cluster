<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { api } from '$lib/services/api';
  import type { Track } from '$lib/types';
  import { Library, Search, ChevronLeft, ChevronRight, Loader2, Music, Play, Pause } from 'lucide-svelte';
  import TrackArtwork from '$lib/components/TrackArtwork.svelte';

  let tracks: Track[] = [];
  let loading = true;
  let limit = 100;
  let offset = 0;
  let total = 0;
  let searchQuery = '';
  let currentTrackId: number | null = null;
  let audio: HTMLAudioElement | null = null;
  let playing = false;

  async function loadTracks() {
    loading = true;
    try {
      const result = await api.getTracks(limit, offset);
      tracks = result.tracks;
      total = result.total;
    } catch (e) {
      console.error('Failed to load tracks:', e);
    } finally {
      loading = false;
    }
  }

  async function search() {
    if (!searchQuery.trim()) {
      loadTracks();
      return;
    }
    loading = true;
    try {
      const result = await api.search(searchQuery, undefined, limit);
      tracks = result.tracks;
      total = result.total;
    } catch (e) {
      console.error('Failed to search:', e);
    } finally {
      loading = false;
    }
  }

  function playTrack(trackId: number) {
    if (currentTrackId === trackId && audio) {
      if (playing) {
        audio.pause();
        playing = false;
      } else {
        audio.play();
        playing = true;
      }
      return;
    }

    // Stop current track
    if (audio) {
      audio.pause();
      audio.src = '';
      audio = null;
    }

    // Start new track
    currentTrackId = trackId;
    audio = new Audio(api.getTrackAudioUrl(trackId));
    audio.addEventListener('play', () => {
      playing = true;
    });
    audio.addEventListener('pause', () => {
      playing = false;
    });
    audio.addEventListener('ended', () => {
      playing = false;
      currentTrackId = null;
      audio = null;
    });
    audio.play().catch(() => {
      playing = false;
      currentTrackId = null;
      audio = null;
    });
  }

  onMount(() => {
    loadTracks();
    
    return () => {
      if (audio) {
        audio.pause();
        audio.src = '';
      }
    };
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
          on:keydown={(e) => e.key === 'Enter' && search()}
          placeholder="Search tracks..."
          class="w-full pl-10 pr-4 py-2 border rounded-lg bg-background"
        />
      </div>
      <button
        on:click={search}
        class="px-4 py-2 bg-primary text-primary-foreground rounded-lg flex items-center gap-2 hover:opacity-90 transition-opacity"
      >
        <Search class="w-4 h-4" />
        Search
      </button>
    </div>
  </div>

  {#if loading}
    <div class="text-center py-12 flex items-center justify-center gap-2">
      <Loader2 class="w-5 h-5 animate-spin" />
      <span>Loading...</span>
    </div>
  {:else if tracks.length > 0}
    <div class="space-y-2">
      {#each tracks as track}
        <div class="bg-card p-4 rounded-lg border hover:border-primary transition-colors">
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
            </div>
            <button
              on:click={() => playTrack(track.id)}
              class="p-3 bg-primary text-primary-foreground rounded-full hover:opacity-90 transition-opacity flex items-center justify-center"
              title={currentTrackId === track.id && playing ? 'Pause' : 'Play'}
            >
              {#if currentTrackId === track.id && playing}
                <Pause class="w-5 h-5" />
              {:else}
                <Play class="w-5 h-5" />
              {/if}
            </button>
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
