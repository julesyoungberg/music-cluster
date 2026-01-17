<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/services/api';
  import type { Track } from '$lib/types';

  let tracks: Track[] = [];
  let loading = true;
  let limit = 100;
  let offset = 0;
  let total = 0;
  let searchQuery = '';

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

  onMount(() => {
    loadTracks();
  });
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8">Library</h1>

  <div class="mb-6">
    <div class="flex gap-2">
      <input
        type="text"
        bind:value={searchQuery}
        on:keydown={(e) => e.key === 'Enter' && search()}
        placeholder="Search tracks..."
        class="flex-1 p-2 border rounded-lg bg-background"
      />
      <button
        on:click={search}
        class="px-4 py-2 bg-primary text-primary-foreground rounded-lg"
      >
        Search
      </button>
    </div>
  </div>

  {#if loading}
    <div class="text-center py-12">Loading...</div>
  {:else if tracks.length > 0}
    <div class="space-y-2">
      {#each tracks as track}
        <div class="bg-card p-4 rounded-lg border">
          <p class="font-medium">{track.filename}</p>
          <p class="text-sm text-muted-foreground">{track.filepath}</p>
          {#if track.cluster}
            <p class="text-sm text-primary">
              Cluster: {track.cluster.name || `Cluster ${track.cluster.index}`}
            </p>
          {/if}
        </div>
      {/each}
    </div>

    <div class="mt-6 flex justify-between items-center">
      <button
        on:click={() => { offset = Math.max(0, offset - limit); loadTracks(); }}
        disabled={offset === 0}
        class="px-4 py-2 bg-secondary rounded-lg disabled:opacity-50"
      >
        Previous
      </button>
      <span class="text-sm text-muted-foreground">
        Showing {offset + 1} - {Math.min(offset + limit, total)} of {total}
      </span>
      <button
        on:click={() => { offset += limit; loadTracks(); }}
        disabled={offset + limit >= total}
        class="px-4 py-2 bg-secondary rounded-lg disabled:opacity-50"
      >
        Next
      </button>
    </div>
  {:else}
    <div class="text-center py-12 text-muted-foreground">No tracks found</div>
  {/if}
</div>
