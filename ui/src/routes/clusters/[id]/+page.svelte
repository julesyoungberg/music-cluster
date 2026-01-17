<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/services/api';
  import type { Cluster, Track } from '$lib/types';

  let cluster: Cluster | null = null;
  let loading = true;
  let error: string | null = null;

  $: clusterId = $page.params.id ? parseInt($page.params.id) : 0;

  onMount(async () => {
    try {
      cluster = await api.getCluster(clusterId);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load cluster';
    } finally {
      loading = false;
    }
  });
</script>

<div class="container mx-auto p-8">
  {#if loading}
    <div class="text-center py-12">Loading...</div>
  {:else if error}
    <div class="bg-destructive/10 text-destructive p-4 rounded-lg">{error}</div>
  {:else if cluster}
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-4">
        {cluster.name || `Cluster ${cluster.cluster_index}`}
      </h1>
      <p class="text-muted-foreground">{cluster.size} tracks</p>
    </div>

    {#if cluster.tracks && cluster.tracks.length > 0}
      <div class="space-y-2">
        <h2 class="text-2xl font-semibold mb-4">Tracks</h2>
        {#each cluster.tracks as track}
          <div class="bg-card p-4 rounded-lg border">
            <p class="font-medium">{track.filename}</p>
            <p class="text-sm text-muted-foreground">{track.filepath}</p>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>
