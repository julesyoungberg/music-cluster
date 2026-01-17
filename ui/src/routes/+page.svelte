<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/services/api';
  import { clusterings } from '$lib/stores/clusters';
  import type { DatabaseInfo, Clustering } from '$lib/types';

  let info: DatabaseInfo | null = null;
  let loading = true;
  let error: string | null = null;

  onMount(async () => {
    try {
      info = await api.getInfo();
      const { clusterings: clusteringsList } = await api.getClusterings();
      clusterings.set(clusteringsList);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load data';
    } finally {
      loading = false;
    }
  });
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8">Music Cluster</h1>

  {#if loading}
    <div class="text-center py-12">Loading...</div>
  {:else if error}
    <div class="bg-destructive/10 text-destructive p-4 rounded-lg">
      Error: {error}
    </div>
  {:else if info}
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <div class="bg-card p-6 rounded-lg border">
        <h2 class="text-2xl font-semibold mb-2">{info.total_tracks}</h2>
        <p class="text-muted-foreground">Total Tracks</p>
      </div>
      <div class="bg-card p-6 rounded-lg border">
        <h2 class="text-2xl font-semibold mb-2">{info.analyzed_tracks}</h2>
        <p class="text-muted-foreground">Analyzed Tracks</p>
      </div>
      <div class="bg-card p-6 rounded-lg border">
        <h2 class="text-2xl font-semibold mb-2">{info.clusterings}</h2>
        <p class="text-muted-foreground">Clusterings</p>
      </div>
    </div>

    <div class="space-y-4">
      <h2 class="text-2xl font-semibold">Recent Clusterings</h2>
      {#each $clusterings.slice(0, 5) as clustering}
        <div class="bg-card p-4 rounded-lg border">
          <h3 class="font-semibold">{clustering.name || 'Unnamed'}</h3>
          <p class="text-sm text-muted-foreground">
            {clustering.num_clusters} clusters • {clustering.algorithm}
          </p>
        </div>
      {/each}
    </div>
  {/if}
</div>
