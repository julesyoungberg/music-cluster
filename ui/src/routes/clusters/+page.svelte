<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/services/api';
  import { clusterings, currentClustering } from '$lib/stores/clusters';
  import type { Clustering, Cluster } from '$lib/types';

  let selectedClusteringId: number | null = null;
  let clusters: Cluster[] = [];
  let loading = false;

  $: if (selectedClusteringId) {
    loadClusters();
  }

  async function loadClusters() {
    if (!selectedClusteringId) return;
    loading = true;
    try {
      const clustering = await api.getClustering(selectedClusteringId);
      clusters = clustering.clusters || [];
      currentClustering.set(clustering);
    } catch (e) {
      console.error('Failed to load clusters:', e);
    } finally {
      loading = false;
    }
  }

  onMount(async () => {
    const { clusterings: list } = await api.getClusterings();
    clusterings.set(list);
    if (list.length > 0) {
      selectedClusteringId = list[0].id;
    }
  });
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8">Clusters</h1>

  <div class="mb-6">
    <label for="clustering-select" class="block text-sm font-medium mb-2">Select Clustering</label>
    <select
      id="clustering-select"
      bind:value={selectedClusteringId}
      class="w-full max-w-xs p-2 border rounded-lg bg-background"
    >
      {#each $clusterings as clustering}
        <option value={clustering.id}>{clustering.name || `Clustering ${clustering.id}`}</option>
      {/each}
    </select>
  </div>

  {#if loading}
    <div class="text-center py-12">Loading clusters...</div>
  {:else if clusters.length > 0}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {#each clusters as cluster}
        <a
          href="/clusters/{cluster.id}"
          class="bg-card p-6 rounded-lg border hover:border-primary transition-colors"
        >
          <h3 class="font-semibold text-lg mb-2">
            {cluster.name || `Cluster ${cluster.cluster_index}`}
          </h3>
          <p class="text-sm text-muted-foreground">{cluster.size} tracks</p>
        </a>
      {/each}
    </div>
  {:else}
    <div class="text-center py-12 text-muted-foreground">No clusters found</div>
  {/if}
</div>
