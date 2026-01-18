<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/services/api';
  import { clusterings, currentClustering } from '$lib/stores/clusters';
  import type { Clustering, Cluster } from '$lib/types';
  import { Network, Loader2, Music2, ChevronDown, Tag } from 'lucide-svelte';
  import ClusterVisualization from '$lib/components/ClusterVisualization.svelte';

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
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <Network class="w-10 h-10" />
    Clusters
  </h1>

  <div class="mb-6">
    <label for="clustering-select" class="block text-sm font-medium mb-2">Select Clustering</label>
    <div class="relative w-full max-w-xs">
      <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
      <select
        id="clustering-select"
        bind:value={selectedClusteringId}
        class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none"
      >
        {#each $clusterings as clustering}
          <option value={clustering.id}>{clustering.name || `Clustering ${clustering.id}`}</option>
        {/each}
      </select>
    </div>
  </div>

  {#if loading}
    <div class="text-center py-12 flex items-center justify-center gap-2">
      <Loader2 class="w-5 h-5 animate-spin" />
      <span>Loading clusters...</span>
    </div>
  {:else if clusters.length > 0}
    {#if selectedClusteringId}
      <div class="mb-8 flex items-center justify-between">
        <div class="flex-1">
          <ClusterVisualization clusteringId={selectedClusteringId} />
        </div>
        <div class="ml-4 flex flex-col gap-2">
          <a
            href="/clusters/{selectedClusteringId}/label"
            class="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
          >
            <Tag class="w-4 h-4" />
            Label Clusters
          </a>
        </div>
      </div>
    {/if}

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {#each clusters as cluster}
        <a
          href="/clusters/{cluster.id}"
          class="bg-card p-6 rounded-lg border hover:border-primary transition-colors group"
        >
          <div class="flex items-start gap-3">
            <div class="p-2 bg-primary/10 rounded-lg group-hover:bg-primary/20 transition-colors">
              <Network class="w-5 h-5 text-primary" />
            </div>
            <div class="flex-1 min-w-0">
              <h3 class="font-semibold text-lg mb-2 truncate">
                {cluster.name || `Cluster ${cluster.cluster_index}`}
              </h3>
              <p class="text-sm text-muted-foreground flex items-center gap-1">
                <Music2 class="w-4 h-4" />
                {cluster.size} tracks
              </p>
            </div>
          </div>
        </a>
      {/each}
    </div>
  {:else}
    <div class="text-center py-12 text-muted-foreground flex flex-col items-center gap-2">
      <Network class="w-12 h-12 opacity-50" />
      <span>No clusters found</span>
    </div>
  {/if}
</div>
