<script lang="ts">
  import { api } from '$lib/services/api';
  import { goto } from '$app/navigation';
  import type { ClusterRequest } from '$lib/types';
  import { CircleDot, AlertCircle, Loader2, Play, ChevronDown } from 'lucide-svelte';

  import { userPreferences } from '$lib/stores/userPreferences';

  let request: ClusterRequest = {
    name: '',
    granularity: $userPreferences.defaultGranularity,
    algorithm: $userPreferences.defaultAlgorithm,
    min_size: $userPreferences.defaultMinClusterSize,
    max_clusters: 100,
    method: 'silhouette',
    show_metrics: true
  };

  let loading = false;
  let error: string | null = null;

  async function createCluster() {
    if (!request.name) {
      error = 'Please enter a name for the clustering';
      return;
    }

    loading = true;
    error = null;

    try {
      const result = await api.createCluster(request);
      goto(`/clusters`);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to create clustering';
    } finally {
      loading = false;
    }
  }
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <CircleDot class="w-10 h-10" />
    Create Clustering
  </h1>

  <div class="max-w-2xl space-y-6">
    {#if error}
      <div class="bg-destructive/10 text-destructive p-4 rounded-lg flex items-center gap-2">
        <AlertCircle class="w-5 h-5" />
        <span>{error}</span>
      </div>
    {/if}

    <div>
      <label for="cluster-name" class="block text-sm font-medium mb-2">Name</label>
      <input
        id="cluster-name"
        type="text"
        bind:value={request.name}
        class="w-full p-2 border rounded-lg bg-background"
        placeholder="Enter clustering name..."
        aria-required="true"
        aria-describedby="cluster-name-help"
      />
      <p id="cluster-name-help" class="text-xs text-muted-foreground mt-1">A unique name for this clustering</p>
    </div>

    <div>
      <label for="algorithm" class="block text-sm font-medium mb-2">Algorithm</label>
      <div class="relative">
        <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
        <select id="algorithm" bind:value={request.algorithm} class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none">
          <option value="kmeans">K-Means</option>
          <option value="hierarchical">Hierarchical</option>
          <option value="hdbscan">HDBSCAN</option>
        </select>
      </div>
    </div>

    <div>
      <label for="granularity" class="block text-sm font-medium mb-2">Granularity</label>
      <div class="relative">
        <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
        <select id="granularity" bind:value={request.granularity} class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none">
          <option value="fewer">Fewer (broader clusters)</option>
          <option value="less">Less</option>
          <option value="normal">Normal</option>
          <option value="more">More</option>
          <option value="finer">Finer (more specific clusters)</option>
        </select>
      </div>
    </div>

    <div>
      <label for="clusters" class="block text-sm font-medium mb-2">Exact Number of Clusters (optional)</label>
      <input
        id="clusters"
        type="number"
        bind:value={request.clusters}
        min="2"
        class="w-full p-2 border rounded-lg bg-background"
        placeholder="Leave empty for auto-detection"
      />
    </div>

    <div>
      <label for="min-size" class="block text-sm font-medium mb-2">Minimum Cluster Size</label>
      <input
        id="min-size"
        type="number"
        bind:value={request.min_size}
        min="1"
        class="w-full p-2 border rounded-lg bg-background"
      />
    </div>

    <div>
      <label class="flex items-center gap-2">
        <input type="checkbox" bind:checked={request.show_metrics} />
        <span>Show quality metrics</span>
      </label>
    </div>

    <button
      on:click={createCluster}
      disabled={loading || !request.name}
      class="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
      aria-label="Create new clustering"
    >
      {#if loading}
        <Loader2 class="w-4 h-4 animate-spin" />
        <span>Creating...</span>
      {:else}
        <Play class="w-4 h-4" />
        <span>Create Clustering</span>
      {/if}
    </button>
  </div>
</div>
