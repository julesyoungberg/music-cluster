<script lang="ts">
  import { api } from '$lib/services/api';
  import { goto } from '$app/navigation';
  import type { ClusterRequest } from '$lib/types';

  let request: ClusterRequest = {
    name: '',
    granularity: 'normal',
    algorithm: 'kmeans',
    min_size: 3,
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
  <h1 class="text-4xl font-bold mb-8">Create Clustering</h1>

  <div class="max-w-2xl space-y-6">
    {#if error}
      <div class="bg-destructive/10 text-destructive p-4 rounded-lg">{error}</div>
    {/if}

    <div>
      <label for="cluster-name" class="block text-sm font-medium mb-2">Name</label>
      <input
        id="cluster-name"
        type="text"
        bind:value={request.name}
        class="w-full p-2 border rounded-lg bg-background"
        placeholder="Enter clustering name..."
      />
    </div>

    <div>
      <label for="algorithm" class="block text-sm font-medium mb-2">Algorithm</label>
      <select id="algorithm" bind:value={request.algorithm} class="w-full p-2 border rounded-lg bg-background">
        <option value="kmeans">K-Means</option>
        <option value="hierarchical">Hierarchical</option>
        <option value="hdbscan">HDBSCAN</option>
      </select>
    </div>

    <div>
      <label for="granularity" class="block text-sm font-medium mb-2">Granularity</label>
      <select id="granularity" bind:value={request.granularity} class="w-full p-2 border rounded-lg bg-background">
        <option value="fewer">Fewer (broader clusters)</option>
        <option value="less">Less</option>
        <option value="normal">Normal</option>
        <option value="more">More</option>
        <option value="finer">Finer (more specific clusters)</option>
      </select>
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
      class="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50"
    >
      {loading ? 'Creating...' : 'Create Clustering'}
    </button>
  </div>
</div>
