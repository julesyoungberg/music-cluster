<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/services/api';
  import { clusterings } from '$lib/stores/clusters';
  import { addNotification } from '$lib/stores/notifications';
  import { GitCompare, Loader2, AlertCircle } from 'lucide-svelte';

  let clustering1Id: number | null = null;
  let clustering2Id: number | null = null;
  let comparing = false;
  let comparisonResult: any = null;

  onMount(async () => {
    const { clusterings: list } = await api.getClusterings();
    clusterings.set(list);
  });

  async function compare() {
    if (!clustering1Id || !clustering2Id) {
      addNotification('error', 'Please select both clusterings to compare');
      return;
    }

    if (clustering1Id === clustering2Id) {
      addNotification('error', 'Please select two different clusterings');
      return;
    }

    comparing = true;
    comparisonResult = null;
    try {
      const clustering1 = await api.getClustering(clustering1Id);
      const clustering2 = await api.getClustering(clustering2Id);
      const result = await api.compare(
        clustering1.name || clustering1Id.toString(),
        clustering2.name || clustering2Id.toString()
      );
      comparisonResult = result;
      addNotification('success', 'Comparison completed');
    } catch (e) {
      addNotification('error', e instanceof Error ? e.message : 'Failed to compare clusterings');
    } finally {
      comparing = false;
    }
  }
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <GitCompare class="w-10 h-10" />
    Compare Clusterings
  </h1>

  <div class="max-w-4xl space-y-6">
    <div class="bg-card p-6 rounded-lg border space-y-4">
      <div>
        <label for="clustering1-select" class="block text-sm font-medium mb-2">First Clustering</label>
        <select
          id="clustering1-select"
          bind:value={clustering1Id}
          class="w-full p-2 border rounded-lg bg-background"
          disabled={comparing}
        >
          <option value={null}>Select clustering...</option>
          {#each $clusterings as clustering}
            <option value={clustering.id}>{clustering.name || `Clustering ${clustering.id}`}</option>
          {/each}
        </select>
      </div>

      <div>
        <label for="clustering2-select" class="block text-sm font-medium mb-2">Second Clustering</label>
        <select
          id="clustering2-select"
          bind:value={clustering2Id}
          class="w-full p-2 border rounded-lg bg-background"
          disabled={comparing}
        >
          <option value={null}>Select clustering...</option>
          {#each $clusterings as clustering}
            <option value={clustering.id}>{clustering.name || `Clustering ${clustering.id}`}</option>
          {/each}
        </select>
      </div>

      <button
        on:click={compare}
        disabled={comparing || !clustering1Id || !clustering2Id}
        class="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
      >
        {#if comparing}
          <Loader2 class="w-4 h-4 animate-spin" />
          <span>Comparing...</span>
        {:else}
          <GitCompare class="w-4 h-4" />
          <span>Compare</span>
        {/if}
      </button>
    </div>

    {#if comparisonResult}
      <div class="bg-card p-6 rounded-lg border">
        <h2 class="text-xl font-semibold mb-4">Comparison Results</h2>
        <pre class="bg-secondary p-4 rounded text-sm overflow-auto">{JSON.stringify(comparisonResult, null, 2)}</pre>
      </div>
    {/if}
  </div>
</div>
