<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/services/api';
  import { goto } from '$app/navigation';
  import { addNotification } from '$lib/stores/notifications';
  import { Tag, Loader2, CheckCircle2, AlertCircle, ArrowLeft } from 'lucide-svelte';
  import type { Clustering } from '$lib/types';

  let clusteringId: number | null = null;
  let clustering: Clustering | null = null;
  let loading = false;
  let labeling = false;
  let options = {
    no_genre: false,
    no_bpm: false,
    no_descriptors: false,
    bpm_average: false,
    dry_run: false
  };
  let previewResults: Array<{ cluster_index: number; name: string; size: number }> | null = null;

  $: clusteringId = $page.params.id ? parseInt($page.params.id) : null;

  onMount(async () => {
    if (clusteringId) {
      try {
        clustering = await api.getClustering(clusteringId);
      } catch (e) {
        addNotification('error', 'Failed to load clustering');
      }
    }
  });

  async function previewLabels() {
    if (!clusteringId) return;
    loading = true;
    try {
      const result = await api.labelClusters(clusteringId, { ...options, dry_run: true });
      previewResults = result.names || [];
      addNotification('info', `Preview generated for ${previewResults.length} clusters`);
    } catch (e) {
      addNotification('error', e instanceof Error ? e.message : 'Failed to preview labels');
    } finally {
      loading = false;
    }
  }

  async function applyLabels() {
    if (!clusteringId) return;
    labeling = true;
    try {
      const result = await api.labelClusters(clusteringId, { ...options, dry_run: false });
      addNotification('success', `Successfully labeled ${result.names?.length || 0} clusters`);
      goto(`/clusters`);
    } catch (e) {
      addNotification('error', e instanceof Error ? e.message : 'Failed to label clusters');
    } finally {
      labeling = false;
    }
  }
</script>

<div class="container mx-auto p-8">
  <button
    on:click={() => goto('/clusters')}
    class="mb-4 flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
  >
    <ArrowLeft class="w-4 h-4" />
    Back to Clusters
  </button>
  <div class="mb-8">
    <h1 class="text-4xl font-bold mb-4 flex items-center gap-3">
      <Tag class="w-10 h-10" />
      Label Clusters
    </h1>
    {#if clustering}
      <p class="text-muted-foreground">
        Auto-generate descriptive names for clusters in "{clustering.name || `Clustering ${clustering.id}`}" based on genre, BPM, and characteristics
      </p>
    {/if}
  </div>

  <div class="max-w-4xl space-y-6">
    <div class="bg-card p-6 rounded-lg border space-y-4">
      <h2 class="text-xl font-semibold">Naming Options</h2>
      <div class="space-y-3">
        <label class="flex items-center gap-2">
          <input type="checkbox" bind:checked={options.no_genre} />
          <span>Exclude genre classification</span>
        </label>
        <label class="flex items-center gap-2">
          <input type="checkbox" bind:checked={options.no_bpm} />
          <span>Exclude BPM information</span>
        </label>
        <label class="flex items-center gap-2">
          <input type="checkbox" bind:checked={options.no_descriptors} />
          <span>Exclude characteristics (Bass-Heavy, Dark, etc.)</span>
        </label>
        <label class="flex items-center gap-2">
          <input type="checkbox" bind:checked={options.bpm_average} />
          <span>Use average BPM instead of range</span>
        </label>
      </div>
    </div>

    <div class="flex gap-4">
      <button
        on:click={previewLabels}
        disabled={loading || !clusteringId}
        class="px-4 py-2 bg-secondary rounded-lg hover:bg-secondary/80 transition-colors disabled:opacity-50 flex items-center gap-2"
      >
        {#if loading}
          <Loader2 class="w-4 h-4 animate-spin" />
        {/if}
        Preview Names
      </button>
      <button
        on:click={applyLabels}
        disabled={labeling || !clusteringId || !previewResults}
        class="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center gap-2"
      >
        {#if labeling}
          <Loader2 class="w-4 h-4 animate-spin" />
        {:else}
          <CheckCircle2 class="w-4 h-4" />
        {/if}
        Apply Labels
      </button>
    </div>

    {#if previewResults}
      <div class="bg-card p-6 rounded-lg border">
        <h2 class="text-xl font-semibold mb-4">Preview</h2>
        <div class="space-y-2">
          {#each previewResults as result}
            <div class="flex items-center justify-between p-3 bg-secondary/50 rounded">
              <div>
                <span class="font-medium">Cluster {result.cluster_index}:</span>
                <span class="ml-2">{result.name}</span>
              </div>
              <span class="text-sm text-muted-foreground">{result.size} tracks</span>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
</div>
