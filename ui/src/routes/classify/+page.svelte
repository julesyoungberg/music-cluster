<script lang="ts">
  import { api } from '$lib/services/api';
  import { clusterings } from '$lib/stores/clusters';
  import { addNotification } from '$lib/stores/notifications';
  import { Scan, FolderOpen, Loader2, CheckCircle2, Music } from 'lucide-svelte';
  import { open } from '@tauri-apps/api/dialog';
  import { onMount } from 'svelte';

  let path = '';
  let selectedClusteringId: number | null = null;
  let recursive = true;
  let threshold: number | null = null;
  let classifying = false;
  let results: Array<{
    filepath: string;
    filename: string;
    status: string;
    cluster_id?: number;
    cluster_index?: number;
    cluster_name?: string;
    distance?: number;
    error?: string;
  }> | null = null;

  onMount(async () => {
    const { clusterings: list } = await api.getClusterings();
    clusterings.set(list);
    if (list.length > 0) {
      selectedClusteringId = list[0].id;
    }
  });

  async function selectDirectory() {
    try {
      const selectedPath = await open({
        directory: true,
        multiple: false
      });
      if (selectedPath && typeof selectedPath === 'string') {
        path = selectedPath;
      }
    } catch (e) {
      console.error('Failed to open directory dialog:', e);
    }
  }

  async function classify() {
    if (!path) {
      addNotification('error', 'Please select a directory or file');
      return;
    }

    classifying = true;
    results = null;
    try {
      const clustering = selectedClusteringId
        ? await api.getClustering(selectedClusteringId)
        : null;
      const result = await api.classify(
        path,
        clustering?.name || selectedClusteringId?.toString(),
        threshold || undefined,
        recursive
      );
      results = result.results || [];
      const successCount = results.filter((r) => r.status === 'classified').length;
      if (successCount > 0) {
        addNotification('success', `Classified ${successCount} tracks`);
      } else {
        addNotification('warning', 'No tracks were classified');
      }
    } catch (e) {
      addNotification('error', e instanceof Error ? e.message : 'Failed to classify tracks');
    } finally {
      classifying = false;
    }
  }
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <Scan class="w-10 h-10" />
    Classify Tracks
  </h1>

  <div class="max-w-4xl space-y-6">
    <div class="bg-card p-6 rounded-lg border space-y-4">
      <div>
        <label for="path-input" class="block text-sm font-medium mb-2">Directory or File</label>
        <div class="flex gap-2">
          <input
            id="path-input"
            type="text"
            bind:value={path}
            class="flex-1 p-2 border rounded-lg bg-background"
            placeholder="Select directory or file..."
            disabled={classifying}
          />
          <button
            on:click={selectDirectory}
            disabled={classifying}
            class="px-4 py-2 bg-secondary rounded-lg hover:bg-secondary/80 transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            <FolderOpen class="w-4 h-4" />
            Browse
          </button>
        </div>
      </div>

      <div>
        <label for="clustering-select" class="block text-sm font-medium mb-2">Clustering</label>
        <select
          id="clustering-select"
          bind:value={selectedClusteringId}
          class="w-full p-2 border rounded-lg bg-background"
          disabled={classifying}
        >
          <option value={null}>Latest</option>
          {#each $clusterings as clustering}
            <option value={clustering.id}>{clustering.name || `Clustering ${clustering.id}`}</option>
          {/each}
        </select>
      </div>

      <div>
        <label for="threshold-input" class="block text-sm font-medium mb-2">
          Distance Threshold (optional)
        </label>
        <input
          id="threshold-input"
          type="number"
          bind:value={threshold}
          step="0.1"
          min="0"
          class="w-full p-2 border rounded-lg bg-background"
          placeholder="Leave empty for no threshold"
          disabled={classifying}
        />
        <p class="text-xs text-muted-foreground mt-1">
          Maximum distance to centroid for classification. Leave empty to classify all tracks.
        </p>
      </div>

      <div>
        <label class="flex items-center gap-2">
          <input type="checkbox" bind:checked={recursive} disabled={classifying} />
          <span>Recursive scan</span>
        </label>
      </div>

      <button
        on:click={classify}
        disabled={classifying || !path}
        class="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
      >
        {#if classifying}
          <Loader2 class="w-4 h-4 animate-spin" />
          <span>Classifying...</span>
        {:else}
          <Scan class="w-4 h-4" />
          <span>Classify Tracks</span>
        {/if}
      </button>
    </div>

    {#if results}
      <div class="bg-card p-6 rounded-lg border">
        <h2 class="text-xl font-semibold mb-4">Results</h2>
        <div class="space-y-2">
          {#each results as result}
            <div class="p-3 bg-secondary/50 rounded">
              <div class="flex items-center gap-2 mb-1">
                <Music class="w-4 h-4 text-muted-foreground" />
                <span class="font-medium">{result.filename}</span>
                {#if result.status === 'classified'}
                  <span class="px-2 py-0.5 bg-green-500/10 text-green-600 dark:text-green-400 rounded text-xs">
                    Classified
                  </span>
                {:else if result.status === 'no_match'}
                  <span class="px-2 py-0.5 bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 rounded text-xs">
                    No Match
                  </span>
                {:else}
                  <span class="px-2 py-0.5 bg-destructive/10 text-destructive rounded text-xs">
                    Error
                  </span>
                {/if}
              </div>
              {#if result.cluster_name}
                <p class="text-sm text-muted-foreground">
                  → {result.cluster_name || `Cluster ${result.cluster_index}`}
                  {#if result.distance !== undefined}
                    <span class="ml-2">(distance: {result.distance.toFixed(3)})</span>
                  {/if}
                </p>
              {/if}
              {#if result.error}
                <p class="text-sm text-destructive">{result.error}</p>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
</div>
