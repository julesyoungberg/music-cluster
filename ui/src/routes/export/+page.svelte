<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/services/api';
  import { clusterings } from '$lib/stores/clusters';
  import { addNotification } from '$lib/stores/notifications';
  import { Download, Loader2, FolderOpen, CheckCircle2 } from 'lucide-svelte';
  import { open } from '@tauri-apps/api/dialog';

  let selectedClusteringId: number | null = null;
  let outputPath = './playlists';
  let format = 'm3u';
  let relativePaths = false;
  let includeRepresentative = true;
  let exporting = false;

  onMount(async () => {
    const { clusterings: list } = await api.getClusterings();
    clusterings.set(list);
    if (list.length > 0) {
      selectedClusteringId = list[0].id;
    }
  });

  async function selectOutputDirectory() {
    try {
      const path = await open({
        directory: true,
        multiple: false
      });
      if (path && typeof path === 'string') {
        outputPath = path;
      }
    } catch (e) {
      console.error('Failed to open directory dialog:', e);
    }
  }

  async function exportPlaylists() {
    if (!selectedClusteringId) {
      addNotification('error', 'Please select a clustering');
      return;
    }

    exporting = true;
    try {
      const clustering = await api.getClustering(selectedClusteringId);
      const result = await api.export({
        output: outputPath,
        format,
        clustering: clustering.name || selectedClusteringId.toString(),
        relative_paths: relativePaths,
        include_representative: includeRepresentative
      });
      addNotification('success', `Exported ${result.files_created || 0} playlists to ${outputPath}`);
    } catch (e) {
      addNotification('error', e instanceof Error ? e.message : 'Failed to export playlists');
    } finally {
      exporting = false;
    }
  }
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <Download class="w-10 h-10" />
    Export Playlists
  </h1>

  <div class="max-w-2xl space-y-6">
    <div>
      <label for="clustering-select" class="block text-sm font-medium mb-2">Clustering</label>
      <select
        id="clustering-select"
        bind:value={selectedClusteringId}
        class="w-full p-2 border rounded-lg bg-background"
        disabled={exporting}
      >
        {#each $clusterings as clustering}
          <option value={clustering.id}>{clustering.name || `Clustering ${clustering.id}`}</option>
        {/each}
      </select>
    </div>

    <div>
      <label for="output-path" class="block text-sm font-medium mb-2">Output Directory</label>
      <div class="flex gap-2">
        <input
          id="output-path"
          type="text"
          bind:value={outputPath}
          class="flex-1 p-2 border rounded-lg bg-background"
          placeholder="Select output directory..."
          disabled={exporting}
        />
        <button
          on:click={selectOutputDirectory}
          disabled={exporting}
          class="px-4 py-2 bg-secondary rounded-lg hover:bg-secondary/80 transition-colors disabled:opacity-50 flex items-center gap-2"
        >
          <FolderOpen class="w-4 h-4" />
          Browse
        </button>
      </div>
    </div>

    <div>
      <label for="format-select" class="block text-sm font-medium mb-2">Format</label>
      <select
        id="format-select"
        bind:value={format}
        class="w-full p-2 border rounded-lg bg-background"
        disabled={exporting}
      >
        <option value="m3u">M3U</option>
        <option value="m3u8">M3U8 (UTF-8)</option>
        <option value="json">JSON</option>
      </select>
    </div>

    <div>
      <label class="flex items-center gap-2">
        <input type="checkbox" bind:checked={relativePaths} disabled={exporting} />
        <span>Use relative paths in playlists</span>
      </label>
    </div>

    <div>
      <label class="flex items-center gap-2">
        <input type="checkbox" bind:checked={includeRepresentative} disabled={exporting} />
        <span>Include representative track first in playlist</span>
      </label>
    </div>

    <button
      on:click={exportPlaylists}
      disabled={exporting || !selectedClusteringId}
      class="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
    >
      {#if exporting}
        <Loader2 class="w-4 h-4 animate-spin" />
        <span>Exporting...</span>
      {:else}
        <Download class="w-4 h-4" />
        <span>Export Playlists</span>
      {/if}
    </button>
  </div>
</div>
