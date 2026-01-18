<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/services/api';
  import type { Cluster, Track } from '$lib/types';
  import { Network, Loader2, AlertCircle, Music, ArrowLeft, Edit2, Save, X } from 'lucide-svelte';
  import { goto } from '$app/navigation';
  import TrackArtwork from '$lib/components/TrackArtwork.svelte';
  import { api } from '$lib/services/api';
  import { addNotification } from '$lib/stores/notifications';

  let editingName = false;
  let editedName = '';
  let savingName = false;

  let cluster: Cluster | null = null;
  let loading = true;
  let error: string | null = null;

  $: clusterId = $page.params.id ? parseInt($page.params.id) : 0;

  async function startEditName() {
    editingName = true;
    editedName = cluster?.name || '';
  }

  async function saveName() {
    if (!cluster) return;
    savingName = true;
    try {
      await api.renameCluster(cluster.id, editedName);
      cluster.name = editedName;
      editingName = false;
      addNotification('success', 'Cluster renamed successfully');
    } catch (e) {
      addNotification('error', e instanceof Error ? e.message : 'Failed to rename cluster');
    } finally {
      savingName = false;
    }
  }

  function cancelEdit() {
    editingName = false;
    editedName = '';
  }

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
    <div class="text-center py-12 flex items-center justify-center gap-2">
      <Loader2 class="w-5 h-5 animate-spin" />
      <span>Loading...</span>
    </div>
  {:else if error}
    <div class="bg-destructive/10 text-destructive p-4 rounded-lg flex items-center gap-2">
      <AlertCircle class="w-5 h-5" />
      <span>{error}</span>
    </div>
  {:else if cluster}
    <div class="mb-8">
      <button
        on:click={() => goto('/clusters')}
        class="mb-4 flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft class="w-4 h-4" />
        Back to Clusters
      </button>
      <div class="flex items-center gap-3 mb-4">
        <div class="p-3 bg-primary/10 rounded-lg">
          <Network class="w-8 h-8 text-primary" />
        </div>
        <div class="flex-1">
          {#if editingName}
            <div class="flex items-center gap-2">
              <input
                type="text"
                bind:value={editedName}
                class="text-4xl font-bold bg-background border rounded px-2 py-1 flex-1"
                on:keydown={(e) => e.key === 'Enter' && saveName()}
                on:keydown={(e) => e.key === 'Escape' && cancelEdit()}
                disabled={savingName}
              />
              <button
                on:click={saveName}
                disabled={savingName}
                class="p-2 text-green-600 hover:bg-green-500/10 rounded transition-colors disabled:opacity-50"
              >
                <Save class="w-5 h-5" />
              </button>
              <button
                on:click={cancelEdit}
                disabled={savingName}
                class="p-2 text-muted-foreground hover:bg-secondary rounded transition-colors"
              >
                <X class="w-5 h-5" />
              </button>
            </div>
          {:else}
            <div class="flex items-center gap-2">
              <h1 class="text-4xl font-bold">
                {cluster.name || `Cluster ${cluster.cluster_index}`}
              </h1>
              <button
                on:click={startEditName}
                class="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors"
                title="Rename cluster"
              >
                <Edit2 class="w-4 h-4" />
              </button>
            </div>
          {/if}
          <p class="text-muted-foreground flex items-center gap-2 mt-1">
            <Music class="w-4 h-4" />
            {cluster.size} tracks
          </p>
        </div>
      </div>
    </div>

    {#if cluster.tracks && cluster.tracks.length > 0}
      <div class="space-y-2">
        <h2 class="text-2xl font-semibold mb-4 flex items-center gap-2">
          <Music class="w-6 h-6" />
          Tracks
        </h2>
        {#each cluster.tracks as track}
          <div class="bg-card p-4 rounded-lg border hover:border-primary transition-colors">
            <div class="flex items-center gap-4">
              <TrackArtwork track={track} size={64} />
              <div class="flex-1 min-w-0">
                <p class="font-medium truncate">{track.filename}</p>
                <p class="text-sm text-muted-foreground truncate">{track.filepath}</p>
              </div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>
