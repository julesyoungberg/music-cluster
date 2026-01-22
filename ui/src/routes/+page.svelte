<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/services/api';
  import { clusterings } from '$lib/stores/clusters';
  import type { DatabaseInfo, Clustering } from '$lib/types';
  import { Music, CheckCircle2, CircleDot } from 'lucide-svelte';
  import { addNotification } from '$lib/stores/notifications';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import ErrorState from '$lib/components/ErrorState.svelte';

  let loadData = async () => {
    loading = true;
    error = null;
    try {
      info = await api.getInfo();
      const { clusterings: clusteringsList } = await api.getClusterings();
      clusterings.set(clusteringsList);
      
      // Check if database is initialized
      if (info.total_tracks === 0 && info.analyzed_tracks === 0) {
        // Database might not be initialized - this is okay, just show the info
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load data';
      if (error.includes('no such table') || error.includes('database')) {
        addNotification('warning', 'Database not initialized. Please run initialization first.');
      }
    } finally {
      loading = false;
    }
  };

  let info: DatabaseInfo | null = null;
  let loading = true;
  let error: string | null = null;

  onMount(() => {
    loadData();
  });
</script>

<div class="container mx-auto p-8">
  {#if loading}
    <LoadingState message="Loading dashboard..." />
  {:else if error}
    <ErrorState error={error} onRetry={loadData} />
  {:else if info}
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <div class="bg-card p-6 rounded-lg border">
        <div class="flex items-center gap-3 mb-2">
          <Music class="w-6 h-6 text-primary" />
          <h2 class="text-2xl font-semibold">{info.total_tracks}</h2>
        </div>
        <p class="text-muted-foreground">Total Tracks</p>
      </div>
      <div class="bg-card p-6 rounded-lg border">
        <div class="flex items-center gap-3 mb-2">
          <CheckCircle2 class="w-6 h-6 text-primary" />
          <h2 class="text-2xl font-semibold">{info.analyzed_tracks}</h2>
        </div>
        <p class="text-muted-foreground">Analyzed Tracks</p>
      </div>
      <div class="bg-card p-6 rounded-lg border">
        <div class="flex items-center gap-3 mb-2">
          <CircleDot class="w-6 h-6 text-primary" />
          <h2 class="text-2xl font-semibold">{info.clusterings}</h2>
        </div>
        <p class="text-muted-foreground">Clusterings</p>
      </div>
    </div>

    <div class="space-y-4">
      <h2 class="text-2xl font-semibold flex items-center gap-2">
        <CircleDot class="w-6 h-6" />
        Recent Clusterings
      </h2>
      {#each $clusterings.slice(0, 5) as clustering}
        <div class="bg-card p-4 rounded-lg border hover:border-primary transition-colors">
          <h3 class="font-semibold">{clustering.name || 'Unnamed'}</h3>
          <p class="text-sm text-muted-foreground">
            {clustering.num_clusters} clusters • {clustering.algorithm}
          </p>
        </div>
      {/each}
    </div>
  {/if}
</div>
