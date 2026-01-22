<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { api } from '$lib/services/api';
  import { clusterings, currentClustering } from '$lib/stores/clusters';
  import type { Clustering, Cluster } from '$lib/types';
  import { CircleDot, Loader2, Music2, ChevronDown, Tag, Download, Play, GitCompareArrows, BarChart3, List, Trash2, AlertTriangle } from 'lucide-svelte';
  import ClusterVisualization from '$lib/components/ClusterVisualization.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import { goto } from '$app/navigation';
  import { addNotification } from '$lib/stores/notifications';

  let selectedClusteringId: number | null = null;
  let clusters: Cluster[] = [];
  let loading = false;
  let activeTab: 'visualization' | 'list' = 'list';
  let showDeleteModal = false;
  let deleting = false;

  // Sort clusterings by created_at in reverse chronological order (newest first)
  $: sortedClusterings = $clusterings.slice().sort((a, b) => {
    const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
    const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
    return dateB - dateA; // Descending order (newest first)
  });

  // Format date for display
  function formatDate(dateString?: string): string {
    if (!dateString) return 'Unknown date';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

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

  async function loadClusterings() {
    const { clusterings: list } = await api.getClusterings();
    clusterings.set(list);
    // Sort and select the first (newest) one if current selection is deleted
    const sorted = list.slice().sort((a, b) => {
      const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
      const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
      return dateB - dateA;
    });
    
    // If current selection was deleted or doesn't exist, select first one
    if (!selectedClusteringId || !list.find(c => c.id === selectedClusteringId)) {
      if (sorted.length > 0) {
        selectedClusteringId = sorted[0].id;
      } else {
        selectedClusteringId = null;
      }
    }
  }

  function openDeleteModal() {
    showDeleteModal = true;
  }

  function closeDeleteModal() {
    if (!deleting) {
      showDeleteModal = false;
    }
  }

  async function confirmDelete() {
    if (!selectedClusteringId) return;
    
    deleting = true;
    try {
      await api.deleteClustering(selectedClusteringId);
      addNotification('success', 'Clustering deleted successfully');
      showDeleteModal = false;
      
      // Reload clusterings list
      await loadClusterings();
      
      // Clear current clustering if it was deleted
      if (currentClustering && currentClustering.id === selectedClusteringId) {
        currentClustering.set(null);
      }
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : 'Failed to delete clustering';
      addNotification('error', errorMsg);
    } finally {
      deleting = false;
    }
  }

  onMount(async () => {
    await loadClusterings();
  });
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <CircleDot class="w-10 h-10" />
    Clusters
  </h1>

  <div class="mb-6">
    <label for="clustering-select" class="block text-sm font-medium mb-2">Select Clustering</label>
    <div class="flex items-center gap-4">
      <div class="flex-1 max-w-xs">
        <div class="relative">
          <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
          <select
            id="clustering-select"
            bind:value={selectedClusteringId}
            class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none"
          >
            {#each sortedClusterings as clustering}
              <option value={clustering.id}>{clustering.name || `Clustering ${clustering.id}`}</option>
            {/each}
          </select>
        </div>
      </div>
      {#if selectedClusteringId && sortedClusterings.length > 0}
        <button
          on:click={openDeleteModal}
          class="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2 disabled:opacity-50"
          disabled={deleting}
          aria-label="Delete clustering"
          title="Delete clustering"
        >
          <Trash2 class="w-4 h-4" />
          Delete
        </button>
      {/if}
    </div>
    {#if $currentClustering?.created_at}
      <p class="text-sm text-muted-foreground mt-2">
        Created: {formatDate($currentClustering.created_at)}
      </p>
    {/if}
  </div>

  <!-- Delete Confirmation Modal -->
  <Modal open={showDeleteModal} title="Delete Clustering" on:close={closeDeleteModal}>
    <div class="space-y-4">
      <div class="flex items-start gap-3">
        <div class="p-2 bg-destructive/10 rounded-lg">
          <AlertTriangle class="w-5 h-5 text-destructive" />
        </div>
        <div class="flex-1">
          <p class="text-sm font-medium mb-2">Are you sure you want to delete this clustering?</p>
          <p class="text-sm text-muted-foreground">
            This will permanently delete <strong>"{$currentClustering?.name || `Clustering ${selectedClusteringId}`}"</strong> and all its clusters. This action cannot be undone.
          </p>
        </div>
      </div>
      <div class="flex gap-2 justify-end pt-4">
        <button
          on:click={closeDeleteModal}
          class="px-4 py-2 text-sm bg-secondary text-secondary-foreground rounded-lg hover:opacity-90 transition-opacity"
          disabled={deleting}
        >
          Cancel
        </button>
        <button
          on:click={confirmDelete}
          class="px-4 py-2 text-sm bg-destructive text-destructive-foreground rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center gap-2"
          disabled={deleting}
        >
          {#if deleting}
            <Loader2 class="w-4 h-4 animate-spin" />
          {/if}
          Delete Clustering
        </button>
      </div>
    </div>
  </Modal>

  {#if loading}
    <div class="text-center py-12 flex items-center justify-center gap-2">
      <Loader2 class="w-5 h-5 animate-spin" />
      <span>Loading clusters...</span>
    </div>
  {:else if clusters.length > 0 && selectedClusteringId}
    <!-- Action buttons -->
    <div class="mb-6 flex flex-wrap gap-2">
      <a
        href="/clusters/{selectedClusteringId}/label"
        class="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
      >
        <Tag class="w-4 h-4" />
        Label Clusters
      </a>
      <button
        on:click={() => goto('/export?clustering=' + ($currentClustering?.name || selectedClusteringId))}
        class="px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
      >
        <Download class="w-4 h-4" />
        Export Playlists
      </button>
      <button
        on:click={() => goto('/classify?clustering=' + ($currentClustering?.name || selectedClusteringId))}
        class="px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
      >
        <Play class="w-4 h-4" />
        Classify Tracks
      </button>
      <button
        on:click={() => goto('/compare?clustering1=' + ($currentClustering?.name || selectedClusteringId))}
        class="px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
      >
        <GitCompareArrows class="w-4 h-4" />
        Compare Clusterings
      </button>
    </div>

    <!-- Tabs -->
    <div class="mb-6 border-b">
      <div class="flex gap-1">
        <button
          on:click={() => activeTab = 'list'}
          class="px-4 py-2 font-medium transition-colors flex items-center gap-2 {activeTab === 'list' ? 'border-b-2 border-primary text-primary' : 'text-muted-foreground hover:text-foreground'}"
        >
          <List class="w-4 h-4" />
          Clusters List
        </button>
        <button
          on:click={() => activeTab = 'visualization'}
          class="px-4 py-2 font-medium transition-colors flex items-center gap-2 {activeTab === 'visualization' ? 'border-b-2 border-primary text-primary' : 'text-muted-foreground hover:text-foreground'}"
        >
          <BarChart3 class="w-4 h-4" />
          Visualization
        </button>
      </div>
    </div>

    <!-- Tab content -->
    {#if activeTab === 'list'}
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {#each clusters as cluster}
          <a
            href="/clusters/{cluster.id}"
            class="bg-card p-6 rounded-lg border hover:border-primary transition-colors group"
          >
            <div class="flex items-start gap-3">
              <div class="p-2 bg-primary/10 rounded-lg group-hover:bg-primary/20 transition-colors">
                <CircleDot class="w-5 h-5 text-primary" />
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
    {:else if activeTab === 'visualization'}
      <div class="bg-card p-6 rounded-lg border">
        <ClusterVisualization clusteringId={selectedClusteringId} />
      </div>
    {/if}
  {:else if selectedClusteringId}
    <div class="text-center py-12 text-muted-foreground flex flex-col items-center gap-2">
      <CircleDot class="w-12 h-12 opacity-50" />
      <span>No clusters found</span>
    </div>
  {/if}
</div>
