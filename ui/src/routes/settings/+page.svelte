<script lang="ts">
  import { theme } from '$lib/stores/settings';
  import { Settings, Palette, ChevronDown, Info, Trash2, RefreshCw } from 'lucide-svelte';
  import { cache } from '$lib/services/cache';
  import { addNotification } from '$lib/stores/notifications';
  import { onMount } from 'svelte';

  let stats = { memoryEntries: 0, storageSize: 0, storageEntries: 0 };
  let clearing = false;

  function updateStats() {
    stats = cache.getStats();
  }

  async function clearCache() {
    if (!confirm('Clear all cached artwork and waveform data? This will require re-downloading from the server.')) {
      return;
    }

    clearing = true;
    try {
      cache.clear();
      updateStats();
      addNotification('success', 'Cache cleared successfully');
    } catch (e) {
      addNotification('error', 'Failed to clear cache');
    } finally {
      clearing = false;
    }
  }

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  }

  onMount(() => {
    updateStats();
  });
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <Settings class="w-10 h-10" />
    Settings
  </h1>

  <div class="max-w-2xl space-y-6">
    <div>
      <label for="theme-select" class="block text-sm font-medium mb-2 flex items-center gap-2">
        <Palette class="w-4 h-4" />
        Theme
      </label>
      <div class="relative">
        <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
        <select
          id="theme-select"
          bind:value={$theme}
          class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none"
        >
          <option value="light">Light</option>
          <option value="dark">Dark</option>
          <option value="system">System</option>
        </select>
      </div>
    </div>

    <div class="bg-card p-6 rounded-lg border">
      <h2 class="text-xl font-semibold mb-4">Cache Management</h2>
      <p class="text-sm text-muted-foreground mb-4">
        Artwork and waveform data are cached locally to improve performance and reduce server load.
      </p>

      <div class="space-y-3 mb-6">
        <div class="flex justify-between items-center">
          <span class="text-sm text-muted-foreground">Memory Cache Entries</span>
          <span class="font-medium">{stats.memoryEntries}</span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-sm text-muted-foreground">Persistent Cache Entries</span>
          <span class="font-medium">{stats.storageEntries}</span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-sm text-muted-foreground">Persistent Cache Size</span>
          <span class="font-medium">{formatBytes(stats.storageSize)}</span>
        </div>
      </div>

      <div class="flex gap-3">
        <button
          on:click={updateStats}
          class="px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
        >
          <RefreshCw class="w-4 h-4" />
          Refresh Stats
        </button>
        <button
          on:click={clearCache}
          disabled={clearing}
          class="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center gap-2"
        >
          <Trash2 class="w-4 h-4" />
          {clearing ? 'Clearing...' : 'Clear Cache'}
        </button>
      </div>
    </div>

    <div class="bg-card p-4 rounded-lg border">
      <h2 class="font-semibold mb-2 flex items-center gap-2">
        <Info class="w-5 h-5" />
        About
      </h2>
      <p class="text-sm text-muted-foreground">Music Cluster v1.0.0</p>
    </div>
  </div>
</div>
