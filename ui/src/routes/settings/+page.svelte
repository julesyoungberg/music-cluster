<script lang="ts">
  import { theme } from '$lib/stores/settings';
  import { Settings, Palette, ChevronDown, Info, Trash2, RefreshCw, Layout, Volume2, Database, Download, CircleDot, RotateCcw } from 'lucide-svelte';
  import { cache } from '$lib/services/cache';
  import { userPreferences } from '$lib/stores/userPreferences';
  import { addNotification } from '$lib/stores/notifications';
  import { onMount } from 'svelte';

  let stats = { memoryEntries: 0, storageSize: 0, storageEntries: 0 };
  let clearing = false;
  let activeSection: 'appearance' | 'playback' | 'display' | 'performance' | 'clustering' | 'export' | 'cache' | 'about' = 'appearance';

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

  <div class="flex gap-6 max-w-6xl">
    <!-- Sidebar Navigation -->
    <div class="w-64 flex-shrink-0">
      <div class="bg-card rounded-lg border p-2 space-y-1">
        <button
          on:click={() => activeSection = 'appearance'}
          class="w-full px-4 py-2 text-left rounded-lg transition-colors flex items-center gap-2 {activeSection === 'appearance' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}"
        >
          <Palette class="w-4 h-4" />
          Appearance
        </button>
        <button
          on:click={() => activeSection = 'playback'}
          class="w-full px-4 py-2 text-left rounded-lg transition-colors flex items-center gap-2 {activeSection === 'playback' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}"
        >
          <Volume2 class="w-4 h-4" />
          Playback
        </button>
        <button
          on:click={() => activeSection = 'display'}
          class="w-full px-4 py-2 text-left rounded-lg transition-colors flex items-center gap-2 {activeSection === 'display' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}"
        >
          <Layout class="w-4 h-4" />
          Display
        </button>
        <button
          on:click={() => activeSection = 'performance'}
          class="w-full px-4 py-2 text-left rounded-lg transition-colors flex items-center gap-2 {activeSection === 'performance' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}"
        >
          <Database class="w-4 h-4" />
          Performance
        </button>
        <button
          on:click={() => activeSection = 'clustering'}
          class="w-full px-4 py-2 text-left rounded-lg transition-colors flex items-center gap-2 {activeSection === 'clustering' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}"
        >
          <CircleDot class="w-4 h-4" />
          Clustering
        </button>
        <button
          on:click={() => activeSection = 'export'}
          class="w-full px-4 py-2 text-left rounded-lg transition-colors flex items-center gap-2 {activeSection === 'export' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}"
        >
          <Download class="w-4 h-4" />
          Export
        </button>
        <button
          on:click={() => activeSection = 'cache'}
          class="w-full px-4 py-2 text-left rounded-lg transition-colors flex items-center gap-2 {activeSection === 'cache' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}"
        >
          <RefreshCw class="w-4 h-4" />
          Cache
        </button>
        <button
          on:click={() => activeSection = 'about'}
          class="w-full px-4 py-2 text-left rounded-lg transition-colors flex items-center gap-2 {activeSection === 'about' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}"
        >
          <Info class="w-4 h-4" />
          About
        </button>
      </div>
    </div>

    <!-- Settings Content -->
    <div class="flex-1 space-y-6">
      {#if activeSection === 'appearance'}
        <div class="bg-card p-6 rounded-lg border space-y-6">
          <h2 class="text-2xl font-semibold flex items-center gap-2">
            <Palette class="w-6 h-6" />
            Appearance
          </h2>
          
          <div>
            <label for="theme-select" class="block text-sm font-medium mb-2">Theme</label>
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
        </div>
      {:else if activeSection === 'playback'}
        <div class="bg-card p-6 rounded-lg border space-y-6">
          <h2 class="text-2xl font-semibold flex items-center gap-2">
            <Volume2 class="w-6 h-6" />
            Playback
          </h2>
          
          <div class="space-y-4">
            <div class="flex items-center justify-between">
              <div>
                <label for="auto-play" class="block text-sm font-medium">Auto-play next track</label>
                <p class="text-sm text-muted-foreground">Automatically play the next track when current track ends</p>
              </div>
              <input
                id="auto-play"
                type="checkbox"
                bind:checked={$userPreferences.autoPlayNext}
                class="w-5 h-5"
              />
            </div>

            <div>
              <label for="default-volume" class="block text-sm font-medium mb-2">
                Default Volume: {Math.round($userPreferences.defaultVolume * 100)}%
              </label>
              <input
                id="default-volume"
                type="range"
                min="0"
                max="1"
                step="0.01"
                bind:value={$userPreferences.defaultVolume}
                class="w-full"
              />
            </div>

            <div class="flex items-center justify-between">
              <div>
                <label for="show-waveforms" class="block text-sm font-medium">Show waveforms</label>
                <p class="text-sm text-muted-foreground">Display waveform visualizations for tracks</p>
              </div>
              <input
                id="show-waveforms"
                type="checkbox"
                bind:checked={$userPreferences.showWaveforms}
                class="w-5 h-5"
              />
            </div>

            <div>
              <label for="waveform-samples" class="block text-sm font-medium mb-2">
                Waveform Samples: {$userPreferences.waveformSamples}
              </label>
              <input
                id="waveform-samples"
                type="range"
                min="50"
                max="500"
                step="50"
                bind:value={$userPreferences.waveformSamples}
                class="w-full"
              />
              <p class="text-xs text-muted-foreground mt-1">More samples = more detail but slower loading</p>
            </div>
          </div>
        </div>
      {:else if activeSection === 'display'}
        <div class="bg-card p-6 rounded-lg border space-y-6">
          <h2 class="text-2xl font-semibold flex items-center gap-2">
            <Layout class="w-6 h-6" />
            Display
          </h2>
          
          <div class="space-y-4">
            <div>
              <label for="items-per-page" class="block text-sm font-medium mb-2">
                Items per page: {$userPreferences.itemsPerPage}
              </label>
              <input
                id="items-per-page"
                type="range"
                min="10"
                max="200"
                step="10"
                bind:value={$userPreferences.itemsPerPage}
                class="w-full"
              />
            </div>

            <div>
              <label for="default-view" class="block text-sm font-medium mb-2">Default View</label>
              <div class="relative">
                <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                <select
                  id="default-view"
                  bind:value={$userPreferences.defaultView}
                  class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none"
                >
                  <option value="list">List</option>
                  <option value="grid">Grid</option>
                </select>
              </div>
            </div>

            <div class="flex items-center justify-between">
              <div>
                <label for="show-artwork" class="block text-sm font-medium">Show artwork</label>
                <p class="text-sm text-muted-foreground">Display album artwork in track lists</p>
              </div>
              <input
                id="show-artwork"
                type="checkbox"
                bind:checked={$userPreferences.showArtwork}
                class="w-5 h-5"
              />
            </div>

            <div>
              <label for="artwork-size" class="block text-sm font-medium mb-2">
                Artwork Size: {$userPreferences.artworkSize}px
              </label>
              <input
                id="artwork-size"
                type="range"
                min="32"
                max="128"
                step="8"
                bind:value={$userPreferences.artworkSize}
                class="w-full"
                disabled={!$userPreferences.showArtwork}
              />
            </div>

            <div class="flex items-center justify-between">
              <div>
                <label for="show-filepath" class="block text-sm font-medium">Show file path</label>
                <p class="text-sm text-muted-foreground">Display file paths in track lists</p>
              </div>
              <input
                id="show-filepath"
                type="checkbox"
                bind:checked={$userPreferences.showFilepath}
                class="w-5 h-5"
              />
            </div>

            <div class="flex items-center justify-between">
              <div>
                <label for="show-cluster-info" class="block text-sm font-medium">Show cluster info</label>
                <p class="text-sm text-muted-foreground">Display cluster information in track lists</p>
              </div>
              <input
                id="show-cluster-info"
                type="checkbox"
                bind:checked={$userPreferences.showClusterInfo}
                class="w-5 h-5"
              />
            </div>

            <div class="flex items-center justify-between">
              <div>
                <label for="compact-view" class="block text-sm font-medium">Compact view</label>
                <p class="text-sm text-muted-foreground">Use more compact spacing in lists</p>
              </div>
              <input
                id="compact-view"
                type="checkbox"
                bind:checked={$userPreferences.compactView}
                class="w-5 h-5"
              />
            </div>
          </div>
        </div>
      {:else if activeSection === 'performance'}
        <div class="bg-card p-6 rounded-lg border space-y-6">
          <h2 class="text-2xl font-semibold flex items-center gap-2">
            <Database class="w-6 h-6" />
            Performance
          </h2>
          
          <div class="space-y-4">
            <div>
              <label for="api-url" class="block text-sm font-medium mb-2">API URL</label>
              <input
                id="api-url"
                type="text"
                bind:value={$userPreferences.apiUrl}
                placeholder="http://localhost:8000"
                class="w-full p-2 border rounded-lg bg-background"
              />
              <p class="text-xs text-muted-foreground mt-1">Base URL for the Music Cluster API</p>
            </div>

            <div class="flex items-center justify-between">
              <div>
                <label for="cache-enabled" class="block text-sm font-medium">Enable cache</label>
                <p class="text-sm text-muted-foreground">Cache artwork and waveform data locally</p>
              </div>
              <input
                id="cache-enabled"
                type="checkbox"
                bind:checked={$userPreferences.cacheEnabled}
                class="w-5 h-5"
              />
            </div>

            <div>
              <label for="max-cache-size" class="block text-sm font-medium mb-2">
                Max Cache Size: {$userPreferences.maxCacheSize} MB
              </label>
              <input
                id="max-cache-size"
                type="range"
                min="10"
                max="500"
                step="10"
                bind:value={$userPreferences.maxCacheSize}
                class="w-full"
                disabled={!$userPreferences.cacheEnabled}
              />
            </div>
          </div>
        </div>
      {:else if activeSection === 'clustering'}
        <div class="bg-card p-6 rounded-lg border space-y-6">
          <h2 class="text-2xl font-semibold flex items-center gap-2">
            <CircleDot class="w-6 h-6" />
            Clustering Defaults
          </h2>
          
          <div class="space-y-4">
            <div>
              <label for="default-algorithm" class="block text-sm font-medium mb-2">Default Algorithm</label>
              <div class="relative">
                <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                <select
                  id="default-algorithm"
                  bind:value={$userPreferences.defaultAlgorithm}
                  class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none"
                >
                  <option value="kmeans">K-Means</option>
                  <option value="hierarchical">Hierarchical</option>
                  <option value="hdbscan">HDBSCAN</option>
                </select>
              </div>
            </div>

            <div>
              <label for="default-granularity" class="block text-sm font-medium mb-2">Default Granularity</label>
              <div class="relative">
                <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                <select
                  id="default-granularity"
                  bind:value={$userPreferences.defaultGranularity}
                  class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none"
                >
                  <option value="fewer">Fewer (broader clusters)</option>
                  <option value="less">Less</option>
                  <option value="normal">Normal</option>
                  <option value="more">More</option>
                  <option value="finer">Finer (more specific clusters)</option>
                </select>
              </div>
            </div>

            <div>
              <label for="default-min-cluster-size" class="block text-sm font-medium mb-2">
                Default Min Cluster Size: {$userPreferences.defaultMinClusterSize}
              </label>
              <input
                id="default-min-cluster-size"
                type="range"
                min="1"
                max="20"
                step="1"
                bind:value={$userPreferences.defaultMinClusterSize}
                class="w-full"
              />
            </div>
          </div>
        </div>
      {:else if activeSection === 'export'}
        <div class="bg-card p-6 rounded-lg border space-y-6">
          <h2 class="text-2xl font-semibold flex items-center gap-2">
            <Download class="w-6 h-6" />
            Export Defaults
          </h2>
          
          <div class="space-y-4">
            <div>
              <label for="default-export-format" class="block text-sm font-medium mb-2">Default Export Format</label>
              <div class="relative">
                <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                <select
                  id="default-export-format"
                  bind:value={$userPreferences.defaultExportFormat}
                  class="w-full p-2 pr-10 border rounded-lg bg-background appearance-none"
                >
                  <option value="m3u">M3U Playlist</option>
                  <option value="json">JSON</option>
                </select>
              </div>
            </div>

            <div class="flex items-center justify-between">
              <div>
                <label for="default-relative-paths" class="block text-sm font-medium">Use relative paths</label>
                <p class="text-sm text-muted-foreground">Export playlists with relative file paths</p>
              </div>
              <input
                id="default-relative-paths"
                type="checkbox"
                bind:checked={$userPreferences.defaultRelativePaths}
                class="w-5 h-5"
              />
            </div>
          </div>
        </div>
      {:else if activeSection === 'cache'}
        <div class="bg-card p-6 rounded-lg border space-y-6">
          <h2 class="text-2xl font-semibold flex items-center gap-2">
            <RefreshCw class="w-6 h-6" />
            Cache Management
          </h2>
          <p class="text-sm text-muted-foreground">
            Artwork and waveform data are cached locally to improve performance and reduce server load.
          </p>

          <div class="space-y-3">
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
      {:else if activeSection === 'about'}
        <div class="bg-card p-6 rounded-lg border space-y-6">
          <h2 class="text-2xl font-semibold flex items-center gap-2">
            <Info class="w-6 h-6" />
            About
          </h2>
          <div class="space-y-4">
            <div>
              <p class="font-medium">Music Cluster</p>
              <p class="text-sm text-muted-foreground">v1.0.0</p>
            </div>
            <div>
              <p class="text-sm text-muted-foreground">
                A music library management and clustering tool for organizing and discovering music through audio analysis.
              </p>
            </div>
            <div class="pt-4 border-t">
              <button
                on:click={() => userPreferences.reset()}
                class="px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
              >
                <RotateCcw class="w-4 h-4" />
                Reset All Settings
              </button>
            </div>
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
