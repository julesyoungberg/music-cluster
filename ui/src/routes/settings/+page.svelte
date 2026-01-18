<script lang="ts">
  import { theme } from '$lib/stores/settings';
  import { Settings, Palette, ChevronDown, Info } from 'lucide-svelte';
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

    <div class="bg-card p-4 rounded-lg border">
      <h2 class="font-semibold mb-2 flex items-center gap-2">
        <Info class="w-5 h-5" />
        About
      </h2>
      <p class="text-sm text-muted-foreground">Music Cluster v1.0.0</p>
    </div>
  </div>
</div>
