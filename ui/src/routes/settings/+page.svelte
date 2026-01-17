<script lang="ts">
  import { theme } from '$lib/stores/settings';
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8">Settings</h1>

  <div class="max-w-2xl space-y-6">
    <div>
      <label for="theme-select" class="block text-sm font-medium mb-2">Theme</label>
      <select
        id="theme-select"
        bind:value={$theme}
        class="w-full p-2 border rounded-lg bg-background"
      >
        <option value="light">Light</option>
        <option value="dark">Dark</option>
        <option value="system">System</option>
      </select>
    </div>

    <div class="bg-card p-4 rounded-lg border">
      <h2 class="font-semibold mb-2">About</h2>
      <p class="text-sm text-muted-foreground">Music Cluster v1.0.0</p>
    </div>
  </div>
</div>
