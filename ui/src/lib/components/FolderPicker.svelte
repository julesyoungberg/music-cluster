<script lang="ts">
  import { api, ApiError } from '$lib/services/api';
  import { ChevronRight, CornerLeftUp, Folder, Loader2, Music } from 'lucide-svelte';

  /**
   * Browse the machine's filesystem for a folder.
   *
   * The desktop shell cannot hand a directory path to the API from an
   * `<input type="file">`, so the API exposes a browse endpoint and this walks it.
   */
  let {
    value = $bindable(''),
    label = 'Folder'
  }: { value?: string; label?: string } = $props();

  let current = $state<string>('');
  let parent = $state<string | null>(null);
  let directories = $state<{ name: string; path: string }[]>([]);
  let audioCount = $state(0);
  let loading = $state(false);
  let error = $state('');

  async function browse(path?: string) {
    loading = true;
    error = '';
    try {
      const result = await api.browse(path);
      current = result.path;
      parent = result.parent;
      directories = result.directories;
      audioCount = result.audio_file_count;
      value = result.path;
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (!current) void browse(value || undefined);
  });
</script>

<div class="space-y-2">
  <div class="flex items-center justify-between gap-2">
    <span class="text-sm font-medium">{label}</span>
    {#if audioCount > 0}
      <span class="flex items-center gap-1 text-xs text-muted-foreground">
        <Music class="h-3 w-3" />
        {audioCount} audio file{audioCount === 1 ? '' : 's'} here
      </span>
    {/if}
  </div>

  <input
    class="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs"
    bind:value
    onchange={() => browse(value)}
    placeholder="/path/to/folder"
    spellcheck="false"
  />

  <div class="h-56 overflow-y-auto rounded-md border border-border">
    {#if loading}
      <div class="flex h-full items-center justify-center text-muted-foreground">
        <Loader2 class="h-4 w-4 animate-spin" />
      </div>
    {:else if error}
      <p class="p-3 text-sm text-destructive">{error}</p>
    {:else}
      {#if parent}
        <button
          class="flex w-full items-center gap-2 border-b border-border px-3 py-2 text-left text-sm hover:bg-secondary"
          onclick={() => browse(parent!)}
        >
          <CornerLeftUp class="h-4 w-4 text-muted-foreground" />
          <span class="text-muted-foreground">Up one level</span>
        </button>
      {/if}
      {#each directories as directory (directory.path)}
        <button
          class="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-secondary"
          onclick={() => browse(directory.path)}
        >
          <Folder class="h-4 w-4 shrink-0 text-muted-foreground" />
          <span class="truncate">{directory.name}</span>
          <ChevronRight class="ml-auto h-4 w-4 shrink-0 text-muted-foreground" />
        </button>
      {:else}
        <p class="p-3 text-sm text-muted-foreground">No subfolders here.</p>
      {/each}
    {/if}
  </div>
</div>
