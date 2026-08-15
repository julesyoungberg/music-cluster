<script lang="ts">
  import type { Group } from '$lib/types';
  import { Search } from 'lucide-svelte';

  /** Searchable list of groups — the fallback when a suggestion is wrong. */
  let {
    groups,
    onSelect,
    selectedId = null
  }: { groups: Group[]; onSelect: (group: Group) => void; selectedId?: number | null } = $props();

  let query = $state('');

  const filtered = $derived(
    query.trim()
      ? groups.filter((group) => group.name.toLowerCase().includes(query.trim().toLowerCase()))
      : groups
  );
</script>

<div class="space-y-2">
  <div class="relative">
    <Search class="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
    <input
      class="w-full rounded-md border border-input bg-background py-2 pl-9 pr-3 text-sm"
      placeholder="Search groups..."
      bind:value={query}
    />
  </div>

  <div class="max-h-64 space-y-1 overflow-y-auto">
    {#each filtered as group (group.id)}
      <button
        class="flex w-full items-center justify-between gap-2 rounded-md border px-3 py-2 text-left text-sm transition-colors
          {selectedId === group.id ? 'border-primary bg-primary/10' : 'border-transparent hover:bg-secondary'}"
        onclick={() => onSelect(group)}
      >
        <span class="truncate">{group.name}</span>
        <span class="shrink-0 text-xs text-muted-foreground">{group.size ?? 0}</span>
      </button>
    {:else}
      <p class="px-3 py-6 text-center text-sm text-muted-foreground">No groups match.</p>
    {/each}
  </div>
</div>
