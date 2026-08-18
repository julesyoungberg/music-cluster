<script lang="ts">
  import { activeCollection, collections, selectCollection } from '$lib/stores/app';
  import { Layers } from 'lucide-svelte';

  function onChange(event: Event) {
    const value = Number((event.target as HTMLSelectElement).value);
    if (Number.isFinite(value)) selectCollection(value);
  }
</script>

{#if $collections.length > 0}
  <label class="flex shrink-0 items-center gap-2 text-sm" title="Active collection">
    <Layers class="h-4 w-4 text-muted-foreground" />
    <span class="sr-only">Active collection</span>
    <select
      class="max-w-[11rem] truncate rounded-md border border-input bg-background px-2 py-1 text-sm"
      value={$activeCollection?.id ?? ''}
      onchange={onChange}
    >
      {#each $collections as collection (collection.id)}
        <option value={collection.id}>
          {collection.name}{collection.profile === 'sample' ? ' · samples' : ''}
        </option>
      {/each}
    </select>
  </label>
{/if}
