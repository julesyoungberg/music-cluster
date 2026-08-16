<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    icon,
    title,
    description = '',
    children
  }: {
    // A lucide-svelte icon. Its exported type differs between the legacy and
    // runes builds, so this stays loose rather than pinning one of them.
    icon?: any;
    title: string;
    description?: string;
    children?: Snippet;
  } = $props();
</script>

<div
  class="flex flex-col items-center gap-3 rounded-lg border border-dashed border-border px-6 py-14 text-center"
>
  {#if icon}
    {@const Icon = icon}
    <Icon class="h-7 w-7 text-muted-foreground" />
  {/if}
  <div>
    <p class="font-medium">{title}</p>
    {#if description}
      <p class="mx-auto mt-1 max-w-md text-sm text-muted-foreground">{description}</p>
    {/if}
  </div>
  {#if children}
    <div class="mt-2 flex flex-wrap justify-center gap-2">{@render children()}</div>
  {/if}
</div>
