<script lang="ts">
  import type { Snippet } from 'svelte';
  import { X } from 'lucide-svelte';
  import { fade, scale } from 'svelte/transition';

  let {
    open = $bindable(false),
    title = '',
    description = '',
    size = 'md',
    children,
    footer
  }: {
    open?: boolean;
    title?: string;
    description?: string;
    size?: 'sm' | 'md' | 'lg';
    children?: Snippet;
    footer?: Snippet;
  } = $props();

  const widths = { sm: 'max-w-md', md: 'max-w-xl', lg: 'max-w-3xl' };

  function onKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') open = false;
  }
</script>

<svelte:window on:keydown={onKeydown} />

{#if open}
  <div class="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4 sm:p-8">
    <button
      class="fixed inset-0 cursor-default bg-black/50"
      transition:fade={{ duration: 120 }}
      onclick={() => (open = false)}
      aria-label="Close dialog"
    ></button>

    <div
      class="relative z-10 w-full {widths[size]} rounded-lg border border-border bg-card shadow-xl"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      transition:scale={{ duration: 130, start: 0.97 }}
    >
      <div class="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
        <div>
          <h2 class="font-semibold">{title}</h2>
          {#if description}
            <p class="mt-0.5 text-sm text-muted-foreground">{description}</p>
          {/if}
        </div>
        <button
          class="rounded p-1 text-muted-foreground hover:bg-secondary"
          onclick={() => (open = false)}
          aria-label="Close"
        >
          <X class="h-4 w-4" />
        </button>
      </div>

      <div class="px-5 py-4">
        {#if children}{@render children()}{/if}
      </div>

      {#if footer}
        <div class="flex justify-end gap-2 border-t border-border px-5 py-3">
          {@render footer()}
        </div>
      {/if}
    </div>
  </div>
{/if}
