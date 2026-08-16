<script lang="ts">
  import { AlertCircle, RotateCw } from 'lucide-svelte';

  let {
    title = 'Something went wrong',
    message = '',
    onRetry = undefined
  }: { title?: string; message?: string; onRetry?: (() => void) | undefined } = $props();
</script>

<div
  class="flex flex-col items-center gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-6 py-10 text-center"
>
  <AlertCircle class="h-6 w-6 text-destructive" />
  <div>
    <p class="font-medium">{title}</p>
    {#if message}
      <p class="mt-1 max-w-md text-sm text-muted-foreground">{message}</p>
    {/if}
  </div>
  {#if onRetry}
    <button
      class="mt-1 inline-flex items-center gap-2 rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary"
      onclick={onRetry}
    >
      <RotateCw class="h-4 w-4" /> Try again
    </button>
  {/if}
</div>
