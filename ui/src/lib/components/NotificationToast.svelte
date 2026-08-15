<script lang="ts">
  import { notifications, dismiss } from '$lib/stores/notifications';
  import { CheckCircle2, AlertTriangle, Info, XCircle, X } from 'lucide-svelte';
  import { fly } from 'svelte/transition';

  const icons = { success: CheckCircle2, error: XCircle, warning: AlertTriangle, info: Info };
  const tones = {
    success: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
    error: 'border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300',
    warning: 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300',
    info: 'border-border bg-card text-foreground'
  };
</script>

<div class="pointer-events-none fixed bottom-24 right-4 z-50 flex w-full max-w-sm flex-col gap-2">
  {#each $notifications as item (item.id)}
    {@const Icon = icons[item.kind]}
    <div
      class="pointer-events-auto flex items-start gap-3 rounded-lg border p-3 shadow-lg {tones[item.kind]}"
      role="status"
      transition:fly={{ y: 12, duration: 150 }}
    >
      <Icon class="mt-0.5 h-4 w-4 shrink-0" />
      <div class="min-w-0 flex-1 text-sm">
        <p class="font-medium">{item.message}</p>
        {#if item.detail}
          <p class="mt-0.5 break-words opacity-80">{item.detail}</p>
        {/if}
      </div>
      <button
        class="shrink-0 rounded p-0.5 opacity-60 hover:opacity-100"
        onclick={() => dismiss(item.id)}
        aria-label="Dismiss"
      >
        <X class="h-4 w-4" />
      </button>
    </div>
  {/each}
</div>
