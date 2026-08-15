<script lang="ts">
  import type { TaskStatus } from '$lib/types';
  import { Loader2 } from 'lucide-svelte';

  let { task, label = '' }: { task: TaskStatus | null; label?: string } = $props();

  const percent = $derived(Math.round((task?.progress ?? 0) * 100));
</script>

{#if task && task.status === 'running'}
  <div class="space-y-2 rounded-md border border-border bg-secondary/40 p-3">
    <div class="flex items-center gap-2 text-sm">
      <Loader2 class="h-4 w-4 animate-spin" />
      <span class="font-medium">{label || task.kind}</span>
      {#if task.total > 0}
        <span class="ml-auto tabular-nums text-muted-foreground">{task.current}/{task.total}</span>
      {/if}
    </div>
    <div class="h-1.5 overflow-hidden rounded-full bg-border">
      <div class="h-full bg-primary transition-all duration-200" style="width: {percent}%"></div>
    </div>
    <p class="truncate text-xs text-muted-foreground">{task.message}</p>
  </div>
{/if}
