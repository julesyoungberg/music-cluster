<script lang="ts">
  import { Loader2 } from 'lucide-svelte';
  let { message = 'Loading...', compact = false }: { message?: string; compact?: boolean } = $props();
</script>

<div class="flex items-center justify-center gap-3 text-muted-foreground {compact ? 'py-4' : 'py-16'}">
  <Loader2 class="h-5 w-5 animate-spin" />
  <span class="text-sm">{message}</span>
</div>
