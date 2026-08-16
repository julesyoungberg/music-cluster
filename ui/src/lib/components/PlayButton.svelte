<script lang="ts">
  import { Loader2, Pause, Play } from 'lucide-svelte';
  import { player, playTrack } from '$lib/stores/player';
  import type { Track } from '$lib/types';
  import { cn } from '$lib/utils';

  let { track, size = 'md' }: { track: Track; size?: 'sm' | 'md' | 'lg' } = $props();

  const isCurrent = $derived($player.track?.id === track.id);
  const isPlaying = $derived(isCurrent && $player.playing);
  const isLoading = $derived(isCurrent && $player.loading);

  const boxes = { sm: 'h-7 w-7', md: 'h-9 w-9', lg: 'h-11 w-11' };
  const icons = { sm: 'h-3 w-3', md: 'h-4 w-4', lg: 'h-5 w-5' };
</script>

<button
  class={cn(
    'flex shrink-0 items-center justify-center rounded-full border transition-colors',
    boxes[size],
    isCurrent
      ? 'border-primary bg-primary text-primary-foreground'
      : 'border-input hover:border-primary hover:text-primary'
  )}
  onclick={() => playTrack(track)}
  aria-label={isPlaying
    ? `Pause ${track.display ?? track.filename}`
    : `Play ${track.display ?? track.filename}`}
>
  {#if isLoading}
    <Loader2 class={cn(icons[size], 'animate-spin')} />
  {:else if isPlaying}
    <Pause class={icons[size]} />
  {:else}
    <Play class={cn(icons[size], 'ml-0.5')} />
  {/if}
</button>
