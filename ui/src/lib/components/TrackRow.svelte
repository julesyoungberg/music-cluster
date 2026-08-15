<script lang="ts">
  import type { Snippet } from 'svelte';
  import PlayButton from './PlayButton.svelte';
  import type { Track } from '$lib/types';
  import { formatDuration, trackLabel } from '$lib/utils';

  let {
    track,
    subtitle = '',
    dense = false,
    actions
  }: { track: Track; subtitle?: string; dense?: boolean; actions?: Snippet } = $props();

  const meta = $derived(
    [
      track.tag_bpm ? `${Math.round(track.tag_bpm)} BPM` : null,
      track.tag_key,
      track.genre,
      formatDuration(track.duration)
    ]
      .filter(Boolean)
      .join(' · ')
  );
</script>

<div class="flex items-center gap-3 {dense ? 'py-1.5' : 'py-2'}">
  <PlayButton {track} size={dense ? 'sm' : 'md'} />
  <div class="min-w-0 flex-1">
    <p class="truncate text-sm font-medium">{trackLabel(track)}</p>
    <p class="truncate text-xs text-muted-foreground">{subtitle || meta}</p>
  </div>
  {#if actions}
    <div class="flex shrink-0 items-center gap-1">{@render actions()}</div>
  {/if}
</div>
