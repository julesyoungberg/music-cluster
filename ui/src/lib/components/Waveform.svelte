<script lang="ts">
  import { api } from '$lib/services/api';
  import { player, playTrack, seekFraction } from '$lib/stores/player';
  import type { Track } from '$lib/types';

  let {
    track,
    height = 48,
    samples = 200
  }: { track: Track; height?: number; samples?: number } = $props();

  let peaks = $state<number[]>([]);
  let loading = $state(false);
  let failed = $state(false);

  const isCurrent = $derived($player.track?.id === track.id);
  const progress = $derived(
    isCurrent && $player.duration > 0 ? $player.currentTime / $player.duration : 0
  );

  // Waveforms are decoded on demand, one track at a time, so switching tracks
  // in a review queue does not queue up decodes for everything on screen.
  $effect(() => {
    const id = track.id;
    peaks = [];
    failed = false;
    loading = true;

    let cancelled = false;
    api
      .waveform(id, samples)
      .then((result) => {
        if (!cancelled) peaks = result.peaks;
      })
      .catch(() => {
        if (!cancelled) failed = true;
      })
      .finally(() => {
        if (!cancelled) loading = false;
      });

    return () => {
      cancelled = true;
    };
  });

  function onClick(event: MouseEvent) {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const fraction = (event.clientX - rect.left) / rect.width;
    if (isCurrent) seekFraction(fraction);
    else void playTrack(track, 0);
  }
</script>

<div
  class="relative w-full cursor-pointer select-none overflow-hidden rounded bg-secondary/50"
  style="height: {height}px"
  onclick={onClick}
  onkeydown={(event) => event.key === 'Enter' && onClick(event as unknown as MouseEvent)}
  role="slider"
  tabindex="0"
  aria-label="Waveform scrubber"
  aria-valuenow={Math.round(progress * 100)}
  aria-valuemin="0"
  aria-valuemax="100"
>
  {#if loading}
    <div class="absolute inset-0 animate-pulse bg-border/40"></div>
  {:else if failed}
    <div class="flex h-full items-center justify-center text-xs text-muted-foreground">
      Preview unavailable
    </div>
  {:else}
    <div class="flex h-full w-full items-center gap-px px-px">
      {#each peaks as peak, index}
        {@const played = peaks.length > 0 && index / peaks.length <= progress}
        <div
          class="flex-1 rounded-sm transition-colors {played ? 'bg-primary' : 'bg-muted-foreground/40'}"
          style="height: {Math.max(6, peak * 100)}%"
        ></div>
      {/each}
    </div>
  {/if}
</div>
