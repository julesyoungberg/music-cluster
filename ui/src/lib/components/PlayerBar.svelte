<script lang="ts">
  import { Pause, Play, Square, Volume2 } from 'lucide-svelte';
  import { player, seekFraction, setVolume, stop, togglePlay } from '$lib/stores/player';
  import { formatDuration, trackLabel } from '$lib/utils';

  const progress = $derived($player.duration > 0 ? $player.currentTime / $player.duration : 0);

  function scrub(event: MouseEvent) {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    seekFraction((event.clientX - rect.left) / rect.width);
  }
</script>

{#if $player.track}
  <div
    class="fixed bottom-0 left-0 right-0 z-40 border-t border-border bg-background/95 backdrop-blur"
  >
    <div class="mx-auto flex max-w-7xl items-center gap-4 px-4 py-2.5">
      <button
        class="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground"
        onclick={togglePlay}
        aria-label={$player.playing ? 'Pause' : 'Play'}
      >
        {#if $player.playing}<Pause class="h-4 w-4" />{:else}<Play class="ml-0.5 h-4 w-4" />{/if}
      </button>

      <div class="min-w-0 flex-1">
        <p class="truncate text-sm font-medium">{trackLabel($player.track)}</p>
        <div class="mt-1 flex items-center gap-2">
          <span class="shrink-0 tabular-nums text-xs text-muted-foreground">
            {formatDuration($player.currentTime)}
          </span>
          <div
            class="h-1.5 flex-1 cursor-pointer overflow-hidden rounded-full bg-secondary"
            onclick={scrub}
            onkeydown={(event) => event.key === 'Enter' && scrub(event as unknown as MouseEvent)}
            role="slider"
            tabindex="0"
            aria-label="Seek"
            aria-valuenow={Math.round(progress * 100)}
            aria-valuemin="0"
            aria-valuemax="100"
          >
            <div class="h-full bg-primary" style="width: {progress * 100}%"></div>
          </div>
          <span class="shrink-0 tabular-nums text-xs text-muted-foreground">
            {formatDuration($player.duration)}
          </span>
        </div>
      </div>

      <label class="hidden shrink-0 items-center gap-2 sm:flex" title="Volume">
        <Volume2 class="h-4 w-4 text-muted-foreground" />
        <span class="sr-only">Volume</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value="1"
          class="w-20 accent-primary"
          oninput={(event) => setVolume(Number(event.currentTarget.value))}
        />
      </label>

      <button
        class="shrink-0 rounded p-2 text-muted-foreground hover:bg-secondary"
        onclick={stop}
        aria-label="Stop playback"
      >
        <Square class="h-4 w-4" />
      </button>
    </div>
  </div>
{/if}
