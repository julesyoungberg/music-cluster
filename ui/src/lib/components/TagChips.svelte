<script lang="ts">
  import type { TrackTags } from '$lib/types';

  let { tags, compact = false }: { tags: TrackTags | null | undefined; compact?: boolean } =
    $props();

  const beat = $derived(
    tags?.beat_label
      ? tags.swung && tags.beat !== 'beatless'
        ? `Swung ${tags.beat_label.toLowerCase()}`
        : tags.beat_label
      : null
  );

  // Energy reads like a meter: cool at 1, hot at 10.
  const energyTone = $derived(
    !tags
      ? ''
      : tags.energy_level >= 8
        ? 'border-rose-500/40 text-rose-600 dark:text-rose-400'
        : tags.energy_level >= 5
          ? 'border-amber-500/40 text-amber-600 dark:text-amber-400'
          : 'border-sky-500/40 text-sky-600 dark:text-sky-400'
  );

  const chip = 'inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] leading-4';
</script>

{#if tags}
  <div class="flex flex-wrap items-center gap-1" aria-label="Auto-tags">
    {#if tags.genre_label}
      <span
        class="{chip} border-primary/40 bg-primary/10 font-medium text-primary"
        title="Genre, from tempo, beat type and sound"
      >
        {tags.genre_label}
      </span>
    {/if}
    {#if beat}
      <span class="{chip} border-border" title="Beat type, from where the kick and snare land">
        {beat}
      </span>
    {/if}
    {#if !compact}
      {#each tags.mood_labels as mood (mood)}
        <span class="{chip} border-border text-muted-foreground" title="Mood">{mood}</span>
      {/each}
    {/if}
    <span
      class="{chip} tabular-nums {energyTone}"
      title="Energy {tags.energy_level}/10 · danceability {Math.round(tags.danceability * 100)}%"
    >
      E{tags.energy_level}
    </span>
    {#if tags.camelot && !compact}
      <span class="{chip} border-border tabular-nums text-muted-foreground" title={tags.key ?? ''}>
        {tags.camelot}
      </span>
    {/if}
  </div>
{/if}
