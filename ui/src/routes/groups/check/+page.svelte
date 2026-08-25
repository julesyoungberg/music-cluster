<script lang="ts">
  import { api, ApiError } from '$lib/services/api';
  import { activeCollection } from '$lib/stores/app';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import type { CheckMetrics } from '$lib/types';
  import { formatDate, formatPercent } from '$lib/utils';
  import { ArrowLeft } from 'lucide-svelte';

  let metrics = $state<CheckMetrics | null>(null);
  let fittedAt = $state<string | null>(null);
  let loading = $state(true);
  let error = $state('');

  const collection = $derived($activeCollection);

  async function load() {
    if (!collection) return;
    loading = true;
    error = '';
    try {
      const result = await api.checkCollection(collection.id);
      metrics = result.metrics;
      fittedAt = result.fitted_at;
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (collection) void load();
  });

  /** Colour a confusion cell by how much of a row it accounts for. */
  function cellTone(value: number, rowTotal: number): string {
    if (value === 0) return '';
    const share = rowTotal > 0 ? value / rowTotal : 0;
    if (share >= 0.6) return 'bg-primary/70 text-primary-foreground';
    if (share >= 0.3) return 'bg-primary/40';
    if (share >= 0.1) return 'bg-primary/20';
    return 'bg-primary/10';
  }
</script>

<svelte:head><title>Overlap report · music-cluster</title></svelte:head>

<a
  href="/groups"
  class="mb-4 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
>
  <ArrowLeft class="h-4 w-4" /> All groups
</a>

{#if loading}
  <LoadingState message="Loading report..." />
{:else if error || !metrics}
  <ErrorState title="No report yet" message={error || 'Fit the collection first.'} onRetry={load} />
{:else}
  <div class="space-y-8">
    <div>
      <h1 class="text-2xl font-semibold">Overlap report</h1>
      <p class="mt-1 max-w-2xl text-sm text-muted-foreground">
        Every reference track is re-sorted with itself hidden from its own group. If a track does
        not come back to where you filed it, those two groups sound alike to the sorter — which
        usually means they sound alike, full stop.
      </p>
      <p class="mt-2 text-xs text-muted-foreground">Fitted {formatDate(fittedAt)}</p>
    </div>

    <div class="grid gap-4 sm:grid-cols-3">
      <div class="rounded-lg border border-border p-4">
        <p class="text-xs uppercase tracking-wide text-muted-foreground">Overall</p>
        <p class="mt-1 text-2xl font-semibold">{formatPercent(metrics.accuracy)}</p>
      </div>
      <div class="rounded-lg border border-border p-4">
        <p class="text-xs uppercase tracking-wide text-muted-foreground">Reference tracks</p>
        <p class="mt-1 text-2xl font-semibold">{metrics.n_tracks}</p>
      </div>
      <div class="rounded-lg border border-border p-4">
        <p class="text-xs uppercase tracking-wide text-muted-foreground">Groups</p>
        <p class="mt-1 text-2xl font-semibold">{metrics.n_groups}</p>
      </div>
    </div>

    {#if metrics.problems && metrics.problems.length > 0}
      <div class="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm">
        <p class="font-medium">Groups needing attention</p>
        <ul class="mt-2 space-y-1">
          {#each metrics.problems as problem}
            <li>{problem.name}: {problem.issue}</li>
          {/each}
        </ul>
      </div>
    {/if}

    {#if metrics.confusable_pairs.length > 0}
      <section>
        <h2 class="mb-3 font-medium">Most easily confused</h2>
        <div class="divide-y divide-border rounded-lg border border-border">
          {#each metrics.confusable_pairs as pair}
            <div class="flex items-center gap-4 px-4 py-3 text-sm">
              <p class="min-w-0 flex-1">
                <span class="font-medium">{pair.count}</span> tracks from
                <a href="/groups/{pair.from_group_id}" class="text-primary hover:underline"
                  >{pair.from_name}</a
                >
                look like
                <a href="/groups/{pair.to_group_id}" class="text-primary hover:underline"
                  >{pair.to_name}</a
                >
              </p>
              <span class="shrink-0 tabular-nums text-muted-foreground"
                >{formatPercent(pair.rate)}</span
              >
            </div>
          {/each}
        </div>
        <p class="mt-2 text-xs text-muted-foreground">
          Heavy overlap is a decision, not a bug: either merge the two groups, or add reference
          tracks that show what actually separates them.
        </p>
      </section>
    {/if}

    <section>
      <h2 class="mb-3 font-medium">Per-group accuracy</h2>
      <div class="divide-y divide-border rounded-lg border border-border">
        {#each [...metrics.per_group].sort((a, b) => a.accuracy - b.accuracy) as entry (entry.group_id)}
          <div class="flex items-center gap-4 px-4 py-3">
            <a
              href="/groups/{entry.group_id}"
              class="min-w-0 flex-1 truncate text-sm hover:underline"
            >
              {entry.name}
            </a>
            <div class="hidden h-2 w-40 overflow-hidden rounded-full bg-secondary sm:block">
              <div class="h-full bg-primary" style="width: {entry.accuracy * 100}%"></div>
            </div>
            <span class="w-24 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
              {entry.correct}/{entry.size}
            </span>
            <span class="w-14 shrink-0 text-right text-sm tabular-nums"
              >{formatPercent(entry.accuracy)}</span
            >
          </div>
        {/each}
      </div>
    </section>

    <section>
      <h2 class="mb-3 font-medium">Confusion matrix</h2>
      <p class="mb-2 text-xs text-muted-foreground">
        Rows are where a track is filed; columns are where it would land.
      </p>
      <div class="overflow-x-auto rounded-lg border border-border">
        <table class="w-full text-xs">
          <thead>
            <tr>
              <th class="sticky left-0 z-10 bg-card px-3 py-2 text-left font-medium">Filed in</th>
              {#each metrics.group_names as name}
                <th class="px-2 py-2 text-center font-medium">
                  <span class="inline-block max-w-[6rem] truncate align-bottom" title={name}
                    >{name}</span
                  >
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each metrics.confusion as row, i}
              {@const rowTotal = row.reduce((sum, value) => sum + value, 0)}
              <tr class="border-t border-border">
                <th
                  class="sticky left-0 z-10 max-w-[10rem] truncate bg-card px-3 py-2 text-left font-normal"
                  title={metrics.group_names[i]}
                >
                  {metrics.group_names[i]}
                </th>
                {#each row as value, j}
                  <td
                    class="px-2 py-2 text-center tabular-nums {i === j
                      ? 'font-semibold'
                      : ''} {cellTone(value, rowTotal)}"
                  >
                    {value || ''}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  </div>
{/if}
