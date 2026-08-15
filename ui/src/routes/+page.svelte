<script lang="ts">
  import { api, ApiError, pollTask } from '$lib/services/api';
  import { activeCollection, loadCollections } from '$lib/stores/app';
  import { notifyError, notifySuccess } from '$lib/stores/notifications';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import EmptyState from '$lib/components/EmptyState.svelte';
  import type { Collection, DiscoveryRun, SortSession, TaskStatus } from '$lib/types';
  import { formatDate, formatPercent, pluralize } from '$lib/utils';
  import { Compass, FolderTree, Inbox, Sparkles, TriangleAlert, Wand2 } from 'lucide-svelte';

  let loading = $state(true);
  let error = $state('');
  let trackCount = $state(0);
  let analyzedCount = $state(0);
  let collections = $state<Collection[]>([]);
  let sessions = $state<SortSession[]>([]);
  let runs = $state<DiscoveryRun[]>([]);
  let fitting = $state(false);

  async function load() {
    loading = true;
    error = '';
    try {
      const info = await api.info();
      trackCount = info.track_count;
      analyzedCount = info.analyzed_count;
      collections = info.collections;
      sessions = info.sessions;
      runs = info.discovery_runs;
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void load();
  });

  const current = $derived(collections.find((c) => c.id === $activeCollection?.id) ?? null);
  const openSessions = $derived(sessions.filter((session) => (session.pending_count ?? 0) > 0));

  async function fit() {
    if (!current) return;
    fitting = true;
    try {
      const { task_id } = await api.fitCollection(current.id);
      const task: TaskStatus = await pollTask(task_id);
      const metrics = task.result as { accuracy: number } | null;
      notifySuccess(
        'Groups re-learned',
        metrics ? `Self-check accuracy ${formatPercent(metrics.accuracy)}` : undefined
      );
      await Promise.all([load(), loadCollections()]);
    } catch (err) {
      notifyError('Could not fit', err instanceof ApiError ? err.message : String(err));
    } finally {
      fitting = false;
    }
  }
</script>

<svelte:head><title>Overview · music-cluster</title></svelte:head>

{#if loading}
  <LoadingState message="Loading library..." />
{:else if error}
  <ErrorState title="Cannot reach the API" message={error} onRetry={load} />
{:else}
  <div class="space-y-8">
    <div>
      <h1 class="text-2xl font-semibold">Overview</h1>
      <p class="mt-1 text-sm text-muted-foreground">
        {pluralize(trackCount, 'track')} in the library, {analyzedCount} analysed.
      </p>
    </div>

    {#if collections.length === 0}
      <EmptyState
        icon={FolderTree}
        title="No collections yet"
        description="A collection holds the folders you sort into. Start by importing the genre folders you already keep."
      >
        <a href="/groups" class="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground">
          Set up your groups
        </a>
      </EmptyState>
    {:else}
      <section class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div class="rounded-lg border border-border p-4">
          <p class="text-xs uppercase tracking-wide text-muted-foreground">Groups</p>
          <p class="mt-1 text-2xl font-semibold">{current?.group_count ?? 0}</p>
          <p class="text-xs text-muted-foreground">in {current?.name ?? '—'}</p>
        </div>
        <div class="rounded-lg border border-border p-4">
          <p class="text-xs uppercase tracking-wide text-muted-foreground">Reference tracks</p>
          <p class="mt-1 text-2xl font-semibold">{current?.track_count ?? 0}</p>
          <p class="text-xs text-muted-foreground">defining those groups</p>
        </div>
        <div class="rounded-lg border border-border p-4">
          <p class="text-xs uppercase tracking-wide text-muted-foreground">Self-check</p>
          <p class="mt-1 text-2xl font-semibold">{formatPercent(current?.accuracy)}</p>
          <p class="text-xs text-muted-foreground">
            {current?.fitted ? `fitted ${formatDate(current.fitted_at)}` : 'not fitted yet'}
          </p>
        </div>
        <div class="rounded-lg border border-border p-4">
          <p class="text-xs uppercase tracking-wide text-muted-foreground">Waiting to review</p>
          <p class="mt-1 text-2xl font-semibold">
            {openSessions.reduce((sum, session) => sum + (session.pending_count ?? 0), 0)}
          </p>
          <p class="text-xs text-muted-foreground">
            across {pluralize(openSessions.length, 'session')}
          </p>
        </div>
      </section>

      {#if current && !current.fitted}
        <div class="flex flex-wrap items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-4">
          <TriangleAlert class="h-5 w-5 text-amber-600 dark:text-amber-400" />
          <p class="flex-1 text-sm">
            <span class="font-medium">{current.name} has not been fitted.</span>
            Sorting needs to learn what each group contains before it can suggest anything.
          </p>
          <button
            class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-60"
            onclick={fit}
            disabled={fitting || (current.group_count ?? 0) < 2}
          >
            {fitting ? 'Fitting...' : 'Fit now'}
          </button>
        </div>
      {/if}

      <section class="grid gap-4 md:grid-cols-3">
        <a href="/sort" class="group rounded-lg border border-border p-5 transition-colors hover:border-primary">
          <Inbox class="h-6 w-6 text-muted-foreground group-hover:text-primary" />
          <p class="mt-3 font-medium">Sort new music</p>
          <p class="mt-1 text-sm text-muted-foreground">
            Point at a folder of new buys and decide where each track goes.
          </p>
        </a>
        <a href="/discover" class="group rounded-lg border border-border p-5 transition-colors hover:border-primary">
          <Compass class="h-6 w-6 text-muted-foreground group-hover:text-primary" />
          <p class="mt-3 font-medium">Break down a pile</p>
          <p class="mt-1 text-sm text-muted-foreground">
            Turn one big unsorted folder into candidate groups you can name and keep.
          </p>
        </a>
        <a href="/groups" class="group rounded-lg border border-border p-5 transition-colors hover:border-primary">
          <FolderTree class="h-6 w-6 text-muted-foreground group-hover:text-primary" />
          <p class="mt-3 font-medium">Manage groups</p>
          <p class="mt-1 text-sm text-muted-foreground">
            Import folders and playlists, set destinations, check for overlap.
          </p>
        </a>
      </section>

      {#if openSessions.length > 0}
        <section>
          <h2 class="mb-3 flex items-center gap-2 font-medium">
            <Sparkles class="h-4 w-4" /> Waiting on you
          </h2>
          <div class="divide-y divide-border rounded-lg border border-border">
            {#each openSessions as session (session.id)}
              <a href="/sort/{session.id}" class="flex items-center gap-4 px-4 py-3 hover:bg-secondary/50">
                <div class="min-w-0 flex-1">
                  <p class="truncate font-medium">{session.name ?? `Session ${session.id}`}</p>
                  <p class="truncate text-xs text-muted-foreground">
                    {session.collection_name} · {formatDate(session.created_at)}
                  </p>
                </div>
                <span class="shrink-0 rounded-full bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary">
                  {session.pending_count} to review
                </span>
              </a>
            {/each}
          </div>
        </section>
      {/if}

      {#if runs.length > 0}
        <section>
          <h2 class="mb-3 flex items-center gap-2 font-medium">
            <Wand2 class="h-4 w-4" /> Recent discovery runs
          </h2>
          <div class="divide-y divide-border rounded-lg border border-border">
            {#each runs.slice(0, 5) as run (run.id)}
              <a href="/discover/{run.id}" class="flex items-center gap-4 px-4 py-3 hover:bg-secondary/50">
                <div class="min-w-0 flex-1">
                  <p class="truncate font-medium">{run.name ?? `Run ${run.id}`}</p>
                  <p class="truncate text-xs text-muted-foreground">
                    {run.algorithm} · {formatDate(run.created_at)}
                  </p>
                </div>
                <span class="shrink-0 text-xs text-muted-foreground">
                  {pluralize(run.candidate_count ?? 0, 'candidate')}
                </span>
              </a>
            {/each}
          </div>
        </section>
      {/if}
    {/if}
  </div>
{/if}
