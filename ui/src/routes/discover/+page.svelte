<script lang="ts">
  import { goto } from '$app/navigation';
  import { api, ApiError, pollTask } from '$lib/services/api';
  import { notifyError, notifySuccess } from '$lib/stores/notifications';
  import EmptyState from '$lib/components/EmptyState.svelte';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import FolderPicker from '$lib/components/FolderPicker.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import TaskProgress from '$lib/components/TaskProgress.svelte';
  import type { DiscoveryRun, TaskStatus } from '$lib/types';
  import { formatDate, pluralize } from '$lib/utils';
  import { Compass, Plus, Trash2 } from 'lucide-svelte';

  let runs = $state<DiscoveryRun[]>([]);
  let loading = $state(true);
  let error = $state('');
  let task = $state<TaskStatus | null>(null);

  let newOpen = $state(false);
  let sourcePath = $state('');
  let runName = $state('');
  let algorithm = $state<'hdbscan' | 'kmeans' | 'hierarchical'>('hdbscan');
  let targetGroups = $state<number | null>(null);
  let minGroupSize = $state(8);
  let useLlm = $state(false);

  async function load() {
    loading = true;
    error = '';
    try {
      runs = (await api.listDiscoveryRuns()).runs;
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void load();
  });

  async function start() {
    if (!sourcePath) return;
    newOpen = false;
    try {
      const { task_id } = await api.discover({
        paths: [sourcePath],
        name: runName || undefined,
        algorithm,
        target_groups: algorithm === 'hdbscan' ? undefined : (targetGroups ?? undefined),
        min_group_size: minGroupSize,
        use_llm: useLlm
      });
      const finished = await pollTask(task_id, (status) => (task = status));
      task = null;
      runName = '';

      const result = finished.result as { run: DiscoveryRun } | null;
      if (result?.run) {
        notifySuccess('Candidates ready to review');
        void goto(`/discover/${result.run.id}`);
      } else {
        await load();
      }
    } catch (err) {
      task = null;
      notifyError('Discovery failed', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function remove(run: DiscoveryRun) {
    try {
      await api.deleteDiscoveryRun(run.id);
      runs = runs.filter((item) => item.id !== run.id);
    } catch (err) {
      notifyError('Could not delete', err instanceof ApiError ? err.message : String(err));
    }
  }
</script>

<svelte:head><title>Discover · music-cluster</title></svelte:head>

<div class="space-y-6">
  <div class="flex flex-wrap items-start justify-between gap-4">
    <div class="max-w-2xl">
      <h1 class="text-2xl font-semibold">Discover</h1>
      <p class="mt-1 text-sm text-muted-foreground">
        Break a big unsorted folder into candidate piles. Each one comes with representative tracks
        to audition — you name and keep the ones that are real, and throw away the rest.
      </p>
    </div>
    <button class="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground" onclick={() => (newOpen = true)}>
      <Plus class="h-4 w-4" /> New run
    </button>
  </div>

  <TaskProgress {task} label="Analysing and grouping" />

  {#if loading}
    <LoadingState message="Loading runs..." />
  {:else if error}
    <ErrorState message={error} onRetry={load} />
  {:else if runs.length === 0}
    <EmptyState
      icon={Compass}
      title="No discovery runs yet"
      description="Use this when you have a pile of music and no structure yet. Nothing becomes a group until you say so."
    >
      <button class="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground" onclick={() => (newOpen = true)}>
        Break down a folder
      </button>
    </EmptyState>
  {:else}
    <div class="divide-y divide-border rounded-lg border border-border">
      {#each runs as run (run.id)}
        <div class="flex items-center gap-4 px-4 py-3">
          <a href="/discover/{run.id}" class="min-w-0 flex-1">
            <p class="truncate font-medium">{run.name ?? `Run ${run.id}`}</p>
            <p class="truncate text-xs text-muted-foreground">
              {run.algorithm} · {formatDate(run.created_at)}
              {#if run.source_path}· <span class="font-mono">{run.source_path}</span>{/if}
            </p>
          </a>
          <span class="shrink-0 text-xs text-muted-foreground">
            {pluralize(run.candidate_count ?? 0, 'candidate')}
          </span>
          <button class="shrink-0 rounded p-1.5 text-muted-foreground hover:bg-secondary hover:text-destructive" onclick={() => remove(run)} aria-label="Delete run">
            <Trash2 class="h-4 w-4" />
          </button>
        </div>
      {/each}
    </div>
  {/if}
</div>

<Modal bind:open={newOpen} title="Break down a folder" description="Candidates are proposals. Nothing is filed and no group is created until you promote one.">
  <div class="space-y-4">
    <FolderPicker bind:value={sourcePath} label="Unsorted folder" />

    <label class="block text-sm">
      <span class="font-medium">Run name</span>
      <input class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2" bind:value={runName} placeholder="Defaults to the folder name" />
    </label>

    <div class="grid gap-3 sm:grid-cols-2">
      <label class="block text-sm">
        <span class="font-medium">Method</span>
        <select class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2" bind:value={algorithm}>
          <option value="hdbscan">Density (finds its own count)</option>
          <option value="kmeans">K-means (fixed count)</option>
          <option value="hierarchical">Hierarchical (fixed count)</option>
        </select>
      </label>

      {#if algorithm === 'hdbscan'}
        <label class="block text-sm">
          <span class="font-medium">Smallest pile</span>
          <input type="number" min="2" class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2" bind:value={minGroupSize} />
        </label>
      {:else}
        <label class="block text-sm">
          <span class="font-medium">How many piles</span>
          <input
            type="number"
            min="2"
            class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2"
            placeholder="Auto"
            value={targetGroups ?? ''}
            oninput={(event) => (targetGroups = event.currentTarget.value ? Number(event.currentTarget.value) : null)}
          />
        </label>
      {/if}
    </div>

    <label class="flex items-start gap-2 text-sm">
      <input type="checkbox" bind:checked={useLlm} class="mt-1 accent-primary" />
      <span>
        <span class="font-medium">Suggest names with an LLM</span>
        <span class="block text-xs text-muted-foreground">
          Sends artist, title, genre tag and tempo for a few tracks per pile. Needs an API key —
          see Settings. Without it, names come from tags and audio characteristics.
        </span>
      </span>
    </label>
  </div>

  {#snippet footer()}
    <button class="rounded-md border border-input px-3 py-1.5 text-sm" onclick={() => (newOpen = false)}>Cancel</button>
    <button class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50" onclick={start} disabled={!sourcePath}>
      Find groups
    </button>
  {/snippet}
</Modal>
