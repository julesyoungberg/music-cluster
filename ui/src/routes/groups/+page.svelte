<script lang="ts">
  import { api, ApiError, pollTask } from '$lib/services/api';
  import { activeCollection, collections, loadCollections, loadGroups, selectCollection } from '$lib/stores/app';
  import { notifyError, notifySuccess } from '$lib/stores/notifications';
  import EmptyState from '$lib/components/EmptyState.svelte';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import FolderPicker from '$lib/components/FolderPicker.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import ProfileBadge from '$lib/components/ProfileBadge.svelte';
  import TaskProgress from '$lib/components/TaskProgress.svelte';
  import type { AudioProfile, CheckMetrics, Group, TaskStatus } from '$lib/types';
  import { formatPercent, pluralize } from '$lib/utils';
  import {
    FolderPlus,
    FolderTree,
    ListPlus,
    Plus,
    RefreshCw,
    TriangleAlert
  } from 'lucide-svelte';

  let groups = $state<Group[]>([]);
  let metrics = $state<CheckMetrics | null>(null);
  let loading = $state(true);
  let error = $state('');
  let task = $state<TaskStatus | null>(null);
  let fitting = $state(false);

  let importTreeOpen = $state(false);
  let importFolderOpen = $state(false);
  let newCollectionOpen = $state(false);

  let treePath = $state('');
  let treeMinTracks = $state(1);
  let folderPath = $state('');
  let folderName = $state('');
  let newCollectionName = $state('');
  let newCollectionProfile = $state<AudioProfile>('music');

  const collection = $derived($activeCollection);
  const isSamples = $derived(collection?.profile === 'sample');

  async function load() {
    if (!collection) {
      loading = false;
      return;
    }
    loading = true;
    error = '';
    try {
      const detail = await api.getCollection(collection.id);
      groups = detail.groups;
      metrics = (detail.model?.metrics as CheckMetrics) ?? null;
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (collection) void load();
    else loading = false;
  });

  const accuracyByGroup = $derived(
    new Map((metrics?.per_group ?? []).map((entry) => [entry.group_id, entry]))
  );
  const missingDestination = $derived(groups.filter((group) => !group.destination_path).length);

  async function runTask(start: () => Promise<{ task_id: string }>, label: string) {
    try {
      const { task_id } = await start();
      const finished = await pollTask(task_id, (status) => (task = status));
      task = null;
      notifySuccess(label);
      await Promise.all([load(), loadCollections(), loadGroups(collection!.id)]);
      return finished;
    } catch (err) {
      task = null;
      notifyError(label + ' failed', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function importTree() {
    if (!collection || !treePath) return;
    importTreeOpen = false;
    await runTask(
      () => api.importTree(collection.id, { path: treePath, min_tracks: treeMinTracks }),
      'Folders imported'
    );
  }

  async function importFolder() {
    if (!collection || !folderPath) return;
    importFolderOpen = false;
    await runTask(
      () => api.importFolder(collection.id, { path: folderPath, name: folderName || undefined }),
      'Folder imported'
    );
    folderName = '';
  }

  async function fit() {
    if (!collection) return;
    fitting = true;
    await runTask(() => api.fitCollection(collection.id), 'Groups re-learned');
    fitting = false;
  }

  async function createCollection() {
    if (!newCollectionName.trim()) return;
    try {
      const created = await api.createCollection(
        newCollectionName.trim(),
        undefined,
        newCollectionProfile
      );
      newCollectionOpen = false;
      newCollectionName = '';
      newCollectionProfile = 'music';
      await loadCollections();
      selectCollection(created.id);
      notifySuccess(`Created ${created.name}`);
    } catch (err) {
      notifyError('Could not create collection', err instanceof ApiError ? err.message : String(err));
    }
  }
</script>

<svelte:head><title>Groups · music-cluster</title></svelte:head>

<div class="space-y-6">
  <div class="flex flex-wrap items-start justify-between gap-4">
    <div>
      <div class="flex items-center gap-2">
        <h1 class="text-2xl font-semibold">Groups</h1>
        {#if collection}<ProfileBadge profile={collection.profile} />{/if}
      </div>
      <p class="mt-1 text-sm text-muted-foreground">
        {#if isSamples}
          The folders new one-shots get sorted into. Their samples are what define each one.
        {:else}
          The folders new music gets sorted into. Their tracks are what define each one.
        {/if}
      </p>
    </div>

    <div class="flex flex-wrap gap-2">
      <button
        class="inline-flex items-center gap-2 rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary"
        onclick={() => (newCollectionOpen = true)}
      >
        <Plus class="h-4 w-4" /> New collection
      </button>
      <button
        class="inline-flex items-center gap-2 rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
        onclick={() => (importFolderOpen = true)}
        disabled={!collection}
      >
        <ListPlus class="h-4 w-4" /> Add one folder
      </button>
      <button
        class="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
        onclick={() => (importTreeOpen = true)}
        disabled={!collection}
      >
        <FolderPlus class="h-4 w-4" /> Import my folders
      </button>
    </div>
  </div>

  <TaskProgress {task} label="Analysing tracks" />

  {#if !collection}
    <EmptyState
      icon={FolderTree}
      title="No collection selected"
      description="Create a collection to hold your groups."
    >
      <button class="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground" onclick={() => (newCollectionOpen = true)}>
        Create a collection
      </button>
    </EmptyState>
  {:else if loading}
    <LoadingState message="Loading groups..." />
  {:else if error}
    <ErrorState message={error} onRetry={load} />
  {:else if groups.length === 0}
    <EmptyState
      icon={FolderTree}
      title="No groups in {collection.name}"
      description="If your library is already a folder of genre folders, import the whole tree at once — each subfolder becomes a group with itself as its destination."
    >
      <button class="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground" onclick={() => (importTreeOpen = true)}>
        Import my folders
      </button>
      <a href="/discover" class="rounded-md border border-input px-4 py-2 text-sm hover:bg-secondary">
        Or break down an unsorted pile
      </a>
    </EmptyState>
  {:else}
    <div class="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-secondary/30 px-4 py-3 text-sm">
      <span>
        <span class="font-medium">{pluralize(groups.length, 'group')}</span>
        · {metrics ? `${formatPercent(metrics.accuracy)} self-check` : 'not fitted yet'}
      </span>
      {#if missingDestination > 0}
        <span class="flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
          <TriangleAlert class="h-4 w-4" />
          {missingDestination} without a destination folder
        </span>
      {/if}
      <div class="ml-auto flex gap-2">
        {#if metrics}
          <a href="/groups/check" class="rounded-md border border-input px-3 py-1.5 hover:bg-secondary">
            Overlap report
          </a>
        {/if}
        <button
          class="inline-flex items-center gap-2 rounded-md border border-input px-3 py-1.5 hover:bg-secondary disabled:opacity-50"
          onclick={fit}
          disabled={fitting || groups.length < 2}
        >
          <RefreshCw class="h-4 w-4 {fitting ? 'animate-spin' : ''}" />
          {metrics ? 'Re-fit' : 'Fit'}
        </button>
      </div>
    </div>

    <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {#each groups as group (group.id)}
        {@const quality = accuracyByGroup.get(group.id)}
        <a
          href="/groups/{group.id}"
          class="group rounded-lg border border-border p-4 transition-colors hover:border-primary"
        >
          <div class="flex items-start justify-between gap-2">
            <p class="truncate font-medium">{group.name}</p>
            {#if quality}
              <span
                class="shrink-0 rounded-full px-2 py-0.5 text-xs font-medium tabular-nums
                  {quality.accuracy >= 0.8
                    ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
                    : quality.accuracy >= 0.5
                      ? 'bg-amber-500/15 text-amber-600 dark:text-amber-400'
                      : 'bg-rose-500/15 text-rose-600 dark:text-rose-400'}"
                title="Share of this group's own tracks that get filed back here"
              >
                {formatPercent(quality.accuracy)}
              </span>
            {/if}
          </div>
          <p class="mt-1 text-sm text-muted-foreground">{pluralize(group.size ?? 0, 'track')}</p>
          <p class="mt-2 truncate font-mono text-xs text-muted-foreground">
            {group.destination_path ?? 'No destination set'}
          </p>
        </a>
      {/each}
    </div>
  {/if}
</div>

<Modal bind:open={importTreeOpen} title="Import your folder tree" description="Every subfolder becomes a group, with itself as the destination for new music.">
  <div class="space-y-4">
    <FolderPicker bind:value={treePath} label="Parent folder (the one containing your genre folders)" />
    <label class="block text-sm">
      <span class="font-medium">Skip folders with fewer than</span>
      <input type="number" min="1" bind:value={treeMinTracks} class="ml-2 w-20 rounded-md border border-input bg-background px-2 py-1" />
      <span class="text-muted-foreground">tracks</span>
    </label>
    <p class="text-xs text-muted-foreground">
      Every track in those folders is analysed. Expect roughly a second per track the first time.
    </p>
  </div>
  {#snippet footer()}
    <button class="rounded-md border border-input px-3 py-1.5 text-sm" onclick={() => (importTreeOpen = false)}>Cancel</button>
    <button class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50" onclick={importTree} disabled={!treePath}>
      Import
    </button>
  {/snippet}
</Modal>

<Modal bind:open={importFolderOpen} title="Add one folder as a group">
  <div class="space-y-4">
    <FolderPicker bind:value={folderPath} label="Folder" />
    <label class="block text-sm">
      <span class="font-medium">Group name</span>
      <input
        class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2"
        bind:value={folderName}
        placeholder="Defaults to the folder name"
      />
    </label>
  </div>
  {#snippet footer()}
    <button class="rounded-md border border-input px-3 py-1.5 text-sm" onclick={() => (importFolderOpen = false)}>Cancel</button>
    <button class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50" onclick={importFolder} disabled={!folderPath}>
      Import
    </button>
  {/snippet}
</Modal>

<Modal bind:open={newCollectionOpen} title="New collection" description="A collection is one sorting scheme — a set of groups that new music gets filed into." size="sm">
  <label class="block text-sm">
    <span class="font-medium">Name</span>
    <input
      class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2"
      bind:value={newCollectionName}
      placeholder={newCollectionProfile === 'sample' ? 'e.g. Drum library' : 'e.g. DJ folders'}
    />
  </label>

  <fieldset class="mt-4 text-sm">
    <legend class="font-medium">What is in it</legend>
    <div class="mt-2 grid gap-2 sm:grid-cols-2">
      {#each [{ value: 'music', title: 'Music', blurb: 'Full-length tracks, mixes and edits.' }, { value: 'sample', title: 'Samples', blurb: 'One-shots and loops: kicks, snares, claps, hats, basses, chords.' }] as option (option.value)}
        <label
          class="cursor-pointer rounded-md border p-3 transition-colors {newCollectionProfile ===
          option.value
            ? 'border-primary bg-primary/5'
            : 'border-input hover:bg-muted/50'}"
        >
          <input
            type="radio"
            class="sr-only"
            name="collection-profile"
            value={option.value}
            bind:group={newCollectionProfile}
          />
          <span class="block font-medium">{option.title}</span>
          <span class="mt-0.5 block text-xs text-muted-foreground">{option.blurb}</span>
        </label>
      {/each}
    </div>
    <p class="mt-2 text-xs text-muted-foreground">
      Fixed once the collection exists — samples and tracks are measured differently, so
      sorting both means two collections.
    </p>
  </fieldset>
  {#snippet footer()}
    <button class="rounded-md border border-input px-3 py-1.5 text-sm" onclick={() => (newCollectionOpen = false)}>Cancel</button>
    <button class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground" onclick={createCollection}>Create</button>
  {/snippet}
</Modal>
