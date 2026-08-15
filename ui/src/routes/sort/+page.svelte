<script lang="ts">
  import { goto } from '$app/navigation';
  import { api, ApiError, pollTask } from '$lib/services/api';
  import { activeCollection } from '$lib/stores/app';
  import { notifyError, notifySuccess } from '$lib/stores/notifications';
  import EmptyState from '$lib/components/EmptyState.svelte';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import FolderPicker from '$lib/components/FolderPicker.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import TaskProgress from '$lib/components/TaskProgress.svelte';
  import type { SortSession, TaskStatus } from '$lib/types';
  import { formatDate } from '$lib/utils';
  import { Inbox, Plus, Trash2 } from 'lucide-svelte';

  let sessions = $state<SortSession[]>([]);
  let loading = $state(true);
  let error = $state('');
  let task = $state<TaskStatus | null>(null);

  let newOpen = $state(false);
  let sourcePath = $state('');
  let sessionName = $state('');
  let autoAccept = $state(false);

  const collection = $derived($activeCollection);

  async function load() {
    loading = true;
    error = '';
    try {
      const result = await api.listSessions();
      sessions = result.sessions;
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void load();
  });

  async function startSort() {
    if (!collection || !sourcePath) return;
    newOpen = false;
    try {
      const { task_id } = await api.createSession({
        collection_id: collection.id,
        paths: [sourcePath],
        name: sessionName || undefined,
        auto_accept: autoAccept
      });
      const finished = await pollTask(task_id, (status) => (task = status));
      task = null;
      sessionName = '';

      const result = finished.result as { session: SortSession } | null;
      if (result?.session) {
        notifySuccess('Ready to review');
        void goto(`/sort/${result.session.id}`);
      } else {
        await load();
      }
    } catch (err) {
      task = null;
      notifyError('Could not sort', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function remove(session: SortSession) {
    try {
      await api.deleteSession(session.id);
      sessions = sessions.filter((item) => item.id !== session.id);
      notifySuccess('Session deleted');
    } catch (err) {
      notifyError('Could not delete', err instanceof ApiError ? err.message : String(err));
    }
  }
</script>

<svelte:head><title>Sort · music-cluster</title></svelte:head>

<div class="space-y-6">
  <div class="flex flex-wrap items-start justify-between gap-4">
    <div>
      <h1 class="text-2xl font-semibold">Sort</h1>
      <p class="mt-1 text-sm text-muted-foreground">
        Point at a folder of new music and decide which group each track belongs to.
      </p>
    </div>
    <button
      class="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
      onclick={() => (newOpen = true)}
      disabled={!collection}
      title={collection ? '' : 'Select a collection first'}
    >
      <Plus class="h-4 w-4" /> New sort
    </button>
  </div>

  <TaskProgress {task} label="Analysing and scoring" />

  {#if loading}
    <LoadingState message="Loading sessions..." />
  {:else if error}
    <ErrorState message={error} onRetry={load} />
  {:else if sessions.length === 0}
    <EmptyState
      icon={Inbox}
      title="No sort sessions yet"
      description="A session takes a folder of incoming music, scores every track against your groups, and holds the suggestions until you have signed off on them."
    >
      <button class="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground disabled:opacity-50" onclick={() => (newOpen = true)} disabled={!collection}>
        Sort a folder
      </button>
    </EmptyState>
  {:else}
    <div class="divide-y divide-border rounded-lg border border-border">
      {#each sessions as session (session.id)}
        <div class="flex items-center gap-4 px-4 py-3">
          <a href="/sort/{session.id}" class="min-w-0 flex-1">
            <p class="truncate font-medium">{session.name ?? `Session ${session.id}`}</p>
            <p class="truncate text-xs text-muted-foreground">
              {session.collection_name} · {formatDate(session.created_at)}
              {#if session.source_path}· <span class="font-mono">{session.source_path}</span>{/if}
            </p>
          </a>

          {#if (session.pending_count ?? 0) > 0}
            <span class="shrink-0 rounded-full bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary">
              {session.pending_count} to review
            </span>
          {:else}
            <span class="shrink-0 rounded-full bg-secondary px-2.5 py-1 text-xs text-muted-foreground">
              {session.status}
            </span>
          {/if}
          <span class="hidden w-20 shrink-0 text-right text-xs text-muted-foreground sm:block">
            {session.item_count} tracks
          </span>

          <button
            class="shrink-0 rounded p-1.5 text-muted-foreground hover:bg-secondary hover:text-destructive"
            onclick={() => remove(session)}
            aria-label="Delete session"
          >
            <Trash2 class="h-4 w-4" />
          </button>
        </div>
      {/each}
    </div>
  {/if}
</div>

<Modal bind:open={newOpen} title="Sort new music" description="Every track is analysed, then scored against the groups in {collection?.name ?? 'your collection'}.">
  <div class="space-y-4">
    <FolderPicker bind:value={sourcePath} label="Folder of new music" />
    <label class="block text-sm">
      <span class="font-medium">Session name</span>
      <input
        class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2"
        bind:value={sessionName}
        placeholder="Defaults to the folder name"
      />
    </label>
    <label class="flex items-start gap-2 text-sm">
      <input type="checkbox" bind:checked={autoAccept} class="mt-1 accent-primary" />
      <span>
        <span class="font-medium">Pre-accept confident suggestions</span>
        <span class="block text-xs text-muted-foreground">
          You will only review the tracks the sorter is unsure about. Everything still goes through
          the commit step before anything is written to disk.
        </span>
      </span>
    </label>
  </div>
  {#snippet footer()}
    <button class="rounded-md border border-input px-3 py-1.5 text-sm" onclick={() => (newOpen = false)}>Cancel</button>
    <button class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50" onclick={startSort} disabled={!sourcePath}>
      Start
    </button>
  {/snippet}
</Modal>
