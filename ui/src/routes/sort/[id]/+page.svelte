<script lang="ts">
  import { page } from '$app/stores';
  import { api, ApiError } from '$lib/services/api';
  import { loadCollections, loadGroups } from '$lib/stores/app';
  import { notifyError, notifyInfo, notifySuccess } from '$lib/stores/notifications';
  import ConfidenceBadge from '$lib/components/ConfidenceBadge.svelte';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import GroupPicker from '$lib/components/GroupPicker.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import Waveform from '$lib/components/Waveform.svelte';
  import PlayButton from '$lib/components/PlayButton.svelte';
  import { playTrack } from '$lib/stores/player';
  import type { CommitPlan, Group, OrganizeMode, SortItem, SortSession } from '$lib/types';
  import { formatDuration, pluralize, trackLabel, trackOf } from '$lib/utils';
  import {
    ArrowLeft,
    Check,
    ChevronLeft,
    ChevronRight,
    FastForward,
    HelpCircle,
    Undo2
  } from 'lucide-svelte';

  const sessionId = $derived(Number($page.params.id));

  let session = $state<SortSession | null>(null);
  let groups = $state<Group[]>([]);
  let items = $state<SortItem[]>([]);
  let summary = $state<Record<string, number>>({});
  let loading = $state(true);
  let error = $state('');

  let cursor = $state(0);
  let filter = $state<'pending' | 'all' | 'accepted' | 'unmatched' | 'skipped'>('pending');
  let pickerOpen = $state(false);
  let commitOpen = $state(false);
  let helpOpen = $state(false);

  let mode = $state<OrganizeMode>('copy');
  let plan = $state<CommitPlan | null>(null);
  let committing = $state(false);

  const current = $derived(items[cursor] ?? null);
  // A sort item's own ID is not its track's — everything that touches audio
  // goes through the track.
  const currentTrack = $derived(current ? trackOf(current) : null);
  const groupsById = $derived(new Map(groups.map((group) => [group.id, group])));
  const pendingCount = $derived(summary.pending ?? 0);
  const acceptedCount = $derived(summary.accepted ?? 0);
  const totalCount = $derived(Object.values(summary).reduce((sum, value) => sum + value, 0));

  async function load(keepCursor = false) {
    loading = !keepCursor;
    error = '';
    try {
      const overview = await api.getSession(sessionId);
      session = overview.session;
      groups = overview.groups;

      const result = await api.sessionItems(sessionId, {
        status: filter === 'all' ? undefined : filter,
        limit: 500
      });
      items = result.items;
      summary = result.summary;
      if (!keepCursor || cursor >= items.length)
        cursor = Math.min(cursor, Math.max(items.length - 1, 0));
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (Number.isFinite(sessionId)) {
      // Re-runs when the filter changes, which is the intended behaviour.
      filter;
      cursor = 0;
      void load();
    }
  });

  /** Record a decision and advance. In the pending view the item leaves the list. */
  async function decide(groupId: number | null, skip = false) {
    const item = current;
    if (!item) return;

    try {
      await api.decideItem(item.id!, { group_id: groupId, skip });
      summary = (await api.getSession(sessionId)).summary;

      if (filter === 'pending') {
        items = items.filter((entry) => entry.id !== item.id);
        if (cursor >= items.length) cursor = Math.max(items.length - 1, 0);
      } else {
        items[cursor] = {
          ...item,
          status: skip ? 'skipped' : 'accepted',
          final_group_id: groupId,
          final_group_name: groupId ? (groupsById.get(groupId)?.name ?? null) : null
        };
        next();
      }
    } catch (err) {
      notifyError('Could not save decision', err instanceof ApiError ? err.message : String(err));
    }
  }

  function next() {
    if (cursor < items.length - 1) cursor += 1;
  }
  function previous() {
    if (cursor > 0) cursor -= 1;
  }

  async function acceptConfident() {
    try {
      const result = await api.acceptConfident(sessionId);
      notifySuccess(`Accepted ${pluralize(result.accepted, 'confident suggestion')}`);
      await load();
    } catch (err) {
      notifyError('Could not accept', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function preview() {
    try {
      const result = await api.commit(sessionId, { mode, dry_run: true });
      plan = result.plan;
      commitOpen = true;
    } catch (err) {
      notifyError('Could not build plan', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function commit() {
    committing = true;
    try {
      const result = await api.commit(sessionId, { mode });
      commitOpen = false;
      const performed = result.result?.performed_count ?? 0;
      const playlists = result.result?.playlists?.length ?? 0;
      notifySuccess(
        'Committed',
        mode === 'playlist'
          ? `Wrote ${pluralize(playlists, 'playlist')}`
          : `${pluralize(performed, 'file')} ${mode === 'move' ? 'moved' : mode === 'copy' ? 'copied' : 'linked'}`
      );
      if (result.learned) {
        notifyInfo(
          `${pluralize(result.learned, 'track')} added as new references`,
          'Re-fit the collection to fold them in.'
        );
      }
      await Promise.all([
        load(),
        loadCollections(),
        session ? loadGroups(session.collection_id) : Promise.resolve([])
      ]);
    } catch (err) {
      notifyError('Commit failed', err instanceof ApiError ? err.message : String(err));
    } finally {
      committing = false;
    }
  }

  async function undo() {
    try {
      const result = await api.undo(sessionId);
      notifySuccess(`Reversed ${pluralize(result.undone, 'operation')}`);
      await load();
    } catch (err) {
      notifyError('Undo failed', err instanceof ApiError ? err.message : String(err));
    }
  }

  function onKeydown(event: KeyboardEvent) {
    const target = event.target as HTMLElement;
    if (pickerOpen || commitOpen || helpOpen) return;
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;

    const ranked = current?.ranked ?? [];
    if (/^[1-5]$/.test(event.key) && ranked[Number(event.key) - 1]) {
      event.preventDefault();
      void decide(ranked[Number(event.key) - 1].group_id);
    } else if (event.key === ' ') {
      event.preventDefault();
      if (currentTrack) void playTrack(currentTrack);
    } else if (event.key === 's') {
      event.preventDefault();
      void decide(null, true);
    } else if (event.key === 'g') {
      event.preventDefault();
      pickerOpen = true;
    } else if (event.key === 'ArrowRight' || event.key === 'j') {
      event.preventDefault();
      next();
    } else if (event.key === 'ArrowLeft' || event.key === 'k') {
      event.preventDefault();
      previous();
    } else if (event.key === '?') {
      helpOpen = true;
    }
  }
</script>

<svelte:head><title>{session?.name ?? 'Review'} · music-cluster</title></svelte:head>
<svelte:window on:keydown={onKeydown} />

<a
  href="/sort"
  class="mb-4 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
>
  <ArrowLeft class="h-4 w-4" /> All sessions
</a>

{#if loading}
  <LoadingState message="Loading session..." />
{:else if error || !session}
  <ErrorState message={error} onRetry={() => load()} />
{:else}
  <div class="space-y-5">
    <div class="flex flex-wrap items-start justify-between gap-4">
      <div>
        <h1 class="text-2xl font-semibold">{session.name ?? `Session ${session.id}`}</h1>
        <p class="mt-1 text-sm text-muted-foreground">
          {pendingCount} to review · {acceptedCount} filed · {totalCount} total
        </p>
      </div>
      <div class="flex flex-wrap gap-2">
        <button
          class="rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary"
          onclick={() => (helpOpen = true)}
        >
          <HelpCircle class="inline h-4 w-4" /> Shortcuts
        </button>
        <button
          class="rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
          onclick={acceptConfident}
          disabled={pendingCount === 0}
          title="Accept every pending suggestion that cleared the confidence threshold"
        >
          <FastForward class="inline h-4 w-4" /> Accept confident
        </button>
        <button
          class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
          onclick={preview}
          disabled={acceptedCount === 0}
        >
          Commit {acceptedCount || ''}
        </button>
        <button
          class="rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary"
          onclick={undo}
          title="Reverse this session's file operations"
        >
          <Undo2 class="inline h-4 w-4" />
        </button>
      </div>
    </div>

    <div class="h-1.5 overflow-hidden rounded-full bg-secondary">
      <div
        class="h-full bg-primary transition-all"
        style="width: {totalCount > 0 ? ((totalCount - pendingCount) / totalCount) * 100 : 0}%"
      ></div>
    </div>

    <div class="flex flex-wrap gap-1 text-sm">
      {#each [['pending', 'To review'], ['accepted', 'Filed'], ['unmatched', 'No match'], ['skipped', 'Skipped'], ['all', 'Everything']] as [value, label]}
        <button
          class="rounded-md px-3 py-1.5 transition-colors {filter === value
            ? 'bg-secondary font-medium'
            : 'text-muted-foreground hover:bg-secondary/60'}"
          onclick={() => (filter = value as typeof filter)}
        >
          {label}
          {#if value !== 'all' && summary[value]}<span class="ml-1 text-xs opacity-70"
              >{summary[value]}</span
            >{/if}
        </button>
      {/each}
    </div>

    {#if items.length === 0}
      <div class="rounded-lg border border-dashed border-border px-6 py-14 text-center">
        <Check class="mx-auto h-7 w-7 text-emerald-500" />
        <p class="mt-3 font-medium">
          {filter === 'pending' ? 'Nothing left to review' : 'Nothing here'}
        </p>
        {#if filter === 'pending' && acceptedCount > 0}
          <p class="mt-1 text-sm text-muted-foreground">
            {pluralize(acceptedCount, 'track')} ready to file. Commit when you are.
          </p>
        {/if}
      </div>
    {:else if current}
      <div class="grid gap-5 lg:grid-cols-[1fr_22rem]">
        <section class="space-y-4 rounded-lg border border-border p-5">
          <div class="flex items-start gap-3">
            <PlayButton track={currentTrack!} size="lg" />
            <div class="min-w-0 flex-1">
              <h2 class="truncate text-lg font-semibold">{trackLabel(current)}</h2>
              <p class="truncate text-sm text-muted-foreground">
                {[
                  current.album,
                  current.genre,
                  current.tag_bpm ? `${Math.round(current.tag_bpm)} BPM` : null,
                  current.tag_key,
                  formatDuration(current.duration)
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </p>
            </div>
            <span class="shrink-0 text-sm tabular-nums text-muted-foreground">
              {cursor + 1} / {items.length}
            </span>
          </div>

          <Waveform track={currentTrack!} height={56} />

          {#if current.status !== 'pending'}
            <p class="rounded-md bg-secondary/60 px-3 py-2 text-sm">
              {current.status === 'accepted'
                ? `Filed into ${current.final_group_name ?? 'a group'}`
                : current.status === 'skipped'
                  ? 'Skipped'
                  : 'No group came close'}
            </p>
          {/if}

          <div class="space-y-2">
            <p class="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Suggested groups
            </p>

            {#if current.ranked.length === 0}
              <p class="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-3 text-sm">
                Nothing in this collection sounds close to this track. It may need a group of its
                own — or it may just be unlike anything you have filed so far.
              </p>
            {:else}
              {#each current.ranked as entry, index (entry.group_id)}
                <button
                  class="flex w-full items-center gap-3 rounded-md border border-border px-3 py-2.5 text-left transition-colors hover:border-primary hover:bg-secondary/50"
                  onclick={() => decide(entry.group_id)}
                >
                  <kbd
                    class="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-border text-xs"
                  >
                    {index + 1}
                  </kbd>
                  <span class="min-w-0 flex-1">
                    <span class="block truncate font-medium">{entry.group_name}</span>
                    <span class="mt-1 block h-1 w-full overflow-hidden rounded-full bg-secondary">
                      <span class="block h-full bg-primary" style="width: {entry.score * 100}%"
                      ></span>
                    </span>
                  </span>
                  <ConfidenceBadge confidence={entry.score} />
                </button>
              {/each}
            {/if}

            <div class="flex gap-2 pt-1">
              <button
                class="flex-1 rounded-md border border-input px-3 py-2 text-sm hover:bg-secondary"
                onclick={() => (pickerOpen = true)}
              >
                Choose another group <kbd class="ml-1 text-xs opacity-60">g</kbd>
              </button>
              <button
                class="rounded-md border border-input px-3 py-2 text-sm text-muted-foreground hover:bg-secondary"
                onclick={() => decide(null, true)}
              >
                Skip <kbd class="ml-1 text-xs opacity-60">s</kbd>
              </button>
            </div>
          </div>

          <div class="flex items-center justify-between border-t border-border pt-3">
            <button
              class="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground disabled:opacity-40"
              onclick={previous}
              disabled={cursor === 0}
            >
              <ChevronLeft class="h-4 w-4" /> Previous
            </button>
            <p
              class="truncate px-3 font-mono text-xs text-muted-foreground"
              title={current.filepath}
            >
              {current.filename}
            </p>
            <button
              class="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground disabled:opacity-40"
              onclick={next}
              disabled={cursor >= items.length - 1}
            >
              Next <ChevronRight class="h-4 w-4" />
            </button>
          </div>
        </section>

        <aside class="space-y-2">
          <p class="text-xs font-medium uppercase tracking-wide text-muted-foreground">Queue</p>
          <div
            class="max-h-[32rem] divide-y divide-border overflow-y-auto rounded-lg border border-border"
          >
            {#each items as item, index (item.id)}
              <button
                class="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors {index ===
                cursor
                  ? 'bg-secondary'
                  : 'hover:bg-secondary/50'}"
                onclick={() => (cursor = index)}
              >
                <span class="min-w-0 flex-1">
                  <span class="block truncate text-sm">{trackLabel(item)}</span>
                  <span class="block truncate text-xs text-muted-foreground">
                    {item.status === 'accepted'
                      ? (item.final_group_name ?? 'filed')
                      : (item.ranked[0]?.group_name ?? 'no match')}
                  </span>
                </span>
                <ConfidenceBadge confidence={item.ranked[0]?.score ?? null} />
              </button>
            {/each}
          </div>
        </aside>
      </div>
    {/if}
  </div>
{/if}

<Modal bind:open={pickerOpen} title="Choose a group" size="sm">
  <GroupPicker
    {groups}
    selectedId={current?.final_group_id ?? null}
    onSelect={(group) => {
      pickerOpen = false;
      void decide(group.id);
    }}
  />
</Modal>

<Modal
  bind:open={commitOpen}
  title="Commit this session"
  description="Nothing has been written yet — this is what would happen."
  size="lg"
>
  <div class="space-y-4">
    <div class="flex flex-wrap gap-2">
      {#each [['playlist', 'Playlists only'], ['copy', 'Copy files'], ['move', 'Move files'], ['symlink', 'Symlink']] as [value, label]}
        <button
          class="rounded-md border px-3 py-1.5 text-sm transition-colors {mode === value
            ? 'border-primary bg-primary/10'
            : 'border-input hover:bg-secondary'}"
          onclick={async () => {
            mode = value as OrganizeMode;
            const result = await api.commit(sessionId, { mode, dry_run: true });
            plan = result.plan;
          }}
        >
          {label}
        </button>
      {/each}
    </div>

    {#if mode === 'move'}
      <p class="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm">
        Moving relocates the original files. The session records every move, so
        <span class="font-medium">Undo</span> can put them back.
      </p>
    {/if}

    {#if plan}
      <div class="max-h-72 overflow-y-auto rounded-md border border-border">
        {#each plan.playlists as entry}
          <div class="border-b border-border px-3 py-2 text-sm last:border-0">
            <span class="font-medium">{entry.group_name}</span>
            <span class="text-muted-foreground"> — {pluralize(entry.track_count, 'track')} → </span>
            <span class="font-mono text-xs">{entry.path}</span>
          </div>
        {/each}
        {#each plan.actions.slice(0, 100) as action}
          <div class="border-b border-border px-3 py-2 text-sm last:border-0">
            <span class="font-medium">{action.group_name}</span>
            <span class="ml-2 font-mono text-xs text-muted-foreground">{action.dest_path}</span>
            {#if action.note}<span class="ml-2 text-xs text-amber-600 dark:text-amber-400"
                >{action.note}</span
              >{/if}
          </div>
        {/each}
        {#if plan.actions.length === 0 && plan.playlists.length === 0}
          <p class="px-3 py-6 text-center text-sm text-muted-foreground">Nothing to write.</p>
        {/if}
      </div>

      {#if plan.problems.length > 0}
        <div class="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
          <p class="font-medium">{pluralize(plan.problems.length, 'track')} will be skipped</p>
          <ul class="mt-1 space-y-0.5 text-xs">
            {#each plan.problems.slice(0, 6) as problem}
              <li>
                {problem.group_name ?? problem.filepath ?? `Track ${problem.track_id}`}: {problem.issue}
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    {/if}
  </div>

  {#snippet footer()}
    <button
      class="rounded-md border border-input px-3 py-1.5 text-sm"
      onclick={() => (commitOpen = false)}>Cancel</button
    >
    <button
      class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-60"
      onclick={commit}
      disabled={committing || !plan || (plan.actions.length === 0 && plan.playlists.length === 0)}
    >
      {committing ? 'Writing...' : 'Commit'}
    </button>
  {/snippet}
</Modal>

<Modal bind:open={helpOpen} title="Keyboard shortcuts" size="sm">
  <dl class="space-y-2 text-sm">
    {#each [['1 – 5', 'File into the nth suggested group'], ['g', 'Pick any group'], ['s', 'Skip this track'], ['Space', 'Play / pause'], ['← / →', 'Move through the queue'], ['? ', 'This help']] as [key, description]}
      <div class="flex items-center gap-4">
        <dt><kbd class="rounded border border-border px-2 py-0.5 text-xs">{key}</kbd></dt>
        <dd class="text-muted-foreground">{description}</dd>
      </div>
    {/each}
  </dl>
</Modal>
