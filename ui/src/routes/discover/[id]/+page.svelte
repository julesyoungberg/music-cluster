<script lang="ts">
  import { page } from '$app/stores';
  import { api, ApiError } from '$lib/services/api';
  import { activeCollection, loadCollections, loadGroups } from '$lib/stores/app';
  import { notifyError, notifySuccess } from '$lib/stores/notifications';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import FolderPicker from '$lib/components/FolderPicker.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import TrackRow from '$lib/components/TrackRow.svelte';
  import ProfileBadge from '$lib/components/ProfileBadge.svelte';
  import type { DiscoveryCandidate, DiscoveryRun, Track } from '$lib/types';
  import { formatSampleLength, pluralize } from '$lib/utils';
  import { ArrowLeft, Check, ChevronDown, ChevronUp, X } from 'lucide-svelte';

  const runId = $derived(Number($page.params.id));

  let run = $state<DiscoveryRun | null>(null);
  let candidates = $state<DiscoveryCandidate[]>([]);
  let loading = $state(true);
  let error = $state('');
  let expanded = $state<number | null>(null);
  let members = $state<Record<number, Track[]>>({});

  let promoteOpen = $state(false);
  let promoting = $state<DiscoveryCandidate | null>(null);
  let groupName = $state('');
  let destination = $state('');
  let seedsOnly = $state(false);
  let destinationOpen = $state(false);

  const collection = $derived($activeCollection);
  const pending = $derived(candidates.filter((candidate) => candidate.status === 'pending'));

  async function load() {
    loading = true;
    error = '';
    try {
      const result = await api.getDiscoveryRun(runId);
      run = result.run;
      candidates = result.candidates;
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (Number.isFinite(runId)) void load();
  });

  async function toggle(candidate: DiscoveryCandidate) {
    if (expanded === candidate.id) {
      expanded = null;
      return;
    }
    expanded = candidate.id;
    if (!members[candidate.id]) {
      try {
        const result = await api.candidateMembers(candidate.id, 60);
        members = { ...members, [candidate.id]: result.tracks };
      } catch (err) {
        notifyError('Could not load tracks', err instanceof ApiError ? err.message : String(err));
      }
    }
  }

  function openPromote(candidate: DiscoveryCandidate) {
    promoting = candidate;
    groupName = candidate.suggested_name;
    destination = '';
    seedsOnly = false;
    promoteOpen = true;
  }

  async function promote() {
    if (!promoting || !collection) return;
    try {
      const result = await api.promoteCandidate(promoting.id, {
        collection_id: collection.id,
        name: groupName || undefined,
        destination_path: destination || undefined,
        seeds_only: seedsOnly
      });
      promoteOpen = false;
      notifySuccess(
        `Created ${result.group.name}`,
        `${pluralize(result.added, 'reference track')} added`
      );
      await Promise.all([load(), loadCollections(), loadGroups(collection.id)]);
    } catch (err) {
      notifyError('Could not promote', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function reject(candidate: DiscoveryCandidate) {
    try {
      await api.rejectCandidate(candidate.id);
      candidates = candidates.map((item) =>
        item.id === candidate.id ? { ...item, status: 'rejected' } : item
      );
    } catch (err) {
      notifyError('Could not discard', err instanceof ApiError ? err.message : String(err));
    }
  }
</script>

<svelte:head><title>{run?.name ?? 'Candidates'} · music-cluster</title></svelte:head>

<a
  href="/discover"
  class="mb-4 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
>
  <ArrowLeft class="h-4 w-4" /> All runs
</a>

{#if loading}
  <LoadingState message="Loading candidates..." />
{:else if error || !run}
  <ErrorState message={error} onRetry={load} />
{:else}
  <div class="space-y-5">
    <div>
      <div class="flex items-center gap-2">
        <h1 class="text-2xl font-semibold">{run.name ?? `Run ${run.id}`}</h1>
        <ProfileBadge profile={run.profile} />
      </div>
      <p class="mt-1 text-sm text-muted-foreground">
        {pluralize(pending.length, 'candidate')} still to review · found with {run.algorithm}
      </p>
      {#if !collection}
        <p class="mt-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm">
          Select or create a collection before promoting candidates into groups.
        </p>
      {:else if (run.profile ?? 'music') !== (collection.profile ?? 'music')}
        <p class="mt-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm">
          This run holds {run.profile ?? 'music'} audio, but
          <strong>{collection.name}</strong> sorts {collection.profile ?? 'music'}. Switch to a
          {run.profile ?? 'music'} collection before promoting these candidates.
        </p>
      {/if}
    </div>

    <div class="space-y-3">
      {#each candidates as candidate (candidate.id)}
        <div
          class="rounded-lg border transition-colors {candidate.status === 'pending'
            ? 'border-border'
            : 'border-border/50 opacity-60'}"
        >
          <div class="flex flex-wrap items-start gap-4 p-4">
            <div class="min-w-0 flex-1">
              <div class="flex flex-wrap items-center gap-2">
                <h2 class="font-medium">{candidate.suggested_name}</h2>
                {#if candidate.status === 'promoted'}
                  <span
                    class="rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs text-emerald-600 dark:text-emerald-400"
                  >
                    kept
                  </span>
                {:else if candidate.status === 'rejected'}
                  <span class="rounded-full bg-secondary px-2 py-0.5 text-xs text-muted-foreground"
                    >discarded</span
                  >
                {/if}
              </div>

              {#if candidate.stats.kind === 'sample'}
                <p class="mt-1 text-sm text-muted-foreground">
                  {pluralize(candidate.size, 'sample')}
                  {#if candidate.stats.median_duration}
                    · {formatSampleLength(candidate.stats.median_duration)}
                  {/if}
                  {#if candidate.stats.attack_ms !== undefined}
                    · {Math.round(candidate.stats.attack_ms)} ms attack
                  {/if}
                  · {candidate.stats.pitched ? 'pitched' : 'unpitched'}
                  {#if candidate.stats.f0_hz}
                    ({Math.round(candidate.stats.f0_hz)} Hz)
                  {/if}
                </p>
                {#if candidate.stats.breakdown?.length}
                  <p class="mt-1 text-xs text-muted-foreground">
                    contains: {candidate.stats.breakdown
                      .slice(0, 4)
                      .map((entry) => `${entry.label} (${entry.count})`)
                      .join(', ')}
                  </p>
                {/if}
              {:else}
                <p class="mt-1 text-sm text-muted-foreground">
                  {pluralize(candidate.size, 'track')}
                  {#if candidate.stats.tag_bpm_median || candidate.stats.detected_bpm}
                    · ~{Math.round(
                      candidate.stats.tag_bpm_median ?? candidate.stats.detected_bpm ?? 0
                    )} BPM
                  {/if}
                  {#if candidate.stats.descriptors?.length}
                    · {candidate.stats.descriptors.join(', ')}
                  {/if}
                </p>

                {#if candidate.stats.top_genres?.length}
                  <p class="mt-1 text-xs text-muted-foreground">
                    tags: {candidate.stats.top_genres
                      .map((genre) => `${genre.name} (${genre.count})`)
                      .join(', ')}
                  </p>
                {/if}
                {#if candidate.stats.top_artists?.length}
                  <p class="text-xs text-muted-foreground">
                    artists: {candidate.stats.top_artists.map((artist) => artist.name).join(', ')}
                  </p>
                {/if}
              {/if}
            </div>

            {#if candidate.status === 'pending'}
              <div class="flex shrink-0 gap-2">
                <button
                  class="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
                  onclick={() => openPromote(candidate)}
                  disabled={!collection}
                >
                  <Check class="h-4 w-4" /> Keep as group
                </button>
                <button
                  class="rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary"
                  onclick={() => reject(candidate)}
                >
                  <X class="h-4 w-4" />
                </button>
              </div>
            {/if}
          </div>

          <div class="border-t border-border px-4 py-3">
            <p class="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Representative tracks
            </p>
            <div class="divide-y divide-border">
              {#each candidate.exemplar_tracks ?? [] as track (track.id)}
                <TrackRow {track} dense />
              {/each}
            </div>

            <button
              class="mt-2 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              onclick={() => toggle(candidate)}
            >
              {#if expanded === candidate.id}
                <ChevronUp class="h-3 w-3" /> Hide all {candidate.size} tracks
              {:else}
                <ChevronDown class="h-3 w-3" /> Show all {candidate.size} tracks
              {/if}
            </button>

            {#if expanded === candidate.id}
              <div
                class="mt-2 max-h-80 divide-y divide-border overflow-y-auto rounded-md border border-border px-3"
              >
                {#each members[candidate.id] ?? [] as track (track.id)}
                  <TrackRow {track} dense />
                {:else}
                  <p class="py-4 text-center text-sm text-muted-foreground">Loading...</p>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  </div>
{/if}

<Modal
  bind:open={promoteOpen}
  title="Keep this as a group"
  description="It becomes a real group in {collection?.name ??
    'your collection'}, and new music can be sorted into it."
>
  <div class="space-y-4">
    <label class="block text-sm">
      <span class="font-medium">Group name</span>
      <input
        class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2"
        bind:value={groupName}
      />
    </label>

    <div class="text-sm">
      <span class="font-medium">Destination folder</span>
      <p
        class="mt-1 break-all rounded-md border border-border bg-secondary/40 px-3 py-2 font-mono text-xs"
      >
        {destination || 'Not set — playlists only'}
      </p>
      <button
        class="mt-1 text-xs text-primary hover:underline"
        onclick={() => (destinationOpen = true)}
      >
        Choose folder
      </button>
    </div>

    <label class="flex items-start gap-2 text-sm">
      <input type="checkbox" bind:checked={seedsOnly} class="mt-1 accent-primary" />
      <span>
        <span class="font-medium">Use only the representative tracks</span>
        <span class="block text-xs text-muted-foreground">
          Adds {promoting?.exemplars.length ?? 0} tracks instead of all {promoting?.size ?? 0}.
          Worth it when the pile is roughly right but you do not trust every track in it to define
          the group.
        </span>
      </span>
    </label>
  </div>

  {#snippet footer()}
    <button
      class="rounded-md border border-input px-3 py-1.5 text-sm"
      onclick={() => (promoteOpen = false)}>Cancel</button
    >
    <button
      class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground"
      onclick={promote}>Create group</button
    >
  {/snippet}
</Modal>

<Modal bind:open={destinationOpen} title="Destination folder">
  <FolderPicker bind:value={destination} label="Folder" />
  {#snippet footer()}
    <button
      class="rounded-md border border-input px-3 py-1.5 text-sm"
      onclick={() => (destinationOpen = false)}>Done</button
    >
  {/snippet}
</Modal>
