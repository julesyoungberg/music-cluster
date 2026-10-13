<script lang="ts">
  import { api, ApiError, pollTask } from '$lib/services/api';
  import {
    activeCollection,
    groups as groupStore,
    loadCollections,
    loadGroups
  } from '$lib/stores/app';
  import { notifyError, notifySuccess } from '$lib/stores/notifications';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import FolderPicker from '$lib/components/FolderPicker.svelte';
  import GroupPicker from '$lib/components/GroupPicker.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import TaskProgress from '$lib/components/TaskProgress.svelte';
  import TrackRow from '$lib/components/TrackRow.svelte';
  import type { AudioProfile, TagVocabulary, TaskStatus, Track } from '$lib/types';
  import { energyLevelToBound, pluralize } from '$lib/utils';
  import { FolderPlus, Search, Sparkles, Tags } from 'lucide-svelte';

  let tracks = $state<Track[]>([]);
  let total = $state(0);
  let loading = $state(true);
  let error = $state('');
  let query = $state('');
  let onlyUnassigned = $state(false);
  let offset = $state(0);
  let task = $state<TaskStatus | null>(null);

  let selected = $state<Set<number>>(new Set());
  let assignOpen = $state(false);
  let analyzeOpen = $state(false);
  let analyzePath = $state('');
  let analyzeProfile = $state<AudioProfile>('music');

  // 'all' is the default so the library stays one place; the filter is for
  // when a database holds both a record collection and a sample library.
  let profileFilter = $state<AudioProfile | 'all'>('all');

  // Auto-tag filters. Empty strings mean "any".
  let vocabulary = $state<TagVocabulary | null>(null);
  let genreFilter = $state('');
  let beatFilter = $state('');
  let moodFilter = $state('');
  let minEnergyLevel = $state(0);
  let tagTask = $state<TaskStatus | null>(null);
  const tagFiltering = $derived(Boolean(genreFilter || beatFilter || moodFilter || minEnergyLevel));

  let similarOpen = $state(false);
  let similarSeed = $state<Track | null>(null);
  let similarResults = $state<{ track: Track; relative_distance: number }[]>([]);
  let similarLoading = $state(false);

  const collection = $derived($activeCollection);
  const pageSize = 100;

  async function load() {
    loading = true;
    error = '';
    try {
      const result = await api.listTracks({
        limit: pageSize,
        offset,
        query: query.trim() || undefined,
        unassignedIn: onlyUnassigned && collection ? collection.id : undefined,
        profile: profileFilter === 'all' ? undefined : profileFilter,
        tags: {
          genre: genreFilter || undefined,
          beat: beatFilter || undefined,
          mood: moodFilter || undefined,
          minEnergy: minEnergyLevel ? energyLevelToBound(minEnergyLevel) : undefined
        }
      });
      tracks = result.tracks;
      total = result.total;
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    // Re-runs whenever the filters change.
    query;
    onlyUnassigned;
    profileFilter;
    genreFilter;
    beatFilter;
    moodFilter;
    minEnergyLevel;
    offset;
    collection;
    void load();
  });

  async function loadVocabulary() {
    try {
      vocabulary = await api.tagVocabulary();
    } catch {
      // Filters are an extra; the library works without them.
      vocabulary = null;
    }
  }

  $effect(() => {
    void loadVocabulary();
  });

  async function tagUntagged() {
    try {
      const { task_id } = await api.runTagging();
      const finished = await pollTask(task_id, (status) => (tagTask = status));
      tagTask = null;
      const tagged = (finished.result as { tagged?: number } | null)?.tagged ?? 0;
      notifySuccess(`Tagged ${pluralize(tagged, 'track')}`);
      await Promise.all([loadVocabulary(), load()]);
    } catch (err) {
      tagTask = null;
      notifyError('Tagging failed', err instanceof ApiError ? err.message : String(err));
    }
  }

  function clearTagFilters() {
    genreFilter = '';
    beatFilter = '';
    moodFilter = '';
    minEnergyLevel = 0;
    offset = 0;
  }

  /** "House (42)" — the count says whether a filter is worth trying. */
  function optionLabel(label: string, count: number | undefined) {
    return count ? `${label} (${count})` : label;
  }

  function toggle(trackId: number) {
    const next = new Set(selected);
    if (next.has(trackId)) next.delete(trackId);
    else next.add(trackId);
    selected = next;
  }

  async function assign(groupId: number) {
    try {
      const result = await api.addGroupMembers(groupId, { track_ids: [...selected] });
      assignOpen = false;
      selected = new Set();
      notifySuccess(`Added ${pluralize(result.added, 'track')} as references`);
      if (collection) await Promise.all([loadGroups(collection.id), loadCollections()]);
      await load();
    } catch (err) {
      notifyError('Could not add', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function analyze() {
    if (!analyzePath) return;
    analyzeOpen = false;
    try {
      const { task_id } = await api.analyze([analyzePath], true, false, analyzeProfile);
      await pollTask(task_id, (status) => (task = status));
      task = null;
      notifySuccess('Analysis complete');
      await Promise.all([load(), loadVocabulary()]);
    } catch (err) {
      task = null;
      notifyError('Analysis failed', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function findSimilar(track: Track) {
    similarSeed = track;
    similarOpen = true;
    similarLoading = true;
    similarResults = [];
    try {
      const result = await api.similar([track.id], 25);
      similarResults = result.results;
    } catch (err) {
      notifyError('Could not search', err instanceof ApiError ? err.message : String(err));
    } finally {
      similarLoading = false;
    }
  }
</script>

<svelte:head><title>Library · music-cluster</title></svelte:head>

<div class="space-y-5">
  <div class="flex flex-wrap items-start justify-between gap-4">
    <div>
      <h1 class="text-2xl font-semibold">Library</h1>
      <p class="mt-1 text-sm text-muted-foreground">
        Every analysed file. Select some to make them references for a group.
      </p>
    </div>
    <div class="flex flex-wrap gap-2">
      {#if vocabulary && vocabulary.untagged > 0}
        <button
          class="inline-flex items-center gap-2 rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
          onclick={tagUntagged}
          disabled={tagTask !== null}
          title="Genre, beat type, mood, energy, danceability and key"
        >
          <Tags class="h-4 w-4" /> Tag {pluralize(vocabulary.untagged, 'untagged track')}
        </button>
      {/if}
      <button
        class="inline-flex items-center gap-2 rounded-md border border-input px-3 py-1.5 text-sm hover:bg-secondary"
        onclick={() => (analyzeOpen = true)}
      >
        <FolderPlus class="h-4 w-4" /> Analyse a folder
      </button>
    </div>
  </div>

  <TaskProgress {task} label="Analysing" />
  <TaskProgress task={tagTask} label="Tagging" />

  <div class="flex flex-wrap items-center gap-3">
    <div class="relative min-w-[16rem] flex-1">
      <Search class="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
      <input
        class="w-full rounded-md border border-input bg-background py-2 pl-9 pr-3 text-sm"
        placeholder="Search artist, title, album, genre..."
        value={query}
        oninput={(event) => {
          query = event.currentTarget.value;
          offset = 0;
        }}
      />
    </div>

    <div class="inline-flex overflow-hidden rounded-md border border-input text-sm">
      {#each [{ value: 'all', label: 'All' }, { value: 'music', label: 'Music' }, { value: 'sample', label: 'Samples' }] as option (option.value)}
        <button
          type="button"
          class="px-3 py-1.5 transition-colors {profileFilter === option.value
            ? 'bg-primary text-primary-foreground'
            : 'hover:bg-secondary'}"
          onclick={() => {
            profileFilter = option.value as AudioProfile | 'all';
            offset = 0;
          }}
        >
          {option.label}
        </button>
      {/each}
    </div>

    {#if collection}
      <label class="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          class="accent-primary"
          checked={onlyUnassigned}
          onchange={(event) => {
            onlyUnassigned = event.currentTarget.checked;
            offset = 0;
          }}
        />
        Not in any {collection.name} group
      </label>
    {/if}

    {#if selected.size > 0}
      <button
        class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground"
        onclick={() => (assignOpen = true)}
      >
        Add {selected.size} to a group
      </button>
    {/if}
  </div>

  {#if vocabulary && vocabulary.counts.tagged > 0}
    <div class="flex flex-wrap items-center gap-2 text-sm" aria-label="Filter by auto-tags">
      <Tags class="h-4 w-4 text-muted-foreground" />
      <select
        class="rounded-md border border-input bg-background px-2 py-1.5"
        bind:value={genreFilter}
        onchange={() => (offset = 0)}
        aria-label="Genre"
      >
        <option value="">Any genre</option>
        {#each vocabulary.genres.filter((term) => vocabulary?.counts.genres[term.key]) as term (term.key)}
          <option value={term.key}
            >{optionLabel(term.label, vocabulary.counts.genres[term.key])}</option
          >
        {/each}
      </select>
      <select
        class="rounded-md border border-input bg-background px-2 py-1.5"
        bind:value={beatFilter}
        onchange={() => (offset = 0)}
        aria-label="Beat type"
      >
        <option value="">Any beat</option>
        {#each vocabulary.beats.filter((term) => vocabulary?.counts.beats[term.key]) as term (term.key)}
          <option value={term.key} title={term.description}>
            {optionLabel(term.label, vocabulary.counts.beats[term.key])}
          </option>
        {/each}
      </select>
      <select
        class="rounded-md border border-input bg-background px-2 py-1.5"
        bind:value={moodFilter}
        onchange={() => (offset = 0)}
        aria-label="Mood"
      >
        <option value="">Any mood</option>
        {#each vocabulary.moods.filter((term) => vocabulary?.counts.moods[term.key]) as term (term.key)}
          <option value={term.key}
            >{optionLabel(term.label, vocabulary.counts.moods[term.key])}</option
          >
        {/each}
      </select>
      <select
        class="rounded-md border border-input bg-background px-2 py-1.5"
        bind:value={minEnergyLevel}
        onchange={() => (offset = 0)}
        aria-label="Minimum energy"
      >
        <option value={0}>Any energy</option>
        {#each [3, 5, 7, 9] as level (level)}
          <option value={level}>Energy {level}+</option>
        {/each}
      </select>
      {#if tagFiltering}
        <button
          class="rounded-md px-2 py-1.5 text-muted-foreground hover:bg-secondary"
          onclick={clearTagFilters}
        >
          Clear
        </button>
      {/if}
    </div>
  {/if}

  {#if loading}
    <LoadingState message="Loading tracks..." />
  {:else if error}
    <ErrorState message={error} onRetry={load} />
  {:else}
    <div class="divide-y divide-border rounded-lg border border-border px-4">
      {#each tracks as track (track.id)}
        <div class="flex items-center gap-3">
          <input
            type="checkbox"
            class="accent-primary"
            checked={selected.has(track.id)}
            onchange={() => toggle(track.id)}
            aria-label="Select {track.display}"
          />
          <div class="min-w-0 flex-1">
            <TrackRow {track}>
              {#snippet actions()}
                <button
                  class="rounded p-1.5 text-muted-foreground hover:bg-secondary hover:text-primary"
                  onclick={() => findSimilar(track)}
                  title="Find tracks that sound like this"
                  aria-label="Find tracks similar to {track.display}"
                >
                  <Sparkles class="h-4 w-4" />
                </button>
              {/snippet}
            </TrackRow>
          </div>
        </div>
      {:else}
        <p class="py-10 text-center text-sm text-muted-foreground">
          {query || tagFiltering
            ? 'No tracks match those filters.'
            : 'No tracks yet — analyse a folder to get started.'}
        </p>
      {/each}
    </div>

    <div class="flex items-center justify-between text-sm text-muted-foreground">
      <span>
        Showing {tracks.length ? offset + 1 : 0}–{offset + tracks.length} of {total}
      </span>
      <div class="flex gap-2">
        <button
          class="rounded-md border border-input px-3 py-1 disabled:opacity-40"
          onclick={() => (offset = Math.max(0, offset - pageSize))}
          disabled={offset === 0}
        >
          Previous
        </button>
        <button
          class="rounded-md border border-input px-3 py-1 disabled:opacity-40"
          onclick={() => (offset += pageSize)}
          disabled={tracks.length < pageSize}
        >
          Next
        </button>
      </div>
    </div>
  {/if}
</div>

<Modal
  bind:open={assignOpen}
  title="Add to a group"
  description="{selected.size} track(s) will become references defining that group."
  size="sm"
>
  <GroupPicker groups={$groupStore} onSelect={(group) => assign(group.id)} />
</Modal>

<Modal
  bind:open={analyzeOpen}
  title="Analyse a folder"
  description="Extracts audio features so these files can be sorted and searched."
>
  <FolderPicker bind:value={analyzePath} label="Folder" />

  <fieldset class="mt-4 text-sm">
    <legend class="font-medium">Analyse as</legend>
    <div class="mt-2 grid gap-2 sm:grid-cols-2">
      {#each [{ value: 'music', title: 'Music', blurb: 'A representative excerpt from the middle of each track.' }, { value: 'sample', title: 'Samples', blurb: 'The whole file from its first sample, plus envelope and pitch.' }] as option (option.value)}
        <label
          class="cursor-pointer rounded-md border p-3 transition-colors {analyzeProfile ===
          option.value
            ? 'border-primary bg-primary/5'
            : 'border-input hover:bg-muted/50'}"
        >
          <input
            type="radio"
            class="sr-only"
            name="analyze-profile"
            value={option.value}
            bind:group={analyzeProfile}
          />
          <span class="block font-medium">{option.title}</span>
          <span class="mt-0.5 block text-xs text-muted-foreground">{option.blurb}</span>
        </label>
      {/each}
    </div>
  </fieldset>
  {#snippet footer()}
    <button
      class="rounded-md border border-input px-3 py-1.5 text-sm"
      onclick={() => (analyzeOpen = false)}>Cancel</button
    >
    <button
      class="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
      onclick={analyze}
      disabled={!analyzePath}
    >
      Analyse
    </button>
  {/snippet}
</Modal>

<Modal
  bind:open={similarOpen}
  title="Sounds like"
  description={similarSeed ? (similarSeed.display ?? similarSeed.filename) : ''}
>
  {#if similarLoading}
    <LoadingState message="Searching..." compact />
  {:else}
    <p class="mb-2 text-xs text-muted-foreground">
      Pick a few of these to seed a new group — this is the manual half of discovery, made faster.
    </p>
    <div class="max-h-96 divide-y divide-border overflow-y-auto">
      {#each similarResults as result (result.track.id)}
        <TrackRow
          track={result.track}
          dense
          subtitle="distance {result.relative_distance.toFixed(2)}"
        />
      {:else}
        <p class="py-6 text-center text-sm text-muted-foreground">Nothing similar found.</p>
      {/each}
    </div>
  {/if}
</Modal>
