<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { api, ApiError } from '$lib/services/api';
  import { loadCollections, loadGroups } from '$lib/stores/app';
  import { notifyError, notifySuccess } from '$lib/stores/notifications';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import TrackRow from '$lib/components/TrackRow.svelte';
  import FolderPicker from '$lib/components/FolderPicker.svelte';
  import type { Group, GroupQuality, Track } from '$lib/types';
  import { formatPercent, pluralize } from '$lib/utils';
  import { ArrowLeft, Save, Trash2, X } from 'lucide-svelte';

  const groupId = $derived(Number($page.params.id));

  let group = $state<Group | null>(null);
  let members = $state<Track[]>([]);
  let size = $state(0);
  let quality = $state<GroupQuality | null>(null);
  let loading = $state(true);
  let error = $state('');

  let name = $state('');
  let destination = $state('');
  let description = $state('');
  let saving = $state(false);
  let destinationOpen = $state(false);
  let deleteOpen = $state(false);

  async function load() {
    loading = true;
    error = '';
    try {
      const detail = await api.getGroup(groupId, 500);
      group = detail.group;
      members = detail.members;
      size = detail.size;
      quality = detail.quality;
      name = detail.group.name;
      destination = detail.group.destination_path ?? '';
      description = detail.group.description ?? '';
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (Number.isFinite(groupId)) void load();
  });

  async function save() {
    if (!group) return;
    saving = true;
    try {
      await api.updateGroup(group.id, {
        name,
        destination_path: destination || undefined,
        description: description || undefined
      });
      notifySuccess('Group updated');
      await Promise.all([load(), loadGroups(group.collection_id), loadCollections()]);
    } catch (err) {
      notifyError('Could not save', err instanceof ApiError ? err.message : String(err));
    } finally {
      saving = false;
    }
  }

  async function removeMember(track: Track) {
    if (!group) return;
    try {
      await api.removeGroupMember(group.id, track.id);
      members = members.filter((member) => member.id !== track.id);
      size -= 1;
      notifySuccess('Removed from group', 'The track stays in your library.');
    } catch (err) {
      notifyError('Could not remove', err instanceof ApiError ? err.message : String(err));
    }
  }

  async function remove() {
    if (!group) return;
    try {
      const collectionId = group.collection_id;
      await api.deleteGroup(group.id);
      await Promise.all([loadGroups(collectionId), loadCollections()]);
      notifySuccess('Group deleted');
      void goto('/groups');
    } catch (err) {
      notifyError('Could not delete', err instanceof ApiError ? err.message : String(err));
    }
  }
</script>

<svelte:head><title>{group?.name ?? 'Group'} · music-cluster</title></svelte:head>

<a
  href="/groups"
  class="mb-4 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
>
  <ArrowLeft class="h-4 w-4" /> All groups
</a>

{#if loading}
  <LoadingState message="Loading group..." />
{:else if error || !group}
  <ErrorState message={error} onRetry={load} />
{:else}
  <div class="grid gap-6 lg:grid-cols-[20rem_1fr]">
    <aside class="space-y-4">
      <div class="space-y-3 rounded-lg border border-border p-4">
        <label class="block text-sm">
          <span class="font-medium">Name</span>
          <input
            class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2"
            bind:value={name}
          />
        </label>

        <label class="block text-sm">
          <span class="font-medium">Notes</span>
          <textarea
            class="mt-1 w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm"
            rows="2"
            bind:value={description}
            placeholder="What belongs in here?"
          ></textarea>
        </label>

        <div class="text-sm">
          <span class="font-medium">Destination folder</span>
          <p
            class="mt-1 break-all rounded-md border border-border bg-secondary/40 px-3 py-2 font-mono text-xs"
          >
            {destination || 'Not set — this group can only export playlists'}
          </p>
          <button
            class="mt-2 text-xs text-primary hover:underline"
            onclick={() => (destinationOpen = true)}
          >
            Choose folder
          </button>
        </div>

        <button
          class="inline-flex w-full items-center justify-center gap-2 rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground disabled:opacity-60"
          onclick={save}
          disabled={saving}
        >
          <Save class="h-4 w-4" />
          {saving ? 'Saving...' : 'Save changes'}
        </button>
      </div>

      <div class="rounded-lg border border-border p-4 text-sm">
        <p class="font-medium">Self-check</p>
        {#if quality}
          <p class="mt-2 text-2xl font-semibold">{formatPercent(quality.accuracy)}</p>
          <p class="mt-1 text-xs text-muted-foreground">
            {quality.correct} of {quality.size} reference tracks get filed back here when tested against
            the other groups.
          </p>
          {#if quality.accuracy < 0.6}
            <p class="mt-2 text-xs text-amber-600 dark:text-amber-400">
              This group overlaps heavily with others. Check the overlap report to see which.
            </p>
          {/if}
        {:else}
          <p class="mt-2 text-xs text-muted-foreground">
            Fit the collection to see how distinct this group is.
          </p>
        {/if}
      </div>

      <button
        class="inline-flex w-full items-center justify-center gap-2 rounded-md border border-destructive/40 px-3 py-2 text-sm text-destructive hover:bg-destructive/10"
        onclick={() => (deleteOpen = true)}
      >
        <Trash2 class="h-4 w-4" /> Delete group
      </button>
    </aside>

    <section>
      <div class="mb-3 flex items-baseline justify-between">
        <h1 class="text-xl font-semibold">{group.name}</h1>
        <p class="text-sm text-muted-foreground">{pluralize(size, 'reference track')}</p>
      </div>

      <div class="divide-y divide-border rounded-lg border border-border px-4">
        {#each members as track (track.id)}
          <TrackRow {track}>
            {#snippet actions()}
              <button
                class="rounded p-1.5 text-muted-foreground hover:bg-secondary hover:text-destructive"
                onclick={() => removeMember(track)}
                title="Remove from this group"
                aria-label="Remove {track.display} from this group"
              >
                <X class="h-4 w-4" />
              </button>
            {/snippet}
          </TrackRow>
        {:else}
          <p class="py-8 text-center text-sm text-muted-foreground">
            No reference tracks yet. Import a folder or add tracks from the library.
          </p>
        {/each}
      </div>

      {#if size > members.length}
        <p class="mt-3 text-center text-sm text-muted-foreground">
          Showing {members.length} of {size}.
        </p>
      {/if}
    </section>
  </div>
{/if}

<Modal
  bind:open={destinationOpen}
  title="Destination folder"
  description="Where music sorted into this group should be filed."
>
  <FolderPicker bind:value={destination} label="Folder" />
  {#snippet footer()}
    <button
      class="rounded-md border border-input px-3 py-1.5 text-sm"
      onclick={() => (destinationOpen = false)}>Done</button
    >
  {/snippet}
</Modal>

<Modal
  bind:open={deleteOpen}
  title="Delete this group?"
  description="Its tracks stay in your library and on disk."
  size="sm"
>
  <p class="text-sm text-muted-foreground">
    {group?.name} will be removed from this collection along with its {size} reference tracks.
  </p>
  {#snippet footer()}
    <button
      class="rounded-md border border-input px-3 py-1.5 text-sm"
      onclick={() => (deleteOpen = false)}>Cancel</button
    >
    <button
      class="rounded-md bg-destructive px-3 py-1.5 text-sm text-destructive-foreground"
      onclick={remove}>Delete</button
    >
  {/snippet}
</Modal>
