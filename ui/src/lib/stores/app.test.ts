import { get } from 'svelte/store';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from '$lib/services/api';
import {
  activeCollection,
  activeCollectionId,
  collections,
  collectionsLoaded,
  groups,
  loadCollections,
  loadGroups,
  selectCollection
} from '$lib/stores/app';

const crates = { id: 1, name: 'Crates' } as never;
const archive = { id: 2, name: 'Archive' } as never;

describe('collection selection', () => {
  beforeEach(() => {
    collections.set([]);
    groups.set([]);
    collectionsLoaded.set(false);
    activeCollectionId.set(null);
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('selects the first collection when nothing is stored', async () => {
    vi.spyOn(api, 'listCollections').mockResolvedValue({ collections: [crates, archive] });

    await loadCollections();

    expect(get(activeCollectionId)).toBe(1);
    expect(get(activeCollection)).toBe(crates);
    expect(get(collectionsLoaded)).toBe(true);
  });

  it('keeps a stored selection that still exists', async () => {
    activeCollectionId.set(2);
    vi.spyOn(api, 'listCollections').mockResolvedValue({ collections: [crates, archive] });

    await loadCollections();

    expect(get(activeCollectionId)).toBe(2);
  });

  it('falls back to the first collection when the stored one was deleted', async () => {
    activeCollectionId.set(99);
    vi.spyOn(api, 'listCollections').mockResolvedValue({ collections: [crates] });

    await loadCollections();

    expect(get(activeCollectionId)).toBe(1);
  });

  it('clears the selection when there are no collections left', async () => {
    activeCollectionId.set(1);
    vi.spyOn(api, 'listCollections').mockResolvedValue({ collections: [] });

    await loadCollections();

    expect(get(activeCollectionId)).toBeNull();
    expect(get(activeCollection)).toBeNull();
  });

  it('persists the selection so it survives a reload', () => {
    activeCollectionId.set(7);
    expect(localStorage.getItem('music-cluster:active-collection')).toBe('7');

    activeCollectionId.set(null);
    expect(localStorage.getItem('music-cluster:active-collection')).toBeNull();
  });

  it('drops stale groups immediately when switching collection', () => {
    // Showing the previous collection's groups under the new name, even for one
    // frame, would invite filing a track into the wrong crate.
    groups.set([{ id: 5, name: 'Old group' } as never]);
    const listGroups = vi.spyOn(api, 'listGroups').mockResolvedValue({ groups: [] });

    selectCollection(2);

    expect(get(groups)).toEqual([]);
    expect(get(activeCollectionId)).toBe(2);
    expect(listGroups).toHaveBeenCalledWith(2);
  });
});

describe('loadGroups', () => {
  afterEach(() => vi.restoreAllMocks());

  it('publishes the groups it fetched', async () => {
    const list = [{ id: 3, name: 'Deep House' } as never];
    vi.spyOn(api, 'listGroups').mockResolvedValue({ groups: list });

    await expect(loadGroups(1)).resolves.toEqual(list);
    expect(get(groups)).toEqual(list);
  });
});
