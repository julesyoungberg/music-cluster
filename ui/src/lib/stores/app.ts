import { derived, get, writable } from 'svelte/store';
import { api } from '$lib/services/api';
import type { Collection, Group } from '$lib/types';

const STORAGE_KEY = 'music-cluster:active-collection';

export const collections = writable<Collection[]>([]);
export const activeCollectionId = writable<number | null>(readStoredCollectionId());
export const groups = writable<Group[]>([]);
export const collectionsLoaded = writable(false);

export const activeCollection = derived(
  [collections, activeCollectionId],
  ([$collections, $id]) => $collections.find((c) => c.id === $id) ?? $collections[0] ?? null
);

function readStoredCollectionId(): number | null {
  if (typeof localStorage === 'undefined') return null;
  const raw = localStorage.getItem(STORAGE_KEY);
  return raw ? Number(raw) : null;
}

activeCollectionId.subscribe((id) => {
  if (typeof localStorage === 'undefined') return;
  if (id === null) localStorage.removeItem(STORAGE_KEY);
  else localStorage.setItem(STORAGE_KEY, String(id));
});

/** Load collections and keep the active selection valid. */
export async function loadCollections(): Promise<Collection[]> {
  const { collections: list } = await api.listCollections();
  collections.set(list);
  collectionsLoaded.set(true);

  const current = get(activeCollectionId);
  if (!list.some((c) => c.id === current)) {
    activeCollectionId.set(list[0]?.id ?? null);
  }
  return list;
}

export async function loadGroups(collectionId: number): Promise<Group[]> {
  const { groups: list } = await api.listGroups(collectionId);
  groups.set(list);
  return list;
}

export function selectCollection(id: number): void {
  activeCollectionId.set(id);
  groups.set([]);
  void loadGroups(id);
}
