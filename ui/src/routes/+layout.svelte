<script lang="ts">
  import '../app.css';
  import type { Snippet } from 'svelte';
  import Navigation from '$lib/components/Navigation.svelte';
  import NotificationToast from '$lib/components/NotificationToast.svelte';
  import PlayerBar from '$lib/components/PlayerBar.svelte';
  import { loadCollections, activeCollectionId, loadGroups } from '$lib/stores/app';
  import { notifyError } from '$lib/stores/notifications';
  import { ApiError } from '$lib/services/api';
  import { player } from '$lib/stores/player';

  let { children }: { children: Snippet } = $props();

  let theme = $state<'light' | 'dark' | 'system'>('system');

  $effect(() => {
    const stored = localStorage.getItem('music-cluster:theme') as typeof theme | null;
    if (stored) theme = stored;
  });

  $effect(() => {
    const dark =
      theme === 'dark' ||
      (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('music-cluster:theme', theme);
  });

  // The collection list is needed by nearly every page, so it is loaded once here.
  $effect(() => {
    loadCollections().catch((error) => {
      notifyError(
        'Could not load collections',
        error instanceof ApiError ? error.message : String(error)
      );
    });
  });

  $effect(() => {
    const id = $activeCollectionId;
    if (id !== null) void loadGroups(id).catch(() => undefined);
  });
</script>

<div class="min-h-screen bg-background text-foreground">
  <Navigation />
  <main class="mx-auto max-w-7xl px-4 py-6 {$player.track ? 'pb-28' : 'pb-10'}">
    {@render children()}
  </main>
  <PlayerBar />
  <NotificationToast />
</div>
