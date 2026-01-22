<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { theme } from '$lib/stores/settings';
  import { isOnline } from '$lib/stores/network';
  import Navigation from '$lib/components/Navigation.svelte';
  import NotificationToast from '$lib/components/NotificationToast.svelte';
  import { WifiOff } from 'lucide-svelte';

  onMount(() => {
    // Apply theme
    theme.subscribe(t => {
      if (t === 'system') {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.classList.toggle('dark', prefersDark);
      } else {
        document.documentElement.classList.toggle('dark', t === 'dark');
      }
    });
  });
</script>

<div class="min-h-screen bg-background">
  <Navigation />
  {#if !$isOnline}
    <div class="bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-b border-yellow-500/20 px-4 py-2 flex items-center gap-2">
      <WifiOff class="w-4 h-4" />
      <span class="text-sm">You're currently offline. Some features may not work.</span>
    </div>
  {/if}
  <main>
    <slot />
  </main>
  <NotificationToast />
</div>
