<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { theme } from '$lib/stores/settings';
  import Navigation from '$lib/components/Navigation.svelte';

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
  <main>
    <slot />
  </main>
</div>
