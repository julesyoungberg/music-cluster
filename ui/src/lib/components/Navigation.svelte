<script lang="ts">
  import { page } from '$app/stores';
  import { FolderTree, Inbox, Compass, Library, Settings2, Disc3 } from 'lucide-svelte';
  import CollectionSwitcher from './CollectionSwitcher.svelte';
  import { cn } from '$lib/utils';

  const links = [
    { href: '/', label: 'Overview', icon: Disc3 },
    { href: '/groups', label: 'Groups', icon: FolderTree },
    { href: '/sort', label: 'Sort', icon: Inbox },
    { href: '/discover', label: 'Discover', icon: Compass },
    { href: '/library', label: 'Library', icon: Library },
    { href: '/settings', label: 'Settings', icon: Settings2 }
  ];

  const isActive = (href: string, path: string) =>
    href === '/' ? path === '/' : path.startsWith(href);
</script>

<header class="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur">
  <div class="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4">
    <a href="/" class="flex shrink-0 items-center gap-2 font-semibold">
      <Disc3 class="h-5 w-5" />
      <span class="hidden sm:inline">music-cluster</span>
    </a>

    <nav class="flex flex-1 items-center gap-1 overflow-x-auto">
      {#each links as link (link.href)}
        {@const active = isActive(link.href, $page.url.pathname)}
        <a
          href={link.href}
          class={cn(
            'flex items-center gap-2 whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors',
            active
              ? 'bg-secondary text-secondary-foreground font-medium'
              : 'text-muted-foreground hover:bg-secondary/60 hover:text-foreground'
          )}
          aria-current={active ? 'page' : undefined}
        >
          <link.icon class="h-4 w-4" />
          <span class="hidden md:inline">{link.label}</span>
        </a>
      {/each}
    </nav>

    <CollectionSwitcher />
  </div>
</header>
