<!--
  What kind of audio a collection holds.

  Worth a persistent badge rather than a line of prose: a sample collection and
  a music collection look identical on screen — groups, counts, confidence bars
  — right up until you point the wrong one at a folder and wait for a thousand
  files to be analysed against a space that cannot read them.
-->
<script lang="ts">
  import type { AudioProfile } from '$lib/types';
  import { AudioWaveform, Disc3 } from 'lucide-svelte';

  let {
    profile = 'music',
    size = 'md'
  }: { profile?: AudioProfile | null; size?: 'sm' | 'md' } = $props();

  const resolved = $derived(profile === 'sample' ? 'sample' : 'music');
  const label = $derived(resolved === 'sample' ? 'Samples' : 'Music');
  const Icon = $derived(resolved === 'sample' ? AudioWaveform : Disc3);
</script>

<span
  class="inline-flex shrink-0 items-center gap-1 rounded-full border font-medium
    {size === 'sm' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-0.5 text-xs'}
    {resolved === 'sample'
    ? 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400'
    : 'border-input bg-muted text-muted-foreground'}"
  title={resolved === 'sample'
    ? 'One-shots and loops: kicks, snares, claps, hats, basses, chords'
    : 'Full-length tracks'}
>
  <Icon class={size === 'sm' ? 'h-2.5 w-2.5' : 'h-3 w-3'} />
  {label}
</span>
