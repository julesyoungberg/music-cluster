<script lang="ts">
  import { crateColor } from '$lib/crateColors';

  /** A group's crate colour as a small square, shown before its name. */
  let { group, size = 'sm' }: { group: { id: number; color?: string | null }; size?: 'sm' | 'md' } =
    $props();
</script>

<span
  class="inline-block shrink-0 rounded-sm {size === 'md' ? 'h-3 w-3' : 'h-2.5 w-2.5'}"
  style="background-color: {crateColor(group)}"
  aria-hidden="true"
></span>
