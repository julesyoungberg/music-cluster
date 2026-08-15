<script lang="ts">
  import { confidenceClasses, confidenceTone, formatPercent } from '$lib/utils';

  let {
    confidence,
    label = ''
  }: { confidence: number | null | undefined; label?: string } = $props();

  const tone = $derived(confidenceTone(confidence));
</script>

<span
  class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium tabular-nums {confidenceClasses[tone]}"
  title="How sure the sorter is, relative to the runner-up group"
>
  {formatPercent(confidence)}
  {#if label}<span class="font-normal opacity-80">{label}</span>{/if}
</span>
