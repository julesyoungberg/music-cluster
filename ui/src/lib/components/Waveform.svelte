<script lang="ts">
  import { ApiError } from '$lib/services/api';
  import {
    cachedWaveform,
    forgetWaveform,
    loadWaveform,
    resampleEnvelope,
    type WaveformData
  } from '$lib/services/waveform';
  import { nudge, player, playTrack, seek } from '$lib/stores/player';
  import type { Track } from '$lib/types';
  import { formatDuration } from '$lib/utils';

  let {
    track,
    height = 48,
    /**
     * How many envelope points to ask the server for. One generous resolution
     * per track redraws at any width without another request, and asking every
     * component for the same number keeps the server's cache warm.
     */
    resolution = 400,
    barWidth = 3,
    barGap = 1
  }: {
    track: Track;
    height?: number;
    resolution?: number;
    barWidth?: number;
    barGap?: number;
  } = $props();

  const MIN_BAR_HEIGHT = 2;

  let canvas = $state<HTMLCanvasElement | null>(null);
  let container = $state<HTMLDivElement | null>(null);
  let width = $state(0);

  let data = $state<WaveformData | null>(null);
  // Bumped on every new envelope, so a redraw is never skipped because two
  // different waveforms happened to agree on duration and bar count. Plain, not
  // reactive: only the drawing reads it, and reading state you also write inside
  // an effect is how loops start.
  let dataVersion = 0;
  let loading = $state(false);
  let failed = $state('');
  let hoverFraction = $state<number | null>(null);
  let scrubbing = $state(false);

  const isCurrent = $derived($player.track?.id === track.id);

  /** The waveform's own duration is authoritative until the audio reports its own. */
  const duration = $derived(
    isCurrent && $player.duration > 0 ? $player.duration : (data?.duration ?? track.duration ?? 0)
  );

  const progress = $derived(
    isCurrent && duration > 0 ? Math.min($player.currentTime / duration, 1) : 0
  );

  const hoverTime = $derived(hoverFraction === null ? null : hoverFraction * duration);

  // Waveforms load one track at a time and are cached, so moving through a
  // review queue does not re-decode anything already seen.
  $effect(() => {
    const id = track.id;
    if (!Number.isFinite(id)) return;

    const ready = cachedWaveform(id, resolution);
    if (ready) {
      data = ready;
      dataVersion += 1;
      loading = false;
      failed = '';
      return;
    }

    const controller = new AbortController();
    data = null;
    failed = '';
    loading = true;

    loadWaveform(id, resolution, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        data = result;
        dataVersion += 1;
        loading = false;
      })
      .catch((error) => {
        if (controller.signal.aborted || error?.name === 'AbortError') return;
        forgetWaveform(id, resolution);
        failed =
          error instanceof ApiError && error.status === 404
            ? 'Audio file not found'
            : 'Preview unavailable';
        loading = false;
      });

    return () => controller.abort();
  });

  /** Track the element's real width so the canvas matches its box exactly. */
  $effect(() => {
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      width = Math.floor(entries[0].contentRect.width);
    });
    observer.observe(container);
    width = Math.floor(container.getBoundingClientRect().width);
    return () => observer.disconnect();
  });

  const bars = $derived(
    width > 0 ? Math.max(1, Math.floor((width + barGap) / (barWidth + barGap))) : 0
  );

  const envelope = $derived(data && bars > 0 ? resampleEnvelope(data.peaks, bars) : []);

  /**
   * Canvas colours come from the same CSS variables as the rest of the UI, so
   * the waveform follows the light/dark theme instead of hard-coding a palette.
   */
  function themeColor(name: string, alpha = 1): string {
    if (typeof window === 'undefined' || !canvas) return '#888';
    const value = getComputedStyle(canvas).getPropertyValue(name).trim();
    if (!value) return '#888';
    const [h, s, l] = value.split(/\s+/);
    return alpha >= 1 ? `hsl(${h}, ${s}, ${l})` : `hsla(${h}, ${s}, ${l}, ${alpha})`;
  }

  let lastDrawn = '';

  function draw(): void {
    const element = canvas;
    if (!element || width <= 0) return;

    // Playback updates every animation frame, but the picture only changes when
    // the playhead crosses a pixel, so most of those frames redraw nothing.
    const signature = [
      width,
      height,
      envelope.length,
      dataVersion,
      track.id,
      isCurrent,
      // Colours come from the theme, so a theme switch has to redraw.
      document.documentElement.className,
      Math.round(progress * width),
      hoverFraction === null ? -1 : Math.round(hoverFraction * width)
    ].join('|');
    if (signature === lastDrawn) return;
    lastDrawn = signature;

    const ratio = window.devicePixelRatio || 1;
    const pixelWidth = Math.floor(width * ratio);
    const pixelHeight = Math.floor(height * ratio);
    if (element.width !== pixelWidth || element.height !== pixelHeight) {
      element.width = pixelWidth;
      element.height = pixelHeight;
    }

    const context = element.getContext('2d');
    if (!context) return;

    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    if (envelope.length === 0) return;

    const played = themeColor('--primary');
    const unplayed = themeColor('--muted-foreground', 0.45);
    const middle = height / 2;
    const step = barWidth + barGap;
    const playedUntil = progress * width;

    for (let index = 0; index < envelope.length; index++) {
      const x = index * step;
      // A bar is "played" once the playhead has passed its middle, which keeps
      // the colour change lined up with the playhead itself.
      context.fillStyle = x + barWidth / 2 <= playedUntil ? played : unplayed;

      const barHeight = Math.max(MIN_BAR_HEIGHT, envelope[index] * (height - 2));
      // Mirrored around the centre line, the way a waveform is normally read.
      const top = middle - barHeight / 2;
      if (context.roundRect) {
        context.beginPath();
        context.roundRect(x, top, barWidth, barHeight, barWidth / 2);
        context.fill();
      } else {
        // Older webviews (some Linux Tauri builds) have no roundRect; square
        // bars are better than a canvas that throws and draws nothing.
        context.fillRect(x, top, barWidth, barHeight);
      }
    }

    if (isCurrent && progress > 0) {
      context.fillStyle = played;
      context.fillRect(Math.min(playedUntil, width - 1.5), 0, 1.5, height);
    }

    if (hoverFraction !== null) {
      context.fillStyle = themeColor('--foreground', 0.35);
      context.fillRect(Math.min(hoverFraction * width, width - 1), 0, 1, height);
    }
  }

  /** Follow light/dark switches, which change the colours but no drawing input. */
  $effect(() => {
    const observer = new MutationObserver(() => draw());
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });
    return () => observer.disconnect();
  });

  // Redraw whenever anything the drawing depends on changes. Reading the values
  // here is what registers them as dependencies.
  $effect(() => {
    envelope;
    progress;
    hoverFraction;
    isCurrent;
    width;
    height;
    draw();
  });

  function fractionFrom(event: { clientX: number }): number {
    if (!container) return 0;
    const rect = container.getBoundingClientRect();
    if (rect.width <= 0) return 0;
    return Math.max(0, Math.min((event.clientX - rect.left) / rect.width, 1));
  }

  /** Seek within the playing track, or start this one from where you clicked. */
  function seekTo(fraction: number): void {
    const target = fraction * duration;
    if (isCurrent) seek(target);
    else void playTrack(track, duration > 0 ? target : 0);
  }

  function onPointerDown(event: PointerEvent): void {
    if (event.button !== 0 || !data) return;
    event.preventDefault();
    container?.focus();
    scrubbing = true;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    const fraction = fractionFrom(event);
    hoverFraction = fraction;
    seekTo(fraction);
  }

  function onPointerMove(event: PointerEvent): void {
    if (!data) return;
    const fraction = fractionFrom(event);
    hoverFraction = fraction;
    // Dragging scrubs, like dragging along a SoundCloud waveform.
    if (scrubbing && isCurrent) seek(fraction * duration);
  }

  function onPointerUp(event: PointerEvent): void {
    if (!scrubbing) return;
    scrubbing = false;
    (event.currentTarget as HTMLElement).releasePointerCapture?.(event.pointerId);
  }

  function onKeydown(event: KeyboardEvent): void {
    if (!data) return;

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      void playTrack(track);
      return;
    }

    const jumps: Record<string, number> = { ArrowRight: 5, ArrowLeft: -5 };
    if (event.key in jumps && isCurrent) {
      // Only when this track is playing, so the arrow keys keep moving through
      // the review queue the rest of the time.
      event.preventDefault();
      event.stopPropagation();
      nudge(jumps[event.key]);
    } else if (event.key === 'Home' && isCurrent) {
      event.preventDefault();
      event.stopPropagation();
      seek(0);
    }
  }

  function retry(event: MouseEvent): void {
    event.stopPropagation();
    forgetWaveform(track.id, resolution);
    failed = '';
    loading = true;
    loadWaveform(track.id, resolution)
      .then((result) => {
        data = result;
        dataVersion += 1;
        loading = false;
      })
      .catch(() => {
        failed = 'Preview unavailable';
        loading = false;
      });
  }
</script>

<div
  bind:this={container}
  class="relative w-full select-none overflow-hidden rounded bg-secondary/40 outline-none ring-offset-2 ring-offset-background focus-visible:ring-2 focus-visible:ring-ring
    {data ? 'cursor-pointer' : 'cursor-default'}"
  style="height: {height}px"
  onpointerdown={onPointerDown}
  onpointermove={onPointerMove}
  onpointerup={onPointerUp}
  onpointercancel={onPointerUp}
  onpointerleave={() => {
    hoverFraction = null;
    scrubbing = false;
  }}
  onkeydown={onKeydown}
  role="slider"
  tabindex="0"
  aria-label="Waveform for {track.display ?? track.filename}"
  aria-valuemin="0"
  aria-valuemax="100"
  aria-valuenow={Math.round(progress * 100)}
  aria-valuetext="{formatDuration($player.currentTime)} of {formatDuration(duration)}"
>
  <!-- The canvas never gets an explicit CSS width: it must follow the container
       when the window shrinks, not hold it open at the last measured size. -->
  <canvas bind:this={canvas} class="block h-full w-full"></canvas>

  {#if loading}
    <!-- Fills the same box the waveform will, so nothing jumps when it arrives,
         and reads as "working" rather than as a track that is silent. -->
    <div
      class="absolute inset-0 animate-pulse bg-gradient-to-r from-transparent via-muted-foreground/20 to-transparent"
    ></div>
  {:else if failed}
    <div class="absolute inset-0 flex items-center justify-center gap-2 text-xs text-muted-foreground">
      <span>{failed}</span>
      <button class="underline underline-offset-2 hover:text-foreground" onclick={retry}>
        Retry
      </button>
    </div>
  {/if}

  {#if hoverTime !== null && data && duration > 0}
    <span
      class="pointer-events-none absolute top-1 rounded bg-foreground/85 px-1 py-0.5 text-[10px] font-medium tabular-nums text-background"
      style="left: {Math.min(Math.max(hoverFraction! * width, 2), Math.max(width - 40, 2))}px"
    >
      {formatDuration(hoverTime)}
    </span>
  {/if}
</div>
