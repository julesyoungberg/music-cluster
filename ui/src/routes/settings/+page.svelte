<script lang="ts">
  import { api, ApiError } from '$lib/services/api';
  import { notifyError, notifySuccess } from '$lib/stores/notifications';
  import ErrorState from '$lib/components/ErrorState.svelte';
  import LoadingState from '$lib/components/LoadingState.svelte';
  import { Save } from 'lucide-svelte';

  let config = $state<Record<string, any> | null>(null);
  let loading = $state(true);
  let error = $state('');
  let saving = $state(false);
  let theme = $state<'light' | 'dark' | 'system'>('system');

  async function load() {
    loading = true;
    error = '';
    try {
      config = await api.getConfig();
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void load();
    theme = (localStorage.getItem('music-cluster:theme') as typeof theme) ?? 'system';
  });

  function setTheme(value: typeof theme) {
    theme = value;
    localStorage.setItem('music-cluster:theme', value);
    const dark =
      value === 'dark' ||
      (value === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.classList.toggle('dark', dark);
  }

  async function save(values: Record<string, unknown>) {
    saving = true;
    try {
      config = await api.updateConfig(values);
      notifySuccess('Settings saved', 'Re-fit your collection for scoring changes to take effect.');
    } catch (err) {
      notifyError('Could not save', err instanceof ApiError ? err.message : String(err));
    } finally {
      saving = false;
    }
  }

  /** Fields grouped so each one can explain what it actually changes. */
  const sortingFields = [
    {
      key: 'sorting.auto_accept_confidence',
      label: 'Auto-accept confidence',
      help: 'How sure the sorter must be before "Accept confident" will file a track for you.',
      min: 0.3,
      max: 0.99,
      step: 0.05
    },
    {
      key: 'sorting.min_margin',
      label: 'Minimum margin',
      help: 'How far ahead the top group must be over the runner-up. Raise it if near-ties are being filed wrongly.',
      min: 0,
      max: 0.6,
      step: 0.02
    },
    {
      key: 'sorting.novelty_factor',
      label: 'Novelty threshold',
      help: 'Beyond this distance a track is reported as matching nothing. Lower it to catch more oddities.',
      min: 1,
      max: 8,
      step: 0.25
    },
    {
      key: 'sorting.neighbors',
      label: 'Neighbours per group',
      help: 'How many of a group’s closest tracks a comparison uses. Higher is steadier on big folders; lower is more sensitive on small ones.',
      min: 1,
      max: 25,
      step: 1
    },
    {
      key: 'sorting.knn_weight',
      label: 'Neighbours vs centre',
      help: '1.0 compares only against nearby tracks; 0 compares only against the group average. Nearby tracks handle folders that contain several distinct sounds.',
      min: 0,
      max: 1,
      step: 0.05
    },
    {
      key: 'sorting.discriminant_weight',
      label: 'Focus on your boundaries',
      help: 'How much the space is tuned to what separates your existing groups, rather than general audio character.',
      min: 0,
      max: 1,
      step: 0.05
    }
  ];

  const weightFields = [
    [
      'sorting.feature_weights.timbre',
      'Timbre',
      'Texture and tone colour — what most genre boundaries follow.'
    ],
    [
      'sorting.feature_weights.rhythm',
      'Rhythm',
      'Tempo and groove. Raise it if BPM matters most to your folders.'
    ],
    ['sorting.feature_weights.harmony', 'Harmony', 'Key and chord content.'],
    ['sorting.feature_weights.dynamics', 'Dynamics', 'Loudness and energy.']
  ] as const;

  function read(path: string): any {
    return path.split('.').reduce<any>((value, key) => value?.[key], config);
  }
</script>

<svelte:head><title>Settings · music-cluster</title></svelte:head>

<div class="space-y-8">
  <div>
    <h1 class="text-2xl font-semibold">Settings</h1>
    <p class="mt-1 text-sm text-muted-foreground">
      Everything here is also in <span class="font-mono text-xs">~/.music-cluster/config.yaml</span
      >.
    </p>
  </div>

  <section>
    <h2 class="mb-2 font-medium">Appearance</h2>
    <div class="flex gap-2">
      {#each ['light', 'dark', 'system'] as option}
        <button
          class="rounded-md border px-3 py-1.5 text-sm capitalize transition-colors {theme ===
          option
            ? 'border-primary bg-primary/10'
            : 'border-input hover:bg-secondary'}"
          onclick={() => setTheme(option as typeof theme)}
        >
          {option}
        </button>
      {/each}
    </div>
  </section>

  {#if loading}
    <LoadingState message="Loading settings..." />
  {:else if error || !config}
    <ErrorState message={error} onRetry={load} />
  {:else}
    <section>
      <h2 class="font-medium">Sorting</h2>
      <p class="mb-3 text-sm text-muted-foreground">
        These govern how confident the sorter has to be before it will decide anything for you.
      </p>
      <div class="space-y-4 rounded-lg border border-border p-4">
        {#each sortingFields as field}
          <div>
            <div class="flex items-baseline justify-between gap-4">
              <label class="text-sm font-medium" for={field.key}>{field.label}</label>
              <span class="tabular-nums text-sm text-muted-foreground">{read(field.key)}</span>
            </div>
            <input
              id={field.key}
              type="range"
              class="mt-1 w-full accent-primary"
              min={field.min}
              max={field.max}
              step={field.step}
              value={read(field.key)}
              onchange={(event) => save({ [field.key]: Number(event.currentTarget.value) })}
            />
            <p class="text-xs text-muted-foreground">{field.help}</p>
          </div>
        {/each}

        <label class="flex items-start gap-2 border-t border-border pt-4 text-sm">
          <input
            type="checkbox"
            class="mt-1 accent-primary"
            checked={read('sorting.learn_on_commit')}
            onchange={(event) => save({ 'sorting.learn_on_commit': event.currentTarget.checked })}
          />
          <span>
            <span class="font-medium">Learn from every commit</span>
            <span class="block text-xs text-muted-foreground">
              Tracks you file become references for that group, so the sorter follows your taste as
              it drifts. Turn it off to keep your reference sets exactly as you curated them.
            </span>
          </span>
        </label>
      </div>
    </section>

    <section>
      <h2 class="font-medium">What counts as similar</h2>
      <p class="mb-3 text-sm text-muted-foreground">
        Relative weight of each kind of audio feature when comparing tracks.
      </p>
      <div class="grid gap-4 rounded-lg border border-border p-4 sm:grid-cols-2">
        {#each weightFields as [key, label, help]}
          <div>
            <div class="flex items-baseline justify-between gap-4">
              <label class="text-sm font-medium" for={key}>{label}</label>
              <span class="tabular-nums text-sm text-muted-foreground">{read(key) ?? 1}</span>
            </div>
            <input
              id={key}
              type="range"
              class="mt-1 w-full accent-primary"
              min="0"
              max="3"
              step="0.1"
              value={read(key) ?? 1}
              onchange={(event) => save({ [key]: Number(event.currentTarget.value) })}
            />
            <p class="text-xs text-muted-foreground">{help}</p>
          </div>
        {/each}
      </div>
    </section>

    <section>
      <h2 class="font-medium">Filing</h2>
      <div class="space-y-4 rounded-lg border border-border p-4">
        <div>
          <span class="text-sm font-medium">Default mode</span>
          <div class="mt-1 flex flex-wrap gap-2">
            {#each [['playlist', 'Playlists only'], ['copy', 'Copy'], ['move', 'Move'], ['symlink', 'Symlink']] as [value, label]}
              <button
                class="rounded-md border px-3 py-1.5 text-sm transition-colors {read(
                  'organize.mode'
                ) === value
                  ? 'border-primary bg-primary/10'
                  : 'border-input hover:bg-secondary'}"
                onclick={() => save({ 'organize.mode': value })}
              >
                {label}
              </button>
            {/each}
          </div>
          <p class="mt-1 text-xs text-muted-foreground">
            You can still pick a different mode at commit time.
          </p>
        </div>

        <label class="block text-sm">
          <span class="font-medium">Playlist folder</span>
          <input
            class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs"
            value={read('organize.playlist_dir')}
            onchange={(event) => save({ 'organize.playlist_dir': event.currentTarget.value })}
          />
        </label>

        <div>
          <span class="text-sm font-medium">If a file of the same name is already there</span>
          <div class="mt-1 flex gap-2">
            {#each [['skip', 'Skip it'], ['rename', 'Rename'], ['overwrite', 'Overwrite']] as [value, label]}
              <button
                class="rounded-md border px-3 py-1.5 text-sm transition-colors {read(
                  'organize.on_conflict'
                ) === value
                  ? 'border-primary bg-primary/10'
                  : 'border-input hover:bg-secondary'}"
                onclick={() => save({ 'organize.on_conflict': value })}
              >
                {label}
              </button>
            {/each}
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2 class="font-medium">Analysis</h2>
      <div class="space-y-4 rounded-lg border border-border p-4">
        <label class="block text-sm">
          <span class="font-medium">Seconds analysed per track</span>
          <input
            type="number"
            min="0"
            step="15"
            class="mt-1 w-32 rounded-md border border-input bg-background px-3 py-2"
            value={read('feature_extraction.excerpt_seconds')}
            onchange={(event) =>
              save({ 'feature_extraction.excerpt_seconds': Number(event.currentTarget.value) })}
          />
          <span class="mt-1 block text-xs text-muted-foreground">
            Taken from the middle of the track, where the material is most representative. 0
            analyses the whole file — slower, rarely better. Changing this means re-analysing to
            compare like with like.
          </span>
        </label>

        <label class="block text-sm">
          <span class="font-medium">Parallel workers</span>
          <input
            type="number"
            min="-1"
            class="mt-1 w-32 rounded-md border border-input bg-background px-3 py-2"
            value={read('performance.num_workers')}
            onchange={(event) =>
              save({ 'performance.num_workers': Number(event.currentTarget.value) })}
          />
          <span class="mt-1 block text-xs text-muted-foreground">-1 uses every CPU core.</span>
        </label>
      </div>
    </section>

    <section>
      <h2 class="font-medium">LLM naming</h2>
      <div class="space-y-3 rounded-lg border border-border p-4">
        <label class="flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            class="mt-1 accent-primary"
            checked={read('labeling.llm_enabled')}
            onchange={(event) => save({ 'labeling.llm_enabled': event.currentTarget.checked })}
          />
          <span>
            <span class="font-medium">Use an LLM to suggest names for discovered groups</span>
            <span class="block text-xs text-muted-foreground">
              Only track metadata is sent — artist, title, genre tag, tempo — never audio. Set
              <span class="font-mono">ANTHROPIC_API_KEY</span> in the environment where the API
              runs, and install the <span class="font-mono">anthropic</span> package.
            </span>
          </span>
        </label>

        <label class="block text-sm">
          <span class="font-medium">Model</span>
          <input
            class="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs"
            value={read('labeling.llm_model')}
            onchange={(event) => save({ 'labeling.llm_model': event.currentTarget.value })}
          />
        </label>
      </div>
    </section>

    <section>
      <h2 class="font-medium">Database</h2>
      <p class="mt-1 break-all rounded-lg border border-border p-4 font-mono text-xs">
        {read('database.path')}
      </p>
    </section>

    {#if saving}
      <p class="flex items-center gap-2 text-sm text-muted-foreground">
        <Save class="h-4 w-4" /> Saving...
      </p>
    {/if}
  {/if}
</div>
