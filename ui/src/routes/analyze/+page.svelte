<script lang="ts">
  import { onDestroy } from 'svelte';
  import { api } from '$lib/services/api';
  import { activeTasks, updateTask } from '$lib/stores/tasks';
  import type { AnalyzeRequest, TaskStatus } from '$lib/types';
  import { open } from '@tauri-apps/api/dialog';
  import { Scan, FolderOpen, Play, CheckCircle2, AlertCircle, Loader2, Clock } from 'lucide-svelte';
  import { addNotification } from '$lib/stores/notifications';

  let request: AnalyzeRequest = {
    path: '',
    recursive: true,
    update: false,
    extensions: 'mp3,flac,wav,m4a,ogg',
    workers: -1,
    skip_errors: true
  };

  let taskId: string | null = null;
  let polling = false;
  let intervalId: ReturnType<typeof setInterval> | null = null;

  async function selectDirectory() {
    try {
      const path = await open({
        directory: true,
        multiple: false
      });
      if (path && typeof path === 'string') {
        request.path = path;
      }
    } catch (e) {
      console.error('Failed to open directory dialog:', e);
    }
  }

  async function startAnalysis() {
    if (!request.path) {
      addNotification('error', 'Please select a directory to analyze');
      return;
    }

    try {
      const result = await api.analyze(request);
      taskId = result.task_id;
      polling = true;
      addNotification('info', 'Analysis started. Processing files...');
      pollTaskStatus();
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : 'Failed to start analysis';
      addNotification('error', errorMsg);
      console.error('Failed to start analysis:', e);
    }
  }

  async function pollTaskStatus() {
    if (!taskId) return;

    // Clear any existing interval
    if (intervalId) {
      clearInterval(intervalId);
    }

    intervalId = setInterval(async () => {
      try {
        const status = await api.getTaskStatus(taskId!);
        updateTask(taskId!, status);

        if (status.status === 'complete') {
          if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
          }
          polling = false;
          const analyzed = status.analyzed || status.completed || 0;
          const errors = status.errors || 0;
          if (errors > 0) {
            addNotification('warning', `Analysis complete: ${analyzed} tracks analyzed, ${errors} errors`);
          } else {
            addNotification('success', `Successfully analyzed ${analyzed} tracks`);
          }
        } else if (status.status === 'error') {
          if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
          }
          polling = false;
          addNotification('error', status.message || 'Analysis failed');
        }
      } catch (e) {
        console.error('Failed to get task status:', e);
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        polling = false;
        addNotification('error', 'Failed to get task status');
      }
    }, 1000);
  }

  onDestroy(() => {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  });

  $: taskStatus = taskId ? $activeTasks.get(taskId) : null;
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8 flex items-center gap-3">
    <Scan class="w-10 h-10" />
    Analyze Music Library
  </h1>

  <div class="max-w-2xl space-y-6">
    <div>
      <label for="directory-path" class="block text-sm font-medium mb-2">Directory Path</label>
      <div class="flex gap-2">
        <div class="flex-1 relative">
          <FolderOpen class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <input
            id="directory-path"
            type="text"
            bind:value={request.path}
            class="w-full pl-10 pr-4 py-2 border rounded-lg bg-background"
            placeholder="Select a directory..."
          />
        </div>
        <button
          on:click={selectDirectory}
          class="px-4 py-2 bg-primary text-primary-foreground rounded-lg flex items-center gap-2 hover:opacity-90 transition-opacity"
        >
          <FolderOpen class="w-4 h-4" />
          Browse
        </button>
      </div>
    </div>

    <div>
      <label class="flex items-center gap-2">
        <input type="checkbox" bind:checked={request.recursive} />
        <span>Recursive scan</span>
      </label>
    </div>

    <div>
      <label class="flex items-center gap-2">
        <input type="checkbox" bind:checked={request.update} />
        <span>Re-analyze existing tracks</span>
      </label>
    </div>

    <div>
      <label for="extensions" class="block text-sm font-medium mb-2">File Extensions</label>
      <input
        id="extensions"
        type="text"
        bind:value={request.extensions}
        class="w-full p-2 border rounded-lg bg-background"
      />
    </div>

    <button
      on:click={startAnalysis}
      disabled={!request.path || polling}
      class="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
    >
      {#if polling}
        <Loader2 class="w-4 h-4 animate-spin" />
        <span>Analyzing...</span>
      {:else}
        <Play class="w-4 h-4" />
        <span>Start Analysis</span>
      {/if}
    </button>

    {#if taskStatus}
      <div class="bg-card p-4 rounded-lg border">
        <div class="flex items-center gap-2 mb-4">
          {#if taskStatus.status === 'complete'}
            <CheckCircle2 class="w-5 h-5 text-green-500" />
          {:else if taskStatus.status === 'error'}
            <AlertCircle class="w-5 h-5 text-destructive" />
          {:else}
            <Loader2 class="w-5 h-5 animate-spin text-primary" />
          {/if}
          <h3 class="font-semibold capitalize">Status: {taskStatus.status}</h3>
        </div>
        {#if taskStatus.total && taskStatus.total > 0}
          <div class="mb-4">
            <div class="flex justify-between text-sm mb-2">
              <span class="font-medium">Progress</span>
              <span class="text-muted-foreground">
                {taskStatus.completed || 0} / {taskStatus.total} files
              </span>
            </div>
            <div class="w-full bg-secondary rounded-full h-3 mb-2">
              <div
                class="bg-primary h-3 rounded-full transition-all duration-300"
                style="width: {(taskStatus.progress || 0)}%"
              ></div>
            </div>
            <div class="flex items-center justify-between text-xs text-muted-foreground">
              <div class="flex items-center gap-4">
                {#if taskStatus.eta_formatted}
                  <div class="flex items-center gap-1">
                    <Clock class="w-3 h-3" />
                    <span>ETA: {taskStatus.eta_formatted}</span>
                  </div>
                {/if}
                {#if taskStatus.elapsed_formatted}
                  <span>Elapsed: {taskStatus.elapsed_formatted}</span>
                {/if}
              </div>
              {#if taskStatus.rate_per_second && taskStatus.rate_per_second > 0}
                <span>{taskStatus.rate_per_second.toFixed(1)} files/sec</span>
              {/if}
            </div>
          </div>
        {/if}
        {#if taskStatus.current_file}
          <div class="mb-2 p-2 bg-secondary/50 rounded text-sm">
            <p class="text-muted-foreground text-xs mb-1">Currently processing:</p>
            <p class="truncate font-mono text-xs">{taskStatus.current_file}</p>
          </div>
        {/if}
        {#if taskStatus.status === 'complete'}
          <div class="space-y-1">
            {#if taskStatus.analyzed !== undefined}
              <p class="text-sm text-green-600 dark:text-green-400">
                ✓ Analyzed: {taskStatus.analyzed} tracks
              </p>
            {/if}
            {#if taskStatus.errors && taskStatus.errors > 0}
              <p class="text-sm text-destructive flex items-center gap-1">
                <AlertCircle class="w-4 h-4" />
                Errors: {taskStatus.errors}
              </p>
            {/if}
          </div>
        {:else if taskStatus.status === 'error'}
          <p class="text-sm text-destructive">{taskStatus.message || 'An error occurred'}</p>
        {/if}
        {#if taskStatus.message && taskStatus.status !== 'complete' && taskStatus.status !== 'error'}
          <p class="text-sm text-muted-foreground mt-2">{taskStatus.message}</p>
        {/if}
      </div>
    {/if}
  </div>
</div>
