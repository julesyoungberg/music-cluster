<script lang="ts">
  import { api } from '$lib/services/api';
  import { activeTasks, updateTask } from '$lib/stores/tasks';
  import type { AnalyzeRequest, TaskStatus } from '$lib/types';
  import { open } from '@tauri-apps/api/dialog';

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
    if (!request.path) return;

    try {
      const result = await api.analyze(request);
      taskId = result.task_id;
      polling = true;
      pollTaskStatus();
    } catch (e) {
      console.error('Failed to start analysis:', e);
    }
  }

  async function pollTaskStatus() {
    if (!taskId) return;

    const interval = setInterval(async () => {
      try {
        const status = await api.getTaskStatus(taskId!);
        updateTask(taskId!, status);

        if (status.status === 'complete' || status.status === 'error') {
          clearInterval(interval);
          polling = false;
        }
      } catch (e) {
        console.error('Failed to get task status:', e);
        clearInterval(interval);
        polling = false;
      }
    }, 1000);
  }

  $: taskStatus = taskId ? $activeTasks.get(taskId) : null;
</script>

<div class="container mx-auto p-8">
  <h1 class="text-4xl font-bold mb-8">Analyze Music Library</h1>

  <div class="max-w-2xl space-y-6">
    <div>
      <label for="directory-path" class="block text-sm font-medium mb-2">Directory Path</label>
      <div class="flex gap-2">
        <input
          id="directory-path"
          type="text"
          bind:value={request.path}
          class="flex-1 p-2 border rounded-lg bg-background"
          placeholder="Select a directory..."
        />
        <button
          on:click={selectDirectory}
          class="px-4 py-2 bg-primary text-primary-foreground rounded-lg"
        >
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
      class="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50"
    >
      Start Analysis
    </button>

    {#if taskStatus}
      <div class="bg-card p-4 rounded-lg border">
        <h3 class="font-semibold mb-2">Status: {taskStatus.status}</h3>
        {#if taskStatus.total}
          <div class="mb-2">
            <div class="flex justify-between text-sm mb-1">
              <span>Progress</span>
              <span>{taskStatus.completed || 0} / {taskStatus.total}</span>
            </div>
            <div class="w-full bg-secondary rounded-full h-2">
              <div
                class="bg-primary h-2 rounded-full transition-all"
                style="width: {(taskStatus.progress || 0)}%"
              ></div>
            </div>
          </div>
        {/if}
        {#if taskStatus.errors}
          <p class="text-sm text-destructive">Errors: {taskStatus.errors}</p>
        {/if}
        {#if taskStatus.message}
          <p class="text-sm text-muted-foreground">{taskStatus.message}</p>
        {/if}
      </div>
    {/if}
  </div>
</div>
