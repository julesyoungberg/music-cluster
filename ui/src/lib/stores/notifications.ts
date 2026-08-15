import { writable } from 'svelte/store';

export type NotificationKind = 'info' | 'success' | 'warning' | 'error';

export interface Notification {
  id: number;
  kind: NotificationKind;
  message: string;
  detail?: string;
}

let nextId = 1;

export const notifications = writable<Notification[]>([]);

export function notify(kind: NotificationKind, message: string, detail?: string): number {
  const id = nextId++;
  notifications.update((list) => [...list, { id, kind, message, detail }]);
  // Errors stay until dismissed; anything else clears itself.
  if (kind !== 'error') {
    setTimeout(() => dismiss(id), 4500);
  }
  return id;
}

export function dismiss(id: number): void {
  notifications.update((list) => list.filter((n) => n.id !== id));
}

export const notifySuccess = (message: string, detail?: string) => notify('success', message, detail);
export const notifyError = (message: string, detail?: string) => notify('error', message, detail);
export const notifyInfo = (message: string, detail?: string) => notify('info', message, detail);
