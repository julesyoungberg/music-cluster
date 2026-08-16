import { get } from 'svelte/store';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  dismiss,
  notifications,
  notify,
  notifyError,
  notifyInfo,
  notifySuccess
} from '$lib/stores/notifications';

describe('notifications', () => {
  beforeEach(() => {
    notifications.set([]);
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('appends a notification and hands back its id', () => {
    const id = notify('info', 'Analysing', '12 tracks');

    expect(get(notifications)).toEqual([
      { id, kind: 'info', message: 'Analysing', detail: '12 tracks' }
    ]);
  });

  it('clears a non-error notification on its own', () => {
    notifySuccess('Folders imported');
    expect(get(notifications)).toHaveLength(1);

    vi.advanceTimersByTime(5000);
    expect(get(notifications)).toHaveLength(0);
  });

  it('keeps an error until it is dismissed', () => {
    // An error a user never saw is an error they cannot act on, so these do not
    // time out.
    const id = notifyError('Commit failed', 'Permission denied');

    vi.advanceTimersByTime(60_000);
    expect(get(notifications)).toHaveLength(1);

    dismiss(id);
    expect(get(notifications)).toHaveLength(0);
  });

  it('gives every notification a distinct id', () => {
    const ids = [notifyInfo('a'), notifyInfo('b'), notifyInfo('c')];
    expect(new Set(ids).size).toBe(3);
  });

  it('dismissing one leaves the others alone', () => {
    const first = notifyError('first');
    const second = notifyError('second');

    dismiss(first);

    expect(get(notifications).map((n) => n.id)).toEqual([second]);
  });

  it('ignores a dismissal for an id that is already gone', () => {
    const id = notifyError('gone');
    dismiss(id);
    expect(() => dismiss(id)).not.toThrow();
    expect(get(notifications)).toHaveLength(0);
  });
});
