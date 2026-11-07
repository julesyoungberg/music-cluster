import '@testing-library/jest-dom/vitest';

// jsdom has no layout engine, so anything that measures or animates needs a
// stand-in before a component under test can mount.
if (!globalThis.requestAnimationFrame) {
  globalThis.requestAnimationFrame = ((callback: FrameRequestCallback) =>
    setTimeout(
      () => callback(performance.now()),
      0
    ) as unknown as number) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame = ((handle: number) =>
    clearTimeout(handle)) as typeof cancelAnimationFrame;
}

if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}

// Node 25+ ships its own global `localStorage`, which shadows jsdom's and is
// unusable unless Node is started with `--localstorage-file`. Put the jsdom
// window's storage back so tests behave the same on every Node version.
const jsdomWindow = (globalThis as { jsdom?: { window: Window } }).jsdom?.window;
if (jsdomWindow && typeof globalThis.localStorage?.clear !== 'function') {
  for (const name of ['localStorage', 'sessionStorage'] as const) {
    Object.defineProperty(globalThis, name, {
      configurable: true,
      value: jsdomWindow[name]
    });
  }
}
