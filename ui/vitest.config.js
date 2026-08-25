import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// Unit tests run against the modules directly rather than through SvelteKit, so
// the `$lib` alias is declared here instead of coming from `svelte-kit sync`.
// That keeps `npm test` runnable without a prior sync step.
export default defineConfig({
  plugins: [svelte({ hot: false })],
  resolve: {
    alias: {
      $lib: fileURLToPath(new URL('./src/lib', import.meta.url))
    },
    // Svelte components must be resolved through their browser entry points for
    // client-side rendering under jsdom.
    conditions: ['browser']
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.test.ts'],
    setupFiles: ['./tests/setup.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/lib/**/*.ts'],
      reporter: ['text', 'lcov']
    }
  }
});
