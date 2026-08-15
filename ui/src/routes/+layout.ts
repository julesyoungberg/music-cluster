// Everything is rendered in the browser: the app reads a local API and the
// user's filesystem, neither of which exists at build time.
export const ssr = false;
export const prerender = false;
export const trailingSlash = 'never';
