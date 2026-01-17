# Music Cluster UI

Modern desktop UI for Music Cluster built with Tauri and Svelte.

## Development

### Prerequisites

- Node.js 18+ and npm
- Rust and Cargo (for Tauri)
- Python dependencies installed (see main README)

### Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the FastAPI backend (in the project root):
   ```bash
   uvicorn music_cluster.api:app --reload --port 8000
   ```

3. Run the UI in development mode:
   ```bash
   npm run dev
   ```

4. Or run as Tauri desktop app:
   ```bash
   npm run tauri:dev
   ```

## Building

Build the desktop app:
```bash
npm run tauri:build
```

This creates native executables in `src-tauri/target/release/bundle/`.

## Project Structure

- `src/routes/` - SvelteKit routes (pages)
- `src/lib/components/` - Reusable components
- `src/lib/stores/` - Svelte stores for state management
- `src/lib/services/` - API client
- `src/lib/types/` - TypeScript type definitions
- `src-tauri/` - Tauri Rust backend
