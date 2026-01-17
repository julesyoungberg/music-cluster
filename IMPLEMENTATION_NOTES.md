# Implementation Notes

## Tauri + Svelte UI Implementation

This document contains notes about the UI implementation.

### Architecture

- **Backend**: FastAPI REST API (`music_cluster/api.py`)
- **Frontend**: Svelte 5 with SvelteKit
- **Desktop Shell**: Tauri (Rust)

### Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node.js dependencies:**
   ```bash
   cd ui
   npm install
   ```

3. **Start the API server:**
   ```bash
   uvicorn music_cluster.api:app --reload --port 8000
   ```

4. **Run the UI:**
   ```bash
   cd ui
   npm run dev  # Web version
   # or
   npm run tauri:dev  # Desktop app
   ```

### Development Notes

- The API runs on `http://localhost:8000` by default
- The UI dev server runs on `http://localhost:1420` (configured for Tauri)
- Tauri will proxy the frontend and can communicate with the API

### Future Enhancements

- Add WebSocket support for real-time progress updates
- Implement shadcn-svelte components for better UI
- Add more visualizations (charts, cluster graphs)
- Implement drag-and-drop file import
- Add keyboard shortcuts
- Add audio preview functionality
