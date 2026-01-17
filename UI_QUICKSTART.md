# UI Quick Start Guide

## Recommended Approach: FastAPI + React/Vue

This is the fastest path to a modern UI that leverages your web development skills.

## Step 1: Set Up API Backend (30 minutes)

### Install Dependencies

```bash
pip install fastapi uvicorn
```

### Create API Module

I've created an example API in `music_cluster/api_example.py`. You can:

1. **Use it as-is** for testing
2. **Rename it** to `api.py` and customize
3. **Extend it** with additional endpoints

### Run the API

```bash
# Development mode (auto-reload)
uvicorn music_cluster.api_example:app --reload --port 8000

# Or add to your CLI
# music-cluster serve  # (you could add this command)
```

### Test the API

Open http://localhost:8000/docs for interactive API documentation (Swagger UI)

Or test with curl:
```bash
curl http://localhost:8000/api/info
curl http://localhost:8000/api/tracks
```

## Step 2: Build Frontend (2-3 days)

### Option A: React (Recommended if you know React)

```bash
# Create React app
npx create-react-app music-cluster-ui
cd music-cluster-ui

# Install dependencies
npm install axios  # For API calls

# Start dev server
npm start
```

### Option B: Vue (If you prefer Vue)

```bash
npm create vue@latest music-cluster-ui
cd music-cluster-ui
npm install
npm install axios
npm run dev
```

### Option C: Svelte (Lightweight alternative)

```bash
npm create svelte@latest music-cluster-ui
cd music-cluster-ui
npm install
npm install axios
npm run dev
```

## Step 3: Connect Frontend to API

Create an API client in your frontend:

```javascript
// src/services/api.js (or .ts)
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
});

export const musicClusterAPI = {
  // Get database info
  getInfo: () => api.get('/info'),
  
  // Get tracks
  getTracks: (limit = 100, offset = 0) => 
    api.get('/tracks', { params: { limit, offset } }),
  
  // Get clusterings
  getClusterings: () => api.get('/clusterings'),
  
  // Get clusters for a clustering
  getClusters: (clusteringId) => 
    api.get(`/clusterings/${clusteringId}`),
  
  // Start analysis
  analyze: (path, options) => 
    api.post('/analyze', { path, ...options }),
  
  // Create clustering
  cluster: (options) => 
    api.post('/cluster', options),
  
  // Search tracks
  search: (query, clustering) => 
    api.post('/search', null, { 
      params: { query, clustering } 
    }),
};
```

## Step 4: Build UI Components

### Example React Component Structure

```
src/
├── App.jsx
├── components/
│   ├── LibraryView.jsx      # Track list
│   ├── ClusteringPanel.jsx   # Create/view clusterings
│   ├── ClusterView.jsx       # Cluster visualization
│   ├── StatsPanel.jsx        # Statistics
│   └── SearchBar.jsx         # Search functionality
├── services/
│   └── api.js                # API client
└── styles/
    └── App.css
```

### Key UI Features to Build

1. **Library Browser**
   - List all tracks
   - Show analysis status
   - Filter/search

2. **Analysis Panel**
   - Select directory
   - Start analysis
   - Show progress
   - View results

3. **Clustering Panel**
   - Create new clustering
   - Configure parameters (granularity, algorithm)
   - View existing clusterings
   - Compare clusterings

4. **Cluster Visualization**
   - List clusters
   - Show cluster details
   - View tracks in cluster
   - Export playlists

5. **Statistics Dashboard**
   - Database stats
   - Clustering metrics
   - Cluster size distribution

## Step 5: Optional - Wrap in Tauri (Later)

Once your web UI is working, you can wrap it in Tauri for a native desktop app:

```bash
# In your frontend directory
npm install --save-dev @tauri-apps/cli
npm install @tauri-apps/api

# Initialize Tauri
npx tauri init

# Build native app
npm run tauri build
```

This creates a native app bundle (~15-25MB) that uses your web UI.

## Development Workflow

1. **Terminal 1**: Run FastAPI backend
   ```bash
   uvicorn music_cluster.api_example:app --reload
   ```

2. **Terminal 2**: Run frontend dev server
   ```bash
   cd music-cluster-ui
   npm start  # or npm run dev
   ```

3. **Browser**: Open http://localhost:3000 (or whatever port your frontend uses)

## Example Frontend Code

### React Component Example

```jsx
// src/components/LibraryView.jsx
import { useState, useEffect } from 'react';
import { musicClusterAPI } from '../services/api';

function LibraryView() {
  const [tracks, setTracks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [info, setInfo] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [infoRes, tracksRes] = await Promise.all([
          musicClusterAPI.getInfo(),
          musicClusterAPI.getTracks(100)
        ]);
        setInfo(infoRes.data);
        setTracks(tracksRes.data.tracks);
      } catch (error) {
        console.error('Failed to load data:', error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div>
      <h1>Music Library</h1>
      <div>
        <p>Total Tracks: {info?.total_tracks}</p>
        <p>Analyzed: {info?.analyzed_tracks}</p>
      </div>
      <ul>
        {tracks.map(track => (
          <li key={track.id}>{track.filename}</li>
        ))}
      </ul>
    </div>
  );
}

export default LibraryView;
```

## Next Steps

1. ✅ Set up FastAPI backend (use `api_example.py`)
2. ✅ Create frontend project
3. ✅ Build basic UI components
4. ✅ Connect to API
5. ✅ Add features incrementally
6. ⏭️ (Optional) Wrap in Tauri for native app

## Tips

- **Start simple**: Build one feature at a time (e.g., track list first)
- **Use the API docs**: Visit http://localhost:8000/docs to see all endpoints
- **Test in browser**: Use browser DevTools to debug API calls
- **Progressive enhancement**: Add features as you need them

## Resources

- FastAPI Docs: https://fastapi.tiangolo.com/
- React Docs: https://react.dev/
- Vue Docs: https://vuejs.org/
- Tauri Docs: https://tauri.app/

Good luck! 🎵
