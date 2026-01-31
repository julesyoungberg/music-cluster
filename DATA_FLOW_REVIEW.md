# UI Data Flow Review & Optimization Plan

## Current Data Flow Analysis

### 1. Artwork Loading
**Current Implementation:**
- Each `TrackArtwork` component makes an individual API call in `onMount`
- For 100 tracks = 100 separate HTTP requests
- Uses client-side caching (memory + localStorage)
- No request deduplication

**Issues:**
- ❌ N+1 query problem - one request per track
- ❌ No batching - each component loads independently
- ❌ Network waterfall - all requests fire simultaneously
- ❌ No server-side batch endpoint exists

### 2. Waveform Loading
**Current Implementation:**
- Preloaded in batches of 5 concurrent requests
- For 100 tracks = 20 sequential batches (5 at a time)
- Uses client-side caching
- Only loads waveforms for visible tracks

**Issues:**
- ⚠️ Better than artwork (batched), but still individual requests
- ⚠️ No server-side batch endpoint exists
- ✅ At least has concurrency limiting

### 3. Track Data Loading
**Current Implementation:**
- Single request for track list: `GET /api/tracks?limit=100&offset=0`
- Returns basic track metadata
- No artwork or waveform data included

**Status:**
- ✅ Efficient - single request for track list

## Optimization Opportunities

### Option 1: Add Batch API Endpoints (Recommended)
**Pros:**
- Reduces HTTP overhead significantly
- Server can optimize database queries
- Single round-trip for multiple resources
- Better for large libraries

**Cons:**
- Requires backend changes
- More complex API design

**Implementation:**
```typescript
// New batch endpoints
GET /api/tracks/batch/artwork?ids=1,2,3,4,5
GET /api/tracks/batch/waveform?ids=1,2,3,4,5&samples=200
```

### Option 2: Include Artwork/Waveform in Track List Response
**Pros:**
- Zero additional requests
- Simplest solution
- All data in one response

**Cons:**
- Large response payloads
- Loads data for tracks user might not view
- Not suitable for large lists (pagination issues)

### Option 3: Lazy Loading with Intersection Observer
**Pros:**
- Only loads what's visible
- Better initial load time
- Progressive enhancement

**Cons:**
- Still individual requests (but fewer)
- More complex implementation
- Doesn't solve the fundamental N+1 problem

### Option 4: Request Deduplication + Better Batching
**Pros:**
- No backend changes needed
- Can implement immediately
- Reduces duplicate requests

**Cons:**
- Still many requests (just better organized)
- Doesn't solve fundamental issue

## Recommended Solution: Hybrid Approach

### Phase 1: Immediate Improvements (No Backend Changes)
1. **Request Deduplication**
   - Track pending requests in a Map
   - Share promises across components
   - Prevents duplicate requests for same track

2. **Better Artwork Batching**
   - Batch artwork requests like waveforms
   - Load in background after tracks load
   - Use intersection observer for viewport-based loading

3. **Centralized Resource Loading**
   - Create a resource manager service
   - Manages all artwork/waveform loading
   - Provides unified caching and request management

### Phase 2: Backend Optimization (Future)
1. **Add Batch Endpoints**
   - `/api/tracks/batch/artwork` - Get artwork for multiple tracks
   - `/api/tracks/batch/waveform` - Get waveforms for multiple tracks
   - Accept comma-separated IDs or array in body

2. **Optional: Include in Track Response**
   - Add `include_artwork` and `include_waveform` query params
   - Only include if explicitly requested
   - Use for small lists or specific use cases

## Implementation Plan

### ✅ Step 1: Create Resource Manager Service (COMPLETED)
- ✅ Centralized loading of artwork and waveforms
- ✅ Request deduplication (prevents duplicate requests)
- ✅ Batch loading with concurrency control
- ✅ Unified caching interface
- ✅ Memory + localStorage caching

### ✅ Step 2: Refactor TrackArtwork Component (COMPLETED)
- ✅ Use resource manager instead of direct API calls
- ✅ Automatic deduplication
- ✅ Shared cache across all components

### ✅ Step 3: Improve Waveform Loading (COMPLETED)
- ✅ Use resource manager
- ✅ Better batching strategy (5 concurrent)
- ✅ Preloads for all visible tracks

### ✅ Step 4: Batch Artwork Loading (COMPLETED)
- ✅ Artwork now batched (10 concurrent)
- ✅ Preloads in background after tracks load
- ✅ Reduces initial request flood

### 🔄 Step 5: Add Intersection Observer (FUTURE)
- Load artwork only when track enters viewport
- Progressive loading for better UX
- Further reduce initial load time

## Expected Improvements

### Current State (100 tracks):
- Artwork: 100 requests (all at once)
- Waveform: 20 batches of 5 = 20 sequential batches
- Total: ~120 requests

### After Phase 1 (No Backend Changes):
- Artwork: Batched in groups of 10 = 10 batches
- Waveform: Same (already batched)
- Request deduplication prevents duplicates
- Intersection observer reduces initial load
- Total: ~30 requests (75% reduction)

### After Phase 2 (With Batch Endpoints):
- Artwork: 1-2 batch requests
- Waveform: 1-2 batch requests  
- Total: 2-4 requests (97% reduction)

## Metrics to Track
- Number of HTTP requests per page load
- Time to first artwork display
- Time to interactive
- Cache hit rate
- Network bandwidth usage
