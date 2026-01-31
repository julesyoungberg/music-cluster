# Audio Player Improvements Plan

## Current Issues

1. **No Visual Feedback**: Play/pause icon doesn't update because `useAudioPlayer` uses getters which aren't reactive in Svelte
2. **Waveform Only Shows for Current Track**: Should be preloaded and visible for all tracks
3. **Missing Controls**: Skip forward/backward buttons only show for current track
4. **Waveform Seeking**: Already implemented but needs proper state management

## Solution Plan

### 1. Fix Reactive State in `useAudioPlayer`
- **Problem**: Getters aren't reactive in Svelte - `$: playing = audioPlayer.playing` won't update
- **Solution**: Convert to Svelte stores (writable) for reactive state
- **Changes**:
  - Replace getters with writable stores
  - Return stores instead of getters
  - Update stores when audio events fire

### 2. Preload Waveforms
- **Problem**: Waveforms only load when track is played
- **Solution**: Preload waveforms for all visible tracks when tracks are loaded
- **Implementation**:
  - Add batch waveform loading function
  - Load waveforms in parallel (with concurrency limit)
  - Show loading state for individual waveforms
  - Cache waveforms in memory

### 3. Always Show Waveform for Playing Track
- **Problem**: Waveform only shows conditionally
- **Solution**: Always show waveform when track is playing, with seek functionality
- **Implementation**:
  - Show waveform for currently playing track
  - Make waveform clickable to seek
  - Show progress indicator on waveform

### 4. Improve Visual Feedback
- **Problem**: Play/pause state not visible
- **Solution**: 
  - Show play icon for all tracks
  - Show pause icon when track is playing
  - Highlight currently playing track
  - Show skip buttons when any track is playing

### 5. Waveform Click-to-Seek
- **Problem**: Need to verify seeking works
- **Solution**: Ensure waveform component properly dispatches seek events and audio player handles them

## Implementation Steps

1. Refactor `useAudioPlayer` to use Svelte stores
2. Add waveform preloading function
3. Update library page to preload waveforms
4. Update cluster detail page to preload waveforms
5. Ensure waveform is always visible for playing track
6. Fix play/pause button state updates
7. Test seek functionality

## Files to Modify

1. `ui/src/lib/composables/useAudioPlayer.ts` - Convert to stores
2. `ui/src/routes/library/+page.svelte` - Add preloading, fix reactivity
3. `ui/src/routes/clusters/[id]/+page.svelte` - Add preloading, fix reactivity
4. `ui/src/lib/components/Waveform.svelte` - Verify seek functionality
