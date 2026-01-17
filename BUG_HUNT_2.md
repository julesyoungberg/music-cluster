# Bug Hunt #2 - Deeper Analysis
## Date: 2026-01-17

## 🐛 Bugs Found

### Bug 1: Division by Zero in BPM Rounding
**Location**: `cluster_namer.py:267-268`
**Severity**: LOW (protected by if bpm > 0 check, but still risky)

```python
bpm_min_rounded = int(stats['bpm_min'] / 5) * 5
```

**Issue**: If `stats['bpm_min']` is 0, this divides by zero (though protected by outer check)

**Status**: Actually safe because wrapped in `if stats['bpm'] > 0` check at line 264

### Bug 2: Empty Array Handling in np.percentile
**Location**: `cluster_namer.py:195-196`
**Severity**: LOW (protected but worth noting)

```python
bpm_p10 = np.percentile(bpms, 10)
bpm_p90 = np.percentile(bpms, 90)
```

**Issue**: If `bpms` has < 2 elements, percentile may behave unexpectedly

**Status**: Protected by `if bpms is not None and len(bpms) > 0` check

### Bug 3: List Indexing Without Bounds Check
**Location**: `cli.py:730` (label-clusters)
**Severity**: MEDIUM

```python
track_idx = track_ids.index(member['id'])
```

**Issue**: `.index()` raises ValueError if not found. Also, using dict key 'id' but should be track_id

**Status**: **NEEDS FIX** - This will crash if track not in list

### Bug 4: Potential Unicode Encoding Issues
**Location**: Multiple CLI output locations
**Severity**: LOW

**Issue**: Using Unicode symbols (✓✗←) without encoding checks

**Status**: May fail on non-UTF8 terminals (low priority)

### Bug 5: No Validation of Cluster Index in rename-cluster
**Location**: `cli.py:690-693`
**Severity**: MEDIUM

```python
cluster = db.get_cluster_by_index(clustering['id'], cluster_index)
if not cluster:
    click.echo(f"Error: Cluster {cluster_index} not found...")
    return
```

**Status**: Actually OK - validates cluster exists

### Bug 6: Missing Error Handling in Search
**Location**: `cli.py:807-830`
**Severity**: MEDIUM

**Issue**: No try-except around database queries

**Status**: **NEEDS FIX** - Should wrap in try-except

### Bug 7: Unchecked Array Slicing
**Location**: `cluster_namer.py:210-219`
**Severity**: LOW

```python
energy_features = centroid[40:60] if len(centroid) > 60 else centroid[20:40]
```

**Issue**: What if centroid has < 20 elements?

**Status**: Edge case, but should handle gracefully

## 🔧 Fixes to Apply

### Fix 1: Handle track_idx properly in label-clusters

**Before:**
```python
track_idx = track_ids.index(member['id'])
labels[track_idx] = cluster['cluster_index']
```

**After:**
```python
try:
    track_idx = track_ids.index(member['id'])
    labels[track_idx] = cluster['cluster_index']
except (ValueError, KeyError):
    # Track not in feature list, skip
    continue
```

### Fix 2: Add try-except to search command

**Add:**
```python
try:
    all_tracks = db.get_all_tracks()
    # ... rest of logic
except Exception as e:
    click.echo(f"Error searching tracks: {e}", err=True)
    return
```

### Fix 3: Graceful handling of small centroids

**Add check:**
```python
if len(centroid) < 20:
    # Not enough features, skip analysis
    return None
```

## ✅ Code Quality Improvements

### Improvement 1: Extract Magic Numbers
**Status**: DONE in genre_classifier.py - thresholds are clear

### Improvement 2: Add Logging
**Status**: TODO - needs implementation

### Improvement 3: Input Validation
**Status**: PARTIAL - some commands validate, others don't

### Improvement 4: Memory Efficiency
**Status**: TODO - label-clusters loads all features

## 🧪 Edge Cases to Test

1. **Empty database**: Run commands on empty DB
2. **Single track**: Cluster with only 1 track
3. **All identical tracks**: Same audio copied multiple times
4. **Corrupted audio**: Files that fail to analyze
5. **Missing files**: Tracks in DB but files deleted
6. **Very large library**: 100k+ tracks
7. **Unicode filenames**: Non-ASCII characters
8. **Very short audio**: < 1 second tracks
9. **Silent audio**: 0 energy tracks
10. **Extreme BPMs**: 0 or 500+ BPM

## 📊 Code Smells

### Smell 1: Long Functions
- `label_clusters_cmd`: 70+ lines
- `analyze`: 150+ lines
- `cluster`: 200+ lines

**Recommendation**: Extract helper functions

### Smell 2: Deep Nesting
- Some CLI commands have 4+ levels of nesting
- Reduces readability

**Recommendation**: Use early returns, extract functions

### Smell 3: Inconsistent Naming
- Some functions use `clustering_name`, others use `name`
- Mix of snake_case and camelCase in some places

**Recommendation**: Standardize naming conventions

### Smell 4: Repeated Database Patterns
- Many functions repeat: `config = Config()`, `db = Database(...)`

**Recommendation**: Create decorator or helper

```python
def with_database(func):
    def wrapper(*args, **kwargs):
        config = Config()
        db = Database(config.get_db_path())
        return func(db, *args, **kwargs)
    return wrapper
```

## 🎯 Priority Actions

### Immediate (Critical Bugs)
1. ✅ Fix track_idx bug in label-clusters
2. ✅ Add error handling to search
3. ✅ Validate centroid size in cluster_namer

### Short Term (Important)
1. Add comprehensive error handling to all CLI commands
2. Extract magic numbers to constants
3. Add input validation helpers
4. Test edge cases (empty DB, single track, etc.)

### Long Term (Nice to Have)
1. Refactor long functions
2. Add logging system
3. Create database helper decorator
4. Improve memory efficiency
5. Add unit tests

## 🎉 Overall Assessment

**Second Review Grade: B (Good)**

Main remaining issues:
- Label-clusters has a critical bug with track indexing
- Some commands lack error handling
- Edge cases not fully tested

However:
- Architecture is solid
- Most code is well-written
- Documentation is comprehensive
- Recent fixes improved quality significantly
