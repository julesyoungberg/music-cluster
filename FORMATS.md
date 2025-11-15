# Supported Audio Formats

## Overview

Music Cluster supports **mixed format libraries** and can analyze collections with different audio file types together. All feature extraction is format-agnostic - the tool converts everything to a common representation for analysis.

## Supported Formats

The tool supports any audio format that **librosa + FFmpeg** can decode. This includes:

### Compressed Formats
| Format | Extension | Notes |
|--------|-----------|-------|
| MP3 | `.mp3` | MPEG Audio Layer 3 |
| AAC | `.aac`, `.m4a` | Advanced Audio Coding |
| Ogg Vorbis | `.ogg` | Open format, good quality |
| Opus | `.opus` | Modern codec, excellent quality |
| WMA | `.wma` | Windows Media Audio |

### Lossless Formats
| Format | Extension | Notes |
|--------|-----------|-------|
| FLAC | `.flac` | Free Lossless Audio Codec - most common |
| WAV | `.wav` | Waveform Audio - uncompressed |
| AIFF | `.aiff`, `.aif` | Audio Interchange File Format (Apple) |
| ALAC | `.alac`, `.m4a` | Apple Lossless (usually in M4A container) |
| APE | `.ape` | Monkey's Audio - high compression |
| WavPack | `.wv` | Hybrid lossless/lossy codec |

### Other Formats
With FFmpeg installed, you can also process:
- `.ac3` - Dolby Digital
- `.dts` - DTS audio
- `.mka` - Matroska audio
- `.webm` - WebM audio (usually Opus or Vorbis)
- And many more...

## How It Works

1. **Loading**: Librosa uses FFmpeg/audioread to decode audio files to PCM
2. **Normalization**: All audio is resampled to a common sample rate (default: 44.1kHz)
3. **Feature Extraction**: Same features extracted regardless of source format
4. **Clustering**: Format-agnostic - MP3s and FLACs cluster together if they sound similar

## Default Extensions

By default, the tool scans for these extensions:
```
.mp3, .flac, .wav, .m4a, .ogg, .aiff, .aif, .opus, .aac, .wma, .ape, .alac, .wv
```

## Custom Extensions

### Analyze Specific Formats Only

```bash
# Only MP3 and FLAC
music-cluster analyze ~/Music --recursive --extensions mp3,flac

# Only lossless formats
music-cluster analyze ~/Music -r --extensions flac,wav,aiff,ape
```

### Add Unusual Formats

If you have files with non-standard extensions:

```bash
# Add custom extensions
music-cluster analyze ~/Music -r --extensions mp3,flac,wav,custom_ext
```

## Mixed Format Libraries

**Example library structure:**
```
~/Music/
├── Rock/
│   ├── song1.mp3
│   ├── song2.flac
│   └── song3.wav
├── Electronic/
│   ├── track1.opus
│   ├── track2.m4a
│   └── track3.ogg
└── Classical/
    ├── symphony1.flac
    └── symphony2.aiff
```

All these files will be analyzed together, and similar-sounding tracks will cluster together **regardless of format**.

## Format Quality Considerations

### Does Format Affect Clustering?

**Short answer**: Minimally, if at all.

**Long answer**: 
- Feature extraction focuses on **perceptual** audio characteristics
- Lossy compression (MP3, AAC, Ogg) removes inaudible information
- Feature extraction works on the decoded audio, so format compression is mostly irrelevant
- Very low bitrate files (< 128kbps) might cluster slightly differently due to artifacts
- Lossless vs lossy generally doesn't affect clustering results

### Recommendations

1. **Mixed libraries are fine** - Don't worry about having different formats
2. **Quality matters more than format** - 320kbps MP3 ≈ FLAC for clustering purposes
3. **Avoid very low bitrates** - Files < 96kbps may have noticeable artifacts
4. **Consistency not required** - The tool handles format differences automatically

## Troubleshooting

### Format Not Recognized

If a format isn't being picked up:

1. **Check FFmpeg installation:**
   ```bash
   ffmpeg -formats | grep <format>
   ```

2. **Test if librosa can load it:**
   ```python
   import librosa
   y, sr = librosa.load('your_file.xyz')
   ```

3. **Add extension explicitly:**
   ```bash
   music-cluster analyze ~/Music -r --extensions mp3,flac,xyz
   ```

### Codec Not Supported

Some exotic codecs may not be supported. Solutions:

1. **Install full FFmpeg** with all codecs:
   ```bash
   # macOS
   brew install ffmpeg --with-all
   
   # Ubuntu
   sudo apt-get install ffmpeg libavcodec-extra
   ```

2. **Convert unsupported files:**
   ```bash
   # Batch convert to FLAC
   for f in *.xyz; do ffmpeg -i "$f" "${f%.xyz}.flac"; done
   ```

## Performance by Format

### Analysis Speed

Approximate relative speeds (same duration):

| Format | Speed | Notes |
|--------|-------|-------|
| WAV | 1.0x | Fastest (no decompression) |
| FLAC | 1.1x | Very fast decompression |
| ALAC | 1.1x | Similar to FLAC |
| MP3 | 1.2x | Efficient decoder |
| AAC | 1.2x | Modern, efficient |
| Ogg/Opus | 1.3x | Slightly slower decode |
| APE | 1.5x | CPU-intensive decompression |
| WMA | 1.3x | Varies by version |

Differences are minor - file I/O usually dominates.

### Storage in Database

**All formats take the same space in the database** (~1-2 KB per track) because only:
- Metadata (filename, path, duration)
- Extracted features (same size regardless of source format)

are stored, not the audio itself.

## Best Practices

1. **Don't convert your library** - Use whatever formats you have
2. **FFmpeg is essential** - Make sure it's properly installed
3. **Test with small sample** - If unsure, test a few files first
4. **Specify extensions** - Use `--extensions` to limit scan if needed
5. **Check analysis output** - Tool will report files it can't process

## Summary

✅ **Mixed format libraries fully supported**  
✅ **15+ common formats work out of the box**  
✅ **Format-agnostic clustering**  
✅ **No quality loss in feature extraction**  
✅ **No conversion required**

The tool is designed to "just work" with whatever audio formats you have!
