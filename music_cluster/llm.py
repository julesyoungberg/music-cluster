"""Optional LLM assist for naming discovered groups.

Entirely opt-in. Nothing else in the application depends on it, and every
failure path falls back to the heuristic labels in :mod:`music_cluster.naming`.
Only track metadata (artist, title, genre tag, BPM) is sent — never audio.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROMPT = """You are helping a DJ name folders in their music library.

Below are candidate groups that were found by clustering audio features. For \
each one you get a few representative tracks and some measured statistics.

Suggest a short folder name for each group: 2-4 words, in the vocabulary a DJ \
would use for crates (genre and feel, e.g. "Deep Rolling House", "Peak Time \
Techno", "Broken Beat / UKG"). Do not include BPM numbers, do not number the \
groups, and make the names distinct from each other.

Respond with JSON only, in the form:
{"names": [{"candidate_index": 0, "name": "..."}, ...]}

Candidates:
%s
"""


def is_available(config) -> bool:
    """True when LLM labelling is enabled and an API key is present."""
    if not config.get("labeling", "llm_enabled", default=False):
        return False
    env_var = config.get("labeling", "llm_api_key_env", default="ANTHROPIC_API_KEY")
    return bool(os.environ.get(env_var))


def suggest_names(
    candidates: List[Dict[str, Any]], config, timeout: float = 60.0
) -> Dict[int, str]:
    """Suggest names for candidate groups, keyed by ``candidate_index``.

    Returns an empty mapping if the LLM is unavailable or the call fails, so
    callers can simply fall back to heuristic names.
    """
    if not is_available(config) or not candidates:
        return {}

    try:
        import anthropic
    except ImportError:
        logger.info("LLM labelling enabled but the 'anthropic' package is not installed")
        return {}

    env_var = config.get("labeling", "llm_api_key_env", default="ANTHROPIC_API_KEY")
    model = config.get("labeling", "llm_model", default="claude-sonnet-5")

    try:
        client = anthropic.Anthropic(api_key=os.environ[env_var], timeout=timeout)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": PROMPT % _render(candidates)}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return _parse(text)
    except Exception as exc:
        logger.warning("LLM labelling failed, using heuristic names: %s", exc)
        return {}


def _render(candidates: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for candidate in candidates:
        lines.append(f"\n## Candidate {candidate['candidate_index']} ({candidate['size']} tracks)")
        stats = candidate.get("stats") or {}
        if stats.get("tag_bpm_median") or stats.get("detected_bpm"):
            lines.append(f"- tempo: ~{stats.get('tag_bpm_median') or stats.get('detected_bpm')} BPM")
        if stats.get("descriptors"):
            lines.append(f"- character: {', '.join(stats['descriptors'])}")
        if stats.get("top_genres"):
            genres = ", ".join(f"{g['name']} ({g['count']})" for g in stats["top_genres"])
            lines.append(f"- genre tags: {genres}")
        for track in candidate.get("exemplar_tracks", [])[:6]:
            artist = track.get("artist") or "Unknown"
            title = track.get("title") or track.get("filename")
            lines.append(f"- track: {artist} - {title}")
    return "\n".join(lines)


def _parse(text: str) -> Dict[int, str]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}

    names: Dict[int, str] = {}
    for entry in data.get("names", []):
        try:
            index = int(entry["candidate_index"])
            name = str(entry["name"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if name:
            names[index] = name[:80]
    return names
