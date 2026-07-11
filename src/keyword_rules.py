"""
Rule-based keyword enhancement layer.

Purpose: after a deep-learning model gives raw probabilities for each
emotion, we nudge those scores using simple keyword matching. This
catches obvious signals the model might miss and is what produces
"mixed emotion" outputs (e.g. Curious + Confused) when two scores end
up close together.
"""

from typing import Dict

EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

KEYWORD_MAP: Dict[str, list] = {
    "Bored": [
        "boring", "bored", "same thing", "nothing new", "tedious",
        "uninterested", "monotonous", "dull",
    ],
    "Confident": ["i understand", "i get it", "this makes sense", "easy",
"i can do this",
    ],
    "Confused": [
        "confused", "don't understand", "dont understand", "lost",
        "no idea", "stuck", "doesn't make sense", "what does this mean",
        "i don't get", "unclear",
    ],
    "Curious": [
        "wonder", "how does", "why does", "what if", "interested in",
        "curious", "want to know more", "how come",
    ],
    "Frustrated": [
        "frustrated", "annoying", "give up", "so hard", "hate this",
        "keep failing", "tried everything", "can't figure", "cant figure",
        "fed up",
    ],
}

# Reserved probability mass guaranteed to matched emotion(s), so a clear
# keyword hit can never be silently outvoted by model noise.
BASE_FLOOR = 0.55        # reserved mass for 1 keyword match
EXTRA_PER_MATCH = 0.10   # extra reserved mass per additional match
MAX_FLOOR = 0.85         # cap so raw model score still has some influence


def apply_keyword_boost(text: str, scores: Dict[str, float]) -> Dict[str, float]:
    """
    Take model output scores (emotion -> probability, should sum to ~1)
    and the original text, and return adjusted scores that also sum to 1.

    Keyword matches reserve a guaranteed floor share of the total
    probability mass (rather than just adding a small bonus), so a
    clear keyword hit reliably wins even against noisy/random model
    output. If multiple emotions have matches, the reserved mass is
    split between them proportional to match count (supports mixed
    emotions like Confused + Frustrated).
    """
    text_lower = text.lower()
    matches = {
        emotion: sum(1 for kw in KEYWORD_MAP.get(emotion, []) if kw in text_lower)
        for emotion in EMOTIONS
    }
    matched = {e: c for e, c in matches.items() if c > 0}

    raw_total = sum(scores.values()) or 1.0
    normalized_raw = {e: scores.get(e, 0.0) / raw_total for e in EMOTIONS}

    if not matched:
        return normalized_raw

    total_match_count = sum(matched.values())
    reserved = min(BASE_FLOOR + EXTRA_PER_MATCH * (total_match_count - 1), MAX_FLOOR)
    remaining = 1.0 - reserved

    result = {e: normalized_raw[e] * remaining for e in EMOTIONS}
    for emotion, count in matched.items():
        share = (count / total_match_count) * reserved
        result[emotion] += share

    return result

def get_mixed_emotions(scores: Dict[str, float], threshold: float = 0.15):
    """
    Return all emotions within `threshold` of the top score, sorted
    descending. This is what powers "Curious + Confused" style output.
    """
    if not scores:
        return []
    top_score = max(scores.values())
    mixed = [
        (emotion, score) for emotion, score in scores.items()
        if top_score - score <= threshold
    ]
    return sorted(mixed, key=lambda x: x[1], reverse=True)