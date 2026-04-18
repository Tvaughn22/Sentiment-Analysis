import re
import html
from typing import List, Optional

# Compiled Regexes
URL_RE   = re.compile(r"https?://\S+|www\.\S+")
USER_RE  = re.compile(r"@\w+")
MULTI_WS = re.compile(r"\s+")

# Split CamelCase hashtags: #ThisIsBad → This Is Bad
HASHTAG_RE  = re.compile(r"#([A-Za-z][A-Za-z0-9]*)")
CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Contractions
# Expanded to full two-word forms so that pre-trained embeddings (GloVe,
# Word2Vec, FastText) find a vector for each token individually.
# e.g. "don't" → "do not"  (both in-vocabulary) not → "dont" (OOV).
CONTRACTIONS: dict[str, str] = {
    # Negative contractions
    "won't":    "will not",
    "can't":    "cannot",
    "don't":    "do not",
    "isn't":    "is not",
    "wasn't":   "was not",
    "aren't":   "are not",
    "weren't":  "were not",
    "haven't":  "have not",
    "hasn't":   "has not",
    "hadn't":   "had not",
    "couldn't": "could not",
    "shouldn't":"should not",
    "wouldn't": "would not",
    "didn't":   "did not",
    "doesn't":  "does not",
    "needn't":  "need not",
    "mustn't":  "must not",
    # Positive contractions (for better embedding coverage)
    "i'm":      "i am",
    "i've":     "i have",
    "i'll":     "i will",
    "i'd":      "i would",
    "you're":   "you are",
    "you've":   "you have",
    "you'll":   "you will",
    "you'd":    "you would",
    "he's":     "he is",
    "she's":    "she is",
    "it's":     "it is",
    "we're":    "we are",
    "we've":    "we have",
    "we'll":    "we will",
    "we'd":     "we would",
    "they're":  "they are",
    "they've":  "they have",
    "they'll":  "they will",
    "they'd":   "they would",
    "that's":   "that is",
    "there's":  "there is",
    "what's":   "what is",
    "who's":    "who is",
    "how's":    "how is",
    "let's":    "let us",
}
# Match longest key first to avoid partial matches (e.g. "won't" before "n't")
_CONTR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Negation
# Only contains forms present AFTER contraction expansion.
# Collapsed forms like "dont"/"wasnt" are no longer needed because
# "don't" → "do not" and "wasn't" → "was not" before this step runs.
NEGATIONS = {
    "not", "no", "never", "neither", "nor",
    "nobody", "nothing", "nowhere",
    "hardly", "scarcely", "barely",
    "cannot",   # from "can't" expansion
}

# Slang / abbreviation normalization
SLANG: dict[str, str] = {
    "lol":    "laughing out loud",
    "lmao":   "laughing",
    "lmfao":  "laughing",
    "rofl":   "laughing",
    "omg":    "oh my god",
    "omfg":   "oh my god",
    "wtf":    "what the heck",
    "wth":    "what the heck",
    "smh":    "shaking my head",
    "imo":    "in my opinion",
    "imho":   "in my honest opinion",
    "tbh":    "to be honest",
    "ngl":    "not going to lie",   # contains "not" — expanded BEFORE negation
    "idk":    "i do not know",      # contains "not" — expanded BEFORE negation
    "ik":     "i know",
    "ikr":    "i know right",
    "irl":    "in real life",
    "afaik":  "as far as i know",
    "afk":    "away from keyboard",
    "brb":    "be right back",
    "btw":    "by the way",
    "fyi":    "for your information",
    "til":    "today i learned",
    "eli5":   "explain like i am five",
    "tldr":   "too long did not read",
    "tho":    "though",
    "rn":     "right now",
    "atm":    "at the moment",
    "nvm":    "never mind",
    "fr":     "for real",
    "lowkey": "somewhat",
    "highkey":"very",
    "salty":  "bitter",
    "lit":    "exciting",
    "goat":   "greatest of all time",
    "af":     "very",
    "fwiw":   "for what it is worth",
    "gg":     "good game",
    "gl":     "good luck",
    "hmu":    "hit me up",
    "dm":     "direct message",
    "sus":    "suspicious",
    "vibe":   "feeling",
    "bc":     "because",
}

# Punctuation / emoji regex
_EMOJI = (
    r"\U00002600-\U000027BF"   # Misc symbols (☀️ ❄️ ✨ ✔️)
    r"\U0001F300-\U0001F5FF"   # Misc symbols & pictographs
    r"\U0001F600-\U0001F64F"   # Emoticons 😀–🙏
    r"\U0001F680-\U0001F6FF"   # Transport & map
    r"\U0001F900-\U0001F9FF"   # Supplemental symbols
    r"\U0001FA00-\U0001FAFF"   # Newer emoji (🫠 🫶)
)
PUNCT_RE = re.compile(rf"[^\w\s<>{_EMOJI}]")


# Internal helpers

def _expand_hashtag(match: re.Match) -> str:
    """#ThisIsBad → 'This Is Bad'  (CamelCase splitting)."""
    return CAMEL_SPLIT.sub(" ", match.group(1))


def _expand_slang(tokens: List[str]) -> List[str]:
    """Replace known slang tokens with their expanded multi-word forms."""
    result: List[str] = []
    for tok in tokens:
        expanded = SLANG.get(tok)
        if expanded:
            result.extend(expanded.split())
        else:
            result.append(tok)
    return result


def _apply_negation(tokens: List[str]) -> List[str]:
    """
    Join a negation word with the immediately following content word so that
    'not good' becomes 'not_good'. Scope resets after one token or at a
    placeholder boundary (<url> / <user>).

    NOTE — For Word Embedding models: disable this step (handle_negation=False)
    because joined tokens like 'not_good' are out-of-vocabulary in GloVe /
    Word2Vec and receive a zero/unknown vector, losing all semantic meaning.
    This strategy benefits Bag-of-Words models only.
    """
    result: List[str] = []
    negate_next = False
    for tok in tokens:
        if negate_next and tok not in ("<url>", "<user>"):
            result.append(f"not_{tok}")
            negate_next = False
        elif tok in NEGATIONS:
            result.append(tok)
            negate_next = True
        else:
            result.append(tok)
            negate_next = False
    return result

# Public API
def clean_text(
    text: Optional[str],
    lowercase: bool = True,
    replace_urls: bool = True,
    replace_users: bool = True,
    keep_punct: bool = False,
    expand_hashtags: bool = True,
    expand_slang: bool = True,
    handle_negation: bool = False,
) -> str:
    """
    Clean social-media text for sentiment analysis.

    Pipeline (in order):
      1.  Unescape HTML entities          (&amp; → &)
      2.  Expand CamelCase hashtags       (#ThisIsBad → This Is Bad)
      3.  Replace URLs                    (→ <url>)
      4.  Replace @usernames              (→ <user>)
      5.  Lowercase
      6.  Expand contractions             (don't → do not,  wasn't → was not)
      7.  Remove punctuation              (emojis and <placeholders> preserved)
      8.  Expand slang                    (ngl → not going to lie)
      9.  Apply negation joining          (not good → not_good)  ← BoW only
     10.  Collapse whitespace

    Steps 8 and 9 are intentionally ordered so that negations hidden inside
    slang abbreviations (e.g. "ngl", "idk") are visible to the negation handler.

    Parameters
    ----------
    text            : Input string. None is treated as an empty string.
    lowercase       : Convert to lowercase (default True).
    replace_urls    : Substitute URLs with the token <url> (default True).
    replace_users   : Substitute @mentions with <user> (default True).
    keep_punct      : Skip punctuation removal (default False).
    expand_hashtags : Split CamelCase hashtags into words (default True).
    expand_slang    : Replace slang abbreviations with full phrases (default True).
    handle_negation : Join negation words with the next token (default False).
                      Leave disabled by default because this heuristic can create
                      unnatural tokens such as 'not_you' or 'not_going'.
                      Enable only for Bag-of-Words experiments if it improves results.
    """
    if text is None:
        return ""

    # 1. HTML entities
    text = html.unescape(text)

    # 2. Hashtag expansion — before lowercasing to preserve CamelCase boundaries
    if expand_hashtags:
        text = HASHTAG_RE.sub(_expand_hashtag, text)

    # 3 & 4. URL / username replacement
    if replace_urls:
        text = URL_RE.sub("<url>", text)
    if replace_users:
        text = USER_RE.sub("<user>", text)

    # 5. Lowercase
    if lowercase:
        text = text.lower()

    # 6. Contraction expansion — done before punctuation removal so the
    #    apostrophe does not split tokens into meaningless fragments.
    #    Produces full in-vocabulary forms ("do not", "was not") that
    #    pre-trained embeddings already have vectors for.
    text = _CONTR_PATTERN.sub(
        lambda m: CONTRACTIONS.get(m.group().lower(), m.group()), text
    )

    # 7. Punctuation removal
    if not keep_punct:
        text = PUNCT_RE.sub(" ", text)

    tokens = text.split()

    # 8. Slang expansion — BEFORE negation so that slang phrases containing
    #    negation words (ngl → "not going to lie", idk → "i do not know")
    #    are exposed to the negation handler in step 9.
    if expand_slang:
        tokens = _expand_slang(tokens)

    # 9. Negation joining — AFTER slang expansion (see step 8 rationale).
    #    Disable for word-embedding pipelines to avoid OOV tokens.
    if handle_negation:
        tokens = _apply_negation(tokens)

    # 10. Reassemble and normalise whitespace
    return MULTI_WS.sub(" ", " ".join(tokens)).strip()


def tokenize_simple(text: str) -> List[str]:
    """
    Whitespace tokenizer with residual punctuation stripping.

    Strips leading/trailing punctuation from each token so that
    'word.' and 'word' are treated identically when keep_punct=True,
    while preserving negation-joined tokens like 'not_good' and
    placeholder tags like '<url>' / '<user>'.
    """
    if not text:
        return []
    tokens = text.split()
    cleaned = [re.sub(r"^[^\w<]+|[^\w>]+$", "", t) for t in tokens]
    return [t for t in cleaned if t]


def batch_clean(texts: List[Optional[str]], **kwargs) -> List[str]:
    """Apply clean_text to a list of texts, forwarding all keyword arguments."""
    return [clean_text(text, **kwargs) for text in texts]



# Quick smoke-test

if __name__ == "__main__":
    samples = [
        ("LOL this is NOT good at all 😡",                          False),
        ("omg @JohnDoe check out https://example.com #ThisIsTerrible", False),
        ("tbh it wasn't that bad tho 🤷",                           False),
        ("ngl the movie was lit af 🔥🔥",                           False),
        ("smh nobody ever listens, idk why I bother",               False),
        ("don't you think it's amazing?",                           False),
        ("LOL this is NOT good at all 😡",                          True),
        ("not good",                                                True),
        ("not bad",                                                 True),
        (None,                                                       False),
    ]

    print(f"{'Neg?':<5}  {'Original':<55} → Cleaned")
    print("-" * 105)
    for text, use_negation in samples:
        mode = "On" if use_negation else "Off"
        cleaned = clean_text(text, handle_negation=use_negation)
        print(f"[{mode:<5}]  {str(text):<55} → {cleaned}")