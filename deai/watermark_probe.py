"""KGW-MinHash style watermark probe for DE-AI.

Watermarks (KGW, Kirchenbauer et al. 2023, and its MinHash variant) split
the vocabulary into green/red lists via a PRNG seeded from the preceding
token, then bias generation toward green. Detection is a z-test on the
green-token count:

    z = (green_count - gamma * n) / sqrt(n * gamma * (1 - gamma))

This probe uses the MinHash membership test (same authors, O(1) per token,
no materialized vocab lists): a token is "green" under a key when

    sha256(key | prev_token | token) mod 2^32 < gamma * 2^32

The real key is unknown, so the probe runs the test over a handful of fixed
probe keys and reports the maximum z: a green-biased text lights up under
the key it was generated with; unwatermarked text stays near 0 under every
key. Pure offline: BPE tokenizer + hashlib, no numpy, no model weights,
deterministic. If the tokenizer cannot be loaded (``tokenizers`` /
``transformers`` missing or offline on first run), ``probe()`` returns
``None`` and callers degrade gracefully.

Honest limits: membership here is the MinHash variant, not the original
choice-based green list, and a watermark applied by a different
model/tokenizer (e.g. Gemini's SynthID) will not line up exactly. An unknown
key can only be probed, never confirmed. It is a signal estimator, not an
oracle.
"""

from __future__ import annotations

import hashlib

_GAMMA = 0.5
_THRESHOLD = 4.0
_MAX_TOKENS = 2000

# Fixed probe keys: uniform randomness, chosen once. A watermarked text
# scores high under at least one of these; unwatermarked text does not.
_DEFAULT_KEYS = (
    "deai-probe-0",
    "deai-probe-1",
    "deai-probe-2",
    "deai-probe-3",
    "deai-probe-4",
)


def _load_tokenizer():
    """Load a GPT-2 BPE tokenizer, preferring the lightest available backend.

    Order: tiktoken (installed, fast, no torch) > tokenizers > transformers.
    All three fetch the GPT-2 vocab on first use and cache it locally, so a
    second call is offline. If none is importable, returns None and
    ``probe()`` degrades gracefully.
    """
    try:
        import tiktoken

        return _TikTokenAdapter(tiktoken.get_encoding("gpt2"))
    except Exception:
        pass
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained("gpt2")
        if tok.get_vocab_size() == 50257:
            return tok
    except Exception:
        pass
    try:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        if len(tok) == 50257:
            return tok
    except Exception:
        pass
    return None


class _TikTokenAdapter:
    """Minimal uniform interface over tiktoken's Encoding for this probe.

    Mirrors the ``tokenizers`` interface used by the rest of ``probe()``:
    ``encode() -> Encoding``-like (with ``.ids``), ``get_vocab() -> {token: id}``.
    """

    def __init__(self, enc):
        self._enc = enc
        self._vocab: dict[str, int] | None = None

    def encode(self, text: str) -> _Ids:
        return _Ids(self._enc.encode(text))

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def get_vocab(self) -> dict[str, int]:
        if self._vocab is None:
            # surrogateescape (not "replace"): byte-tokens are arbitrary
            # bytes; "replace" collapses many ids onto U+FFFD and breaks the
            # id->token inversion. These strings are only hash inputs for the
            # green-list test, so any injective encoding is fine.
            self._vocab = {
                self._enc.decode_single_token_bytes(i).decode(
                    "utf-8", errors="surrogateescape"
                ): i
                for i in range(self._enc.n_vocab)
            }
        return self._vocab

    def get_vocab_size(self) -> int:
        return self._enc.n_vocab


class _Ids:
    """Minimal stand-in for the ``ids`` attribute of a tokenizers Encoding."""

    __slots__ = ("ids",)

    def __init__(self, ids: list[int]):
        self.ids = ids


_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _load_tokenizer()
    return _tokenizer


def _is_green(key: str, prev_token: str, token_id: int) -> bool:
    h = hashlib.sha256(
        f"{key}|{prev_token}|{token_id}".encode("utf-8", errors="surrogateescape")
    ).digest()
    # first 4 bytes as uint32, little-endian
    value = int.from_bytes(h[:4], "little")
    return value < _GAMMA * (1 << 32)


def probe(text: str, keys: tuple[str, ...] = _DEFAULT_KEYS) -> dict | None:
    """Return watermark stats for *text*, or None if unavailable/short.

    Fields: z (max over keys), z_per_key, n_tokens, verdict, threshold,
    scheme. Verdict "clean" means z <= threshold: no detectable green-list
    bias under any probe key.
    """
    tok = _get_tokenizer()
    if tok is None:
        return None

    try:
        if hasattr(tok, "encode"):
            enc = tok.encode(text)
            ids = enc.ids
        else:
            ids = tok(text, add_special_tokens=False)["input_ids"]
    except Exception:
        return None
    if len(ids) < 51:
        return None

    ids = ids[:_MAX_TOKENS + 1]

    try:
        id_to_tok = {v: k for k, v in tok.get_vocab().items()}
        tok_strings = [id_to_tok.get(i, f"<{i}>") for i in ids]
    except Exception:
        return None

    n = len(ids) - 1
    z_per_key: list[float] = []
    denom = (n * _GAMMA * (1 - _GAMMA)) ** 0.5
    for key in keys:
        green = 0
        for i in range(1, len(ids)):
            if _is_green(key, tok_strings[i - 1], ids[i]):
                green += 1
        z_per_key.append((green - _GAMMA * n) / denom if denom else 0.0)

    z = max(z_per_key)
    return {
        "z": round(z, 2),
        "z_per_key": [round(v, 2) for v in z_per_key],
        "n_tokens": n,
        "threshold": _THRESHOLD,
        "verdict": "watermarked" if z > _THRESHOLD else "clean",
        "scheme": "KGW-MinHash n=1 (multi-key probe)",
    }