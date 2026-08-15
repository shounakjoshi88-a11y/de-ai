"""Deep rewrite engine for DE-AI.

Four deterministic, dependency-light passes on top of the rule bank:

1. Rule bank   - the existing construction/tic-word rewrites (via paraphrase).
2. Lexical     - WordNet synonym substitution, frequency-gated by wordfreq
                 (only common synonyms, never much rarer than the original),
                 with surface-form inflection preserved via lemminflect.
3. Clausal     - clause-order moves: because/although fronting, "It is X that
                 P" extraposition flips, "the N, which was ADJ," relative
                 reductions, time/place-PP fronting. Every move is gated so
                 the open-class token set never changes (meaning survives)
                 and no transformed shape can re-match its own pattern.
4. Syntactic   - guarded structure moves: negation-antonym flips, adverb
                 fronting, long compound-clause splits.
5. Burstiness  - rhythm shaping: paragraphs whose sentence lengths are too
                 uniform get their longest sentence split at a safe clause
                 boundary (detectors key on sentence-length variance).
6. Verification - re-scans the output and reverts any swap that introduced a
                 tell the original did not have.

Every pass is deterministic (highest-frequency first sense, no randomness),
so results are reproducible. Missing optional deps degrade gracefully:
without `wn`/`wordfreq`/`lemminflect` the deep passes skip and only the rule
bank runs.
"""

from __future__ import annotations

import re

try:
    import wn as _wn
    _wn.config.lexicon = "oewn:2024"  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _wn = None

try:
    from wordfreq import zipf_frequency as _zipf
except Exception:  # pragma: no cover
    _zipf = None

try:
    import lemminflect as _lf
except Exception:  # pragma: no cover
    _lf = None

from .paraphraser import COVERED, Change, paraphrase

_SKIP_WORDS = frozenset(
    """a an the this that these those my your his her its our their i you he she it we
    they me him us them am is are was were be been being do does did have has had
    will would shall should can could may might must of in on at to for with by
    from up down over under about into through between out off as so than then
    but or and nor not no yes if when while because although though unless
    there here what which who whom whose how why very just too also even only
    still yet again more most less least all any both each either neither some
    any such own same other another like as if one two three four five six seven eight nine
    ten first second third well now oh okay ok yeah um uh""".split()
)

_MIN_WORD_LEN = 4
_MAX_SWAPS_PER_SENTENCE = 10
_MIN_ZIPF = 2.7
_ZIPF_TOLERANCE = 2.0
_ING_ED_TOLERANCE = 1.2  # tighter gate for participles: drift is costlier
_ADJ_TOLERANCE = 1.4  # adjective swaps: mild drift is acceptable
_TIE_BAND = 0.4  # near-tie synonyms get varied deterministically (diversity)

# ultra-polysemous words whose first-sense synonymy drifts: never auto-swap
_AMBIGUOUS = frozenset(
    """last kind news sign change fail look face make take get set run hold
    point case way work play stand form word time matter order line hand head
    mind course part place end right ground school field beat back light
    close interest charge concern issue state position power space system
    level rate note value view side kind type sort put come go turn pass
    break draw drive leave move open raise reach return serve show support
    clear direct natural common fine great good hard high long low major
    ready small strong young serious real simply truly honestly message main
    adult life speed stride thing stuff guy guys dude dudes friend friends
    result child peak adrenaline class picture straight single quiet song
    handle grip trade grow contain find grand crisp remain wait golden
    jagged mountain autumn terrace courtyard cosmos identical revenue
    metabolism quality management team report decision determination
    execution let entire model document adjust predict""".split()
)

# technical / domain vocabulary that must NEVER be auto-swapped by the lexical
# pass. These are the "elf-attention / unreal intelligence / stallion context
# window" class of errors: first-sense synonymy drifts a technical term into a
# wrong-sense word (server -> waiter, endpoint -> terminus, corpus -> principal,
# convolution -> swirl). Multiword technical phrases get fixed by _AMBIGUOUS
# parts (token/model/... ) plus these single-word terms.
_TECH_TERMS = frozenset(
    """token tokens tokenized tokenization embedding embeddings embed embedder
    vector vectors vectorized matrix matrices tensor tensors scalar scalars
    dimension dimensions dimensional latent latent-space representation
    representations feature features label labels annotation annotated
    dataset datasets datapoint datapoints sample samples sampling sampler
    batch batches epoch epochs iteration iterations step steps
    hyperparameter hyperparameters parameter parameters weight weights bias
    biases activation activations gradient gradients descent
    backpropagation backprop feedforward backprop optimizer optimize
    optimization optimized learning-rate learning loss losses
    objective objectives metric metrics benchmark benchmarks
    baseline baselines state-of-the-art SOTA accuracy precision recall f1
    cross-validation cross-entropy entropy kl-divergence
    softmax sigmoid relu tanh logits layer layers hidden-layer
    convolution convolutional pooling pool stride padding
    recurrent lstm gru transformer transformers attention
    self-attention attention-head head heads multi-head
    encoder decoders decoder encoder-decoder autoencoder
    sequence sequences sequential context context-window
    prompt prompts prompting prompt-engineering temperature top-k top-p
    nucleus sampling beam-search greedy decoding decode decoding
    tokenizer tokenizers vocab vocabulary corpus corpora
    fine-tuning fine-tune pretrain pretrained finetune training trained
    generative generation generated generate synthetic
    inference inferring inferencing quantization quantize quantized distilled sparse sparsity pruning pruned
    pruning distillation pruning distillation
    generalization generalize overfitting overfit underfitting underfit
    regularization regularize dropout normalization normalize
    model models architecture architectures parameter-count
    neuron neurons network networks neural deep-learning
    machine-learning ml ai artificial-intelligence nlp
    computer-vision cv speech-recognition asr text-to-speech tts
    retrieval rag agent agents multi-agent tool-use function-calling
    hallucination hallucinate reasoning chain-of-thought cot
    system systems algorithm algorithms computation computational
    data database databases dataset tables table rows columns schema schemas
    sql nosql query queries queried index indexes indexed key keys
    primary-key foreign-key join joins transaction transactions
    cache caches cached caching cache-miss buffer buffers buffered
    pointer pointers array arrays stack stacks heap heaps queue queues
    linked-list hash-table tree trees binary-tree graph graphs node nodes
    edge edges vertex vertices recursion recursive iterate iteration
    compile compiled compiler runtime syntax semantic semantics
    variable variables function functions method methods class classes
    object objects instance instances interface interfaces api apis
    endpoint endpoints request requests response responses
    json xml yaml http https tcp udp ip port ports socket sockets
    protocol protocols packet packets latency bandwidth throughput
    server servers client clients backend frontend middleware
    load-balancer load-balancing horizontal-scaling vertical-scaling
    shard sharding partition partitioning replica replicas replication
    deployment deploy deployed deployable container containers docker
    kubernetes k8s pod pods microservice microservices service-mesh
    orchestration orchestrator registry image images ci cd pipeline pipelines
    test tests testing testing-unit unit-test integration-test e2e
    mock mocks mocking assertion assertions coverage coverage-test
    debug debugging debugger breakpoint breakpoints log logs logging
    exception exceptions error errors stack-trace traceback
    thread threads process processes concurrency concurrent
    parallel parallelism asynchronous async synchronous sync
    deadlock deadlocks race-condition race-conditions semaphore mutex
    memory heap-memory stack-memory garbage-collection gc leak leaks
    storage filesystem fs file files directory directories path paths
    encryption encrypt decrypt decryption cipher ciphertext plaintext
    hash hashes hashing salted salt rainbow-table
    authentication authenticate authenticator authorization authorize
    credential credentials password passwords passphrase passphrases
    token-based session sessions cookie cookies sso oauth jwt
    key-rotation certificate certificates cert certs public-key
    private-key asymmetric symmetric rsa aes des 3des
    digital-signature signature signatures signing verified verify
    firewall firewalls ids ips waf intrusion-detection
    malware virus viruses worm worms trojan ransomware spyware adware
    rootkit bootkit keylogger backdoor botnet
    vulnerability vulnerabilities exploit exploits exploited exploitation
    exploit-kit cve zero-day zeroday threat threats threat-actor
    attacker attackers adversary adversaries compromise compromised
    penetration pentest pentesting red-team blue-team
    phishing spear-phishing vishing smishing pretexting baiting
    social-engineering mitm replay-attack brute-force dictionary-attack
    sql-injection xss csrf injection deserialization
    sandbox sandboxing isolation quarantined quarantine
    compliance regulatory gdpr hipaa pci-dss sox
    audit auditing auditor forensic forensics artifact artifacts
    incident incidents ir incidence response breach breached
    malware-analysis reverse-engineering obfuscate obfuscated obfuscation
    packing packed unpacking unpacked disassembly disassembler
    dump dumps core-dump memory-dump volatility
    proxy proxies vpn tunnel tunneling tor relay relays onion
    certificate-authority ca csr tls ssl dtls handshake
    packet-capture pcap sniffing sniffer port-scan scanning
    enumeration enumerate fingerprinting osint ioc indicators""".split()
)

_STILTED_TARGETS = frozenset(
    """halt vale ere unto nigh whence perchance behold hither thence
    whereat wherefore""".split()
)

# verbs that require a direct object; unsafe as targets when the source
# verb is used intransitively (said -> *told)
_NEEDS_OBJECT = frozenset(
    "tell give hand offer send pass lend rent buy sell bring throw award"
    " grant owe promise teach feed show serve assign allocate".split()
)
# noun-sense swaps that are traps ("defeat" -> "licking"); verb sense is fine
_NOUN_ONLY_TRAPS = frozenset(
    "defeat mutter sweep burst stroke".split()
)
# specific source -> target blacklist for wrong-sense synonyms
# (sources that are fully blacklisted live in _AMBIGUOUS instead)
_TARGET_TRAPS = {
    "moment": frozenset("minute second instant".split()),
    "frustration": frozenset("defeat".split()),
    "floor": frozenset("flooring".split()),
    "edge": frozenset("borderline".split()),
    "border": frozenset("borderline".split()),
    "mundane": frozenset("everyday".split()),
    "routine": frozenset("everyday".split()),
    "artificial": frozenset("unreal synthetic contrived fabricated".split()),
}

_LEMMA_CACHE: dict[str, list[str]] = {}
_FIRST_POS_CACHE: dict[str, str] = {}
_LEXFILE_CACHE: dict[str, str | None] = {}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")
_SENT_END = re.compile(r"[.!?]\s+")
_ORDINAL_BEFORE = re.compile(r"\d+(?:th|st|nd|rd)\s+$")
_DETERMINERS = frozenset(
    "a an the this that these those my your his her its our their some any no "
    "every each another other both few several many much".split()
)
_PRONOUNS = frozenset(
    "i you he she it we they who what which that one everyone nobody".split()
)
# negated auxiliaries force the following word to be a verb
# ("don't battle" must not swap battle as a noun -> conflict)
_NEG_AUX = frozenset(
    "don't doesn't didn't won't wouldn't shouldn't couldn't can't cannot "
    "mustn't needn't ain't isn't aren't wasn't weren't haven't hasn't hadn't".split()
)
# intensifiers/degree words: "not just", "not really" are NOT adjective flips
_INTENSIFIERS = frozenset(
    "just only exactly simply merely really quite very so too that as even "
    "all quite actually definitely certainly".split()
)


def _prev_token(seg: str, start: int) -> str | None:
    before = seg[:start]
    m = list(_TOKEN_RE.finditer(before))
    return m[-1].group(0).lower() if m else None

# ------------------------------------------------------------- lexical pass


def _synonyms(word: str, pos: str) -> list[str]:
    """Return lemma synonyms from the FIRST sense of `word` for `pos`."""
    if _wn is None:
        return []
    key = word + "/" + pos
    if key in _LEMMA_CACHE:
        return _LEMMA_CACHE[key]
    out: list[str] = []
    try:
        objs = _wn.words(word, pos=pos)
    except Exception:
        objs = []
    if objs:
        for sense in objs[0].senses()[:1]:
            for lemma in sense.synset().lemmas():
                low = lemma.lower()
                if low == word or " " in low or "_" in low:
                    continue
                out.append(low)
    _LEMMA_CACHE[key] = out
    return out


def _first_pos(word: str) -> str | None:
    """First sense's POS from WordNet ('v'/'n'/'a'...), or None."""
    if _wn is None:
        return None
    if word in _FIRST_POS_CACHE:
        return _FIRST_POS_CACHE[word]
    out: str | None = None
    try:
        objs = _wn.words(word)
        if objs and objs[0].senses():
            out = objs[0].senses()[0].synset().pos
    except Exception:
        out = None
    _FIRST_POS_CACHE[word] = out or ""
    return out


def _first_lexfile(word: str, pos: str) -> str | None:
    """WordNet lexfile domain of `word`'s first sense for `pos`.

    E.g. 'spine' -> noun.body, 'backbone' (first sense!) -> noun.cognition.
    Used to reject synonyms whose own dominant sense lives in a different
    domain than the source (spine -> backbone -> keystone compounding).
    """
    if _wn is None:
        return None
    key = word + "/" + pos
    if key in _LEXFILE_CACHE:
        return _LEXFILE_CACHE[key]
    out: str | None = None
    try:
        objs = _wn.words(word, pos=pos)
        if objs and objs[0].senses():
            lf = objs[0].senses()[0].synset().lexfile
            if callable(lf):
                lf = lf()
            out = str(lf)
    except Exception:
        out = None
    _LEXFILE_CACHE[key] = out
    return out


def _base(word: str, upos: str) -> str:
    if _lf is None:
        return word
    try:
        lemmas = _lf.getLemma(word, upos)  # type: ignore[call-overload]
        return lemmas[0] if lemmas else word
    except Exception:
        return word


def _best_swap(word: str, pos: str, used: set[str], tol: float | None = None,
               salt: str = "") -> str | None:
    """Closest-frequency synonym within tolerance of the original.

    Near-ties (dist within _TIE_BAND of the closest) are varied deterministically
    by a hash of (word, salt) so the same text location gets the same synonym,
    but adjacent sentences don't all resolve to the same word.
    """
    if _zipf is None:
        return None
    limit = _ZIPF_TOLERANCE if tol is None else min(_ZIPF_TOLERANCE, tol)
    base_zipf = _zipf(word, "en")
    src_lf = _first_lexfile(word, pos)
    ranked: list[tuple[float, str]] = []
    for syn in _synonyms(word, pos):
        if syn in used or syn in _SKIP_WORDS or len(syn) < _MIN_WORD_LEN:
            continue
        # compounding guard: the synonym's OWN dominant sense must live in the
        # same WordNet domain as the source's (spine=noun.body vs
        # backbone's first sense = noun.cognition => reject, else a re-run
        # drifts spine -> backbone -> keystone). Unknown lexfiles are
        # permissive (adjectives/satellite senses often resolve oddly).
        syn_lf = _first_lexfile(syn, pos)
        if src_lf and syn_lf and syn_lf != src_lf:
            continue
        if word in _TECH_TERMS or syn in _TECH_TERMS:
            continue
        if word in _AMBIGUOUS or _base(word, "NOUN") in _AMBIGUOUS:
            continue
        if _base(word, "VERB") in _AMBIGUOUS:
            continue
        if pos == "n" and word in _NOUN_ONLY_TRAPS:
            continue
        if pos == "v" and syn in _NEEDS_OBJECT:
            continue
        if word in _TARGET_TRAPS and syn in _TARGET_TRAPS[word]:
            continue
        if syn in _STILTED_TARGETS:
            continue
        upos = "VERB" if pos == "v" else "NOUN"
        if _base(syn, upos) == _base(word, upos):
            continue
        if pos == "a" and _base(syn, "VERB") != syn:
            continue
        z = _zipf(syn, "en")
        if z < _MIN_ZIPF:
            continue
        dist = abs(base_zipf - z)
        if dist > limit:
            continue
        ranked.append((dist, syn))
    if not ranked:
        return None
    ranked.sort(key=lambda r: (r[0], r[1]))
    if len(ranked) >= 2 and ranked[1][0] - ranked[0][0] < _TIE_BAND:
        h = sum(ord(c) for c in word + salt)
        return ranked[h % 2][1]
    return ranked[0][1]


def _inflect_like(synonym: str, target: str, pos: str, prev_low: str | None = None) -> str:
    """Inflect `synonym` to match the surface form of `target` (ran -> sprinted)."""
    if _lf is None:
        return synonym
    upos = {"v": "VERB", "n": "NOUN"}.get(pos)
    if upos is None:
        return synonym
    try:
        lemmas = _lf.getLemma(target, upos)  # type: ignore[call-overload]
        base = lemmas[0] if lemmas else ""
        if not base or base == target:
            return synonym
        forms = _lf.getAllInflections(base, upos)  # type: ignore[call-overload]
        if prev_low in ("has", "have", "had"):
            got = _lf.getInflection(synonym, "VBN")
            if got and got[0] and got[0] != synonym:
                return got[0]
        tag = next((t for t, f in forms.items() if target in f), None)
        if tag is None:
            return synonym
        got = _lf.getInflection(synonym, tag)
        if got and got[0]:
            return got[0]
    except Exception:
        pass
    return synonym


def _lexical_pass(text: str, flagged: set[str], swaps: list[tuple[int, int, str, str]],
                  used: set[str] | None = None) -> str:
    if _wn is None or _zipf is None:
        return text
    work = text
    delta = 0
    if used is None:
        used = set()
    prev_end = 0
    for end in _SENT_END.finditer(text):
        seg_start = prev_end + delta
        seg_end = end.end() + delta
        seg = work[seg_start:seg_end]
        if len(seg) < 2:
            prev_end = end.end()
            continue
        toks = list(_TOKEN_RE.finditer(seg))
        budget = min(_MAX_SWAPS_PER_SENTENCE, max(0, len(toks) // 4))
        verb_counts: dict[str, int] = {}
        for t in toks:
            w = t.group(0).lower()
            if w.isalpha() and len(w) >= _MIN_WORD_LEN and w not in _SKIP_WORDS:
                vb = _base(w, "VERB")
                verb_counts[vb] = verb_counts.get(vb, 0) + 1
        intra = 0
        for tok in toks:
            if budget <= 0:
                break
            word = tok.group(0)
            low = word.lower()
            if (len(word) < _MIN_WORD_LEN or low in _SKIP_WORDS or low in flagged
                    or low in used or low in _TECH_TERMS or not word.isalpha()):
                continue
            if word[0].isupper() and tok.start() > 0:
                continue
            if _ORDINAL_BEFORE.search(seg[: tok.start()]):
                continue
            prev_tok = _prev_token(seg, tok.start())
            prev_low = prev_tok.lower() if prev_tok else None
            noun_base = _base(low, "NOUN")
            verb_base = _base(low, "VERB")
            if verb_counts.get(verb_base, 0) > 1:
                continue  # deliberate repetition device (sweeping to sweep)
            verb_common = verb_base != low and (
                _zipf is None or _zipf(verb_base, "en") >= _zipf(noun_base, "en")
            )
            if low.endswith(("ing", "ed")):
                if prev_low in _PRONOUNS or prev_low == "to" or (
                    prev_low not in _DETERMINERS and verb_common
                ):
                    pos_order = ("v",)
                else:
                    continue
            elif prev_low in _DETERMINERS:
                pos_order = ("n",)
            elif prev_low in _PRONOUNS or prev_low in _NEG_AUX or prev_low == "to":
                pos_order = ("v",)
            else:
                fp = _first_pos(low)
                if fp in ("v", "n"):
                    pos_order = (fp,)
                elif fp in ("a", "s"):
                    pos_order = ("a",)
                elif verb_common:
                    pos_order = ("v",)
                else:
                    continue
            replacement = None
            chosen_pos = None
            for pos in pos_order:
                if low.endswith(("ing", "ed")):
                    tol = _ING_ED_TOLERANCE
                elif pos == "a":
                    tol = _ADJ_TOLERANCE
                else:
                    tol = None
                cand = _best_swap(low, pos, used, tol=tol, salt=str(tok.start()))
                if cand:
                    replacement, chosen_pos = cand, pos
                    break
            if not replacement or replacement == low:
                continue
            replacement = _inflect_like(replacement, low, chosen_pos or "v", prev_low)
            if replacement == word:
                continue
            if _zipf is not None and _zipf(replacement, "en") < 2.0:
                continue  # inflected form is not a real word (hugging -> bosoming)
            start = seg_start + tok.start() + intra
            span_end = start + len(word)
            if work[start - 1 : start] == '"':
                continue
            work = work[:start] + replacement + work[span_end:]
            intra += len(replacement) - len(word)
            used.add(low)
            used.add(replacement)
            swaps.append((start, span_end, word, replacement))
            budget -= 1
        delta += intra
        prev_end = end.end()
    return work


# ------------------------------------------------------------ syntactic pass

_NEGATION = re.compile(
    r"\b(is|was|are|were|am|be)\s+(not|n'?t)\s+([a-z][a-z]{2,})",
    re.IGNORECASE,
)
_NEGATION_CC = re.compile(
    r"\b(isn'?t|wasn'?t|aren'?t|weren'?t|ain'?t)\s+([a-z][a-z]{2,})",
    re.IGNORECASE,
)
_ADVERB_END = re.compile(
    r"^([A-Z][^.!?\"]{4,}?)\s+(quickly|slowly|silently|suddenly|finally|"
    r"eventually|immediately|cautiously|gently|calmly|softly|eagerly|"
    r"reluctantly|anxiously|patiently|impatiently|gracefully|awkwardly)\.$",
    re.DOTALL,
)
_COMPOUND_SPLIT = re.compile(
    r"^(.{15,}?[A-Za-z]), (and|but) "
    r"((?:he|she|it|they|we|you|i|there|anyone|everyone|somebody|someone|"
    r"people|[A-Z][a-z]+)\b.{12,}?[A-Za-z])\.$",
    re.DOTALL,
)


def _negation_flip(sentence: str) -> tuple[str, bool]:
    m = _NEGATION.search(sentence) or _NEGATION_CC.search(sentence)
    if not m or _wn is None or '"' in sentence:
        return sentence, False
    if m.re is _NEGATION:
        be, adj = m.group(1), m.group(3).lower()
    else:
        be, adj = m.group(1), m.group(2).lower()
    if adj in _INTENSIFIERS:
        return sentence, False
    try:
        objs = _wn.words(adj, pos="a")
    except Exception:
        return sentence, False
    if not objs:
        return sentence, False
    antonyms: list[str] = []
    for sense in objs[0].senses()[:2]:
        for rel in sense.get_related("antonym"):
            lemma = rel.word().lemma()
            if len(lemma) >= 3:
                antonyms.append(lemma)
    if not antonyms:
        return sentence, False
    if _zipf is not None:
        antonyms = [a for a in antonyms if _zipf(a, "en") >= _MIN_ZIPF - 1.0]
    if not antonyms:
        return sentence, False
    antonym = sorted(antonyms, key=lambda a: -(_zipf(a, "en") if _zipf else 0))[0]
    if m.re is _NEGATION:
        return sentence[: m.start(1)] + be + " " + antonym + sentence[m.end(3) :], True
    return sentence[: m.start(1)] + be + " " + antonym + sentence[m.end(2) :], True


def _front_adverb(sentence: str) -> tuple[str, bool]:
    m = _ADVERB_END.match(sentence)
    if not m:
        return sentence, False
    body, adv = m.group(1), m.group(2)
    if "," in body or "?" in body:
        return sentence, False
    return f"{adv.capitalize()}, {body[0].lower()}{body[1:]}." , True


def _split_compound(sentence: str) -> tuple[str, bool]:
    m = _COMPOUND_SPLIT.match(sentence)
    if not m:
        return sentence, False
    a, b = m.group(1), m.group(3)
    if "," in a or ";" in a or "?" in a:
        return sentence, False
    return f"{a}. {b[0].upper()}{b[1:]}." , True


# ------------------------------------------------------ clausal structure moves

# Detectors also key on clause order: LLMs over-produce "A because B.",
# "It is X that P.", "the N, which was ADJ," templates. These moves
# reorder/restructure clauses WITHOUT touching any content word (the
# survival gate compares open-class tokens before/after), and every
# transformed shape is a shape the pattern cannot re-match, so re-runs
# are stable. Dialogue (") is never touched.
_CLAUSAL_WHY = re.compile(
    r"^([^,]{8,}?[A-Za-z]) (because|although|though|even though) "
    r"([^,]{6,}?[A-Za-z])\.$",
    re.DOTALL,
)
_IT_FLIP = re.compile(
    r"^It (is|was) ([A-Za-z][a-z'-]+(?: [A-Za-z][a-z'-]+){0,2}) "
    r"(that|who) ([^,]{6,}?[A-Za-z])\.$",
    re.DOTALL,
)
_REL_REDUCE = re.compile(
    r"\b([Aa]|[Aa]n|[Tt]he) ((?:[A-Za-z]{3,} ){0,1}[A-Za-z]{3,}), (which|who) "
    r"(was|is|were|are) ([A-Za-z]{3,}),",
)
_PP_FRONT = re.compile(
    r"^([^,]{12,}?[A-Za-z]) (in|on|at|by|before|after|during) the "
    r"(morning|afternoon|evening|night|dawn|dusk|day|week|month|year|"
    r"spring|summer|autumn|winter|end|beginning|start|top|bottom|left|"
    r"right|front|back)\.$",
    re.DOTALL,
)
# words that are safe to lowercase as a re-ordered clause's head
_COMMON_STARTERS = (
    _PRONOUNS | _DETERMINERS
    | frozenset(
        "there here it this that am is are was were has have had will would "
        "can could should must may might do does did people everyone nobody "
        "someone anyone".split()
    )
)


def _content_sig(sentence: str) -> frozenset[str]:
    """Open-class token set (drops function words); used as a survival gate."""
    out = set()
    for t in re.findall(r"[A-Za-z][a-z'-]{3,}", sentence.lower()):
        if t in _SKIP_WORDS:
            continue
        out.add(t)
    return frozenset(out)


def _uncap_if_safe(sentence: str, names: frozenset[str] | None = None) -> str:
    """Lowercase a moved clause's head unless it looks like a proper noun.

    A word is protected (kept capitalized) if it appears capitalized
    mid-sentence anywhere in the document (names like Tenzin repeat in
    prose), or if its lowercase form is too rare to be a common word.
    """
    first = sentence.split()[0]
    if first.lower() in _COMMON_STARTERS:
        return first.lower() + sentence[len(first):]
    if names and first in names:
        return sentence
    lf = first.lower()
    if _zipf is not None and _zipf(lf, "en") >= 3.5:
        return lf + sentence[len(first):]
    return sentence


def _clausal_why(sentence: str, names: frozenset[str]) -> tuple[str, bool]:
    m = _CLAUSAL_WHY.match(sentence)
    if not m or '"' in sentence or "?" in sentence:
        return sentence, False
    a, conn, b = m.group(1), m.group(2), m.group(3)
    if len(a.split()) < 2 or len(b.split()) < 3:
        return sentence, False
    new = f"{conn.capitalize()} {b}, {_uncap_if_safe(a, names)}."
    if _content_sig(sentence) != _content_sig(new):
        return sentence, False
    return new, True


def _it_flip(sentence: str, names: frozenset[str]) -> tuple[str, bool]:
    m = _IT_FLIP.match(sentence)
    if not m or '"' in sentence or "?" in sentence:
        return sentence, False
    be, x, conn, p = m.group(1), m.group(2), m.group(3), m.group(4)
    words = x.split()
    head_pos = _first_pos(words[0].lower())
    is_adj = len(words) == 1 and head_pos in ("a", "s")
    is_noun = words[0][0].isupper() or head_pos == "n"
    if not (is_adj or is_noun):
        return sentence, False
    if len(p.split()) < 3:
        return sentence, False
    if is_adj:
        new = f"That {p} {be} {x}."
    else:
        new = f"{x[0].upper()}{x[1:]} {p}."
    if _content_sig(sentence) != _content_sig(new):
        return sentence, False
    return new, True


def _rel_reduce(sentence: str, names: frozenset[str]) -> tuple[str, bool]:
    m = _REL_REDUCE.search(sentence)
    if not m or '"' in sentence or "?" in sentence:
        return sentence, False
    art, noun, rel, be, adj = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    head = noun.split()[-1]
    if head.lower() in _SKIP_WORDS or adj in _SKIP_WORDS:
        return sentence, False
    if _first_pos(head.lower()) != "n" or _first_pos(adj) not in ("a", "s"):
        return sentence, False
    if rel == "who" and be not in ("was", "is"):
        return sentence, False
    art_out = art
    if art.lower() == "a" and adj[0] in "aeiou":
        art_out = "an"
    if art[0].isupper():
        art_out = art_out.capitalize()
    new = sentence[: m.start()] + f"{art_out} {adj} {noun}" + sentence[m.end():]
    if _content_sig(sentence) != _content_sig(new):
        return sentence, False
    return new, True


def _pp_front(sentence: str, names: frozenset[str]) -> tuple[str, bool]:
    m = _PP_FRONT.match(sentence)
    if not m or '"' in sentence or "?" in sentence:
        return sentence, False
    a, prep, time = m.group(1), m.group(2), m.group(3)
    if len(a.split()) < 5:
        return sentence, False
    new = f"{prep.capitalize()} the {time}, {_uncap_if_safe(a, names)}."
    if _content_sig(sentence) != _content_sig(new):
        return sentence, False
    return new, True


_CLAUSAL = (_clausal_why, _it_flip, _rel_reduce, _pp_front)


# ----------------------------------------------------------- rule-of-three pass

# Detectors also key on parallel "A, B, and C" triads: LLMs over-produce
# same-POS lists in their canonical order. Human editors reorder such lists
# freely (list order carries no meaning). This pass deterministically sorts
# triad items by (length desc, alpha) -- idempotent, so a re-run finds the
# canonical order and changes nothing. Order-sensitive sets (numbers, days,
# months, time-of-day, life stages) are never touched, and any capitalized
# item (names) blocks the move.
_TRIAD_AND = re.compile(
    r"\b([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?), "
    r"([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?), and "
    r"([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?)\b"
)
_TRIAD_AND_BARE = re.compile(
    r"\b([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?), "
    r"([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?) and "
    r"([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?)\b"
)
_TRIAD_CHAIN = re.compile(
    r"\b([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?) and "
    r"([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?) and "
    r"([A-Za-z][A-Za-z'-]*(?: [A-Za-z][A-Za-z'-]*)?)\b"
)
_SEQUENCE_WORDS = frozenset(
    """one two three four five six seven eight nine ten first second third fourth
    morning afternoon evening night dawn dusk noon midnight day week month year
    january february march april may june july august september october november
    december monday tuesday wednesday thursday friday saturday sunday spring
    summer autumn winter childhood youth adulthood birth life death beginning
    middle end start finish""".split()
)


def _triad_heads(items: list[str]) -> list[str]:
    """POS-resolvable head word per item (lemmatizes plurals/inflections)."""
    out: list[str] = []
    for w in items:
        head = w.split()[-1].lower()
        if _first_pos(head) in ("n", "a", "s", "v"):
            out.append(head)
        else:
            for upos in ("NOUN", "VERB"):
                base = _base(head, upos)
                if base != head and _first_pos(base) in ("n", "a", "s", "v"):
                    out.append(base)
                    break
            else:
                out.append(head)
    return out


def _triad_pos(head: str) -> str | None:
    """POS class for a triad head: 'a' (incl. satellite 's'), 'n', or 'v'.

    Words whose first sense is a noun but that also have a real adjective
    sense (e.g. 'wise', 'metal') count as adjectives when their triad peers
    are adjectives, so "brave, strong, and wise" is not blocked. Inflected
    forms that only exist as verbs (ran, climbed, chanted) count as verbs
    even when their lemma base is noun-first.
    """
    p = _first_pos(head)
    if p in ("a", "s"):
        return "a"
    if p in ("n", "v"):
        if p == "n":
            try:
                if _wn and _wn.words(head, pos="a"):
                    return "a"
            except Exception:
                pass
        return p
    if _base(head, "VERB") != head and _first_pos(_base(head, "VERB")) in ("v", "n"):
        return "v"
    return None


def _triad_sort(sentence: str) -> tuple[str, bool]:
    for pat, bare in ((_TRIAD_AND, False), (_TRIAD_AND_BARE, True), (_TRIAD_CHAIN, False)):
        pos = 0
        while True:
            m = pat.search(sentence, pos)
            if not m:
                break
            pos = m.start() + 1
            items = [m.group(1), m.group(2), m.group(3)]
            shapes = {len(w.split()) for w in items}
            if len(shapes) != 1:
                continue
            if len(items[0].split()) == 2 and any(
                w.split()[0].lower() not in _DETERMINERS for w in items
            ):
                continue
            heads = _triad_heads(items)
            if any(h in _SEQUENCE_WORDS for h in heads):
                continue
            if any(w[0].isupper() for w in items):
                continue
            pos_cls = _triad_pos(heads[0])
            if pos_cls not in ("n", "a", "v"):
                continue
            if any(_triad_pos(h) != pos_cls for h in heads):
                continue
            ordered = sorted(items, key=lambda w: (-len(w), w))
            if ordered == items:
                continue
            if pat is _TRIAD_CHAIN:
                repl = f"{ordered[0]} and {ordered[1]} and {ordered[2]}"
            elif bare:
                repl = f"{ordered[0]}, {ordered[1]} and {ordered[2]}"
            else:
                repl = f"{ordered[0]}, {ordered[1]}, and {ordered[2]}"
            new = sentence[: m.start()] + repl + sentence[m.end():]
            if m.start() == 0 and new[0].islower():
                new = new[0].upper() + new[1:]
            return new, True
    return sentence, False


def _triad_pass(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", line)
        changed: list[str] = []
        for sent in sentences:
            sent, _ = _triad_sort(sent)
            changed.append(sent)
        out.append(" ".join(changed))
    return "\n".join(out)


def _clausal_pass(text: str) -> str:
    # Document-level proper-noun evidence: words capitalized mid-sentence
    # anywhere in the text are treated as names and never lowercased by the
    # clause reorder moves.
    names: set[str] = set()
    for line in text.split("\n"):
        for sent in re.split(r"(?<=[.!?])\s+", line):
            words = sent.split()
            for w in words[1:]:
                if w[0].isupper() and w[0].isalpha() and not w.isupper():
                    names.add(w)
    name_set = frozenset(names)
    out: list[str] = []
    for line in text.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", line)
        changed: list[str] = []
        for sent in sentences:
            for fn in _CLAUSAL:
                sent, applied = fn(sent, name_set)
                if applied:
                    break
            changed.append(sent)
        out.append(" ".join(changed))
    return "\n".join(out)


_SYNTACTIC = (_negation_flip, _front_adverb, _split_compound)


def _syntactic_pass(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", line)
        changed: list[str] = []
        for sent in sentences:
            for fn in _SYNTACTIC:
                sent, applied = fn(sent)
                if applied:
                    break
            changed.append(sent)
        out.append(" ".join(changed))
    return "\n".join(out)


# ------------------------------------------------------------- burstiness pass

# Detectors (ZeroGPT/GPTZero/Originality) key on two published signals:
# per-token perplexity (predictability under a language model) and
# "burstiness" -- variance in sentence length/structure across a paragraph.
# AI prose is rhythmically uniform; human prose mixes short punchy sentences
# with long flowing ones. This pass injects that variance deterministically:
# a paragraph whose sentence lengths are too uniform gets its longest sentence
# split at the most balanced safe clause boundary. Only independent-clause
# conjunctions (and/but/so/yet/or/nor/because/though/although) are split
# points, so neither half is ever a fragment. Applied after the per-sentence
# syntactic moves, before register shift.
_RHYTHM_SPLIT = re.compile(
    r"^(.{10,}?[A-Za-z]), (and|but|so|yet|or|nor|because|though|although) "
    r"([A-Za-z].{8,}?[A-Za-z])\.$",
    re.DOTALL,
)


def _rhythm_split(sentence: str) -> tuple[str, bool]:
    m = _RHYTHM_SPLIT.match(sentence)
    if not m:
        return sentence, False
    if '"' in sentence:
        return sentence, False
    a, b = m.group(1), m.group(3)
    if len(a.split()) < 6 or len(b.split()) < 5:
        return sentence, False
    return f"{a}. {b[0].upper()}{b[1:]}." , True


def _burstiness_pass(text: str, profile: dict | None = None) -> str:
    """Inject human-style rhythm variance. Detectors key on coefficient of
    variation (std/mean) of sentence length: AI prose is CV-flat (~0.4-0.6)
    even when raw variance is high, because every sentence sits in the same
    10-30 word band. Human prose mixes short punches with long flows
    (CV ~0.85+). This pass splits a flat paragraph's longest sentence at the
    most balanced safe clause boundary -- and only commits the split if it
    actually raises the line's CV (a split into two mid-length halves would
    flatten rhythm further, which is why raw-variance gating was wrong).
    With a writer profile, the target CV is the writer's own, so output
    rhythm matches *that* human."""
    if profile is None:
        target_cv = 0.85
        min_longest = 18
    else:
        target_cv = max(0.6, (profile.get("burstiness", 0.0) or 0.0) * 0.85)
        target_mean = profile.get("avg_sentence_len", 0.0) or 0.0
        min_longest = max(18, int(target_mean * 0.6))
    out: list[str] = []
    for line in text.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", line)
        lens = [len(s.split()) for s in sentences]
        if len(lens) < 3:
            out.append(line)
            continue
        mean = sum(lens) / len(lens)
        variance = sum((l - mean) ** 2 for l in lens) / len(lens)
        cv = (variance ** 0.5) / mean if mean else 0.0
        if cv >= target_cv:
            out.append(line)
            continue
        longest = max(range(len(sentences)), key=lambda i: lens[i])
        if lens[longest] < min_longest:
            out.append(line)
            continue
        new, ok = _rhythm_split(sentences[longest])
        if not ok:
            out.append(line)
            continue
        split_lens = [len(p.split()) for p in new.split(". ") if p.strip()]
        new_lens = lens[:longest] + split_lens + lens[longest + 1:]
        new_mean = sum(new_lens) / len(new_lens)
        new_var = sum((l - new_mean) ** 2 for l in new_lens) / len(new_lens)
        new_cv = (new_var ** 0.5) / new_mean if new_mean else 0.0
        if new_cv <= cv:
            out.append(line)
            continue
        sentences[longest] = new
        out.append(" ".join(sentences))
    return "\n".join(out)


# ------------------------------------------------------------ humanize pass

# (pattern, contraction): expands the text's register away from the stiff,
# perfectly-formal surface that classifiers key on. Applied only outside
# quoted spans, so dialogue is untouched.
_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    (r"\bI am\b", "I'm"),
    (r"\bI have\b", "I've"),
    (r"\bI will\b", "I'll"),
    (r"\bI would\b", "I'd"),
    (r"\byou are\b", "you're"),
    (r"\byou have\b", "you've"),
    (r"\byou will\b", "you'll"),
    (r"\byou would\b", "you'd"),
    (r"\bwe are\b", "we're"),
    (r"\bwe have\b", "we've"),
    (r"\bwe will\b", "we'll"),
    (r"\bwe would\b", "we'd"),
    (r"\bthey are\b", "they're"),
    (r"\bthey have\b", "they've"),
    (r"\bthey will\b", "they'll"),
    (r"\bthey would\b", "they'd"),
    (r"\bhe is\b", "he's"),
    (r"\bhe has\b", "he's"),
    (r"\bshe is\b", "she's"),
    (r"\bshe has\b", "she's"),
    (r"\bit is\b", "it's"),
    (r"\bthat is\b", "that's"),
    (r"\bthere is\b", "there's"),
    (r"\bhere is\b", "here's"),
    (r"\bwhat is\b", "what's"),
    (r"\bwho is\b", "who's"),
    (r"\bcannot\b", "can't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bhave not\b", "haven't"),
    (r"\bhas not\b", "hasn't"),
    (r"\bhad not\b", "hadn't"),
    (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"),
    (r"\bcould not\b", "couldn't"),
    (r"\bmight not\b", "mightn't"),
    (r"\bshould have\b", "should've"),
    (r"\bwould have\b", "would've"),
    (r"\bcould have\b", "could've"),
    (r"\bmight have\b", "might've"),
    (r"\bit will\b", "it'll"),
    (r"\bthat will\b", "that'll"),
    (r"\bthey have got\b", "they've got"),
    (r"\bwe have got\b", "we've got"),
    (r"\byou have got\b", "you've got"),
    (r"\bI have got\b", "I've got"),
)


def _humanize(text: str) -> tuple[str, list[Change]]:
    work = text
    changes: list[Change] = []
    for pat, rep in _CONTRACTIONS:
        rx = re.compile(pat, re.IGNORECASE)
        matched: list[str] = []
        line = 1
        try:
            def _repl(m: re.Match[str]) -> str:
                nonlocal line
                if work.count('"', 0, m.start()) % 2 == 1:
                    close = work.find('"', m.start())
                    span = work[work.rfind('"', 0, m.start()) + 1 : close if close != -1 else m.end()]
                    if not re.search(r"[.!?,;:]", span):
                        return m.group(0)
                matched.append(m.group(0))
                line = work.count("\n", 0, m.start()) + 1
                if m.group(0)[0].isupper() and rep[0].islower():
                    return rep[0].upper() + rep[1:]
                return rep
            new = rx.sub(_repl, work)
        except Exception:
            continue
        if matched:
            changes.append(Change(f"contraction: {matched[-1]} → {rep}", matched[-1], rep, line))
            work = new
    return work, changes


# ---------------------------------------------------------------- pipeline


def deep_rewrite(text: str, max_scrub: int = 0, profile: dict | None = None) -> dict:
    from .watermark_probe import probe as _wprobe

    used_pool: set[str] = set()
    result = _deep_once(text, used_pool, profile)
    result["scrub_iterations"] = 0
    wm = result["watermark"]
    if max_scrub <= 0 or not wm.get("available") or not wm.get("after"):
        return result
    if wm["after"]["z"] <= wm["after"]["threshold"]:
        return result

    current = result["text"]
    for i in range(1, max_scrub + 1):
        nxt = _deep_once(current, used_pool, profile)
        wm_after = nxt["watermark"].get("after")
        result = nxt
        result["scrub_iterations"] = i
        if wm_after is None or wm_after["z"] <= wm_after["threshold"]:
            break
        if nxt["applied"] == 0:
            break
        current = nxt["text"]
    return result


def _deep_once(text: str, used: set[str] | None = None, profile: dict | None = None) -> dict:
    base = paraphrase(text)
    work = base["text"]

    flagged = {m.match_text.lower() for m in __import__("deai.detector", fromlist=["scan"]).scan(text)}
    flagged |= {m.match_text.lower() for m in __import__("deai.detector", fromlist=["scan"]).scan(work)}

    swaps: list[tuple[int, int, str, str]] = []
    work = _lexical_pass(work, flagged, swaps, used)
    lex_changes = [
        Change(f"synonym: {b} → {a}", b, a, work.count("\n", 0, s) + 1)
        for s, _e, b, a in swaps
    ]

    before_syn = work
    work = _clausal_pass(work)
    work = _triad_pass(work)
    work = _syntactic_pass(work)
    work = _burstiness_pass(work, profile)
    syn_changes: list[Change] = [
        Change("structure", o[:80], n[:80], i + 1)
        for i, (o, n) in enumerate(zip(before_syn.split("\n"), work.split("\n")))
        if o != n
    ]

    work, hum_changes = _humanize(work)

    from .detector import scan as _scan

    orig_ids = {m.rule_id for m in _scan(text)}
    introduced = [m for m in _scan(work) if m.rule_id not in orig_ids]
    if introduced:
        revert: set[tuple[int, int]] = set()
        for m in introduced:
            for s, e, _b, _a in swaps:
                if s <= m.start < e or s < m.end <= e:
                    revert.add((s, e))
                    break
        for s, e, b, a in sorted(swaps, key=lambda x: x[0], reverse=True):
            if (s, e) in revert:
                work = work[:s] + b + work[e:]
                lex_changes = [c for c in lex_changes if not (c.before == b and c.after == a)]

    work = re.sub(r" {2,}", " ", work)
    work = re.sub(r"\s+([,.;:!?])", r"\1", work)
    work = re.sub(r"[,;:]\s*([.!?])", r"\1", work)
    work = re.sub(r"[ \t]+\n", "\n", work).rstrip()

    from .paraphraser import _matched_tokens

    tokens_orig = re.findall(r"\S+", text)
    tokens_new = re.findall(r"\S+", work)
    total = len(tokens_orig)
    changed = total - _matched_tokens(tokens_orig, tokens_new)
    remaining = len([m for m in _scan(work) if m.rule_id not in COVERED])

    from .watermark_probe import probe as _wprobe

    wm_before = _wprobe(text)
    wm_after = _wprobe(work) if wm_before else None

    all_changes = [
        Change(c["name"], c["before"], c["after"], c["line"]) for c in base["changes"]
    ] + lex_changes + syn_changes + hum_changes

    return {
        "text": work,
        "mechanical": base["mechanical"],
        "rewrites": len(all_changes),
        "applied": base["applied"] + len(lex_changes) + len(syn_changes) + len(hum_changes),
        "changes": sorted(all_changes, key=lambda c: c.line),
        "remaining_flags": remaining,
        "watermark": {
            "available": wm_before is not None,
            "before": wm_before,
            "after": wm_after,
        },
        "disruption": {
            "tokens_total": total,
            "tokens_changed": changed,
            "token_pct": round(100 * changed / total, 1) if total else 0.0,
            "remaining_flags": remaining,
            "note": (
                "Deep rewrite: rule bank + WordNet synonym substitution "
                "(frequency-gated by wordfreq, inflection preserved) + "
                "guarded structure moves + sentence-rhythm shaping + "
                "register shift (contractions). Deterministic, no LLM; "
                "every swap is verified against the tell scanner and "
                "reverted if it introduces a new tell."
            ),
        },
    }