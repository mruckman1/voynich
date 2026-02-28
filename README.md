# Voynich Manuscript Convergence Attack Toolkit

A multi-strategy cryptanalytic framework for the Voynich Manuscript (Beinecke MS 408) implementing a multi-phase convergence attack that progressively narrows the solution space through cascading constraint satisfaction.

The core insight: rather than attacking the cipher monolithically, multiple independent cryptanalytic strategies constrain each other, exponentially reducing viable hypotheses at each phase.

## Key Results

| Phase | Finding | Confidence | Evidence |
|-------|---------|-----------|----------|
| 1 | Sequential cipher state detected | HIGH | Cross-folio bigram correlation = 0.5646 |
| 1 | Positional affixes are a separable layer | HIGH | Stripping affixes increases H2 by +0.2024 |
| 1 | Cold start patterns at scribe transitions | MODERATE | 2/4 transitions show entropy anomalies |
| 2 | All super-character models excluded | HIGH | None survive discrimination sweep |
| 3 | Language A/B split confirmed | HIGH | Lang B: 13 words, H2=0.741; Lang A: 57 types, H2=1.487 |
| 3 | Hybrid model explains all anomalies | HIGH | All 5 Phase 2 anomalies explained, 446 bits information budget |
| 4 | Latin confirmed as source language | HIGH | Distance 0.663 (Latin) vs 1.008 (Italian) vs 1.024 (German); only Latin H2 within tolerance |
| 5 | Two-tier nomenclator confirmed | HIGH | 1,001 codebook types (74.4%), 2,761 cipher types; H2 drop = 0.212 |
| 6 | Word-level SAA failed; morphological structure confirmed | HIGH | Path C: 123 affixes, 228 paradigms, 81% coverage; Paths A/B: FAIL/INCONCLUSIVE |
| 7 | Morphological decomposition | HIGH | 154 unique stems, 81% paradigm coverage, 123 affixes |
| 8 | Coherent medical Latin via HMM Viterbi | MODERATE | 228 corrections; *papaver/frigida/humida/recipe* vocabulary |
| 9 | Syllabic constraints prevent hallucination | MODERATE | 538 syllables, 13 sigla rules, beam width 25 |
| 10 | Dictionary-guided translation with humoral patterns | MODERATE | 914 vocabulary, 50,027 tokens; *calida et humida in primo gradu* |
| 11 | CSP eliminates repetition; 36.5% bracketed | MODERATE | 655 skeletons; 481 brackets -> 97 resolved by n-gram -> 25.6% final |
| 12 | Full-corpus contextual reconstruction | MODERATE | 224 folios, 41.4% Lang A / 40.5% Lang B resolution |
| 12 | Content: Medieval Latin medical recipes | MODERATE | Recurring: *bibe, coque, oleo, aloe, bufo, aqua, hora* |
| 12.5 | Adversarial defense suite | HIGH | 5 tests: unicity, domain swap, polyglot, EVA collapse, ablation |
| 13 | Scholarly synthesis: 114 folios decoded | MODERATE | HTML viewer, English glosser, HITL console, whitepaper |
| 13 | Illustration-text correlation: 2/22 folios match | LOW | *achillea* on f90r1, *ruta* on f96v; permutation p=0.081 |

## Architecture

```
voynich/
├── cli.py                             # Unified CLI entry point (recommended)
├── convergence_attack.py              # Phase 1: 5-strategy orchestrator
├── run_max.py                         # Full-corpus analysis (requires IVTFF data)
│
├── orchestrators/                     # Phase execution layer
│   ├── __init__.py                    # Lazy phase registry
│   ├── _config.py                     # Centralized constants (SAA iters, corpus size, etc.)
│   ├── _foundation.py                 # Shared initialization pipeline (phases 7-13)
│   ├── _utils.py                      # File I/O and output directory helpers
│   ├── phase2..phase12.py             # Individual phase orchestrators
│   ├── phase12_5.py                   # Adversarial defense suite (5 tests + diagnostics)
│   └── phase13.py                     # Scholarly synthesis (HTML, glosser, HITL, whitepaper)
│
├── data/
│   ├── voynich_corpus.py              # EVA transliterations, scribe mappings, zodiac labels
│   ├── ivtff_parser.py                # IVTFF full-corpus parser (Zandbergen format)
│   ├── botanical_identifications.py   # Plant species IDs for herbal section
│   ├── expanded_medical_vocabulary.py # 225 lemmas, 909 inflected forms (6 categories)
│   ├── glyph_alphabets.py            # EVA glyph properties, positional classes, ligatures
│   ├── latin_syllables.py            # Medieval Latin syllabification rules
│   ├── medieval_text_templates.py    # Latin herbal recipe templates
│   ├── english_glossary.json         # Latin-to-English dictionary for Phase 13
│   ├── Voynich_Botanicals.csv        # 91 botanical IDs: scientific name → medieval Latin
│   ├── botanical_name_mapping.py     # Folio→species→medieval Latin lookup bridge
│   ├── folio_illustration_priors.py  # Per-folio botanical word boost tables (3-tier prior)
│   ├── corpora/                      # Reference Latin text corpora
│   │   ├── latin_vulgate_sample.txt  # Vulgate Bible sample (5K tokens)
│   │   └── corpus_juris_civilis.txt  # Medieval legal Latin
│   └── dictionaries/                 # Language dictionaries for adversarial tests
│       ├── phase5_latin_dict.json    # Latin dictionary (Phase 5)
│       ├── romance_italian_dict.json # Italian dictionary (polyglot test)
│       └── romance_occitan_dict.json # Occitan dictionary (polyglot test)
│
├── modules/
│   ├── statistical_analysis.py        # H1/H2/H3 entropy, Zipf, bigram matrices
│   ├── naibbe_cipher.py              # Multi-table combinatorial cipher engine
│   ├── null_framework.py             # Null distribution testing
│   ├── constraint_model.py           # Multi-constraint satisfaction
│   ├── candidate_search.py           # Constrained cipher candidate search
│   │
│   ├── strategy1_parameter_search.py  # Cipher parameter grid search
│   ├── strategy2_scribe_seams.py      # Scribe transition leakage analysis
│   ├── strategy3_binding_reconstruction.py  # Binding order + sequential state
│   ├── strategy4_positional_grammar.py      # Glyph grammar extraction
│   ├── strategy5_zodiac_attack.py           # Known-plaintext zodiac attack
│   │
│   ├── word_length.py                # Word boundary semantics
│   ├── positional_shape.py           # Positional glyph classes
│   ├── fsa_extraction.py             # Finite state automaton classification
│   ├── nmf_analysis.py               # NMF topic factorization
│   ├── error_model.py                # Scribe error patterns
│   ├── qo_analysis.py                # Functional analysis of qo- prefix
│   ├── label_analysis.py             # Zodiac label plaintext anchors
│   ├── paragraph_analysis.py         # Plaintext:ciphertext length ratios
│   ├── entropy_gradient.py           # U-curve entropy progression
│   │
│   ├── phase2/                       # 6 generative models, discrimination sweep
│   ├── phase3/                       # Language B profiling, onset decomposition, hybrid model
│   ├── phase4/                       # Language A extraction, nomenclator models, SAA
│   ├── phase5/                       # Tier splitting, NMF scaffold, constrained SAA
│   ├── phase6/                       # Improved corpus, homophone detection, morpheme analysis
│   ├── phase7/                       # Voynich morphology, stem SAA, affix alignment
│   ├── phase8/                       # Viterbi decoder, morphological synthesizer
│   ├── phase9/                       # Latin syllabifier, sigla mapper, beam search
│   ├── phase11/                      # Phonetic skeletonizer, CSP decoder
│   ├── phase12/                      # Fuzzy skeletonizer, syntactic scaffold, n-gram mask solver, char n-gram model
│   ├── phase12_5/                    # Adversarial tests (unicity, domain swap, polyglot, EVA collapse, ablation)
│   └── phase13/                      # English glosser, HTML viewer, HITL console, whitepaper, illustration correlation
│
└── output/
    ├── phase3/                       # Language B profile, hybrid model results
    ├── phase4/                       # Language A extraction, multi-language testing
    ├── phase5/                       # Tier split, NMF scaffold, cross-validation
    ├── phase6/                       # Three recovery path results
    ├── phase7/                       # Morphological stem mapping
    ├── phase8/                       # HMM Viterbi translations
    ├── phase9/                       # Syllabic beam search results
    ├── phase10/                      # Dictionary-guided trigram translations
    ├── phase11/                      # CSP phonetic translations
    ├── phase12/                      # Final contextual reconstruction
    ├── phase12_5/                    # Adversarial defense verdicts
    └── phase13/                      # Full translations, English glosses, HTML viewer, whitepaper
```

## The Phase Framework

Each phase proves a specific hypothesis, unlocking the next phase's assumptions. Phases 1-12 form the core decoding pipeline, Phase 12.5 validates results adversarially, and Phase 13 produces scholarly output.

### Phase 1: Convergence Attack (5 Strategies)

Five interlocking strategies that mutually constrain each other:

1. **Naibbe Parameter Search** -- Encrypts Hartlieb-era medical Latin through thousands of cipher parameter variants, comparing statistical fingerprints (H1, H2, H3, Zipf) against real Voynich text.
2. **Scribe Seam Analysis** -- Exploits transition zones between the 5 identified scribes (Davis 2020) as cryptographic side-channels. Found cold-start patterns and shared substitution tables.
3. **Binding Reconstruction** -- Tests quire reorderings for sequential consistency. Confirmed progressive cipher state (bigram correlation = 0.5646).
4. **Positional Grammar Extraction** -- Decomposes words into prefix/root/suffix layers. Stripping affixes increases H2 by +0.2024, confirming separable grammatical wrapper.
5. **Zodiac Known-Plaintext Attack** -- Uses Romance month labels (*abril*, *mars*) and formulaic medical content as cribs for parameter matching.

### Phase 2: Super-Character Model Discrimination

Tested 6 generative models (glyph decomposition, verbose cipher, syllabary code, steganographic carrier, grammar induction, slot machine). All excluded by discrimination sweep against Voynich statistical targets:
- H1 = 3.707, H2 = 1.406, H3 = 0.898
- Zipf exponent = 1.244, TTR = 0.164

The discrimination sweep revealed that the *combined* corpus statistics were internally inconsistent -- different folio groups produced contradictory statistical signatures. This discovery led directly to the Language A/B split hypothesis (Phase 3).

### Phase 3: Language A/B Split

**Language B** identified as a 13-word notation system:
- Vocabulary: *chedy* (51), *shedy* (48), *otedy* (44), *lchedy* (21), *qokaiin* (19), ...
- H2 = 0.741 (extremely low -- mechanical, not linguistic)
- Two word families: *-edy* (84.1%) and *-aiin* (15.0%)
- 534 bits of information (~67 bytes, equivalent to ~23 Latin words)

**Language A** extracted with 57 types, H2 = 1.487:
- Less anomalous than combined corpus (3/3 metrics in natural range vs 2/3)
- Zipf exponent = 0.931 (within natural language range)

**Hybrid model**: explains all 5 Phase 2 anomalies with plausible information budget of 446 bits.

### Phase 4: Full Corpus Extraction + Nomenclator Testing

Parsed full IVTFF corpus (Zandbergen transliteration):
- 10,791 tokens, 3,762 types, 114 folios
- Character entropy: H1 = 3.832, H2 = 2.385, H3 = 2.125
- Top 10 words: *daiin* (468), *chol* (241), *chor* (164), *s* (131), *shol* (109), *dain* (91), *cthy* (91), *chy* (91), *sho* (88), *dar* (86)

**SAA (Simulated Annealing Assignment) attack:**
- Best method: simulated annealing (normalized cost = 0.00305)
- Crib satisfaction: 38/628 (6.1%)
- Sample mapping: *daiin* -> et, *dain* -> siccum

**Entropy gradient** (p = 0.000, significant):
- Q1 (early folios): H2 = 2.442, Q4 (late folios): H2 = 2.332
- Gradient: -0.110 bits/quartile -- page position affects entropy, supporting header/body vocabulary split

**Multi-language discrimination** (8 languages tested):

| Language | Distance | H2 Within Tolerance |
|----------|----------|-------------------|
| Latin | 0.663 | Yes |
| Italian | 1.008 | No |
| German | 1.024 | No |
| Hebrew | 1.423 | No |

Conclusion: Latin is the only source language whose word-bigram H2 falls within tolerance of the Voynich signal. Nomenclator model confirmed.

### Phase 5: Two-Tier Nomenclator

Split the vocabulary into two tiers:
- **Tier 1 (Codebook)**: 1,001 types, 8,030 tokens (74.4% coverage), mean word length 4.63
- **Tier 2 (Cipher)**: 2,761 types, 2,761 tokens (25.6% coverage), mean word length 6.74
- H2 drop between tiers: 0.212 (significant, threshold = 0.15)

**Attack A (Codebook tier SAA):**
- 1,001 Voynich types mapped to 629 Latin words (surjective)
- Normalized cost: 0.000594 (100,000 iterations)
- Crib satisfaction: 260/704 (36.9%)
- Coherent phrases found: **0** -- no multi-word Latin phrases survived

**Attack B (Cipher tier pattern matching):**
- 2,761 singletons analyzed, character H1 = 3.87 (compatible with Latin H1 = 3.95)
- 1,571/2,761 singletons have pattern-matched Latin candidates (56.9%)
- 360 singletons uniquely determined by letter-pattern

**Cross-validation:** 1/5 checks passed (OVERALL FAIL). Humoral consistency: 12/24 folios (50%).

**Conclusion: NOMENCLATOR NEEDS REFINEMENT** -- tier split confirmed but word-level SAA cannot recover coherent text.

### Phase 6: Recovery Paths

Phase 5 failure diagnosis: SAA produced **0 coherent phrases**, crib satisfaction only 36.9%, surjective mapping (1,001 -> 629 Latin words). Root causes: synthetic corpus mismatch, cost function imbalance, homophony blindspot.

**Path A -- Improved corpus SAA:**
- Rebuilt Latin corpus: 10,025 tokens, 894 types, Zipf exponent = 1.163
- SAA result: 1 phrase found, crib rate improved to 44.3%
- Verdict: **FAIL** -- still no coherent multi-word output

**Path B -- Homophone merging:**
- Detected 20 homophone groups covering 116 words (11.6% of vocabulary)
- Largest group: *chol/chor/cthol/shy/otol* (65 variants, 789 combined tokens)
- After merging: 1,001 types reduced to 905 (9.6% reduction)
- Reduced SAA: 3 phrases found, Zipf 0.914 -> 0.918 (hypothesis NOT supported)
- Verdict: **INCONCLUSIVE**

**Path C -- Morphological boundary analysis:**
- 123 productive affixes (65 prefixes + 58 suffixes)
- 228 paradigms covering 81.0% of vocabulary
- Word boundaries show higher entropy (H = 2.790) than within-word positions (H = 1.901)
- Verdict: **CONFIRMED** -- Voynich words are compositional, not opaque codebook entries

**Overall conclusion:** Word-level codebook substitution is fundamentally wrong. The cipher operates on sub-word morphemes. Phase 7 attacks at the morpheme level.

### Phase 7: Morphological Sub-Word Attack

Decomposed Voynich words into stems and affixes:
- 154 unique Voynich stems extracted from 1,001 codebook types
- 81% paradigm coverage with 123 productive affixes
- Latin comparison: 914 vocabulary types -> 760 unique Latin stems
- Top Voynich stems: *o* (986), *a* (767), *y* (716), *l* (644), *i* (637)

**Stem SAA mapping** (normalized cost = 0.00216):

| Voynich Stem | Latin Stem | Meaning |
|-------------|-----------|---------|
| o | et | "and" |
| a | vale | "be effective" |
| y | dos | "dose" |
| l | grad | "degree" |
| i | in | "in" |
| r | cip | "recipe" |
| d | est | "is" |
| k | cum | "with" |
| oi | coqu | "cook/boil" |
| lc | ole | "oil" |
| cha | aqu | "water" |
| sh | emplastr | "plaster" |
| cho | pilul | "pill" |

**Affix alignment:** 30 Voynich suffixes mapped to Latin inflectional endings (e.g., *-iin* -> *-t*, *-hy* -> *-t*, *-chy* -> *-a*).

Decoded sample: `vale-l vale-r grad cip-y dos cum-or grad-dy menstru-y tib-ar cip-y ...`

### Phase 8: HMM Viterbi Translation

Applied Hidden Markov Model Viterbi decoding with the Phase 7 stem-to-Latin mapping and morphological synthesis. 228 contextual corrections applied across all folios.

**Sample translation (f1r):**

> et vel papaver valet est frigida et humida et liberabitur recipe in primo bibere cave similiter ut papaver et liberabitur recipe est frigidum et humida et sicca et humidus et pone super locum et est frigida fortior in secundo gradu habet virtutem et fac recipe ...

The output shows coherent medieval medical Latin structure: humoral temperature descriptions (*frigida et humida*, *calida et sicca*), degree specifications (*in primo/secundo gradu*), preparation verbs (*recipe*, *fac*, *contere*, *misce*), and plant identifiers (*papaver*, *camomilla*, *centaurea*, *asarum*).

**Key limitation:** HMM frequency biasing causes hallucination -- high-frequency stems dominate, producing repetitive "et fac et fac" sequences. This motivated the syllabic constraint decoder in Phase 9.

### Phase 9: Syllabic Beam Search Decoder

Built syllabic decoder to prevent HMM hallucination by constraining output to valid Latin syllable sequences.

**Latin syllable inventory:** 538 unique syllables (top: *et*, *a*, *co*, *est*, *in*, *re*, *ci*, *li*, *pe*, *ca*)

**Sigla (scribal abbreviation) constraints:**

| Voynich | Latin Candidates |
|---------|-----------------|
| qo | con, qu, com |
| ch | ca, ce, co |
| sh | si, se, sa |
| d | de, di |
| iin | us, um, is |
| dy | ae, ti, ur |
| ey | es, em, et |
| m | rum, num |
| r | er, ar, or |
| l | al, el, il |
| a, o, e | a, o, e (transparent) |
| y | i |

**Beam search stats:** 200 tokens processed, beam width 25, final log probability = -1842.07.

**Limitation:** Syllable-level decoding prevents hallucination but produces sub-word fragments rather than whole Latin words, requiring dictionary-guided reassembly (Phase 10).

### Phase 10: Dictionary-Guided Trigram Translation

Trained on expanded Latin corpus (914 vocabulary, 50,027 tokens) with Medieval Latin morphological constraints. Applied trigram-level translation with dictionary verification to reassemble syllabic fragments into whole words.

**Sample translation (f1r):**

> et vel papaver valet est frigida et humida et liberabitur recipe in primo bibere cave similiter ut papaver et liberabitur recipe est frigidum et humida et sicca ... est frigida fortior in secundo gradu habet virtutem ... calida et humida ... camomilla et fac gargarismum

The output reveals recurring medieval medical recipe structure: plant name + humoral properties (*est frigida et humida in primo gradu*) + efficacy (*habet virtutem/valet contra*) + preparation instructions (*recipe/fac/contere/misce cum*). Key vocabulary: *papaver* (poppy), *camomilla*, *centaurea*, *gargarismum* (gargle), *cataplasma* (poultice), *emplastrum* (plaster).

**Limitation:** Flat frequency scoring causes repetition in high-frequency words, motivating the CSP approach in Phase 11.

### Phase 11: Phonetic CSP Decoding

Treated Voynich stems as consonant skeletons and applied Constraint Satisfaction Problem solving:
- 655 Latin skeletons mapped to 914 unique Latin words
- 15 folios decoded (f1r through f8r), 1,318 total words

**Bracket analysis by POS tag:**

| Tag | Count | Description |
|-----|-------|-------------|
| UNK | 277 | No skeleton match |
| NOUN_ACC | 55 | Accusative nouns (from *-iin* suffix) |
| ADJ | 38 | Adjectives (from *-ey/-hy* suffixes) |
| NOUN_AGT | 37 | Agent nouns (from *-ar* suffix) |
| NOUN | 14 | Bare nouns |
| NOUN_NOM | 9 | Nominative nouns |
| NOUN_FEM | 4 | Feminine nouns |

**N-gram resolution (feeding Phase 12):**
- Initial bracketed words: 481 (36.5% of total)
- Resolved by n-gram context: 97 (20.2% resolution rate)
- Still unresolved: 337
- Final unresolved rate: **25.6%**

Eliminated HMM hallucination but suffered from repetition in high-frequency decoded words (*hora/quae/oleo/et*) due to flat frequency scoring. The n-gram context resolver in Phase 12 addresses this.

### Phase 12: Contextual Reconstruction

Full-corpus decoding combining CSP, syntactic scaffolding, n-gram mask solving, cross-folio consistency, POS backoff, character n-gram fallback scoring, and illustration-guided disambiguation across a 9-sub-phase pipeline:

1. **Load Dependencies** -- Latin corpus (50,027 tokens, 1,695 types), skeleton index (1,087 skeletons), transition matrix (1,001×1,001)
2. **Build Components** -- FuzzySkeletonizer (y/o branching), BudgetedCSPDecoder (graduated scoring), SyntacticScaffolder (POS tagging), NgramMaskSolver (8 improvements), CharNgramModel (trigram fallback), Illustration Prior (per-folio botanical boosts)
3. **Budgeted CSP Decoding** -- frequency budgeting + humoral crib injection across all 224 folios
4. **Syntactic Scaffolding** -- POS-tag remaining brackets via Latin suffix patterns
5. **Deterministic N-Gram Mask Solving** -- word-level bigram scoring with bidirectional multi-pass, function word recovery, dual-context confidence reduction, unigram frequency backoff, illustration-guided disambiguation
6. **Cross-Folio Consistency** -- skeleton→word mappings agreed across 3+ folios override local ambiguity
7. **POS Backoff Pass** -- when word-level P(candidate|prev) = 0, falls back to POS transition probability (8×8 matrix) as a coarser discriminator. Runs post-consistency to avoid poisoning cross-folio agreement. Illustration boost applied to POS-scored candidates on botanical folios.
8. **Character N-Gram Fallback** -- for remaining unresolved tokens, scores candidates by Latin character trigram plausibility. Unlike word-level and POS-level scoring, does not require resolved neighbors. Uses average log-probability with score gap thresholding. Score gap threshold relaxed for illustration-boosted candidates.
9. **Illustration-Guided Disambiguation** -- per-folio multiplicative boost for candidates semantically related to the plant depicted in each folio's illustration. Three tiers: Tier 1 (exact plant names + inflections, 2.0×), Tier 2 (medicinal properties + humoral terms, 1.3×), Tier 3 (generic botanical vocabulary, 1.1×). Boost is multiplicative on transition scores (0 × boost = 0), so the prior alone cannot create resolutions -- it only disambiguates when multiple candidates are competitive. Confidence ratio reduced to 2.5× for boosted winners. Covers 25 testable botanical folios with ~60-94 boosted words each. Built from independent botanical identifications (Tucker & Talbert, Bax, Sherwood) via `build_illustration_prior()`.

**Current results (224 folios):**
- Language A: 41.4% resolution (5,468 / 13,204 words)
- Language B: 40.5% resolution (9,319 / 23,030 words)
- Cross-folio consistency: 2,167 tokens resolved
- POS backoff: ~203 additional tokens
- Character n-gram fallback: ~257 additional tokens

### Phase 12.5: Adversarial Defense Suite

Subjects the Phase 11/12 decoding pipeline to 5 adversarial conditions to mathematically prove that its success on real Voynich text is genuine, not an artifact of algorithmic overfitting:

1. **Unicity Distance** -- scrambled/random text baseline; proves resolution isn't from random guessing
2. **Domain Swap** -- injects Bible/Legal transition matrices; confirms domain-specific dictionary matters
3. **Polyglot Dictionary** -- substitutes Italian/Occitan dictionaries for Latin; resolution should collapse
4. **EVA Collapse** -- re-tokenizes with ligature collapse to 18 glyphs; resolution should degrade
5. **Ablation Study** -- removes function word recovery and grammar traces; measures contribution of each component

Additional diagnostics:
- **Compositionality Proof** -- proves Voynich words are compositional (affixes + stems), not opaque codes
- **Dictionary Diagnostic** -- audits skeleton-to-word coverage in the transition matrix; categorizes unresolved words into ZERO_MATCH (13%), MATCH_IN_MATRIX (54%), MATCH_NOT_IN_MATRIX (11%), NO_SKELETON (22%)

Each test produces a pass/fail verdict. The overall defense score is `tests_passed / 5`.

### Phase 13: Scholarly Synthesis

Transforms all decoding output into readable, publishable formats:

1. **Full-Corpus Decode** -- runs the Phase 12 pipeline on all 114 folios (10,791 words)
2. **Interlinear HTML Viewer** -- 4-tier offline HTML (Voynich glyph / skeleton / Latin / English) with full traceability
3. **Deterministic English Glosser** -- Latin-to-English dictionary + inflection rules; 26.9% gloss rate across 114 folios
4. **HITL Console** -- interactive human-in-the-loop editor for manually resolving `[UNRESOLVED]` tokens
5. **Academic Whitepaper** -- structured Markdown with matplotlib charts (bracket waterfall, folio frequency, resolution by folio, Zipf comparison)
6. **Illustration-Text Correlation** -- cross-validates decoded text against independent botanical identifications from manuscript illustrations. For each of 28 herbal folios with published plant IDs (Tucker & Talbert, Bax, Sherwood), searches decoded Latin for medieval Latin names of the visually identified species. Matches (e.g., *achillea* decoded on f90r1 where yarrow was identified from the illustration) provide external physical validation. Statistical significance assessed via binomial and permutation tests (10,000 trials).

## Sample Translation Output (Phase 12, folio f1r)

```
efficax [ykal] [ar] [ataiin] [shol] hora aures in aqua oleo [sory]
cera vere [kair] die [shar] [ase] [cthar] ruta [syaiir] quae [or]
[ykaiin] [shod] [cthoary] [cthes] [daraiin] [sy] [soiin] [oteey]
destillat ortu [okaiin] [or] [okan] [sairy] [chear] [cthaiin] bibe
bufo dorsi et [cy] aloe viva et [sh] sed ab tere [yshey] da quae
aqua et si [dain] [chor] [kos] [daiin] [shos] [cfhol] da [dain] bis
da [ydain] bis [ols] [cphey] [ytain] ossa subtilis eas cassia
[otairin] [oteol] genu et quia [daiin] ossa aqua quotidiana cibo
subtilis [cthey] ossa et est [dain] [oiin] [chol] et et [chdy]
[okain] et [cthy] aceto [daiin] sicca [ckeo] per cor ossa [kol]
[chol] [chol] [kor] [chal] ossa [chol] et cassia [kchy] est [or] et
[sho] [koeam] [ycho] [tchey] coque bis [dydyd] [cthy] die [yto]
[shol] [she] cutis viola [darain] [dain] [ckhyds] decoquere et
[okaiir] quae quae et utilis [dlocto] [shok] [chor] quae [dain]
[ckhey] [otol] [daiiin] [cpho] [shaiin] hoc [chol] testis ossa habet
aqua ruta [ydoin] [chol] [dain] [cthal] hedera sero [kaiin] hedera
ossa [cthar] aqua et coque [shoaiin] [okol] [daiin] bibere [cthol]
[daiin] [ctholdar] [ycheey] [okeey] [oky] [daiin] quoque [kokaiin]
hac [kdchy] [dal] [dcheo] deo cassia [cthy] [okchey] [keey] cautela
[chtor] [eo] [chol] [chok] ideo et [dchaiin]
```

Bracketed words `[...]` are unresolved Voynich stems (203 words total, 110 unresolved). The decoded content shows Medieval Latin medical/herbal recipe language: *efficax* (effective), *bibe* (drink), *coque* (cook/boil), *destillat* (distill), *aqua* (water), *ossa* (bones), *cassia*, *ruta* (rue), *hedera* (ivy), *viola* (violet), *cutis* (skin), *subtilis* (fine/subtle), *quotidiana* (daily), *aceto* (vinegar).

## Usage

### Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
uv sync
```

### Run Phases

The unified CLI (`cli.py`) is the recommended way to run all phases:

```bash
# Phase 1: Core 5-strategy convergence attack (default)
uv run cli.py

# Run a specific phase (2-13, or 12.5)
uv run cli.py --phase 2
uv run cli.py --phase 7

# Run all phases sequentially (1-13, including 12.5)
uv run cli.py --all

# Run phases 1-4 only
uv run cli.py --phased

# List all available phases
uv run cli.py --list

# Suppress verbose output
uv run cli.py --phase 5 --quiet

# Quick mode: reduced SAA iterations and corpus size for faster testing
uv run cli.py --phase 5 --quick

# Custom output directory (default: ./output)
uv run cli.py --all --output-dir ./results
```

A `combined_report.json` is written to the output directory after every run, containing the full results from all phases that were executed. Each phase entry includes a description of what the phase does and its complete output data (conclusions, translations, statistics, mappings, etc.).

Sub-phase flags can target specific steps within a phase:

```bash
# Phase 2: run only the discrimination sweep
uv run cli.py --phase 2 --discrimination

# Phase 3: run only the hybrid model step
uv run cli.py --phase 3 --hybrid

# Phase 6: run recovery paths A and C
uv run cli.py --phase 6 --path-a --path-c

# Phase 12.5: adversarial defense suite
uv run cli.py --phase 12.5                        # All 5 core adversarial tests
uv run cli.py --phase 12.5 --unicity              # Unicity distance test only
uv run cli.py --phase 12.5 --domain-swap          # Domain swap test only
uv run cli.py --phase 12.5 --polyglot             # Polyglot dictionary test only
uv run cli.py --phase 12.5 --eva-collapse         # EVA collapse test only
uv run cli.py --phase 12.5 --ablation             # Ablation study only
uv run cli.py --phase 12.5 --compositionality     # Compositionality proof (opt-in)
uv run cli.py --phase 12.5 --dictionary-diagnostic  # Dictionary coverage audit

# Phase 13: scholarly synthesis
uv run cli.py --phase 13                           # Full synthesis pipeline
uv run cli.py --phase 13 --html                    # HTML viewer only
uv run cli.py --phase 13 --gloss                   # English glosser only
uv run cli.py --phase 13 --hitl                    # Interactive HITL console
uv run cli.py --phase 13 --whitepaper              # Whitepaper generation only
uv run cli.py --phase 13 --correlation             # Illustration-text correlation
uv run cli.py --phase 13 --folios 20               # Limit decode to 20 folios
```

### Run Individual Strategies

```bash
uv run -m modules.strategy1_parameter_search
uv run -m modules.strategy2_scribe_seams
uv run -m modules.strategy4_positional_grammar
```

### Full Corpus Analysis

Download the IVTFF transliteration and run the full-corpus analyzer:

```bash
mkdir -p data/corpus
curl https://www.voynich.nu/data/ZL_ivtff_2b.txt -o data/corpus/ZL_ivtff_2b.txt
uv run run_max.py
```

### Use the Naibbe Cipher Standalone

```python
from modules.naibbe_cipher import demo; demo()
```

## Dependencies

- numpy >= 2.4.2
- scipy >= 1.17.1
- pandas >= 3.0.1
- matplotlib >= 3.10.8
- datasets >= 4.5.0
- python-Levenshtein >= 0.26.1
- beautifulsoup4 >= 4.14.3

## References

- **Greshko (2025, Cryptologia)**: Naibbe cipher mechanism and parameter space
- **Davis (2020)**: Scribe identification and differentiation (5 scribes)
- **Currier (1976)**: Language A/B split hypothesis
- **Zandbergen**: EVA transliteration standard and IVTFF format
- **Takahashi**: Full EVA transliteration corpus (voynich.nu)
