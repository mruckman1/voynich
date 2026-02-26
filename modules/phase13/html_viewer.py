"""
Phase 13, Module 13.1: Interlinear HTML Viewer
================================================
Generates a self-contained offline HTML file displaying decoded folios
in a 4-tier interlinear layout:
  Tier 1: Original EVA transliteration
  Tier 2: Morphological decomposition (prefix | stem | suffix)
  Tier 3: Latin decryption (from Phase 12)
  Tier 4: English gloss (from Module 13.2's glossary)

Color coding:
  Green  — structurally decoded (function words, sigla matches)
  Blue   — probabilistically decoded (n-gram skeleton match)
  Red    — unresolved ([token_UNRESOLVED])

Phase 13  ·  Voynich Convergence Attack
"""

import json
import re
import os
from html import escape
from typing import Dict, List, Optional


# Known function words from Phase 11 CSP decoder
STRUCTURAL_LATIN = {'et', 'in', 'ad', 'sed', 'est', 'vel', 'per'}

# Regex for unresolved tokens
UNRESOLVED_RE = re.compile(r'\[([^\]]+?)_(UNRESOLVED|VERB_3P|NOUN_\w+|ADJ|UNK)\]')
BRACKET_RE = re.compile(r'[\[<]([^\]>]+)[\]>]')


class InterlinearToken:
    """Represents a single token across all four display tiers."""
    __slots__ = ('eva', 'morphology', 'latin', 'english', 'confidence', 'derivation')

    def __init__(self, eva, morphology, latin, english, confidence, derivation):
        self.eva = eva
        self.morphology = morphology
        self.latin = latin
        self.english = english
        self.confidence = confidence       # 'structural' | 'probabilistic' | 'unresolved'
        self.derivation = derivation       # tooltip text


def _classify_confidence(latin_word: str) -> str:
    """Determine confidence level for a decoded token."""
    if UNRESOLVED_RE.match(latin_word) or BRACKET_RE.match(latin_word):
        return 'unresolved'
    if latin_word.lower() in STRUCTURAL_LATIN:
        return 'structural'
    return 'probabilistic'


def _build_morphology_str(eva_token: str, parsed_samples: List[Dict]) -> str:
    """Build a morphological decomposition string for an EVA token."""
    for sample in parsed_samples:
        if sample.get('word') == eva_token:
            prefix = sample.get('prefix', '')
            stem = sample.get('stem', '')
            suffix = sample.get('suffix', '')
            parts = [p for p in [prefix, stem, suffix] if p]
            return '-'.join(parts)
    # Fallback: just return the token itself
    return eva_token


class InterlinearBuilder:
    """Constructs interlinear token sequences from phase outputs."""

    def __init__(self, phase7_data, phase12_data, glosser):
        self.phase7 = phase7_data
        self.phase12 = phase12_data
        self.glosser = glosser
        self.parsed_samples = phase7_data.get('voynich_morphology', {}).get('parsed_sample', [])

    def build_folio(self, folio_id, eva_tokens, final_text, english_text) -> List[InterlinearToken]:
        """Build the interlinear token sequence for one folio.

        Aligns EVA tokens with Phase 12 decoded words by index.
        """
        latin_words = final_text.split()
        english_words = english_text.split()
        tokens = []

        for i in range(max(len(eva_tokens), len(latin_words))):
            eva = eva_tokens[i] if i < len(eva_tokens) else ''
            latin = latin_words[i] if i < len(latin_words) else ''
            eng = english_words[i] if i < len(english_words) else ''

            morphology = _build_morphology_str(eva, self.parsed_samples)
            confidence = _classify_confidence(latin)
            derivation = f'EVA: {eva} | Morph: {morphology} | Latin: {latin} | Conf: {confidence}'

            tokens.append(InterlinearToken(
                eva=eva,
                morphology=morphology,
                latin=latin,
                english=eng,
                confidence=confidence,
                derivation=derivation,
            ))
        return tokens


# ── CSS Template ─────────────────────────────────────────────────────
CSS = """
:root {
  --structural: #2ecc71;
  --probabilistic: #3498db;
  --unresolved: #e74c3c;
  --bg: #1a1a2e;
  --card-bg: #16213e;
  --text: #e0e0e0;
  --text-dim: #8899aa;
  --border: #0f3460;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  display: flex;
  min-height: 100vh;
}
nav {
  width: 220px;
  position: fixed;
  top: 0; left: 0; bottom: 0;
  background: var(--card-bg);
  border-right: 1px solid var(--border);
  padding: 20px 12px;
  overflow-y: auto;
}
nav h2 { font-size: 14px; color: var(--text-dim); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
nav a {
  display: block;
  padding: 6px 10px;
  color: var(--text);
  text-decoration: none;
  font-size: 13px;
  border-radius: 4px;
  margin-bottom: 2px;
}
nav a:hover { background: var(--border); }
main { margin-left: 220px; padding: 30px 40px; flex: 1; max-width: 1200px; }
h1 { font-size: 24px; margin-bottom: 8px; }
.subtitle { color: var(--text-dim); margin-bottom: 30px; font-size: 14px; }
.folio { margin-bottom: 40px; }
.folio h2 { font-size: 18px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.interlinear-row {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  margin: 2px 4px 10px 0;
  padding: 4px 6px;
  border-radius: 4px;
  vertical-align: top;
  position: relative;
  cursor: default;
  min-width: 40px;
}
.interlinear-row:hover { background: rgba(255,255,255,0.05); }
.tier { display: block; font-size: 11px; line-height: 1.4; text-align: center; white-space: nowrap; }
.tier-eva { color: var(--text-dim); font-style: italic; font-size: 10px; }
.tier-morph { color: #aaa; font-size: 10px; }
.tier-latin { font-weight: 600; font-size: 13px; }
.tier-english { color: #ccc; font-size: 11px; }
.conf-structural .tier-latin { color: var(--structural); }
.conf-probabilistic .tier-latin { color: var(--probabilistic); }
.conf-unresolved .tier-latin { color: var(--unresolved); }
.legend {
  display: flex; gap: 20px; margin-bottom: 24px;
  font-size: 12px; color: var(--text-dim);
}
.legend span::before {
  content: '';
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
  margin-right: 5px;
  vertical-align: middle;
}
.legend .lg-struct::before { background: var(--structural); }
.legend .lg-prob::before { background: var(--probabilistic); }
.legend .lg-unres::before { background: var(--unresolved); }
.stats { color: var(--text-dim); font-size: 12px; margin-top: 6px; }
.tooltip-text {
  display: none;
  position: absolute;
  bottom: 105%;
  left: 50%;
  transform: translateX(-50%);
  background: #222;
  color: #eee;
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 11px;
  white-space: nowrap;
  z-index: 100;
  box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.interlinear-row:hover .tooltip-text { display: block; }
"""


def _render_token_html(token: InterlinearToken) -> str:
    """Render a single interlinear token as HTML."""
    conf_class = f'conf-{token.confidence}'
    tooltip = escape(token.derivation)
    return (
        f'<div class="interlinear-row {conf_class}">'
        f'<span class="tooltip-text">{tooltip}</span>'
        f'<span class="tier tier-eva">{escape(token.eva)}</span>'
        f'<span class="tier tier-morph">{escape(token.morphology)}</span>'
        f'<span class="tier tier-latin">{escape(token.latin)}</span>'
        f'<span class="tier tier-english">{escape(token.english)}</span>'
        f'</div>'
    )


def _render_folio_html(folio_id: str, tokens: List[InterlinearToken]) -> str:
    """Render a full folio section."""
    token_html = '\n'.join(_render_token_html(t) for t in tokens)
    n_struct = sum(1 for t in tokens if t.confidence == 'structural')
    n_prob = sum(1 for t in tokens if t.confidence == 'probabilistic')
    n_unres = sum(1 for t in tokens if t.confidence == 'unresolved')
    return (
        f'<section class="folio" id="{escape(folio_id)}">'
        f'<h2>Folio {escape(folio_id)}</h2>'
        f'<div class="stats">{len(tokens)} tokens: '
        f'{n_struct} structural, {n_prob} probabilistic, {n_unres} unresolved</div>'
        f'<div style="margin-top:10px">{token_html}</div>'
        f'</section>'
    )


def _render_full_html(folio_data: Dict[str, List[InterlinearToken]]) -> str:
    """Render the complete HTML document."""
    # Nav sidebar
    nav_links = '\n'.join(
        f'<a href="#{escape(fid)}">{escape(fid)}</a>'
        for fid in folio_data
    )

    # Folio sections
    folio_sections = '\n'.join(
        _render_folio_html(fid, tokens)
        for fid, tokens in folio_data.items()
    )

    total_tokens = sum(len(t) for t in folio_data.values())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voynich Manuscript — Interlinear Decryption Viewer</title>
<style>{CSS}</style>
</head>
<body>
<nav>
<h2>Folios</h2>
{nav_links}
</nav>
<main>
<h1>Voynich Manuscript — Interlinear Viewer</h1>
<p class="subtitle">Phase 13 Scholarly Presentation &middot; {len(folio_data)} folios &middot; {total_tokens} tokens</p>
<div class="legend">
  <span class="lg-struct">Structural (function words)</span>
  <span class="lg-prob">Probabilistic (skeleton match)</span>
  <span class="lg-unres">Unresolved</span>
</div>
{folio_sections}
</main>
</body>
</html>"""


def generate_interlinear_html(
    phase7_data: Dict,
    phase12_data: Dict,
    english_translations: Dict[str, str],
    eva_by_folio: Dict[str, List[str]],
    output_path: str,
    verbose: bool = False,
) -> Dict:
    """Top-level entry point for Module 13.1.

    Args:
        phase7_data:  Phase 7 report dict
        phase12_data: Phase 12/13 translation data dict with 'final_translations'
        english_translations: Dict of {folio_id: english_text} from glosser
        eva_by_folio: Dict of {folio_id: [eva_tokens]} from extractor
        output_path:  Path for output HTML file

    Returns:
        Dict with metrics: folios_rendered, tokens_rendered, file_size_kb
    """
    from modules.phase13.english_glosser import EnglishGlosser

    final_translations = phase12_data.get('final_translations', {})

    builder = InterlinearBuilder(phase7_data, phase12_data, None)

    folio_data = {}
    total_tokens = 0

    for folio_id, final_text in final_translations.items():
        eva_tokens = eva_by_folio.get(folio_id, [])
        english_text = english_translations.get(folio_id, '')
        tokens = builder.build_folio(folio_id, eva_tokens, final_text, english_text)
        folio_data[folio_id] = tokens
        total_tokens += len(tokens)

    html = _render_full_html(folio_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    file_size_kb = os.path.getsize(output_path) / 1024

    if verbose:
        print(f'  → {len(folio_data)} folios, {total_tokens} tokens, {file_size_kb:.1f} KB')

    return {
        'folios_rendered': len(folio_data),
        'tokens_rendered': total_tokens,
        'file_size_kb': round(file_size_kb, 1),
    }
