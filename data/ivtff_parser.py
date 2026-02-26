"""
IVTFF Parser
=============
Parses IVTFF (Intermediate Voynich Transliteration File Format) files
as defined by René Zandbergen (version 2.0+).

Handles:
- Page headers with variables ($Q=quire, $L=language, $H=hand, $I=illustration type)
- Locus identifiers (paragraph text, labels, circular text, radial text)
- EVA transliterated text with word separators (periods)
- Uncertain readings [a:o], drawing breaks <->, alignment marks <~>
- Comments (# lines and inline <! > blocks)

Primary input: ZL_ivtff_2b.txt from https://www.voynich.nu/data/
"""

import re
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class VoynichPage:
    """Represents a single page (folio) of the manuscript."""

    def __init__(self, folio: str):
        self.folio = folio          # e.g. 'f1r', 'f72v3'
        self.quire = 0              # $Q variable
        self.language = ''          # $L: 'A' or 'B' (Currier)
        self.hand = 0               # $H: LFD scribe hand (1-5)
        self.illustration = ''      # $I: illustration type
        self.page_num = 0           # $P: page position
        self.cluster = ''           # $C: cluster assignment
        self.loci: List[VoynichLocus] = []
        self.comments: List[str] = []

    @property
    def all_text(self) -> str:
        """All transliterated text on this page, space-separated."""
        return ' '.join(loc.clean_text for loc in self.loci if loc.clean_text)

    @property
    def all_tokens(self) -> List[str]:
        """All tokens on this page."""
        return self.all_text.split()

    @property
    def paragraph_text(self) -> str:
        """Only paragraph (running) text, excluding labels and circular text."""
        return ' '.join(
            loc.clean_text for loc in self.loci
            if loc.locus_type.startswith('P') and loc.clean_text
        )

    @property
    def section(self) -> str:
        """Infer section from illustration type and folio."""
        type_map = {
            'H': 'herbal',
            'A': 'astronomical',
            'B': 'biological',
            'C': 'cosmological',
            'P': 'pharmaceutical',
            'S': 'recipes',
            'T': 'text_only',
            'Z': 'zodiac',
        }
        return type_map.get(self.illustration, 'unknown')


class VoynichLocus:
    """A single text item (line/label/circular text) on a page."""

    def __init__(self, locus_id: str, locus_type: str, raw_text: str):
        self.locus_id = locus_id    # e.g. 'f1r.P1.1'
        self.locus_type = locus_type  # P=paragraph, L=label, C=circular, R=radial
        self.raw_text = raw_text
        self._clean = None

    @property
    def clean_text(self) -> str:
        """Text with annotations stripped, periods→spaces, cleaned."""
        if self._clean is None:
            self._clean = clean_eva_text(self.raw_text)
        return self._clean


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_eva_text(raw: str) -> str:
    """
    Clean raw IVTFF text into pure EVA tokens.
    
    Conversions:
    - Periods (word separators) → spaces
    - Remove <-> (drawing breaks)
    - Remove <~> (alignment marks)
    - Remove <! ... > comments
    - Resolve uncertain readings [a:o] → take first option
    - Remove capitalisation (Sh → sh)
    - Remove curly-brace ligature markers {ao} → ao
    - Remove line-end markers (- = <$>)
    - Strip * ? markers for uncertain characters
    - Remove , (comma) spacing markers
    """
    text = raw

    # Remove inline comments
    text = re.sub(r'<![^>]*>', '', text)

    # Remove drawing breaks and alignment marks
    text = text.replace('<->', ' ')
    text = text.replace('<~>', ' ')
    text = text.replace('<$>', '')

    # Remove line-end markers
    text = text.rstrip('-=')

    # Resolve uncertain readings: [a:o] → a (take first option)
    text = re.sub(r'\[([^:\]]*?):[^\]]*?\]', r'\1', text)
    # Also handle [ao] format (no colon) → ao
    text = re.sub(r'\[([^\]]*?)\]', r'\1', text)

    # Remove ligature braces
    text = re.sub(r'\{([^}]*)\}', r'\1', text)

    # Lowercase (handles capitalised EVA: Sh → sh)
    text = text.lower()

    # Convert periods to spaces
    text = text.replace('.', ' ')

    # Remove uncertain markers
    text = text.replace('*', '')
    text = text.replace('?', '')
    text = text.replace(',', '')
    text = text.replace("'", '')

    # Remove @ high-ascii codes (rare characters)
    text = re.sub(r'@\d+;?', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove any remaining non-EVA characters
    # Keep only: a-z and spaces
    text = re.sub(r'[^a-z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================================
# IVTFF PARSER
# ============================================================================

def parse_ivtff(filepath: str, verbose: bool = False) -> Dict[str, VoynichPage]:
    """
    Parse an IVTFF transliteration file into structured VoynichPage objects.
    
    Parameters:
        filepath: Path to .txt IVTFF file (e.g., ZL_ivtff_2b.txt)
        verbose: Print parsing statistics
    
    Returns:
        Dict[folio_id, VoynichPage]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IVTFF file not found: {filepath}")

    pages = {}
    current_page = None
    line_count = 0
    locus_count = 0
    skipped = 0

    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        for raw_line in fh:
            line_count += 1
            line = raw_line.rstrip('\n\r')

            # Skip empty lines
            if not line.strip():
                continue

            # Skip file header
            if line.startswith('#=IVTFF') or line.startswith('#='):
                continue

            # Comment lines
            if line.startswith('#'):
                if current_page:
                    current_page.comments.append(line[1:].strip())
                continue

            # Page start: <f1r> or <f1r> <! $Q=1 $P=1 $I=T $L=A $C=1 $H=1 >
            page_match = re.match(r'^<(f\d+[rv]\d?)>\s*(.*)', line)
            if page_match:
                folio = page_match.group(1)
                rest = page_match.group(2)

                current_page = VoynichPage(folio)
                pages[folio] = current_page

                # Parse page variables from dedicated comment
                _parse_page_variables(current_page, rest)
                continue

            # Dedicated comment with page variables (sometimes on next line)
            if line.strip().startswith('<!') and current_page:
                _parse_page_variables(current_page, line)
                continue

            # Data line: locus identifier + transliterated text
            # Format: <f1r.P1.1;H> text.text.text-
            locus_match = re.match(r'^<([^>]+)>\s*(.*)', line)
            if locus_match:
                locus_id = locus_match.group(1)
                text = locus_match.group(2)

                if not current_page:
                    skipped += 1
                    continue

                # Extract locus type from identifier
                # e.g., f1r.P1.1;H → type = P (paragraph)
                locus_type = _extract_locus_type(locus_id)

                locus = VoynichLocus(locus_id, locus_type, text)
                current_page.loci.append(locus)
                locus_count += 1
                continue

            # Lines without angle brackets might be continuation text
            # or unlabeled content — skip these
            skipped += 1

    if verbose:
        total_tokens = sum(len(p.all_tokens) for p in pages.values())
        total_chars = sum(len(p.all_text.replace(' ', '')) for p in pages.values())
        print(f"  Parsed {filepath}")
        print(f"  Pages: {len(pages)}")
        print(f"  Loci: {locus_count}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total characters: {total_chars}")
        print(f"  Lines read: {line_count}, skipped: {skipped}")

        # Language distribution
        lang_counts = Counter(p.language for p in pages.values() if p.language)
        print(f"  Language distribution: {dict(lang_counts)}")

        # Hand distribution
        hand_counts = Counter(p.hand for p in pages.values() if p.hand)
        print(f"  Hand (scribe) distribution: {dict(hand_counts)}")

        # Section distribution
        section_counts = Counter(p.section for p in pages.values())
        print(f"  Section distribution: {dict(section_counts)}")

    return pages


def _parse_page_variables(page: VoynichPage, text: str):
    """Extract $Q, $L, $H, $I, $P, $C variables from page header."""
    var_patterns = {
        'Q': (r'\$Q=(\d+)', 'quire', int),
        'P': (r'\$P=(\d+)', 'page_num', int),
        'L': (r'\$L=([AB])', 'language', str),
        'H': (r'\$H=(\d+)', 'hand', int),
        'I': (r'\$I=(\w)', 'illustration', str),
        'C': (r'\$C=(\w+)', 'cluster', str),
    }

    for var_name, (pattern, attr, conv) in var_patterns.items():
        match = re.search(pattern, text)
        if match:
            setattr(page, attr, conv(match.group(1)))


def _extract_locus_type(locus_id: str) -> str:
    """
    Extract locus type from an IVTFF locus identifier.
    
    Examples:
        f1r.P1.1;H  → P (paragraph)
        f1r.L1.1;H  → L (label)
        f67r.C1.1;H → C (circular)
        f67r.R1.1;H → R (radial)
    """
    # Look for the type letter after the folio reference
    match = re.search(r'\.([PLCRTX])\d', locus_id)
    if match:
        return match.group(1)

    # Fallback: try just finding a capital letter after a period
    match = re.search(r'\.([A-Z])', locus_id)
    if match:
        return match.group(1)

    return 'P'  # default to paragraph


# ============================================================================
# CORPUS BUILDER
# ============================================================================

class VoynichCorpus:
    """
    High-level corpus interface built from parsed IVTFF data.
    Provides filtered access by section, scribe, language, quire.
    """

    def __init__(self, pages: Dict[str, VoynichPage]):
        self.pages = pages

    @classmethod
    def from_file(cls, filepath: str, verbose: bool = False) -> 'VoynichCorpus':
        """Load corpus from an IVTFF file."""
        pages = parse_ivtff(filepath, verbose=verbose)
        return cls(pages)

    def get_text(self,
                 section: Optional[str] = None,
                 language: Optional[str] = None,
                 hand: Optional[int] = None,
                 quire: Optional[int] = None,
                 paragraph_only: bool = True) -> str:
        """Get filtered text as a single string."""
        parts = []
        for page in self.pages.values():
            if section and page.section != section:
                continue
            if language and page.language != language:
                continue
            if hand and page.hand != hand:
                continue
            if quire and page.quire != quire:
                continue

            text = page.paragraph_text if paragraph_only else page.all_text
            if text:
                parts.append(text)

        return ' '.join(parts)

    def get_tokens(self, **kwargs) -> List[str]:
        """Get filtered tokens."""
        return self.get_text(**kwargs).split()

    def get_page(self, folio: str) -> Optional[VoynichPage]:
        """Get a specific page."""
        return self.pages.get(folio)

    def get_pages_by_section(self, section: str) -> List[VoynichPage]:
        """Get all pages in a section."""
        return [p for p in self.pages.values() if p.section == section]

    def get_pages_by_hand(self, hand: int) -> List[VoynichPage]:
        """Get all pages by a specific scribe."""
        return [p for p in self.pages.values() if p.hand == hand]

    def get_scribe_transitions(self) -> List[Tuple[VoynichPage, VoynichPage]]:
        """Find adjacent pages where the scribe hand changes."""
        sorted_pages = sorted(
            [p for p in self.pages.values() if p.hand > 0],
            key=lambda p: (p.quire, p.page_num)
        )
        transitions = []
        for i in range(len(sorted_pages) - 1):
            p1, p2 = sorted_pages[i], sorted_pages[i + 1]
            if p1.hand != p2.hand:
                transitions.append((p1, p2))
        return transitions

    def get_folio_sequence(self, quire_order: Optional[List[int]] = None) -> List[VoynichPage]:
        """
        Get pages in a specific quire ordering.
        Default: current binding order.
        """
        if quire_order is None:
            return sorted(
                self.pages.values(),
                key=lambda p: (p.quire, p.page_num)
            )

        ordered = []
        for q in quire_order:
            quire_pages = sorted(
                [p for p in self.pages.values() if p.quire == q],
                key=lambda p: p.page_num
            )
            ordered.extend(quire_pages)
        return ordered

    def summary(self) -> Dict:
        """Return corpus summary statistics."""
        all_tokens = self.get_tokens(paragraph_only=False)
        return {
            'total_pages': len(self.pages),
            'total_tokens': len(all_tokens),
            'total_characters': sum(len(t) for t in all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'type_token_ratio': len(set(all_tokens)) / max(1, len(all_tokens)),
            'languages': dict(Counter(p.language for p in self.pages.values() if p.language)),
            'hands': dict(Counter(p.hand for p in self.pages.values() if p.hand)),
            'sections': dict(Counter(p.section for p in self.pages.values())),
            'quires': sorted(set(p.quire for p in self.pages.values() if p.quire)),
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def load_corpus(data_dir: str = 'data/corpus', verbose: bool = True) -> VoynichCorpus:
    """
    Load the best available corpus from the data directory.
    Prefers: ZL3b-n.txt > RF1b-e.txt > IT2a-n.txt
    """
    preferred = [
        'ZL3b-n.txt',
        'RF1b-e.txt',
        'IT2a-n.txt',
    ]

    for filename in preferred:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            if verbose:
                print(f"Loading corpus from {filepath}...")
            return VoynichCorpus.from_file(filepath, verbose=verbose)

    raise FileNotFoundError(
        f"No IVTFF files found in {data_dir}. Download with:\n"
        f"  curl https://www.voynich.nu/data/ZL_ivtff_2b.txt -o {data_dir}/ZL_ivtff_2b.txt"
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/corpus/ZL_ivtff_2b.txt'
    corpus = VoynichCorpus.from_file(path, verbose=True)
    print("\nSummary:")
    for k, v in corpus.summary().items():
        print(f"  {k}: {v}")

    print("\nFirst 10 tokens from Language A:")
    tokens_a = corpus.get_tokens(language='A')
    print(f"  {tokens_a[:10]}")

    print("\nFirst 10 tokens from Language B:")
    tokens_b = corpus.get_tokens(language='B')
    print(f"  {tokens_b[:10]}")