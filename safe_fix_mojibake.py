#!/usr/bin/env python3
"""
safe_fix_mojibake.py
- Safely fixes mojibake ONLY in Python comments and string literals.
- Default: dry-run (no writes). Use --apply to write changes.
- Creates .bak backups for modified files.
- Verifies AST parse before writing; if parse fails, file is not changed.
"""

import sys, os, argparse, tokenize, io, ast
from pathlib import Path

def build_mojibake_pairs():
    """Generate mojibake variants dynamically to avoid pasting broken literals."""
    targets = ['\u2192', '\u2013', '\u2014', '\u2018', '\u2019', '\u201C', '\u201D', '\u2026', '\u2022']
    # →, – , — , ‘ , ’ , “ , ” , … , •
    pairs = set()
    for t in targets:
        good = t
        bad1 = t.encode('utf-8').decode('cp1252', errors='ignore')
        bad2 = bad1.encode('utf-8', errors='ignore').decode('cp1252', errors='ignore')
        if bad1 and bad1 != good:
            pairs.add((bad1, good))
        if bad2 and bad2 not in (bad1, good):
            pairs.add((bad2, good))
    # common one-directional forms sometimes seen
    extras = {
        'â†’':'\u2192','â€“':'\u2013','â€”':'\u2014',
        'â€˜':'\u2018','â€™':'\u2019','â€œ':'\u201C','â€\x9d':'\u201D','â€¦':'\u2026'
    }
    for b,g in extras.items():
        pairs.add((b,g))
    # return in stable order
    return sorted(pairs, key=lambda x: (-len(x[0]), x[0]))

PAIRS = build_mojibake_pairs()

def fix_text_in_tokens(src_bytes):
    """Replace mojibake only in STRING and COMMENT tokens; keep others intact."""
    out_tokens = []
    changed = False
    for tok in tokenize.tokenize(io.BytesIO(src_bytes).readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            val = tok.string
            new_val = val
            for bad, good in PAIRS:
                new_val = new_val.replace(bad, good)
            if new_val != val:
                changed = True
                tok = tokenize.TokenInfo(tok.type, new_val, tok.start, tok.end, tok.line)
        out_tokens.append(tok)
    new_src = tokenize.untokenize(out_tokens)
    return new_src, changed

def process_file(path: Path, apply=False):
    try:
        raw = path.read_bytes()
    except Exception as e:
        return False, f"read error: {e}"

    new_src, changed = fix_text_in_tokens(raw)

    if not changed:
        return False, "no changes"

    # Ensure AST is still valid Python (and was before)
    try:
        ast.parse(raw.decode('utf-8', errors='replace'))
    except Exception as e:
        # original was not valid; skip writing
        return False, f"original not parseable: {e}"
    try:
        ast.parse(new_src.decode('utf-8'))
    except Exception as e:
        return False, f"post-fix not parseable: {e}"

    if apply:
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            path.replace(bak)               # move original to .bak
            path.write_bytes(new_src)       # write fixed UTF-8
        except Exception as e:
            # try to restore original
            if bak.exists():
                bak.replace(path)
            return False, f"write error: {e}"
        return True, f"fixed (backup: {bak.name})"
    else:
        return True, "would fix (dry-run)"

def main():
    ap = argparse.ArgumentParser(description="Safely fix mojibake in .py comments and strings.")
    ap.add_argument("--root", default=".", help="Project root (default: current dir)")
    ap.add_argument("--apply", action="store_true", help="Write changes (otherwise dry-run)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    py_files = list(root.rglob("*.py"))

    changed_any = False
    for p in py_files:
        ok, msg = process_file(p, apply=args.apply)
        if ok:
            changed_any = True
            print(f"[{p}] {msg}")
    if not changed_any:
        print("No mojibake issues detected, or nothing limited to comments/strings.")

if __name__ == "__main__":
    sys.exit(main())
