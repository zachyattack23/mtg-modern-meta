#!/usr/bin/env python3
"""Fold the dashboard into one self-contained HTML file.

web/index.html is written for the Artifact host, which wraps it in a document
skeleton and supplies the charset. A file that has to survive being emailed,
dropped on an arbitrary static host, or opened straight off disk needs to carry
that skeleton itself -- without an explicit charset the em-dashes and the
Chinese event name render as mojibake on any host that omits the header.

Inlining data.js also removes the second request, so the file works from a
file:// URL where a fetch would be blocked.

    python3 scripts/build_standalone.py            # -> dist/modern-ceiling-report.html
"""

from __future__ import annotations

import argparse
import datetime
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

SKELETON_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
  :root { color-scheme: light; }
  html { -webkit-text-size-adjust: 100%; }
  body { margin: 0; font: 14px system-ui, -apple-system, sans-serif; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
"""

SKELETON_TAIL = """
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "dist" / "modern-ceiling-report.html")
    args = ap.parse_args()

    page = (WEB / "index.html").read_text(encoding="utf-8")
    data = (WEB / "data.js").read_text(encoding="utf-8")

    # </script> anywhere inside the payload would close the inline block early.
    data = data.replace("</script>", "<\\/script>")

    if '<script src="data.js"></script>' not in page:
        raise SystemExit("web/index.html no longer references data.js as expected")
    page = page.replace('<script src="data.js"></script>',
                        f"<script>\n{data}</script>")

    stamp = datetime.date.today().isoformat()
    note = (f"\n<!-- Modern Ceiling Report - self-contained build {stamp}.\n"
            f"     Open directly, host anywhere, no server required. -->\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(SKELETON_HEAD + page + note + SKELETON_TAIL,
                        encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
