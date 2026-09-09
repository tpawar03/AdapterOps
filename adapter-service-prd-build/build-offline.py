#!/usr/bin/env python3
"""
Regenerate the self-contained offline copy of the PRD from source.html.

  source.html                            the single source of truth; also what gets
                                         published as the Artifact (lean: Google
                                         Fonts link + a file://-guarded mermaid
                                         CDN loader)
  ../adapter-service-prd.html            output: everything inlined, zero network
                                         (pass a path as argv[1] to override)

Run:  python3 build-offline.py
Both files must be regenerated together whenever source.html changes.
"""
import base64
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, 'source.html')
LIB  = os.path.join(HERE, 'vendor', 'mermaid-10.9.1.min.js')
FDIR = os.path.join(HERE, 'vendor', 'fonts')

# The deliverables sit in the repo root, one level up from this folder. Keeping the
# path relative means the build still works wherever the repo is cloned or moved.
# Override with:  python3 build-offline.py /some/other/path.html
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, '..', 'adapter-service-prd.html'))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

INIT = '''
<script>
/* Offline build: mermaid 10.9.1 is inlined above; render unconditionally. */
(function(){
  if (!document.querySelector('pre.mermaid') || !window.mermaid) return;
  var stamped = document.documentElement.getAttribute('data-theme');
  var dark = stamped === 'dark' ||
    (stamped !== 'light' && window.matchMedia &&
     window.matchMedia('(prefers-color-scheme: dark)').matches);
  try {
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      theme: dark ? 'dark' : 'default',
      fontFamily: '"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif',
      flowchart: { htmlLabels: true, curve: 'basis', useMaxWidth: true }
    });
    window.mermaid.run({ querySelector: 'pre.mermaid' });
  } catch (err) {
    var d = document.querySelector('.diagram');
    if (!d) return;
    var el = document.createElement('p');
    el.textContent = 'Diagram could not be rendered: ' + err.message + '. The source is below.';
    el.style.cssText = 'font-family:var(--mono);font-size:.7rem;color:var(--muted);margin:0 0 .7rem';
    d.insertBefore(el, d.firstChild);
  }
})();
</script>
'''


def font_css():
    """One @font-face per unique woff2. Variable files covering several weights
    are declared once with a weight range, so no payload is embedded twice."""
    with open(os.path.join(FDIR, 'manifest.json'), encoding='utf-8') as fh:
        manifest = json.load(fh)
    rules = []
    for f in manifest:
        with open(os.path.join(FDIR, f['file']), 'rb') as fh:
            data = fh.read()
        if data[:4] != b'wOF2':
            sys.exit('not a woff2 file: ' + f['file'])
        uri = 'data:font/woff2;base64,' + base64.b64encode(data).decode('ascii')
        rules.append(
            f"@font-face {{\n  font-family: '{f['family']}';\n"
            f"  font-style: {f['style']};\n"
            f"  font-weight: {f['weight']};\n  font-display: swap;\n"
            f"  src: url({uri}) format('woff2');\n}}")
    header = ("/* Latin subset of IBM Plex Mono / Sans / Sans Condensed and Source Serif 4,\n"
              "   embedded as woff2 data URIs so the page needs no network.\n"
              "   Variable files covering several weights are declared once with a weight range. */\n")
    return header + '\n'.join(rules)


def main():
    with open(SRC, encoding='utf-8') as fh:
        src = fh.read()

    # 1. fonts: drop the preconnect + Google Fonts link, inline @font-face rules
    src = src.replace('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n', '')
    m = re.search(r'<link rel="stylesheet" href="https://fonts\.googleapis\.com/[^"]+">\n', src)
    if not m:
        sys.exit('source.html: Google Fonts <link> not found — did the head change?')
    src = src.replace(m.group(0), '<style>\n' + font_css() + '\n</style>\n')

    # 2. mermaid: strip the CDN loader, inline the library + an unconditional init
    start_marker = '\n/* Mermaid: the published artifact'
    end_marker   = '  document.head.appendChild(s);\n})();\n</script>'
    if start_marker not in src or end_marker not in src:
        sys.exit('source.html: mermaid CDN loader not found — did the script block change?')
    start = src.index(start_marker)
    end   = src.index(end_marker)
    if src[end + len(end_marker):].strip():
        sys.exit('source.html: unexpected content after the final </script>')
    lean = src[:start] + '\n</script>'

    with open(LIB, encoding='utf-8') as fh:
        lib = fh.read()
    if '<script' in lib or '</script' in lib:
        sys.exit('vendored mermaid contains a script tag — cannot inline raw')

    out = lean + '\n<script>' + lib + '</script>\n' + INIT
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(out)

    leftover = [u for u in re.findall(r'https?://[^\s"\')]+', out)
                if any(h in u for h in ('gstatic', 'googleapis', 'cdnjs'))]
    if leftover:
        sys.exit('offline build still references: ' + ', '.join(sorted(set(leftover))))

    size_mb = len(out.encode()) / 1024 / 1024
    print(f'wrote {OUT}  ({size_mb:.2f} MB, {out.count("@font-face")} @font-face, '
          '0 external refs)')


if __name__ == '__main__':
    main()
