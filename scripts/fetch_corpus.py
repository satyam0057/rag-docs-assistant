"""Download 5 FastAPI tutorial pages into data/docs.

Newer FastAPI docs reference code with  {* ../../docs_src/xxx.py hl[..] *}  instead of
inline code blocks. We inline those files so the corpus contains the actual code.
"""
import re
import pathlib
import requests

RAW = "https://raw.githubusercontent.com/fastapi/fastapi/master/"
TUTORIAL = RAW + "docs/en/docs/tutorial/"
PAGES = ["first-steps.md", "path-params.md", "query-params.md", "body.md", "dependencies/index.md"]
INCLUDE = re.compile(r"\{\*\s*(\S+?)\s*(?:hl\[[^\]]*\]|ln\[[^\]]*\]|title=\"[^\"]*\"|\s)*\*\}")

out = pathlib.Path(__file__).resolve().parent.parent / "data" / "docs"
out.mkdir(parents=True, exist_ok=True)


def inline_code(match: re.Match) -> str:
    path = match.group(1)
    if "docs_src/" not in path:
        return ""
    rel = "docs_src/" + path.split("docs_src/", 1)[1]
    try:
        r = requests.get(RAW + rel, timeout=30)
        r.raise_for_status()
        return f"```python\n{r.text.strip()}\n```"
    except Exception:
        return ""


for p in PAGES:
    r = requests.get(TUTORIAL + p, timeout=30)
    r.raise_for_status()
    text = INCLUDE.sub(inline_code, r.text)
    name = "fastapi_" + p.replace("/", "_")
    (out / name).write_text(text, encoding="utf-8")
    print("saved", name, f"({len(text)} chars)")
