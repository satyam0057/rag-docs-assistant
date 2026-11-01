"""Small evaluation: run the graph on test questions and check expected behaviour.
Usage: python scripts/evaluate.py   (after ingesting the corpus)
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.graph.workflow import graph  # noqa: E402

# (question, expected_behaviour, keyword that should appear in answer or None)
CASES = [
    ("How do I declare a path parameter in FastAPI?", "answer", "path"),
    ("How do I define optional query parameters?", "answer", "query"),
    ("How do I receive a JSON request body with Pydantic?", "answer", "BaseModel"),
    ("What is a dependency in FastAPI?", "answer", "depend"),
    ("How do I run a FastAPI app?", "answer", "fastapi dev"),
    ("How do I validate the type of a path parameter?", "answer", None),
    ("How do I bake sourdough bread?", "idk", None),
    ("Who won the 2018 football world cup?", "idk", None),
    ("What is the capital of France?", "idk", None),
    ("how do i send stuff to the api", "answer", None),  # vague
]


def is_idk(text: str) -> bool:
    return "don't know" in text.lower() or "do not know" in text.lower()


rows, passed = [], 0
for q, expected, kw in CASES:
    out = graph.invoke({"question": q, "documents": [], "retries": 0})
    ans = out["generation"]
    got = "idk" if is_idk(ans) else "answer"
    ok = got == expected and (kw is None or got == "idk" or kw.lower() in ans.lower())
    passed += ok
    rows.append((q[:48], expected, got, out.get("retries", 0), out.get("grounded"), "PASS" if ok else "FAIL"))
    time.sleep(1)  # be gentle with free-tier rate limits

print(f"\n{'question':50}{'expected':10}{'got':8}{'retries':8}{'grounded':9}result")
for r in rows:
    print(f"{r[0]:50}{r[1]:10}{r[2]:8}{r[3]!s:8}{r[4]!s:9}{r[5]}")
print(f"\n{passed}/{len(CASES)} passed")

if passed != len(CASES):
    sys.exit(1)
