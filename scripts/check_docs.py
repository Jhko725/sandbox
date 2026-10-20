"""Enforce the documentation bookkeeping rules of CLAUDE.md.

Checks (all under docs/):
  1. every *.md has a frontmatter block with `type`, `status`, `updated` (ISO date);
  2. `status` is one of the allowed values; a change document also has `branch`,
     an experiment document also has `id` and `affects`;
  3. docs/index.md links every document (by relative path);
  4. the Done ledger in docs/roadmap.md is chronological (dates non-decreasing);
  5. --files: every file in `git diff --name-only <base>...HEAD` is covered by a
     backtick-quoted entry in the change document's `## Files` section. Entries may
     be exact paths, basenames, directory prefixes, or globs with `*`, `**`, `…`
     and `{a,b}` brace lists (e.g. `docs/decisions/0001…0009-*.md`,
     `notebooks/**/*.ipynb`).

Usage:
  uv run python scripts/check_docs.py            # 1-4
  uv run python scripts/check_docs.py --files    # 1-5 (needs a branch and its change document)
  uv run python scripts/check_docs.py --files --base origin/master
Exit status 1 on any problem; problems are printed one per line.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ALLOWED_STATUS = {
    "current",
    "planned",
    "in-progress",
    "landed",
    "merged",
    "agreed",
    "accepted",
    "amended",
    "tentative",
    "historical",
    "running",
    "concluded",
    "superseded",
    "template",
}
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def frontmatter(path: Path) -> dict[str, str] | None:
    m = FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not m:
        return None
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def check_frontmatter(problems: list[str]) -> None:
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(ROOT)
        fm = frontmatter(path)
        if fm is None:
            problems.append(f"{rel}: missing frontmatter block")
            continue
        for key in ("type", "status", "updated"):
            if key not in fm:
                problems.append(f"{rel}: frontmatter lacks `{key}`")
        status = fm.get("status", "").split(";")[0].strip()
        if status and status not in ALLOWED_STATUS:
            problems.append(f"{rel}: status `{status}` not in {sorted(ALLOWED_STATUS)}")
        try:
            date.fromisoformat(fm.get("updated", ""))
        except ValueError:
            problems.append(f"{rel}: `updated` is not an ISO date")
        if fm.get("type") == "change" and "branch" not in fm:
            problems.append(f"{rel}: change document lacks `branch`")
        if fm.get("type") == "experiment":
            for key in ("id", "affects"):
                if key not in fm:
                    problems.append(f"{rel}: experiment document lacks `{key}`")


def check_index(problems: list[str]) -> None:
    index = DOCS / "index.md"
    if not index.exists():
        problems.append("docs/index.md missing")
        return
    text = index.read_text(encoding="utf-8")
    for path in sorted(DOCS.rglob("*.md")):
        if path == index:
            continue
        rel = path.relative_to(DOCS).as_posix()
        if rel not in text:
            problems.append(f"docs/index.md does not list {rel}")


def check_done_ledger(problems: list[str]) -> None:
    roadmap = DOCS / "roadmap.md"
    if not roadmap.exists():
        problems.append("docs/roadmap.md missing")
        return
    text = roadmap.read_text(encoding="utf-8")
    if "## Done" not in text:
        problems.append("docs/roadmap.md has no `## Done` ledger")
        return
    ledger = text.split("## Done", 1)[1]
    dates = []
    for line in ledger.splitlines():
        m = re.match(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|", line)
        if m:
            dates.append(m.group(1))
    if dates != sorted(dates):
        problems.append(
            "docs/roadmap.md Done ledger is not chronological (append at the end)"
        )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def _expand_braces(token: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", token)
    if not m:
        return [token]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out.extend(_expand_braces(token[: m.start()] + alt.strip() + token[m.end() :]))
    return out


def _listed_patterns(change_doc_text: str) -> list[str]:
    """Backtick-quoted tokens in the `## Files` section, with {a,b} expanded and … → *."""
    if "## Files" not in change_doc_text:
        return []
    section = change_doc_text.split("## Files", 1)[1].split("\n## ", 1)[0]
    patterns: list[str] = []
    for token in re.findall(r"`([^`]+)`", section):
        token = token.strip().replace("…", "*")
        patterns.extend(_expand_braces(token))
    return patterns


def _is_listed(path: str, patterns: list[str]) -> bool:
    name = Path(path).name
    for pat in patterns:
        if "*" in pat or "?" in pat:
            if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(name, pat):
                return True
            # allow a directory prefix pattern such as `notebooks/**/*.ipynb` or `docs/decisions/*`
            if fnmatch.fnmatch(path, pat.replace("**/", "")):
                return True
        elif pat == path or pat == name or path.startswith(pat.rstrip("/") + "/"):
            return True
    return False


def check_files_vs_diff(problems: list[str], base: str) -> None:
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if branch in {"master", "main", "HEAD"}:
        problems.append("--files needs a feature branch (not master/detached)")
        return
    docs = sorted(DOCS.glob(f"changes/*-{branch}.md"))
    if not docs:
        problems.append(
            f"no change document for branch `{branch}` (docs/changes/<date>-{branch}.md)"
        )
        return
    change_doc = docs[-1]
    patterns = _listed_patterns(change_doc.read_text(encoding="utf-8"))
    try:
        diff = git("diff", "--name-only", f"{base}...HEAD")
    except subprocess.CalledProcessError:
        problems.append(f"cannot diff against `{base}` (fetch it first?)")
        return
    changed = {p for p in diff.split() if p}
    changed.discard(change_doc.relative_to(ROOT).as_posix())
    for path in sorted(changed):
        if not _is_listed(path, patterns):
            problems.append(
                f"{change_doc.relative_to(ROOT)}: changed file not listed in Files: {path}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--files",
        action="store_true",
        help="also check the change document against git diff",
    )
    ap.add_argument(
        "--base",
        default="origin/master",
        help="base ref for --files (default origin/master)",
    )
    args = ap.parse_args()

    problems: list[str] = []
    check_frontmatter(problems)
    check_index(problems)
    check_done_ledger(problems)
    if args.files:
        check_files_vs_diff(problems, args.base)

    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
