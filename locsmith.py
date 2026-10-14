#!/usr/bin/env python3
"""locsmith: private, local pre-translation of game strings with an open-weight model.

Talks to any OpenAI-compatible endpoint (Ollama, llama.cpp `llama-server`, vLLM, LM Studio).
No dependencies beyond the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

LANG_NAMES = {
    "en": "English", "ru": "Russian", "ar": "Arabic", "de": "German", "fr": "French",
    "es": "Spanish", "pt-br": "Brazilian Portuguese", "it": "Italian", "pl": "Polish",
    "tr": "Turkish", "uk": "Ukrainian", "ja": "Japanese", "ko": "Korean",
    "zh-cn": "Simplified Chinese", "zh-tw": "Traditional Chinese",
}

# One alternation so overlapping matches are tokenized identically in source and target.
TOKEN_RE = re.compile(
    r"\{\{|\}\}"                                         # escaped braces
    r"|\{[A-Za-z0-9_.:\-]*\}"                            # {0} {name} {count:d}
    r"|%(?:\d+\$)?[-+ #0]*\d*(?:\.\d+)?[sdifuxXeEgGc%]"  # printf: %s %d %1$s %.2f %%
    r"|</?[A-Za-z][A-Za-z0-9]*(?:[=\s][^<>]*)?/?>"       # <b> </b> <color=#f00> <br/>
    r"|\[/?[A-Za-z]+(?:=[^\]]*)?\]"                      # [b] [color=red] [/color]
    r"|\\[nt]"                                           # literal \n \t escapes
)


@dataclass
class Item:
    key: str
    source: str
    context: str = ""
    max_len: int | None = None


@dataclass
class GlossaryEntry:
    source: str
    target: str
    strict: bool = False


@dataclass
class Result:
    item: Item
    target: str
    issues: list[str] = field(default_factory=list)
    attempts: int = 0
    cached: bool = False

    @property
    def status(self) -> str:
        return "ok" if not self.issues else "review"


# ---------------------------------------------------------------- validation

def tokens(text: str) -> Counter:
    return Counter(TOKEN_RE.findall(text))


def term_in_source(term: str, text: str) -> bool:
    # Allow English plural/possessive on the source side: "Aether Shard" matches "Aether Shards".
    pattern = r"(?<!\w)" + re.escape(term) + r"(?:'s|es|s)?(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def term_in_target(entry: GlossaryEntry, text: str) -> bool:
    low = text.lower()
    if entry.strict:
        return entry.target.lower() in low
    # Inflected languages (ru, uk, pl...) change word endings, so match per-word stems.
    for word in entry.target.lower().split():
        stem = word[: max(3, len(word) - 2)] if len(word) >= 5 else word
        if stem not in low:
            return False
    return True


def validate(item: Item, target: str | None, glossary: list[GlossaryEntry]) -> list[str]:
    if target is None:
        return ["missing in model output"]
    if not target.strip():
        return ["empty translation"]
    issues = []
    src_t, tgt_t = tokens(item.source), tokens(target)
    if src_t != tgt_t:
        lost = list((src_t - tgt_t).elements())
        extra = list((tgt_t - src_t).elements())
        if lost:
            issues.append(f"lost placeholders/tags: {' '.join(lost)}")
        if extra:
            issues.append(f"unexpected placeholders/tags: {' '.join(extra)}")
    if item.source.count("\n") != target.count("\n"):
        issues.append("line break count changed")
    if item.max_len and len(target) > item.max_len:
        issues.append(f"too long: {len(target)} > {item.max_len} chars")
    if target.strip() == item.source.strip() and re.search(r"[A-Za-z]{3,}", item.source):
        issues.append("left untranslated")
    for g in glossary:
        if term_in_source(g.source, item.source) and not term_in_target(g, target):
            issues.append(f"glossary: '{g.source}' must be '{g.target}'")
    return issues


# ---------------------------------------------------------------- model I/O

LLM = Callable[[list[dict]], str]


def make_llm(base_url: str, model: str, api_key: str | None, temperature: float, timeout: int,
             reasoning: str | None = None) -> LLM:
    url = base_url.rstrip("/") + "/chat/completions"

    def call(messages: list[dict]) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
        # Reasoning models (e.g. gemma4) otherwise spend hundreds of "thinking" tokens per batch.
        # reasoning_effort is an OpenAI-standard field; servers that don't support it ignore it.
        if reasoning:
            payload["reasoning_effort"] = reasoning
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
        except urllib.error.URLError as e:
            sys.exit(f"error: cannot reach model endpoint {url}: {e}")
        return data["choices"][0]["message"]["content"]

    return call


def parse_json_object(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def system_prompt(src: str, tgt: str, style: str) -> str:
    return (
        f"You are a senior video game localizer translating UI and dialogue strings "
        f"from {src} to {tgt}.\n"
        "Rules:\n"
        "1. Copy every placeholder, tag and escape EXACTLY as in the source: "
        "{0}, {name}, %s, %d, %1$s, <b>, </b>, <color=...>, [b], \\n. Never translate, "
        "reorder inside, or drop them. You may move them within the sentence if grammar requires.\n"
        "2. Use the glossary translations exactly (inflect only if the glossary term is a common noun).\n"
        "3. If max_len is given, the translation must not exceed it; shorten naturally.\n"
        "4. Use the context field to pick the right meaning and tone. Keep it short for UI.\n"
        f"5. Style: {style}\n"
        "Return ONLY a JSON object mapping each key to its translation. No commentary."
    )


def batch_payload(batch: list[Item], glossary: list[GlossaryEntry]) -> str:
    strings = []
    for it in batch:
        d = {"key": it.key, "text": it.source}
        if it.context:
            d["context"] = it.context
        if it.max_len:
            d["max_len"] = it.max_len
        strings.append(d)
    hits = [{"source": g.source, "target": g.target} for g in glossary
            if any(term_in_source(g.source, it.source) for it in batch)]
    return json.dumps({"glossary": hits, "strings": strings}, ensure_ascii=False, indent=1)


def retry_payload(item: Item, previous: str | None, issues: list[str],
                  glossary: list[GlossaryEntry]) -> str:
    base = json.loads(batch_payload([item], glossary))
    base["previous_attempt"] = previous
    base["problems_to_fix"] = issues
    return json.dumps(base, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- pipeline

def translate(items: list[Item], glossary: list[GlossaryEntry], llm: LLM, *,
              src: str, tgt: str, style: str, batch_size: int, retries: int,
              cache: dict[str, dict], log=print) -> list[Result]:
    sysmsg = {"role": "system", "content": system_prompt(src, tgt, style)}
    results: dict[str, Result] = {}
    todo = []
    for it in items:
        c = cache.get(it.key)
        if c and c.get("source") == it.source and c.get("status") == "ok":
            results[it.key] = Result(it, c["target"], cached=True)
        else:
            todo.append(it)
    if len(todo) < len(items):
        log(f"reusing {len(items) - len(todo)} unchanged strings from previous run")

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        out = parse_json_object(llm([sysmsg, {"role": "user", "content": batch_payload(batch, glossary)}]))
        for it in batch:
            tr = out.get(it.key)
            tr = tr if isinstance(tr, str) else None
            res = Result(it, tr or "", validate(it, tr, glossary), attempts=1)
            while res.issues and res.attempts <= retries:
                fix = parse_json_object(llm([sysmsg, {"role": "user",
                                         "content": retry_payload(it, tr, res.issues, glossary)}]))
                tr = fix.get(it.key) if isinstance(fix.get(it.key), str) else tr
                res = Result(it, tr or "", validate(it, tr, glossary), attempts=res.attempts + 1)
            results[it.key] = res
        log(f"translated {min(i + batch_size, len(todo))}/{len(todo)}")
    return [results[it.key] for it in items]


# ---------------------------------------------------------------- files

def load_items(path: str) -> list[Item]:
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [Item(str(k), str(v)) for k, v in data.items()]
        return [Item(str(d["key"]), d["source"], d.get("context", ""),
                     int(d["max_len"]) if d.get("max_len") else None) for d in data]
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [Item(r["key"], r["source"], r.get("context") or "",
                     int(r["max_len"]) if (r.get("max_len") or "").strip() else None)
                for r in csv.DictReader(f)]


def load_glossary(path: str | None) -> list[GlossaryEntry]:
    if not path:
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [GlossaryEntry(r["source"], r["target"],
                              (r.get("strict") or "").strip().lower() in ("1", "true", "yes"))
                for r in csv.DictReader(f)]


def load_cache(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def write_results(path: str, results: list[Result]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "source", "target", "status", "issues"])
        for r in results:
            w.writerow([r.item.key, r.item.source, r.target, r.status, "; ".join(r.issues)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local, private pre-translation of game strings.")
    p.add_argument("input", help="CSV (key,source[,context,max_len]) or JSON")
    p.add_argument("--to", required=True, help="target language code or name, e.g. ru, ar, German")
    p.add_argument("--from", dest="src", default="en", help="source language (default: en)")
    p.add_argument("-o", "--out", help="output CSV (default: <input>.<to>.csv); reused as cache on reruns")
    p.add_argument("-g", "--glossary", help="CSV: source,target[,strict]")
    p.add_argument("--style", default="natural, concise, consistent with a fantasy action game",
                   help="tone/style instruction for the model")
    p.add_argument("--model", default=os.environ.get("LOCSMITH_MODEL", "gemma4:12b"))
    p.add_argument("--base-url", default=os.environ.get("LOCSMITH_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--api-key", default=os.environ.get("LOCSMITH_API_KEY"))
    p.add_argument("--batch", type=int, default=15)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--reasoning", choices=["none", "low", "medium", "high"], default="none",
                   help="reasoning_effort for thinking models (default: none; disables Gemma's "
                        "thinking so a batch costs ~10x fewer tokens)")
    a = p.parse_args(argv)

    items = load_items(a.input)
    glossary = load_glossary(a.glossary)
    out = a.out or f"{os.path.splitext(a.input)[0]}.{a.to}.csv"
    llm = make_llm(a.base_url, a.model, a.api_key, a.temperature, a.timeout, a.reasoning)

    t0 = time.monotonic()
    results = translate(items, glossary, llm,
                        src=LANG_NAMES.get(a.src.lower(), a.src), tgt=LANG_NAMES.get(a.to.lower(), a.to),
                        style=a.style, batch_size=a.batch, retries=a.retries,
                        cache=load_cache(out), log=lambda m: print(m, file=sys.stderr))
    dt = time.monotonic() - t0
    write_results(out, results)

    fresh = [r for r in results if not r.cached]
    ok = sum(r.status == "ok" for r in results)
    fixed = sum(r.attempts > 1 and r.status == "ok" for r in fresh)
    print(f"\n{a.model} @ {a.base_url}", file=sys.stderr)
    print(f"strings: {len(results)}  ok: {ok}  needs review: {len(results) - ok}  "
          f"auto-fixed on retry: {fixed}  cached: {len(results) - len(fresh)}", file=sys.stderr)
    if fresh:
        print(f"time: {dt:.1f}s  ({dt / len(fresh):.2f}s per new string)", file=sys.stderr)
    for r in results:
        if r.status != "ok":
            print(f"  REVIEW {r.item.key}: {'; '.join(r.issues)}", file=sys.stderr)
    print(f"written: {out}", file=sys.stderr)
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
