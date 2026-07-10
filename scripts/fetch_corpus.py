# -*- coding: utf-8 -*-
"""从中国政府网抓取法规全文，扩充知识库语料（v0.4 评测集需要更大文档池）。

只用标准库：urllib 抓取 + 正则抽取正文。
"""
import html
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "docs"

SOURCES = [
    ("中华人民共和国劳动合同法", "https://www.gov.cn/flfg/2007-06/29/content_669394.htm"),
    ("中华人民共和国社会保险法", "https://www.gov.cn/flfg/2010-10/28/content_1732964.htm"),
    ("女职工劳动保护特别规定", "https://www.gov.cn/zwgk/2012-05/07/content_2131446.htm"),
    ("住房公积金管理条例", "https://www.gov.cn/zhengce/2020-12/26/content_5574588.htm"),
]

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def extract_body(html_text: str) -> str:
    html_text = SCRIPT_RE.sub("", html_text)
    # 优先取正文容器；取不到就全文去标签兜底
    for pattern in (
        r'<div[^>]*id="UCAP-CONTENT"[^>]*>(.*?)</div>',
        r'<div[^>]*class="pages_content"[^>]*>(.*?)</div>',
        r'<td[^>]*class="b12c"[^>]*>(.*?)</td>',
    ):
        m = re.search(pattern, html_text, re.S)
        if m:
            html_text = m.group(1)
            break
    text = TAG_RE.sub("\n", html_text)
    text = html.unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES:
        try:
            body = extract_body(fetch(url))
            if len(body) < 2000:
                print(f"[skip] {name}: 正文过短({len(body)}字)，可能页面结构变化")
                continue
            out = OUT_DIR / f"{name}.txt"
            out.write_text(body, encoding="utf-8")
            print(f"[ok]   {name}: {len(body)} 字 → {out.name}")
        except Exception as e:
            print(f"[fail] {name}: {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
