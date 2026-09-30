# -*- coding: utf-8 -*-
"""特許の図面に、請求項の構成要素を色で示す（公報は Google Patents から自動で取得する）。

流れ
  1. 文献番号から Google Patents の公報を探す（特開・特表・特許・再表・WO の番号を変換）。
     見つからないときは、発明の名称と出願人で検索し、請求項の本文がいちばん近い公報を選ぶ。
  2. 公報のページから、図面の画像・【符号の説明】・図面の簡単な説明・ファミリーを読む。
     図面や符号の説明が無い公報（新しい特許公報など）は、ファミリー（再表・WO など）のものを使う。
  3. 請求項の構成要素（SAO の部品の名前）と、符号の説明の名前を照らし合わせて、部品ごとに符号を決める。
     符号の説明に無いときは、明細書の本文の「管１１２」のような書き方から補う。
  4. 図面の中の符号の位置は、Google Patents が図面から読み取った位置（callouts）と、
     Tesseract の OCR（入っているときだけ）で求める。
  5. 構成要素の符号を、その部品の色の枠で囲み、引き出し線の先の部品の領域を塗る。
     請求項に出てこない符号は灰色の枠にする。図面にあるのに符号の説明に無い数字も一覧にする。

注意：図面は実施形態であり、権利範囲そのものではない。この図は「請求項の各構成要件が、
実施形態のどの部品に当たるか」を示す（クレームチャートを図面に描いたもの）。

アプリ（app.py の「🎨 図面で見る」）から使う。単独でも動く：
  python drawing_tool.py --patent 特開2022-144957 --claim claim1.txt --out 出力フォルダ
  python drawing_tool.py --pdf 公報.pdf --fugo fugo.txt --claim claim1.txt --out 出力フォルダ
"""
import argparse
import base64
import hashlib
import html as _html
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import urllib.parse
from collections import Counter, deque

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:  # 塗りつぶしだけが使えなくなる
    cv2 = None

GOOGLE = "https://patents.google.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d",
           "#ea580c", "#4f46e5", "#0d9488", "#57534e", "#c026d3", "#0284c7", "#a16207", "#e11d48"]
GREY = "#9aa0a6"
NOTE = ("図面は実施形態であり、権利範囲そのものではありません。この図は、請求項の各構成要件が"
        "実施形態のどの部品に当たるかを示します。")


class FetchError(Exception):
    pass


# =============================================================== 名前と符号をそろえる
def nfkc(s):
    return unicodedata.normalize("NFKC", s or "")


def norm(name):
    x = re.sub(r"\s", "", nfkc(name))
    x = re.sub(r"^(前記|該|当該|上記)", "", x)
    x = re.sub(r"^(複数の|一対の|少なくとも[0-9一二]+つの|[0-9]+つの|各)", "", x)
    return x.replace("前記", "")


_ORD = re.compile(r"第[0-9一二三四五六七八九十]+の?")


def name_match(comp, fugo_name):
    """3＝同じ名前。2＝符号の説明の名前の方が詳しい（請求項「多穴管」と「第1多穴管」、「ファン」と「軸流ファン」）。
    1＝請求項の方が番号つき（「第1ヘッダ」と符号の説明「ヘッダ」）。0＝別の部品。"""
    a, b = norm(comp), norm(fugo_name)
    if not a or not b:
        return 0
    if a == b:
        return 3
    sa, sb = _ORD.sub("", a), _ORD.sub("", b)
    oa, ob = bool(_ORD.search(a)), bool(_ORD.search(b))
    if sa == sb and sa:
        if oa and ob:
            return 0          # 「第1ヘッダ」と「第2ヘッダ」は別の部品
        return 2 if ob else 1
    # 符号の説明の名前の方が長く、請求項の名前で終わるときだけ一致とする（「軸流ファン」と「ファン」）。
    # 逆（請求項「横回転軸」と符号「回転軸」）は、別の部品のことが多いので一致としない
    if len(sa) >= 2 and sb.endswith(sa):
        return 2
    return 0


# 符号：「5」「30a」「30b1」「30aB」「30af2」「AX1」「112'」のほか、文字だけの「SPin」も（表の中だけ）
CODE = r"(?:[A-Za-z]{0,4}[0-9]{1,4}(?:[A-Za-z]{1,3}[0-9]{0,2}){0,2}'?)"
CODE_ALPHA = r"(?:[A-Z][A-Za-z]{1,5})"
_CODE_ONLY = re.compile(rf"^({CODE}|{CODE_ALPHA})$")
_CODE_NAME = re.compile(rf"^({CODE}|{CODE_ALPHA})(?:\s+|(?=[^\x00-\x7F]))(.+)$")


def canon_code(c):
    return nfkc(c).strip().replace("'", "").replace("’", "")


def _clean_name(name):
    name = name.strip(" 　…・:：.。")
    name = re.sub(r"[（(][^）)]*[）)]$", "", name).strip()
    return name


def parse_sign_list(text, min_items=3):
    """【符号の説明】の並び（「１、１０１  半導体装置<改行>５  電力変換装置…」や
    「１　送風機、２　送風機、１０　歯車…」）を {符号: 名前} にする。
    読点は「符号どうしの区切り」と「項目どうしの区切り」の両方に使われるので、
    符号だけのかたまりは次の項目の符号にまとめる。"""
    t = nfkc(text)
    chunks = [c.strip() for c in re.split(r"[\n、,，;；]+", t) if c.strip()]
    table, pending, n_ok, n_bad = {}, [], 0, 0
    for ch in chunks:
        ch = ch.strip(" 。.")
        if _CODE_ONLY.match(ch):
            pending.append(canon_code(ch))
            continue
        m = _CODE_NAME.match(ch)
        name = _clean_name(m.group(2)) if m else ""
        if m and name and re.search(r"[^\x00-\x7F]", name) and not re.fullmatch(r"[0-9A-Za-z'\s]+", name):
            for c in pending + [canon_code(m.group(1))]:
                table.setdefault(c, name)
            n_ok += 1
        else:
            n_bad += 1
        pending = []
    if n_ok < min_items or n_ok < 2 * n_bad:
        return {}, n_ok, n_bad
    return table, n_ok, n_bad


def parse_fugo_text(text):
    """貼り付けた【符号の説明】（見出しがあってもなくてもよい）を {符号: 名前} にする。"""
    t = nfkc(text)
    m = re.search(r"符号の説明[】\]]?(.*?)(?:【(?!符号)|$)", t, re.S)
    body = m.group(1) if m else t
    table, _, _ = parse_sign_list(body, min_items=1)
    return table


_BODY_SIGN = re.compile(rf"((?:第[0-9]+の?)?[一-龥々ァ-ヴー]*[一-龥々ァ-ヴー](?<!第))({CODE})(?![0-9A-Za-z.])")
_BODY_STOP = re.compile(r"(特開|特表|特許|公報|図|表|式|項|文献|形態|例|ステップ|工程|段落|番号|数|量|率|係数|度|角|比|倍|個|本|枚|層目|番目)$")


def body_signs(text):
    """明細書の本文の「管１１２」「第１多穴管３０ｂ１」のような書き方から、{符号: 名前} を作る（補助）。"""
    t = nfkc(text)
    t = re.sub(r"(前記|当該|上記|該)", "、", t)
    votes = {}
    for m in _BODY_SIGN.finditer(t):
        name, code = norm(m.group(1)), canon_code(m.group(2))
        if len(name) < 1 or _BODY_STOP.search(name) or not re.search(r"[0-9]", code):
            continue
        votes.setdefault(code, Counter())[name] += 1
    return {c: v.most_common(1)[0][0] for c, v in votes.items()}


def claim_similarity(a, b):
    """請求項の本文どうしの近さ（文字の2字組の Jaccard）。"""
    def grams(s):
        s = re.sub(r"[\s、。，．,.（）()「」]", "", norm(s))
        return {s[i:i + 2] for i in range(len(s) - 1)}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(1, len(ga | gb))


# =============================================================== 文献番号 → Google Patents の番号
_ERA = {"平": "H", "昭": "S", "令": "R"}


def doc_candidates(pid):
    """J-PlatPat などの文献番号を、Google Patents の番号の候補（順に試す）にする。"""
    s = nfkc(pid).upper()
    s = re.sub(r"[\s　]", "", s).replace("第", "").replace("号", "").replace("公報", "")
    m = re.fullmatch(r"(JP|WO|US|EP|CN|KR|DE)[A-Z0-9/\-]+", s)
    if m:
        t = s.replace("/", "").replace("-", "")
        if re.search(r"[A-Z][0-9]?$", t[2:]) and not t[2:].isdigit():
            return [t]
        if t.startswith("WO"):
            return [t + "A1"]
        return [t + "A", t + "B2", t + "B1"]
    m = re.fullmatch(r"(特開|特表|特公|再表|再公表|実開|実登|実用新案登録|特許)(平|昭|令)?([0-9]+)?[\-/]?([0-9]+)", s)
    if not m:
        m2 = re.fullmatch(r"([0-9]{4})[\-/]([0-9]{5,6})", s)       # 「2022-144957」だけのとき
        if m2:
            return [f"JP{m2.group(1)}{int(m2.group(2)):06d}A"]
        m3 = re.fullmatch(r"([0-9]{7})", s)                        # 「7800720」だけのとき
        return [f"JP{m3.group(1)}B2", f"JP{m3.group(1)}B1"] if m3 else []
    kind, era, year, num = m.groups()
    if kind in ("特許",):
        n = (year or "") + num
        return [f"JP{n}B2", f"JP{n}B1"]
    if kind in ("実登", "実用新案登録"):
        return [f"JP{(year or '') + num}U"]
    if not year:
        return []
    if era:
        y = f"{_ERA[era]}{int(year):02d}"
    else:
        y = year
    n = f"{int(num):06d}"
    if kind == "特開" or kind == "特表":
        return [f"JP{y}{n}A"]
    if kind == "特公":
        return [f"JP{y}{n}B2", f"JP{y}{n}B"]
    if kind == "実開":
        return [f"JP{y}{n}U"]
    if kind in ("再表", "再公表"):
        return [f"JPWO{y}{n}A1", f"WO{y}{n}A1"]
    return []


# =============================================================== 取得（キャッシュつき）
class Fetcher:
    """Google Patents のページと図面の画像を取り、フォルダに保存しておく（同じものは2回取らない）。"""

    def __init__(self, cache_dir=None, wait=1.0, timeout=25):
        import requests
        self.requests = requests
        self.dir = cache_dir or os.path.join(tempfile.gettempdir(), "sao_drawing_cache")
        os.makedirs(self.dir, exist_ok=True)
        self.wait, self.timeout, self._last = wait, timeout, 0.0
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "ja,en;q=0.8"})
        self.n_net = 0

    def path_for(self, url, ext):
        return os.path.join(self.dir, hashlib.sha1(url.encode("utf-8")).hexdigest()[:20] + ext)

    def get(self, url, ext=".html", binary=False):
        """取れたらその中身（binary なら保存先のパス）。無いページ（404）は None。"""
        p = self.path_for(url, ext)
        miss = p + ".404"
        if os.path.exists(miss):
            return None
        if os.path.exists(p):
            return p if binary else open(p, encoding="utf-8").read()
        for attempt in range(3):
            dt = time.time() - self._last
            if dt < self.wait and "patents.google.com" in url:
                time.sleep(self.wait - dt)
            try:
                r = self.s.get(url, timeout=self.timeout)
            except self.requests.RequestException as e:
                if attempt == 2:
                    raise FetchError(f"接続できませんでした：{url}（{e.__class__.__name__}）")
                time.sleep(2 + 3 * attempt)
                continue
            finally:
                self._last = time.time()
                self.n_net += 1
            if r.status_code == 404:
                open(miss, "w").close()
                return None
            if r.status_code in (429, 503):
                if attempt == 2:
                    raise FetchError("Google Patents から「アクセスが多すぎる」と返されました。数分おいてから、もう一度試してください。")
                time.sleep(5 + 10 * attempt)
                continue
            if r.status_code != 200:
                raise FetchError(f"取得できませんでした（HTTP {r.status_code}）：{url}")
            if binary:
                open(p, "wb").write(r.content)
                return p
            r.encoding = "utf-8"
            open(p, "w", encoding="utf-8").write(r.text)
            return r.text
        return None

    def get_json(self, url):
        t = self.get(url, ext=".json")
        if not t:
            return None
        try:
            return json.loads(t)
        except ValueError:
            return None


# =============================================================== 公報のページを読む
# =============================================================== HTML を読む（標準ライブラリだけ・bs4 は使わない）
from html.parser import HTMLParser as _HTMLParser

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class _Node:
    """HTML の要素（BeautifulSoup の find / find_all / get_text と同じ使い方ができる小さな木）。"""

    def __init__(self, tag, attrs=None, parent=None):
        self.name, self.attrs, self.parent, self.children = tag, dict(attrs or {}), parent, []

    def get(self, k, default=None):
        return self.attrs.get(k, default)

    def __getitem__(self, k):
        return self.attrs[k]

    def _match(self, tag, attrs, class_):
        if tag and self.name != tag:
            return False
        for k, v in (attrs or {}).items():
            if self.attrs.get(k) != v:
                return False
        if class_ and class_ not in (self.attrs.get("class") or "").split():
            return False
        return True

    def _iter(self):
        for c in self.children:
            if isinstance(c, _Node):
                yield c
                yield from c._iter()

    def find_all(self, tag=None, attrs=None, class_=None):
        return [n for n in self._iter() if n._match(tag, attrs, class_)]

    def find(self, tag=None, attrs=None, class_=None):
        for n in self._iter():
            if n._match(tag, attrs, class_):
                return n
        return None

    def select(self, sel):
        tag, _, cls = sel.partition(".")
        return self.find_all(tag or None, class_=cls or None)

    def find_parent(self, tag=None, class_=None):
        p = self.parent
        while p is not None:
            if p._match(tag, None, class_):
                return p
            p = p.parent
        return None

    def _texts(self):
        for c in self.children:
            if isinstance(c, str):
                yield c
            elif c.name == "br":
                yield "\n"
            else:
                yield from c._texts()

    def get_text(self, sep="", strip=False):
        ts = list(self._texts())
        if strip:
            ts = [t.strip() for t in ts if t.strip()]
        return sep.join(ts)

    @property
    def title(self):
        return self.find("title")


class _TreeBuilder(_HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("[document]")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = _Node(tag, [(k, v if v is not None else "") for k, v in attrs], self.stack[-1])
        self.stack[-1].children.append(n)
        if tag not in _VOID:
            self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        n = _Node(tag, [(k, v if v is not None else "") for k, v in attrs], self.stack[-1])
        self.stack[-1].children.append(n)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].name == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _parse_html(page):
    b = _TreeBuilder()
    b.feed(page)
    b.close()
    return b.root


def _meta(el, prop):
    m = el.find("meta", attrs={"itemprop": prop})
    return m.get("content") if m else None


def parse_patent_html(page, gid=""):
    """Google Patents の公報ページ（HTML）から、名称・請求項・符号の説明・図面・ファミリーを取り出す。"""
    soup = _parse_html(page)
    doc = {"id": gid, "url": f"{GOOGLE}/patent/{gid}/ja" if gid else "", "title": "", "claims": [],
           "fugo": {}, "fugo_src": "", "body_signs": {}, "images": [], "figrefs": [], "family": [], "pdf": ""}
    t = soup.find("meta", attrs={"name": "DC.title"})
    if t and t.get("content"):
        doc["title"] = re.sub(r"\s+", " ", t["content"]).strip()
    elif soup.title:
        doc["title"] = re.sub(r"\s+", " ", soup.title.get_text()).split(" - ")[1:2][0] if " - " in soup.title.get_text() else ""
    # 請求項
    cs = soup.find("section", attrs={"itemprop": "claims"})
    if cs:
        for d in cs.select("div.claim"):
            if d.find_parent("div", class_="claim") is not None:
                continue
            txt = d.get_text("", strip=False)
            txt = re.sub(r"\s+", "", txt)
            if txt:
                doc["claims"].append(txt)
    # 明細書
    ds = soup.find("section", attrs={"itemprop": "description"})
    if ds:
        rsl = ds.find("reference-signs-list")
        if rsl is not None:
            tab, _, _ = parse_sign_list(rsl.get_text("\n"), min_items=1)
            if tab:
                doc["fugo"], doc["fugo_src"] = tab, "符号の説明"
        paras = [p.get_text("\n") for p in ds.select("div.description-paragraph")]
        if not doc["fugo"]:
            # 見出しの無い符号の並び（WO・再表に多い）。後ろの段落から探す
            for p in reversed(paras[-6:]):
                tab, ok, bad = parse_sign_list(p)
                if tab:
                    doc["fugo"], doc["fugo_src"] = tab, "符号の説明（見出しなしの段落）"
                    break
        if not doc["fugo"]:
            whole = ds.get_text("\n")
            if "符号の説明" in nfkc(whole):
                tab = parse_fugo_text(whole)
                if len(tab) >= 2:
                    doc["fugo"], doc["fugo_src"] = tab, "符号の説明"
        doc["body_signs"] = body_signs("\n".join(paras))
        for f in ds.find_all("figref"):
            tx = re.sub(r"\s+", "", f.get_text())
            if re.match(r"^[【\[]?図", nfkc(tx)):
                doc["figrefs"].append(tx)
    # 図面
    for li in soup.find_all("li", attrs={"itemprop": "images"}):
        full = _meta(li, "full")
        th = li.find("img", attrs={"itemprop": "thumbnail"})
        if not full and th is not None:
            full = th.get("src")
        if not full:
            continue
        callouts = []
        for c in li.find_all("li", attrs={"itemprop": "callouts"}):
            b = c.find(attrs={"itemprop": "bounds"})
            try:
                box = tuple(int(float(_meta(b, k))) for k in ("left", "top", "right", "bottom"))
            except (TypeError, ValueError):
                continue
            cid = _meta(c, "id")
            if cid:
                callouts.append({"id": canon_code(cid), "label": _meta(c, "label") or "", "box": box})
        doc["images"].append({"full": full, "thumb": th.get("src") if th is not None else "", "callouts": callouts})
    # ファミリー
    # 同じ出願・同じファミリーの公報だけ（引用文献・似た文献の欄は使わない）
    fam = []
    for prop in ("pubs", "docdbFamily", "countryStatus"):
        for tr in soup.find_all("tr", attrs={"itemprop": prop}):
            for sp in tr.find_all("span", attrs={"itemprop": "publicationNumber"}):
                if sp.get_text(strip=True):
                    fam.append(sp.get_text(strip=True))
    seen = set()
    doc["family"] = [x for x in fam if x != gid and not (x in seen or seen.add(x))]
    a = soup.find("a", attrs={"itemprop": "pdfLink"})
    if a and a.get("href"):
        doc["pdf"] = a["href"]
    return doc


def fetch_doc(fetcher, gid):
    page = fetcher.get(f"{GOOGLE}/patent/{gid}/ja")
    if page is None:
        return None
    return parse_patent_html(page, gid)


def search_ids(fetcher, title="", applicant="", claim="", limit=6):
    """発明の名称（＋出願人）で Google Patents を検索し、公報番号の候補を返す。"""
    qs = []
    title = re.sub(r"\s+", " ", nfkc(title)).strip()
    if title:
        q = f'q=("{title}")&country=JP&language=JAPANESE'
        if applicant:
            q += "&assignee=" + re.sub(r"\s+", " ", nfkc(applicant)).strip()
        qs.append(q)
        qs.append(f'q=("{title}")&country=JP&language=JAPANESE')
    if claim:
        key = re.sub(r"[\s、。]", "", norm(claim))[:40]
        if key:
            qs.append(f'q=("{key}")&country=JP')
    out = []
    for q in qs:
        j = fetcher.get_json(f"{GOOGLE}/xhr/query?url=" + urllib.parse.quote(q, safe="") + "&exp=")
        for cl in ((j or {}).get("results") or {}).get("cluster") or []:
            for r in cl.get("result") or []:
                pn = (r.get("patent") or {}).get("publication_number")
                if pn and pn not in out:
                    out.append(pn)
        if len(out) >= limit:
            break
    return out[:limit]


def fig_label(figref):
    """「図１は、本実施形態に係る電力変換装置の斜視図である。」→「図1：本実施形態に係る電力変換装置の斜視図」"""
    t = nfkc(figref).strip()
    m = re.match(r"^[【\[]?(図[0-9]+(?:\([a-zA-Z0-9]\))?)[】\]]?\s*(?:は、?|:|、)?\s*(.*)$", t)
    if not m:
        return t[:40]
    desc = re.sub(r"(を示す図|を示す|の図)?(である|です)?。?$", "", m.group(2)).strip()
    desc = desc if len(desc) <= 36 else desc[:35] + "…"
    return f"{m.group(1)}：{desc}" if desc else m.group(1)


def best_claim_sim(doc, claim):
    return max([claim_similarity(claim, c) for c in doc["claims"]] or [0.0])


def find_patent(fetcher, pid, claim="", title="", applicant="", log=print):
    """文献番号（無ければ名称と出願人の検索）から公報を探し、図面と符号の説明のある公報をそろえる。
    返り値：{"doc": 本体, "fugo_doc": 符号の説明を取った公報, "image_doc": 図面を取った公報, "found_by", "sim", "tried"}"""
    tried, doc, found_by = [], None, ""
    for gid in doc_candidates(pid):
        tried.append(gid)
        d = fetch_doc(fetcher, gid)
        if d:
            doc, found_by = d, "文献番号"
            break
    if doc is None and (title or claim):
        log("文献番号から見つからないので、名称と出願人で検索します")
        best = None
        for gid in search_ids(fetcher, title, applicant, claim):
            tried.append(gid)
            d = fetch_doc(fetcher, gid)
            if not d:
                continue
            s = best_claim_sim(d, claim) if claim else 0.5
            if best is None or s > best[0]:
                best = (s, d)
            if s >= 0.8:
                break
        if best and best[0] >= 0.35:
            doc, found_by = best[1], "検索（名称・出願人）"
    if doc is None:
        return {"doc": None, "tried": tried}
    sim = best_claim_sim(doc, claim) if claim else None
    fugo_doc = doc if doc["fugo"] else None
    image_doc = doc if doc["images"] else None
    if not (fugo_doc and image_doc):
        fam = sorted(doc["family"], key=lambda x: (0 if x.startswith("JP") else 1 if x.startswith("WO") else 2,
                                                     x.endswith("A5")))
        for gid in fam[:6]:
            if fugo_doc and image_doc:
                break
            d = fetch_doc(fetcher, gid)
            tried.append(gid)
            if not d:
                continue
            if not fugo_doc and d["fugo"] and gid[:2] in ("JP", "WO"):
                fugo_doc = d
            if not image_doc and d["images"]:
                # 番号の付け方が同じか（図面の符号が符号の説明に載っているか）を確かめてから使う
                ids = {c["id"] for im in d["images"] for c in im["callouts"]}
                ref = (fugo_doc or d)["fugo"] or doc["fugo"]
                if not ids or not ref or len(ids & set(ref)) >= 0.3 * len(ids):
                    image_doc = d
    return {"doc": doc, "fugo_doc": fugo_doc, "image_doc": image_doc, "found_by": found_by, "sim": sim, "tried": tried}


# =============================================================== OCR（入っているときだけ）
_OCR_OK = None


def ocr_available():
    global _OCR_OK
    if _OCR_OK is not None:
        return _OCR_OK
    try:
        import pytesseract
    except ImportError:
        _OCR_OK = False
        return False
    if not shutil.which("tesseract"):
        for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                  os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")):
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
    try:
        pytesseract.get_tesseract_version()
        _OCR_OK = True
    except Exception:  # noqa: BLE001
        _OCR_OK = False
    return _OCR_OK


def _ocr_tokens(img):
    import pytesseract
    data = pytesseract.image_to_data(
        img, lang="eng", output_type=pytesseract.Output.DICT,
        config="--psm 11 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    out = []
    for i, txt in enumerate(data["text"]):
        t = (txt or "").strip()
        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1
        if not t or conf < 40:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if w * h > 0 and h < img.height * 0.06:
            out.append({"text": canon_code(t), "box": (x, y, x + w, y + h), "conf": conf})
    return out


def ocr_numerals(path):
    """図面の中の文字のかたまり（符号の候補）を読む。引き出し線が触れていると読めないことがあるので、
    長い直線を消した画像でも読み、合わせる。まわりが線で混んでいるもの（図の模様の読み違い）は捨てる。"""
    img = Image.open(path).convert("L")
    toks = _ocr_tokens(img)
    arr = np.array(img)
    if cv2 is not None:
        edges = (arr < 140).astype(np.uint8) * 255
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(70, img.height // 45), maxLineGap=3)
        if lines is not None:
            clean = arr.copy()
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):   # OpenCV の版で形が (N,1,4) と (N,4) に分かれる
                cv2.line(clean, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)
            for t in _ocr_tokens(Image.fromarray(clean)):
                if not any(o["text"] == t["text"] and abs(o["box"][0] - t["box"][0]) < 30 and abs(o["box"][1] - t["box"][1]) < 30
                           for o in toks):
                    toks.append(t)
    ink = arr < 140
    keep = []
    for t in toks:
        x0, y0, x1, y1 = t["box"]
        m = max(8, (y1 - y0) // 2)
        X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(ink.shape[1], x1 + m), min(ink.shape[0], y1 + m)
        ring = ink[Y0:Y1, X0:X1].copy()
        ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        area = ring.size - (y1 - y0) * (x1 - x0)
        if area > 0 and ring.sum() / area > 0.18:
            continue
        keep.append(t)
    # 「112」の一部だけを「12」と読んだもののように、ほかの読みの枠の中にある短い読みは捨てる
    def inside(a, b):
        return a["box"][0] >= b["box"][0] - 3 and a["box"][2] <= b["box"][2] + 3 and \
            a["box"][1] >= b["box"][1] - 3 and a["box"][3] <= b["box"][3] + 3
    return [t for t in keep if not any(o is not t and len(o["text"]) > len(t["text"]) and inside(t, o) for o in keep)]


def match_code(text, known):
    """読んだ文字を、符号の表の符号に合わせる（大文字・小文字の違いは許す）。"""
    if text in known:
        return text
    low = {k.lower(): k for k in known}
    return low.get(text.lower())


# =============================================================== 引き出し線をたどって部品の領域を塗る
def _segments(gray):
    edges = (gray < 140).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40, minLineLength=40, maxLineGap=6)
    return [] if segs is None else [tuple(map(float, x)) for x in np.asarray(segs).reshape(-1, 4)]


def _fill_at_end(gray, box, end, v, max_frac=0.2, min_px=300):
    h, w = gray.shape
    ink = gray < 140
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    vx, vy = v
    px, py = -vy, vx
    fill = (~ink).astype(np.uint8) * 255
    for dt, dp in ((4, 0), (8, 0), (-6, 6), (-6, -6), (0, 8), (0, -8), (14, 0)):
        sx, sy = int(end[0] + vx * dt + px * dp), int(end[1] + vy * dt + py * dp)
        if not (0 <= sx < w and 0 <= sy < h) or ink[sy, sx]:
            continue
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(fill.copy(), mask, (sx, sy), 128, flags=4 | (255 << 8))
        region = mask[1:-1, 1:-1] > 0
        size = int(region.sum())
        if min_px < size < max_frac * h * w and not region[min(h - 1, int(cy)), min(w - 1, int(cx))]:
            return region
    return None


def _region_by_segment(gray, box, segs):
    """一方の端が符号のすぐ近くにある直線を引き出し線とみなし、もう一方の端の先の閉じた領域を返す。"""
    x0, y0, x1, y1 = box
    near = []
    for (ax, ay, bx, by) in segs:
        for (nx, ny, fx, fy) in ((ax, ay, bx, by), (bx, by, ax, ay)):
            dx = max(x0 - nx, 0, nx - x1)
            dy = max(y0 - ny, 0, ny - y1)
            d = (dx * dx + dy * dy) ** 0.5
            inside_far = x0 - 5 <= fx <= x1 + 5 and y0 - 5 <= fy <= y1 + 5
            if d <= 45 and not inside_far:
                near.append((d, ((fx - nx) ** 2 + (fy - ny) ** 2) ** 0.5, (nx, ny, fx, fy)))
    if not near:
        return None
    # ハフ変換は1本の線を短く切って何本も返すことがあるので、いちばん近いものに近い距離の中で、いちばん長いものを使う
    dmin = min(x[0] for x in near)
    _, _, (nx, ny, fx, fy) = max((x for x in near if x[0] <= dmin + 12), key=lambda x: x[1])
    n = max(1e-6, ((fx - nx) ** 2 + (fy - ny) ** 2) ** 0.5)
    return _fill_at_end(gray, box, (fx, fy), ((fx - nx) / n, (fy - ny) / n))


def _region_by_trace(gray, box, ring):
    """符号のまわりの黒い画素の方向を引き出し線の向きとし、線に沿って先端までたどって塗る（曲がった線用）。"""
    h, w = gray.shape
    ink = gray < 140
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    Y0, Y1, X0, X1 = max(0, y0 - ring), min(h, y1 + ring), max(0, x0 - ring), min(w, x1 + ring)
    ys, xs = np.nonzero(ink[Y0:Y1, X0:X1])
    ys, xs = ys + Y0, xs + X0
    outside = (xs < x0 - 8) | (xs > x1 + 8) | (ys < y0 - 8) | (ys > y1 + 8)
    ys, xs = ys[outside], xs[outside]
    if len(xs) < 5:
        return None
    vx, vy = xs.mean() - cx, ys.mean() - cy
    n = (vx * vx + vy * vy) ** 0.5
    if n < 1:
        return None
    vx, vy = vx / n, vy / n
    t = max(x1 - x0, y1 - y0) / 2 + 4
    X, Y = cx + vx * t, cy + vy * t
    pts, gap, last, step = [], 0, None, 0
    while 0 <= X < w and 0 <= Y < h and step < max(h, w):
        px, py = -vy, vx
        found = None
        for k in (0, -1, 1, -2, 2, -3, 3):
            yy, xx = int(round(Y + py * k)), int(round(X + px * k))
            if 0 <= yy < h and 0 <= xx < w and ink[yy, xx]:
                found = (X + px * k, Y + py * k)
                break
        if found is not None:
            last, gap = found, 0
            pts.append(found)
            if len(pts) >= 15 and len(pts) % 5 == 0:
                fx, fy = pts[0]
                dx, dy = found[0] - fx, found[1] - fy
                nn = (dx * dx + dy * dy) ** 0.5
                if nn > 10:
                    vx, vy = dx / nn, dy / nn
            X, Y = found[0] + vx, found[1] + vy
        else:
            gap += 1
            if last is not None and gap > 10:
                break
            X, Y = X + vx, Y + vy
        step += 1
    if last is None or len(pts) < 20:
        return None
    return _fill_at_end(gray, box, last, (vx, vy))


def _leader_end_isolated(ink, box, pad=16, win=700):
    """符号から出ている引き出し線が、ほかの線とつながっていない1本の細い線のとき、その線を最後までたどり、
    先端の位置と向きを返す（曲がった線・波線でもよい）。つながっているときは None。"""
    h, w = ink.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    X0, Y0, X1, Y1 = max(0, x0 - win), max(0, y0 - win), min(w, x1 + win), min(h, y1 + win)
    sub = ink[Y0:Y1, X0:X1].copy()
    bx0, by0, bx1, by1 = x0 - X0, y0 - Y0, x1 - X0, y1 - Y0
    sub[max(0, by0 - 3):by1 + 3, max(0, bx0 - 3):bx1 + 3] = False       # 符号の文字そのものを消す
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    ring = lab[max(0, by0 - pad):by1 + pad, max(0, bx0 - pad):bx1 + pad]
    best = None
    for L in set(np.unique(ring)) - {0}:
        area, bw, bh = stats[L, cv2.CC_STAT_AREA], stats[L, cv2.CC_STAT_WIDTH], stats[L, cv2.CC_STAT_HEIGHT]
        span = max(bw, bh)
        if span < 25 or area > 8 * (bw + bh) or area > 12000:
            continue            # 短すぎる（文字の一部）か、太い・大きい（図の線とつながっている）
        ys, xs = np.nonzero(lab == L)
        dx = np.maximum(np.maximum(bx0 - xs, 0), xs - bx1)
        dy = np.maximum(np.maximum(by0 - ys, 0), ys - by1)
        d = np.hypot(dx, dy)
        k = int(np.argmin(d))
        if best is None or d[k] < best[0]:
            best = (d[k], L, ys, xs, k)
    if best is None:
        return None
    _, L, ys, xs, k = best
    # 線の上を、符号に近い端から幅優先でたどり、いちばん遠い点を先端とする
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    dist = np.full(len(ys), -1, np.int32)
    par = np.full(len(ys), -1, np.int32)
    dist[k] = 0
    dq = deque([k])
    while dq:
        i = dq.popleft()
        y, x = int(ys[i]), int(xs[i])
        for dy_ in (-1, 0, 1):
            for dx_ in (-1, 0, 1):
                j = idx.get((y + dy_, x + dx_))
                if j is not None and dist[j] < 0:
                    dist[j] = dist[i] + 1
                    par[j] = i
                    dq.append(j)
    e = int(np.argmax(dist))
    if dist[e] < 25:
        return None
    b = e
    for _ in range(15):
        if par[b] < 0:
            break
        b = int(par[b])
    vx, vy = float(xs[e] - xs[b]), float(ys[e] - ys[b])
    nn = max(1e-6, (vx * vx + vy * vy) ** 0.5)
    ex, ey = float(xs[e] + X0), float(ys[e] + Y0)
    # 先端が符号の近くに戻ってきている（文字のまわりの枠など）ものは使わない
    if x0 - 20 <= ex <= x1 + 20 and y0 - 20 <= ey <= y1 + 20:
        return None
    return (ex, ey), (vx / nn, vy / nn)


def _arc_runs(ink, p, v, r, span=75, step=5):
    """点 p から向き v のまわり ±span 度・半径 r の弧の上で、黒い画素がある角度のまとまりを返す。"""
    h, w = ink.shape
    base = np.arctan2(v[1], v[0])
    hits = []
    for a in range(-span, span + 1, step):
        t = base + np.radians(a)
        x, y = p[0] + r * np.cos(t), p[1] + r * np.sin(t)
        xi, yi = int(round(x)), int(round(y))
        on = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = yi + dy, xi + dx
                if 0 <= yy < h and 0 <= xx < w and ink[yy, xx]:
                    on = True
        hits.append((a, on))
    runs, cur = [], []
    for a, on in hits:
        if on:
            cur.append(a)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _walk_leader(ink, box, max_len=3000):
    """引き出し線を、符号の近くから曲がりに沿ってたどる。線が途中でほかの線と交わるときは乗り越え、
    部品の線に当たって終わるときは、その線の向こう側の点を返す。
    返り値：(塗り始める点, 向き) または None"""
    h, w = ink.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # 符号の枠のすぐ外（3〜18px の帯）にある黒い画素のかたまりを、線の出だしの候補にする
    pad = 18
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    band = ink[Y0:Y1, X0:X1].copy()
    band[max(0, y0 - 3 - Y0):y1 + 3 - Y0, max(0, x0 - 3 - X0):x1 + 3 - X0] = False
    n, lab, stats, cents = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
    best = None
    for L in range(1, n):
        if stats[L, cv2.CC_STAT_AREA] < 4:
            continue
        sx, sy = cents[L][0] + X0, cents[L][1] + Y0
        vx, vy = sx - cx, sy - cy
        nn = (vx * vx + vy * vy) ** 0.5
        if nn < 1:
            continue
        res = _walk_from(ink, (sx, sy), (vx / nn, vy / nn), max_len)
        if res is None:
            continue
        length, end, v, kind = res
        if length < 30:
            continue
        # 出だしが符号に近く、長くたどれたものを選ぶ
        dx = max(x0 - sx, 0, sx - x1)
        dy = max(y0 - sy, 0, sy - y1)
        score = length - 6 * (dx * dx + dy * dy) ** 0.5
        if best is None or score > best[0]:
            best = (score, end, v, kind)
    if best is None:
        return None
    _, end, v, kind = best
    if x0 - 15 <= end[0] <= x1 + 15 and y0 - 15 <= end[1] <= y1 + 15:
        return None
    return end, v, kind


def _walk_from(ink, start, v, max_len):
    h, w = ink.shape
    p = (float(start[0]), float(start[1]))
    vx, vy = v
    length = 0
    r = 9
    first = True
    while length < max_len:
        runs = _arc_runs(ink, p, (vx, vy), r, span=90 if first else 75)
        if not runs:
            return length, p, (vx, vy), "free"          # 線の先端（部品の面の中で終わる）
        runs.sort(key=lambda ru: min(abs(a) for a in ru))
        ru = runs[0]
        wide = (max(ru) - min(ru)) > 55 and not first
        first = False
        if wide:
            # ほかの線に当たった。少し先（半径 14〜26px）に、同じ向きの細い線（引き出し線の続き）があれば乗り越える
            jumped = False
            for rr in (14, 20, 26):
                cand = [c for c in _arc_runs(ink, p, (vx, vy), rr, span=40)
                        if (max(c) - min(c)) <= 30 and abs(sum(c) / len(c)) <= 25]
                if not cand:
                    continue
                c = min(cand, key=lambda c: abs(sum(c) / len(c)))
                a = np.radians(sum(c) / len(c))
                base = np.arctan2(vy, vx)
                q = (p[0] + rr * np.cos(base + a), p[1] + rr * np.sin(base + a))
                if _walk_probe(ink, q, (np.cos(base + a), np.sin(base + a))):
                    # 乗り越えた先と今の点のあいだに白い画素があること（同じ太い線の上を進んだだけではないこと）
                    p, length, jumped = q, length + rr, True
                    break
            if jumped:
                continue
            # 部品の線に当たって終わった：線を横切った向こう側を返す
            q = p
            for k in range(20):
                q = (q[0] + vx, q[1] + vy)
                if not (0 <= q[0] < w and 0 <= q[1] < h):
                    return length, p, (vx, vy), "edge"
                if k >= 2 and not ink[int(q[1]), int(q[0])]:
                    break
            return length, (q[0] + vx * 2, q[1] + vy * 2), (vx, vy), "edge"
        a = np.radians(sum(ru) / len(ru))
        base = np.arctan2(vy, vx)
        nx, ny = np.cos(base + a), np.sin(base + a)
        p = (p[0] + r * nx, p[1] + r * ny)
        if not (0 <= p[0] < w and 0 <= p[1] < h):
            return None
        vx, vy = 0.6 * vx + 0.4 * nx, 0.6 * vy + 0.4 * ny
        nn = (vx * vx + vy * vy) ** 0.5
        vx, vy = vx / nn, vy / nn
        length += r
    return None


def _walk_probe(ink, q, v, steps=4):
    """横切った先で、細い線が数歩ぶん続くか（部品の面の模様ではなく、引き出し線の続きか）。"""
    p, vx, vy = q, v[0], v[1]
    for _ in range(steps):
        runs = [c for c in _arc_runs(ink, p, (vx, vy), 9, span=35)
                if (max(c) - min(c)) <= 40 and abs(sum(c) / len(c)) <= 25]
        if not runs:
            return False
        ru = min(runs, key=lambda c: abs(sum(c) / len(c)))
        a = np.radians(sum(ru) / len(ru))
        base = np.arctan2(vy, vx)
        vx, vy = np.cos(base + a), np.sin(base + a)
        p = (p[0] + 9 * vx, p[1] + 9 * vy)
        if not (0 <= p[0] < ink.shape[1] and 0 <= p[1] < ink.shape[0]):
            return False
    return True


def region_from_leader(gray, box, segs):
    ink = gray < 140
    r = _walk_leader(ink, box)
    if r is not None:
        (ex, ey), v, kind = r
        reg = _fill_at_end(gray, box, (ex, ey), v)
        if reg is not None:
            return reg
    r = _leader_end_isolated(ink, box)
    if r is not None:
        reg = _fill_at_end(gray, box, r[0], r[1])
        if reg is not None:
            return reg
    r = _region_by_segment(gray, box, segs)
    if r is not None:
        return r
    for ring in (20, 35, 50):
        r = _region_by_trace(gray, box, ring)
        if r is not None:
            return r
    return None


# =============================================================== 描く
def hex_rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


UNKNOWN = "#f59e0b"


def render(path, hits, code_colors, fill=True, no_fill=(), unknown=()):
    """符号を色の枠で囲み（請求項に出てこない符号は灰色）、引き出し線の先の部品を塗った画像を返す。"""
    base = Image.open(path).convert("RGB")
    gray = np.array(base.convert("L"))
    over = np.zeros((base.height, base.width, 4), np.uint8)
    filled = []
    if fill and cv2 is not None:
        segs = _segments(gray)
        for hct in hits:
            col = code_colors.get(hct["code"])
            if not col or hct["code"] in no_fill:
                continue
            reg = region_from_leader(gray, hct["box"], segs)
            if reg is not None:
                over[reg] = hex_rgb(col) + (90,)
                filled.append(hct["code"])
    ov = Image.fromarray(over, "RGBA")
    od = ImageDraw.Draw(ov)
    lw = max(3, base.height // 400)
    for hct in hits:
        x0, y0, x1, y1 = hct["box"]
        col = code_colors.get(hct["code"])
        m = 6
        if col:
            od.rounded_rectangle((x0 - m, y0 - m, x1 + m, y1 + m), radius=8, outline=hex_rgb(col) + (255,), width=lw + 2)
        elif hct["code"] in unknown:   # 符号の説明に無い符号（橙の点線）
            for k in range(0, int(x1 - x0 + 2 * m), 10):
                od.line((x0 - m + k, y0 - m, min(x1 + m, x0 - m + k + 5), y0 - m), fill=hex_rgb(UNKNOWN) + (255,), width=3)
                od.line((x0 - m + k, y1 + m, min(x1 + m, x0 - m + k + 5), y1 + m), fill=hex_rgb(UNKNOWN) + (255,), width=3)
            for k in range(0, int(y1 - y0 + 2 * m), 10):
                od.line((x0 - m, y0 - m + k, x0 - m, min(y1 + m, y0 - m + k + 5)), fill=hex_rgb(UNKNOWN) + (255,), width=3)
                od.line((x1 + m, y0 - m + k, x1 + m, min(y1 + m, y0 - m + k + 5)), fill=hex_rgb(UNKNOWN) + (255,), width=3)
        else:
            od.rounded_rectangle((x0 - m, y0 - m, x1 + m, y1 + m), radius=8, outline=hex_rgb(GREY) + (200,), width=max(2, lw - 1))
    out = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    return out, filled


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


# =============================================================== PDF の図面ページ（図面の画像が無いとき）
def pdf_to_images(pdf_path, dpi=200):
    """PDF をページの画像にする（pypdfium2 → PyMuPDF → pdftoppm の順に、使えるものを使う）。"""
    d = tempfile.mkdtemp()
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        out = []
        for i in range(len(pdf)):
            p = os.path.join(d, f"p{i + 1:03d}.png")
            pdf[i].render(scale=dpi / 72).to_pil().convert("L").save(p)
            out.append(p)
        return out
    except ImportError:
        pass
    try:
        import fitz
        doc = fitz.open(pdf_path)
        out = []
        for i, page in enumerate(doc):
            p = os.path.join(d, f"p{i + 1:03d}.png")
            page.get_pixmap(dpi=dpi).save(p)
            out.append(p)
        return out
    except ImportError:
        pass
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", pdf_path, os.path.join(d, "p")], check=True)
        return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]
    raise RuntimeError("PDF を画像にする部品がありません（pip install pypdfium2 で入ります）。図面の画像で読み込んでください。")


def looks_like_drawing(path):
    """文字のページか図面のページかを、文字くらいの大きさの黒いかたまりの数で見分ける（おおよそ）。"""
    if cv2 is None:
        return True
    g = np.array(Image.open(path).convert("L"))
    n, _, stats, _ = cv2.connectedComponentsWithStats((g < 140).astype(np.uint8), connectivity=8)
    h = g.shape[0]
    small = sum(1 for s in stats[1:] if h * 0.004 < s[3] < h * 0.025 and s[2] < h * 0.03)
    return small < 900


# =============================================================== 構成要素 → 符号
def extra_components(claim, fugo, comps):
    """符号の説明の名前が請求項の本文にそのまま出てくれば、構成要素に加える（前の文字が別の語の一部でないときだけ）。"""
    flat = norm(claim)
    have = {norm(c) for c in comps}
    out = []
    for nm in dict.fromkeys(fugo.values()):
        k = norm(nm)
        if len(k) < 2 or k in have:
            continue
        for m_ in re.finditer(re.escape(k), flat):
            prev = flat[m_.start() - 1] if m_.start() > 0 else ""
            if not prev or not re.match(r"[一-龥々ァ-ヴー]", prev):
                out.append(nm)
                have.add(k)
                break
    return out


def link_components(comps, fugo, body):
    """構成要素ごとに、符号の説明（無ければ本文）の名前が合う符号を決める。"""
    legend = []
    for comp in comps:
        row = {"comp": comp, "codes": [], "names": [], "source": ""}
        for table, src in ((fugo, "符号の説明"), (body, "明細書の本文")):
            sc = [(name_match(comp, nm), c) for c, nm in table.items()]
            sc = [x for x in sc if x[0] > 0]
            if not sc:
                continue
            top = max(s for s, _ in sc)
            codes = sorted({c for s, c in sc if s == top}, key=lambda z: (len(z), z))
            row.update(codes=codes, names=sorted({table[c] for c in codes}), source=src)
            break
        legend.append(row)
    return legend


# =============================================================== まとめて実行
def build(fetcher, pid, claim, comps, comp_colors=None, title="", applicant="", use_ocr=True, fill=True,
          progress=None, max_figures=40, fugo_override=None):
    """文献番号と請求項（と構成要素の名前・色）から、色を付けた図面と対応表を作る。"""
    log = progress or (lambda *a, **k: None)
    log("公報を探しています…", 0.05)
    found = find_patent(fetcher, pid, claim, title, applicant, log=lambda m: log(m, 0.1))
    if not found.get("doc"):
        return {"ok": False, "error": "Google Patents で公報が見つかりませんでした。", "tried": found.get("tried", [])}
    doc, fdoc, idoc = found["doc"], found["fugo_doc"], found["image_doc"]
    fugo = dict(fugo_override or (fdoc["fugo"] if fdoc else {}))
    body = {}
    for d in (doc, fdoc, idoc):
        for c, n in ((d or {}).get("body_signs") or {}).items():
            body.setdefault(c, n)
    notes = []
    if found["found_by"].startswith("検索"):
        notes.append(f"文献番号 {pid} から直接は見つからなかったので、名称と出願人で検索して {doc['id']} を使いました。")
    if found.get("sim") is not None and found["sim"] < 0.5:
        notes.append(f"データの請求項と、この公報の請求項の近さが {found['sim']:.2f} と低めです（補正で変わった・別の公報の可能性）。")
    if fdoc and fdoc is not doc:
        notes.append(f"{doc['id']} に【符号の説明】が無いので、ファミリーの {fdoc['id']} のものを使いました。")
    if idoc and idoc is not doc:
        notes.append(f"{doc['id']} に図面の画像が無いので、ファミリーの {idoc['id']} の図面を使いました。")
    if not fugo:
        notes.append("【符号の説明】が見つからないので、明細書の本文の「部品名＋符号」の書き方だけで対応を決めました。")

    comps = list(dict.fromkeys(comps))
    extra = extra_components(claim, fugo, comps) if fugo else []
    all_comps = comps + extra
    legend = link_components(all_comps, fugo, body)
    comp_colors = dict(comp_colors or {})
    k = 0
    for row in legend:
        if row["comp"] not in comp_colors:
            while PALETTE[k % len(PALETTE)] in comp_colors.values() and k < len(PALETTE):
                k += 1
            comp_colors[row["comp"]] = PALETTE[k % len(PALETTE)]
            k += 1
        row["color"] = comp_colors[row["comp"]] if row["codes"] else None
        row["extra"] = row["comp"] in extra
        row["figs"] = []
    code_colors = {}
    for row in legend:
        for c in row["codes"]:
            code_colors.setdefault(c, row["color"])
    known = set(fugo) | set(body)
    no_fill = {c for c, nm in {**body, **fugo}.items() if re.search(r"(軸|方向|面|線|対象|空間|間隔|距離|角度|中心)$", nm)}

    # 図面の画像
    figures, pages = [], []
    if idoc and idoc["images"]:
        figrefs = idoc["figrefs"] if len(idoc["figrefs"]) == len(idoc["images"]) else []
        for i, im in enumerate(idoc["images"][:max_figures]):
            log(f"図面を取得しています（{i + 1}/{min(len(idoc['images']), max_figures)}）", 0.15 + 0.35 * i / max(1, len(idoc["images"])))
            p = fetcher.get(im["full"], ext=os.path.splitext(urllib.parse.urlparse(im["full"]).path)[1] or ".png", binary=True)
            if p:
                label = fig_label(figrefs[i]) if figrefs else f"図面 {i + 1}"
                pages.append({"path": p, "label": label, "callouts": im["callouts"]})
    elif doc.get("pdf") or (fdoc or {}).get("pdf"):
        pdf_url = doc.get("pdf") or fdoc.get("pdf")
        log("図面の画像が無いので、公報の PDF から図面のページを探します", 0.2)
        p = fetcher.get(pdf_url, ext=".pdf", binary=True)
        if p:
            for j, q in enumerate(pdf_to_images(p)):
                if looks_like_drawing(q):
                    pages.append({"path": q, "label": f"PDF {j + 1} ページ", "callouts": []})
            notes.append("図面は公報の PDF のページから取りました（図面のページかどうかは自動の見分けです）。")
    if not pages:
        return {"ok": False, "error": "図面の画像が見つかりませんでした。", "doc": doc, "tried": found["tried"],
                "legend": legend, "fugo": fugo}

    ocr = use_ocr and ocr_available()
    if use_ocr and not ocr:
        notes.append("OCR（Tesseract）が入っていないので、Google Patents が読み取った符号の位置だけを使いました。"
                     "英字つきの符号（30a など）は見つからないことがあります。")
    unknown = {}
    for i, pg in enumerate(pages):
        log(f"図面に色を付けています（{i + 1}/{len(pages)}）", 0.5 + 0.48 * i / len(pages))
        hits = []
        for c in pg["callouts"]:
            code = match_code(c["id"], known) or c["id"]
            hits.append({"code": code, "box": c["box"], "src": "google"})
            if code not in fugo:
                unknown.setdefault(code, {"figs": [], "label": c.get("label", "")})["figs"].append(pg["label"])
        if ocr:
            for t in ocr_numerals(pg["path"]):
                code = match_code(t["text"], known)
                box = t["box"]
                def _over(a, b):
                    return not (a[2] < b[0] - 5 or b[2] < a[0] - 5 or a[3] < b[1] - 5 or b[3] < a[1] - 5)
                dup = [h for h in hits if _over(h["box"], box)]
                if dup:
                    # Google の枠は矢印まで含んで大きいことがあるので、同じ符号なら OCR の小さい枠に置きかえる
                    for h in dup:
                        if code and h["code"] == code and (box[2] - box[0]) * (box[3] - box[1]) < \
                                (h["box"][2] - h["box"][0]) * (h["box"][3] - h["box"][1]):
                            h["box"] = box
                    continue
                if code:
                    hits.append({"code": code, "box": box, "src": "ocr"})
                    if code not in fugo:
                        unknown.setdefault(code, {"figs": [], "label": ""})["figs"].append(pg["label"])
                elif re.fullmatch(r"[0-9]{1,4}[a-z]?", t["text"]) and t["conf"] >= 85:
                    unknown.setdefault(t["text"], {"figs": [], "label": ""})["figs"].append(pg["label"])
                    hits.append({"code": t["text"], "box": box, "src": "ocr"})
        img, filled = render(pg["path"], hits, code_colors, fill=fill, no_fill=no_fill, unknown=set(unknown))
        codes = sorted({h["code"] for h in hits}, key=lambda z: (len(z), z))
        claim_codes = [c for c in codes if c in code_colors]
        for row in legend:
            if any(c in claim_codes for c in row["codes"]):
                row["figs"].append(pg["label"])
        figures.append({"label": pg["label"], "png": png_bytes(img), "codes": codes, "claim_codes": claim_codes,
                        "filled": filled, "n_google": sum(h["src"] == "google" for h in hits),
                        "n_ocr": sum(h["src"] == "ocr" for h in hits)})
    for c, u in unknown.items():
        u["body_name"] = body.get(c, "")
        u["figs"] = list(dict.fromkeys(u["figs"]))
    others = [(c, nm) for c, nm in sorted(fugo.items(), key=lambda z: (len(z[0]), z[0])) if c not in code_colors]
    log("できました", 1.0)
    return {"ok": True, "doc": {k: doc[k] for k in ("id", "url", "title", "pdf")}, "found_by": found["found_by"],
            "sim": found.get("sim"), "fugo_doc": (fdoc or {}).get("id"), "fugo_src": (fdoc or {}).get("fugo_src", ""),
            "image_doc": (idoc or {}).get("id") or doc["id"], "image_url": (idoc or doc)["url"],
            "fugo": fugo, "body": body, "legend": legend, "figures": figures,
            "unknown": dict(sorted(unknown.items(), key=lambda z: (len(z[0]), z[0]))), "others": others,
            "notes": notes, "ocr": ocr, "tried": found["tried"]}


def build_local(pages_paths, claim, comps, fugo_text, comp_colors=None, use_ocr=True, fill=True, progress=None):
    """公報を自分で用意したとき（PDF・図面の画像＋貼り付けた符号の説明）。"""
    log = progress or (lambda *a, **k: None)
    fugo = parse_fugo_text(fugo_text) if fugo_text else {}
    comps = list(dict.fromkeys(comps))
    extra = extra_components(claim, fugo, comps) if fugo else []
    legend = link_components(comps + extra, fugo, {})
    comp_colors = dict(comp_colors or {})
    for i, row in enumerate(legend):
        comp_colors.setdefault(row["comp"], PALETTE[(len(comp_colors) + i) % len(PALETTE)])
        row["color"] = comp_colors[row["comp"]] if row["codes"] else None
        row["extra"], row["figs"] = row["comp"] in extra, []
    code_colors = {}
    for row in legend:
        for c in row["codes"]:
            code_colors.setdefault(c, row["color"])
    no_fill = {c for c, nm in fugo.items() if re.search(r"(軸|方向|面|線|対象|空間|間隔|距離|角度|中心)$", nm)}
    ocr = use_ocr and ocr_available()
    figures, unknown = [], {}
    for i, p in enumerate(pages_paths):
        log(f"図面に色を付けています（{i + 1}/{len(pages_paths)}）", 0.1 + 0.85 * i / len(pages_paths))
        label = f"図面 {i + 1}"
        hits = []
        if ocr:
            for t in ocr_numerals(p):
                code = match_code(t["text"], set(fugo))
                if code:
                    hits.append({"code": code, "box": t["box"], "src": "ocr"})
                elif re.fullmatch(r"[0-9]{1,4}[a-z]?", t["text"]) and t["conf"] >= 85:
                    unknown.setdefault(t["text"], {"figs": [], "label": "", "body_name": ""})["figs"].append(label)
                    hits.append({"code": t["text"], "box": t["box"], "src": "ocr"})
        img, filled = render(p, hits, code_colors, fill=fill, no_fill=no_fill, unknown=set(unknown))
        codes = sorted({h["code"] for h in hits}, key=lambda z: (len(z), z))
        claim_codes = [c for c in codes if c in code_colors]
        for row in legend:
            if any(c in claim_codes for c in row["codes"]):
                row["figs"].append(label)
        figures.append({"label": label, "png": png_bytes(img), "codes": codes, "claim_codes": claim_codes,
                        "filled": filled, "n_google": 0, "n_ocr": len(hits)})
    others = [(c, nm) for c, nm in sorted(fugo.items(), key=lambda z: (len(z[0]), z[0])) if c not in code_colors]
    notes = [] if ocr else ["OCR（Tesseract）が入っていないので、図面の中の符号の位置を読めませんでした。"]
    return {"ok": True, "doc": {"id": "", "url": "", "title": "", "pdf": ""}, "found_by": "読み込んだファイル", "sim": None,
            "fugo_doc": None, "fugo_src": "貼り付け", "image_doc": "", "image_url": "", "fugo": fugo, "body": {},
            "legend": legend, "figures": figures, "unknown": unknown, "others": others, "notes": notes, "ocr": ocr,
            "tried": []}


# =============================================================== 報告の HTML（画像を埋め込んだ1ファイル）
def report_html(res, claim="", only_claim_figs=True):
    e = _html.escape
    rows = []
    for L in res["legend"]:
        sw = (f'<span class="sw" style="background:{L["color"]}"></span>' if L["color"] else '<span class="sw none"></span>')
        codes = "、".join(L["codes"]) or "―"
        names = "、".join(L["names"])
        src = L["source"] + ("（請求項に同じ名前）" if L.get("extra") else "")
        where = "、".join(L["figs"]) if L["figs"] else ("図面で見つからない" if L["codes"] else "")
        rows.append(f"<tr><td>{sw}{e(L['comp'])}</td><td>{e(codes)}</td><td>{e(names)}</td><td>{e(src)}</td><td>{e(where)}</td></tr>")
    figs = [f for f in res["figures"] if f["claim_codes"] or not only_claim_figs]
    imgs = "".join(
        f'<figure><img src="data:image/png;base64,{base64.b64encode(f["png"]).decode()}" alt="{e(f["label"])}">'
        f'<figcaption>{e(f["label"])}：請求項の構成要素の符号 {e("、".join(f["claim_codes"]) or "なし")}</figcaption></figure>'
        for f in figs)
    unk = "、".join(f"{e(c)}（{e('・'.join(u['figs'][:3]))}{'・本文では「' + e(u['body_name']) + '」' if u.get('body_name') else ''}）"
                   for c, u in res["unknown"].items()) or "なし"
    others = "、".join(f"{e(c)} {e(n)}" for c, n in res["others"]) or "なし"
    d = res["doc"]
    title = d.get("title") or "請求項1"
    src = f'<a href="{e(d["url"])}">{e(d["id"])}</a>' if d.get("url") else ""
    notes = "".join(f"<li>{e(n)}</li>" for n in res["notes"])
    return f"""<!doctype html><html lang="ja"><meta charset="utf-8"><title>図面の構成要件</title>
<style>body{{font-family:"Meiryo","Noto Sans CJK JP",sans-serif;margin:24px;color:#1b2330;background:#fff;line-height:1.6}}
table{{border-collapse:collapse;margin:12px 0}}td,th{{border:1px solid #ccd2da;padding:4px 10px;text-align:left;vertical-align:top}}
.sw{{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:8px;vertical-align:-2px}}.none{{border:1px dashed #9aa0a6}}
figure{{margin:18px 0}}img{{max-width:100%;border:1px solid #ccd2da}}.note{{color:#556072;font-size:13px;max-width:80ch}}
.claim{{white-space:pre-wrap;border:1px solid #ccd2da;border-radius:6px;padding:10px;font-size:14px}}</style>
<h1>{e(title)}：構成要件と図面の対応</h1>
<p>公報：{src}　図面：{e(res.get("image_doc") or "")}　符号の説明：{e(res.get("fugo_doc") or res.get("fugo_src") or "")}</p>
<p class="note">色つきの枠と塗り＝請求項の構成要素に当たる部品。灰色の枠＝請求項には出てこない部品（実施形態だけの部品）。{e(NOTE)}</p>
{('<ul class="note">' + notes + '</ul>') if notes else ''}
{('<div class="claim">' + e(claim) + '</div>') if claim else ''}
<table><tr><th>請求項の構成要素</th><th>符号</th><th>符号の説明の名前</th><th>対応の根拠</th><th>図面</th></tr>{''.join(rows)}</table>
<p class="note">請求項に出てこない符号：{others}</p>
<p class="note">図面にあるのに【符号の説明】に無い符号（書き漏れの手がかり。読み違いのこともある）：{unk}</p>
{imgs}</html>"""


# =============================================================== コマンドとして使う
def _components_from_pipeline(claim, pipeline_dir):
    import sys
    sys.path.insert(0, pipeline_dir)
    import patent_pipeline as pp
    info, judged, _ = pp.llm_select.analyze_claim_a2(pp, pp, claim, use_llm=False)
    rels = [{"source": c["source"], "relation": c["relation"], "target": c["target"]}
            for c, j in zip(info["cands"], judged) if j["status"] in ("採用", "要確認")]
    cols = {k: v["border"] for k, v in pp.component_colors(rels).items()}
    return list(cols), cols


def main():
    ap = argparse.ArgumentParser(description="特許の図面に、請求項の構成要素を色で示す")
    ap.add_argument("--patent", default="", help="文献番号（例：特開2022-144957、特許7800720、JP2022144957A）")
    ap.add_argument("--title", default="")
    ap.add_argument("--applicant", default="")
    ap.add_argument("--pdf", default="", help="公報の PDF（自分で用意したとき）")
    ap.add_argument("--images", nargs="*", default=[], help="図面の画像（自分で用意したとき）")
    ap.add_argument("--fugo", default="", help="【符号の説明】のテキストファイル（自分で用意したとき）")
    ap.add_argument("--claim", required=True, help="請求項の本文のテキストファイル")
    ap.add_argument("--components", default="", help="構成要素の名前（読点区切り）。省くと規則（LLM なし）で取り出す")
    ap.add_argument("--out", default="drawing_out")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--cache", default="")
    ap.add_argument("--pipeline-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    claim = open(args.claim, encoding="utf-8").read().strip()
    if args.components:
        comps, cols = [c.strip() for c in re.split(r"[、,]", args.components) if c.strip()], {}
    else:
        comps, cols = _components_from_pipeline(claim, args.pipeline_dir)

    def prog(msg, frac=None):
        print(f"[{int((frac or 0) * 100):3d}%] {msg}")

    if args.patent:
        res = build(Fetcher(args.cache or None), args.patent, claim, comps, cols, args.title, args.applicant,
                    use_ocr=not args.no_ocr, fill=not args.no_fill, progress=prog)
    else:
        pages = pdf_to_images(args.pdf) if args.pdf else list(args.images)
        if args.pdf:
            pages = [p for p in pages if looks_like_drawing(p)]
        fugo_text = open(args.fugo, encoding="utf-8").read() if args.fugo else ""
        res = build_local(pages, claim, comps, fugo_text, cols, use_ocr=not args.no_ocr, fill=not args.no_fill, progress=prog)
    if not res["ok"]:
        raise SystemExit(res["error"] + " 試した番号：" + "、".join(res.get("tried", [])))
    for i, f in enumerate(res["figures"], 1):
        open(os.path.join(args.out, f"図面_{i:02d}.png"), "wb").write(f["png"])
    open(os.path.join(args.out, "図面の構成要件.html"), "w", encoding="utf-8").write(report_html(res, claim))
    slim = {k: v for k, v in res.items() if k != "figures"}
    slim["figures"] = [{k: v for k, v in f.items() if k != "png"} for f in res["figures"]]
    json.dump(slim, open(os.path.join(args.out, "結果.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_link = sum(1 for L in res["legend"] if L["codes"])
    print(f"{res['doc'].get('id') or ''} 構成要素 {len(res['legend'])} 個のうち、符号と結び付いたもの {n_link} 個。"
          f"図面 {len(res['figures'])} 枚 → {args.out}")
    for n in res["notes"]:
        print("・" + n)


if __name__ == "__main__":
    main()
