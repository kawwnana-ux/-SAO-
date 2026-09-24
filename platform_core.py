# -*- coding: utf-8 -*-
"""
platform_core.py
=================
特許分析プラットフォーム（app.py）のデータ処理部分。
画面（Streamlit）に依存しない処理だけをまとめる。

  ・コーパス（532件）の読み込み：corpus_sao_532.json
  ・採用／要確認／除外の判定（選別モデルの確率を、交差検証で較正した帯で区切る）
  ・人手の確認・修正結果の反映
  ・構造的特徴量（レーダーチャート用。特許の「強さ」ではなく請求項の書き方の構造）
  ・SAOネットワーク（全特許の構成要素を基本語にまとめたグラフ）
  ・類似性（SAOトリプルのTF-IDFとコサイン類似度）
  ・Excel／CSVへの書き出し
"""
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORPUS_FILE = HERE / "corpus_sao_532.json"

STATUS_ACCEPT = "採用"
STATUS_REVIEW = "要確認"
STATUS_REJECT = "除外"
STATUS_ORDER = [STATUS_ACCEPT, STATUS_REVIEW, STATUS_REJECT]

HAS_WORDS = ("有する", "備える", "具備する", "含む", "含める")

# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------


def load_corpus(path=CORPUS_FILE):
    """corpus_sao_532.json を読み込む。
    戻り値: dict(meta=..., bands=..., patents=[{id, title, applicant, company, group,
    fi, fi_sub, year, url, text, x, y, z, map_x, map_y, relations=[...]}, ...])
    relations の各要素: source, relation, target, prob, status, origin"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify(prob, selected, bands):
    """選別モデルの確率を、採用／要確認／除外に振り分ける。
    bands = {"accept": 採用の下限, "threshold": 選別のしきい値, "review_low": 要確認の下限}
      ・選ばれた関係で確率が accept 以上 → 採用
      ・選ばれた関係で accept 未満、または選ばれなかったが review_low 以上 → 要確認
      ・それ以外 → 除外"""
    if selected and prob >= bands["accept"]:
        return STATUS_ACCEPT
    if selected or prob >= bands["review_low"]:
        return STATUS_REVIEW
    return STATUS_REJECT


# ---------------------------------------------------------------------------
# 人手の確認結果の反映
# ---------------------------------------------------------------------------


def effective_relations(patent, reviews=None):
    """人手の確認結果があればそれを、無ければ「採用」＋「要確認のうち選別モデルが
    選んだもの」を、その特許の現在のSAOとして返す（分析ページ共通の入力）。"""
    if reviews and patent["id"] in reviews:
        return [r for r in reviews[patent["id"]] if r.get("keep", True)]
    return [r for r in patent["relations"] if r.get("selected")]


def review_table(patent, reviews=None):
    """人手確認用の表（DataFrame）。確認済みならその内容を、未確認なら
    採用・要確認の候補を、採用=True/要確認は選別モデルの判断を初期値にして返す。"""
    if reviews and patent["id"] in reviews:
        rows = reviews[patent["id"]]
        return pd.DataFrame([{
            "採用する": bool(r.get("keep", True)), "主語(S)": r["source"], "関係(A)": r["relation"],
            "目的語(O)": r["target"], "確率": r.get("prob"), "判定": r.get("status", "人手追加"),
            "抽出元": r.get("origin", "人手"),
        } for r in rows])
    rows = [r for r in patent["relations"] if r["status"] != STATUS_REJECT]
    return pd.DataFrame([{
        "採用する": bool(r.get("selected")), "主語(S)": r["source"], "関係(A)": r["relation"],
        "目的語(O)": r["target"], "確率": round(float(r["prob"]), 3), "判定": r["status"],
        "抽出元": origin_label(r.get("origin", "")),
    } for r in sorted(rows, key=lambda r: -r["prob"])],
        columns=["採用する", "主語(S)", "関係(A)", "目的語(O)", "確率", "判定", "抽出元"])


def table_to_review(df):
    """人手確認の表（data_editor の結果）を、保存用のリストに戻す。"""
    out = []
    for _, row in df.iterrows():
        s, a, o = (str(row.get(k) or "").strip() for k in ("主語(S)", "関係(A)", "目的語(O)"))
        if not (s and a and o):
            continue
        prob = row.get("確率")
        out.append({
            "source": s, "relation": a, "target": o, "keep": bool(row.get("採用する", True)),
            "prob": None if prob is None or (isinstance(prob, float) and math.isnan(prob)) else float(prob),
            "status": row.get("判定") if isinstance(row.get("判定"), str) and row.get("判定") else "人手追加",
            "origin": row.get("抽出元") if isinstance(row.get("抽出元"), str) and row.get("抽出元") else "人手",
        })
    return out


def reviews_to_csv(reviews):
    rows = [{"特許番号": pid, **r} for pid, rs in reviews.items() for r in rs]
    return pd.DataFrame(rows, columns=["特許番号", "source", "relation", "target", "keep", "prob",
                                       "status", "origin"]).to_csv(index=False).encode("utf-8-sig")


def reviews_from_csv(data):
    df = pd.read_csv(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    out = defaultdict(list)
    for _, r in df.iterrows():
        out[str(r["特許番号"])].append({
            "source": str(r["source"]), "relation": str(r["relation"]), "target": str(r["target"]),
            "keep": str(r.get("keep", True)).lower() in ("true", "1", "yes"),
            "prob": None if pd.isna(r.get("prob")) else float(r["prob"]),
            "status": r.get("status") if isinstance(r.get("status"), str) else "人手追加",
            "origin": r.get("origin") if isinstance(r.get("origin"), str) else "人手",
        })
    return dict(out)


# ---------------------------------------------------------------------------
# 基本語（ネットワーク・類似度用のノード名の正規化）
# ---------------------------------------------------------------------------

_NUM = r"[0-9０-９一二三四五六七八九十]+"
_PREFIX_RE = re.compile(r"^(前記|当該|該|上記|各|複数の|少なくとも(一|１|1)つの|一対の|１対の|1対の)+")
_ORD_RE = re.compile(r"第" + _NUM + r"の?")
_TAIL_RE = re.compile(r"(" + _NUM + r"|[A-Za-zＡ-Ｚａ-ｚ]+)$")
_SUFFIX_RE = re.compile(r"(の各々|のそれぞれ|の各|それぞれ|各々)$")


def base_term(node):
    """「前記第１トランジスタの第２端」→「トランジスタの端」のように、
    番号・指示語を除いて、特許をまたいで比べられる基本語にする。"""
    t = str(node).strip()
    t = _PREFIX_RE.sub("", t)
    t = _SUFFIX_RE.sub("", t)
    t = _ORD_RE.sub("", t)
    t = _TAIL_RE.sub("", t)
    t = re.sub(r"\s+", "", t)
    return t or str(node)


_ORIGIN = [("LLM", ("LLMraw", "E1:llm_direct")),
           ("GiNZA補完", ("E1:claim_title_ginza", "E1:ginza_has_fallback", "E1:attribute",
                        "E1:ginza_has_fallback_conflict", "E1:claim_title_ginza_conflict")),
           ("GiNZA", ("G",)), ("分割GiNZA", ("GS",)), ("ノード結合", ("MRG",)), ("係り受け", ("DEP",)),
           ("持ち主つき", ("OWN",))]


def origin_label(srcs):
    """候補の出どころ（内部の記号）を、画面用の短い名前にする。
    srcs: 記号のリスト（"E1:llm_direct" など）か、"+" でつないだ文字列。"""
    items = srcs.split("+") if isinstance(srcs, str) else list(srcs)
    out = []
    for name, keys in _ORIGIN:
        for x in items:
            if x in keys or (x.split(":")[0] in keys and ":" not in keys[0]):
                out.append(name)
    return "・".join(dict.fromkeys(out)) or (srcs if isinstance(srcs, str) else "")


def is_has(rel):
    return rel in ("の",) + HAS_WORDS or any(h in rel for h in HAS_WORDS)


# ---------------------------------------------------------------------------
# 構造的特徴量（レーダーチャート用）
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(
    r"[0-9０-９]+(\.[0-9０-９]+)?\s*(%|％|μm|um|ｍｍ|mm|nm|ｎｍ|℃|°C|V|Ｖ|A|Ａ|Ω|Hz|倍|度|原子%|atoms|cm)"
    r"|以上|以下|未満|より大きい|より小さい|超え")

RADAR_AXES = ["構成要素数", "SAO関係数", "階層の深さ", "分岐の多さ", "関係の多様性", "機能・配置の記述", "数値限定"]


def _longest_path(edges):
    children = defaultdict(set)
    nodes = set()
    for s, t in edges:
        if s != t:
            children[s].add(t)
            nodes |= {s, t}
    memo = {}

    def depth(n, stack):
        if n in memo:
            return memo[n]
        best = 0
        for c in children[n]:
            if c in stack:
                continue
            best = max(best, 1 + depth(c, stack | {c}))
        memo[n] = best
        return best

    return max((depth(n, {n}) for n in nodes), default=0)


def structural_features(relations, text=""):
    """1件の請求項のSAOから構造的特徴量を求める（値の大小は特許の優劣ではない）。"""
    nodes = {x for r in relations for x in (r["source"], r["target"])}
    has_edges = [(r["source"], r["target"]) for r in relations if is_has(r["relation"])]
    out_deg = Counter(r["source"] for r in relations)
    rel_kinds = {re.sub(r"(され|させ|して|する|した|される)+$", "", r["relation"]) for r in relations}
    non_has = sum(1 for r in relations if not is_has(r["relation"]))
    return {
        "構成要素数": len(nodes),
        "SAO関係数": len(relations),
        "階層の深さ": _longest_path(has_edges),
        "分岐の多さ": round(float(np.mean(list(out_deg.values()))), 2) if out_deg else 0.0,
        "関係の多様性": len(rel_kinds),
        "機能・配置の記述": round(non_has / len(relations), 3) if relations else 0.0,
        "数値限定": len(_NUMERIC_RE.findall(text or "")),
        "請求項の文字数": len(text or ""),
    }


def feature_table(corpus, reviews=None):
    rows = []
    for p in corpus["patents"]:
        f = structural_features(effective_relations(p, reviews), p["text"])
        rows.append({"特許番号": p["id"], "発明の名称": p["title"], "企業": p["company"], "出願年": p["year"], **f})
    return pd.DataFrame(rows)


def percentile_scores(df, axes=RADAR_AXES):
    """各軸をコーパス内の順位（0〜100のパーセンタイル）に変換する。"""
    out = df.copy()
    for a in axes:
        out[a] = (df[a].rank(pct=True, method="average") * 100).round(1)
    return out


# ---------------------------------------------------------------------------
# SAOネットワーク
# ---------------------------------------------------------------------------


def build_network(corpus, patent_ids=None, reviews=None, min_patents=2, top_n=60):
    """全特許（または指定特許）のSAOを基本語でまとめたネットワーク。
    ノード: 基本語。重み = その語が現れる特許の件数。
    エッジ: 基本語どうしのSAO関係。重み = その組が現れる特許の件数、代表の関係名。"""
    node_patents = defaultdict(set)
    edge_patents = defaultdict(set)
    edge_rel = defaultdict(Counter)
    surface = defaultdict(Counter)
    ids = set(patent_ids) if patent_ids else None
    for p in corpus["patents"]:
        if ids is not None and p["id"] not in ids:
            continue
        for r in effective_relations(p, reviews):
            s, t = base_term(r["source"]), base_term(r["target"])
            if s == t:
                continue
            node_patents[s].add(p["id"])
            node_patents[t].add(p["id"])
            surface[s][r["source"]] += 1
            surface[t][r["target"]] += 1
            edge_patents[(s, t)].add(p["id"])
            edge_rel[(s, t)][r["relation"]] += 1
    keep = [n for n, ps in node_patents.items() if len(ps) >= min_patents]
    keep = set(sorted(keep, key=lambda n: -len(node_patents[n]))[:top_n])
    nodes = [{"id": n, "patents": sorted(node_patents[n]), "count": len(node_patents[n]),
              "surfaces": [s for s, _ in surface[n].most_common(8)]} for n in keep]
    edges = [{"source": s, "target": t, "count": len(ps), "relation": edge_rel[(s, t)].most_common(1)[0][0],
              "patents": sorted(ps)}
             for (s, t), ps in edge_patents.items() if s in keep and t in keep]
    return nodes, edges


def layout_network(nodes, edges, seed=0):
    import networkx as nx

    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"])
    for e in edges:
        w = e["count"]
        if g.has_edge(e["source"], e["target"]):
            g[e["source"]][e["target"]]["weight"] += w
        else:
            g.add_edge(e["source"], e["target"], weight=w)
    if not len(g):
        return {}
    k = 2.4 / math.sqrt(max(len(g), 1))
    return nx.spring_layout(g, k=k, iterations=120, seed=seed, weight="weight")


def patents_with_node(corpus, term, reviews=None):
    """基本語 term がSAOに現れる特許と、そのSAO・本文中の出現表記を返す。"""
    out = []
    for p in corpus["patents"]:
        rels = [r for r in effective_relations(p, reviews)
                if base_term(r["source"]) == term or base_term(r["target"]) == term]
        if rels:
            surf = sorted({x for r in rels for x in (r["source"], r["target"]) if base_term(x) == term},
                          key=len, reverse=True)
            out.append({"patent": p, "relations": rels, "surfaces": surf})
    return out


def highlight(text, words):
    """本文中の words をマーカーで強調した HTML を返す。"""
    import html as _html

    esc = _html.escape(text)
    for w in sorted({w for w in words if w}, key=len, reverse=True):
        esc = esc.replace(_html.escape(w), "\u0000" + _html.escape(w) + "\u0001")
    esc = esc.replace("\u0000", '<mark style="background:#fde68a;padding:0 2px;border-radius:3px">')
    esc = esc.replace("\u0001", "</mark>")
    return esc.replace("\n", "<br>")


# ---------------------------------------------------------------------------
# 類似性（SAOトリプルのTF-IDF）
# ---------------------------------------------------------------------------


def sao_tokens(relations):
    toks = []
    for r in relations:
        s, t = base_term(r["source"]), base_term(r["target"])
        rel = "有する" if is_has(r["relation"]) else re.sub(r"(される|させる|する|され|して)$", "", r["relation"])
        toks += ["T:%s→%s→%s" % (s, rel, t), "P:%s→%s" % (s, t), "N:" + s, "N:" + t]
    return toks


def similarity_matrix(corpus, reviews=None):
    """全特許のSAOをTF-IDFベクトルにし、コサイン類似度の行列を返す。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [sao_tokens(effective_relations(p, reviews)) for p in corpus["patents"]]
    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, min_df=1)
    X = vec.fit_transform(docs)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)
    return S


def similarity_explain(rel_a, rel_b):
    """2件のSAOの共通部分（基本語のトリプル・組・ノード）。"""
    ta, tb = set(sao_tokens(rel_a)), set(sao_tokens(rel_b))
    common = ta & tb
    return {
        "共通のSAO": sorted(x[2:] for x in common if x.startswith("T:")),
        "共通の組": sorted(x[2:] for x in common if x.startswith("P:")),
        "共通の構成要素": sorted(x[2:] for x in common if x.startswith("N:")),
    }


# ---------------------------------------------------------------------------
# 集計（ダッシュボード・バブル・ヒートマップ）
# ---------------------------------------------------------------------------


def status_counts(corpus, reviews=None):
    c = Counter()
    for p in corpus["patents"]:
        for r in p["relations"]:
            c[r["status"]] += 1
        c[STATUS_REJECT] += p.get("n_rejected", 0)
    confirmed = sum(sum(1 for r in rs if r.get("keep", True)) for rs in (reviews or {}).values())
    return {
        "特許件数": len(corpus["patents"]),
        STATUS_ACCEPT: c[STATUS_ACCEPT], STATUS_REVIEW: c[STATUS_REVIEW], STATUS_REJECT: c[STATUS_REJECT],
        "抽出SAO": sum(1 for p in corpus["patents"] for r in p["relations"] if r.get("selected")),
        "人手確認済みの特許": len(reviews or {}), "確定SAO（人手確認済み）": confirmed,
    }


def company_year_bubble(corpus, reviews=None):
    rows = []
    for p in corpus["patents"]:
        if p.get("year") is None:
            continue
        rows.append({"企業": p["company"], "出願年": p["year"],
                     "SAO数": len(effective_relations(p, reviews))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["企業", "出願年"]).agg(請求項数=("SAO数", "size"), 平均SAO数=("SAO数", "mean"),
                                              SAO数合計=("SAO数", "sum")).reset_index())


def company_tech_matrix(corpus, axis="FIサブクラス", reviews=None, top_tech=15):
    rows = []
    for p in corpus["patents"]:
        if axis == "FIサブクラス":
            techs = p.get("fi_sub") or []
        elif axis == "FIメイングループ":
            techs = p.get("fi_main") or []
        else:
            techs = sorted({base_term(x) for r in effective_relations(p, reviews)
                            for x in (r["source"], r["target"])})
        for t in techs:
            rows.append({"企業": p["company"], "技術": t, "特許番号": p["id"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    top = df.groupby("技術")["特許番号"].nunique().sort_values(ascending=False).head(top_tech).index
    m = df[df["技術"].isin(top)].groupby(["企業", "技術"])["特許番号"].nunique().unstack(fill_value=0)
    return m[list(top)]


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------


def export_excel(corpus, reviews=None, sim=None, top_k=5):
    """分析結果一式を1つのExcelファイル（bytes）にする。"""
    buf = io.BytesIO()
    feats = feature_table(corpus, reviews)
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([
            {"項目": k, "値": v} for k, v in status_counts(corpus, reviews).items()
        ] + [{"項目": "採用の下限確率", "値": corpus["bands"]["accept"]},
             {"項目": "選別のしきい値", "値": corpus["bands"]["threshold"]},
             {"項目": "要確認の下限確率", "値": corpus["bands"]["review_low"]},
             {"項目": "抽出手法", "値": corpus["meta"].get("method", "")}]).to_excel(xw, sheet_name="概要", index=False)
        pd.DataFrame([{
            "特許番号": p["id"], "発明の名称": p["title"], "出願人": p["applicant"], "企業": p["company"],
            "出願年": p["year"], "FI": p["fi"], "人手確認": p["id"] in (reviews or {}), "URL": p.get("url", ""),
        } for p in corpus["patents"]]).to_excel(xw, sheet_name="特許一覧", index=False)
        pd.DataFrame([{
            "特許番号": p["id"], "主語(S)": r["source"], "関係(A)": r["relation"], "目的語(O)": r["target"],
            "確率": round(float(r["prob"]), 4), "判定": r["status"], "選別モデルの選択": bool(r.get("selected")),
            "抽出元": origin_label(r.get("origin", "")),
        } for p in corpus["patents"] for r in p["relations"]]).to_excel(xw, sheet_name="SAO（AI判定）", index=False)
        if reviews:
            pd.DataFrame([{
                "特許番号": pid, "主語(S)": r["source"], "関係(A)": r["relation"], "目的語(O)": r["target"],
                "採用": r.get("keep", True), "元の判定": r.get("status", ""),
            } for pid, rs in reviews.items() for r in rs]).to_excel(xw, sheet_name="SAO（人手確認済み）", index=False)
        feats.to_excel(xw, sheet_name="構造的特徴", index=False)
        if sim is not None:
            ids = [p["id"] for p in corpus["patents"]]
            rows = []
            for i, pid in enumerate(ids):
                for j in np.argsort(-sim[i])[:top_k]:
                    rows.append({"特許番号": pid, "類似特許": ids[j], "類似度": round(float(sim[i, j]), 4)})
            pd.DataFrame(rows).to_excel(xw, sheet_name="類似特許", index=False)
    return buf.getvalue()


def relations_csv(corpus, reviews=None):
    rows = []
    for p in corpus["patents"]:
        for r in effective_relations(p, reviews):
            rows.append({"特許番号": p["id"], "発明の名称": p["title"], "企業": p["company"],
                         "主語(S)": r["source"], "関係(A)": r["relation"], "目的語(O)": r["target"],
                         "判定": "人手確認済み" if reviews and p["id"] in reviews else r.get("status", "")})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
