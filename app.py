# -*- coding: utf-8 -*-
"""
特許分析プラットフォーム（卒業研究）
====================================

「AIを用いた日本語特許文献の構造分析に関する研究
 ―SAO構造を用いた半導体関連特許の類似性分析―」

請求項からSAO（主語―関係―目的語）構造を取り出す抽出エンジン（LLM＋GiNZA＋候補選別
モデル）を中心に、人が確認・修正しながら使う「人とAIの協働型」の特許分析
プラットフォームとしてまとめたもの。

ページ構成（サイドバーのナビゲーション）
  概要
    🏠 ダッシュボード          … 抽出・確認の進み具合と、判定の根拠
  抽出と確認
    🧪 AI解析                 … 1件の請求項を解析し、処理の各段階と採用／要確認／除外を表示
    ✍️ 人手確認（532件）       … AIの判定を人が確認・修正して確定する
  可視化・分析
    🌍 Patent World           … オズの世界（3次元の技術ランドスケープ）
    🕸️ SAOネットワーク         … 全特許の構成要素のつながり。ノードをクリックすると該当請求項へ
    🗺️ 類似性マップ            … SAOの類似度で特許を配置。点をクリックすると似た特許と共通部分
    🧭 構造レーダー            … 請求項の構造的特徴（特許の強さではない）
    🫧 技術分布               … 出願年×SAO数×請求項数のバブル、企業×技術のヒートマップ
  個別ツール
    🐚 2つの請求項を比較 / 🪼 従属請求項を展開 / 🔦 まとめて検索 / 📊 特許統計分析（CSV）/ ✅ 精度検証
  出力
    📤 エクスポート            … Excel／CSV、人手確認結果の保存と読み込み

【実行方法（自分のPC・Ollama）】
    pip install -r requirements.txt
    ollama pull qwen2.5:7b
    streamlit run app.py
同じフォルダに、translate_sao.py・patent_pipeline.py・en_relation_rules.py・
platform_core.py・corpus_sao_532.json・oz_world_embed.html・sao_selector*.py・
sao_selector*_train.npz・claim_segmenter.py・dep_pairs.py・node_match_eval.py・
nested_graph.py を置いてください。

【Streamlit Community Cloud】
Secrets に OPENROUTER_API_KEY を設定すると、translate_sao._ollama_chat を OpenRouter
経由の qwen2.5-7b-instruct に差し替えて動かす（translate_sao.py 自体は変更しない）。
532件の分析ページ（ダッシュボード・人手確認・可視化）は事前に抽出した
corpus_sao_532.json を使うので、LLMを呼ばずに動く。
"""

import csv
import html
import io
import json
import os
import re
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import patent_pipeline as pp
import platform_core as PC
import translate_sao as ts

try:
    from nested_graph import relations_to_nested_dot
except ImportError:
    relations_to_nested_dot = None

SELECTOR_MODULES = {}
for _name in ("sao_selector12", "sao_selector11", "sao_selector10", "sao_selector"):
    try:
        SELECTOR_MODULES[_name] = __import__(_name)
    except Exception:  # noqa: BLE001
        pass

st.set_page_config(page_title="特許分析プラットフォーム", layout="wide", page_icon="🔬")

APP_DIR = Path(__file__).resolve().parent
OZ_WORLD_HTML_PATH = APP_DIR / "oz_world_embed.html"
CLOUD_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# 抽出手法（プリセット）
# ---------------------------------------------------------------------------
M12 = "実験12：係り受け候補＋2段階選別"
M11 = "実験11：ノード結合＋完全一致学習"
M10 = "実験10：請求項分割＋候補選別"
M9 = "実験9：候補選別モデル"
M4 = "実験4相当（旧ベースライン）"
M1 = "実験1相当（検証なし）"
METHOD_MODULE = {M12: "sao_selector12", M11: "sao_selector11", M10: "sao_selector10", M9: "sao_selector"}
RECOMMENDED = M12  # 主指標（トリプル完全一致）で最良の手法
METHODS = [m for m in (M12, M11, M10, M9) if METHOD_MODULE[m] in SELECTOR_MODULES] + [M4, M1]
if RECOMMENDED not in METHODS:
    RECOMMENDED = METHODS[0]

# 主指標：トリプル完全一致（主語・関係・目的語がすべて一致。表記の揺れのみ吸収）。
# 実験9以降は5分割の入れ子交差検証（評価する請求項を学習に使っていない）。
EXACT_HISTORY = [
    {"手法": "本文を読まないでたらめ出力", "適合率": "19.7%", "再現率": "19.7%", "F1": "19.7%"},
    {"手法": "実験4（旧ベースライン）", "適合率": "35.4%", "再現率": "40.5%", "F1": "37.8%"},
    {"手法": "実験9 候補選別", "適合率": "50.6%", "再現率": "46.2%", "F1": "48.3%"},
    {"手法": "実験10a 請求項分割", "適合率": "51.1%", "再現率": "48.1%", "F1": "49.6%"},
    {"手法": "実験11 ノード結合＋完全一致学習", "適合率": "59.5%", "再現率": "45.1%", "F1": "51.3%"},
    {"手法": "実験12 係り受け候補＋2段階選別", "適合率": "60.5%", "再現率": "48.1%", "F1": "53.6%"},
]

SRC_GROUPS = [
    ("LLM抽出（qwen2.5）", ("LLMraw", "E1:llm_direct")),
    ("GiNZA補完", ("E1:claim_title_ginza", "E1:ginza_has_fallback", "E1:attribute",
                 "E1:ginza_has_fallback_conflict", "E1:claim_title_ginza_conflict")),
    ("GiNZA単体（係り受け・位置・所有）", ("G:",)),
    ("請求項の分割（手がかり句）", ("GS:",)),
    ("ノード結合（XのY）", ("MRG",)),
    ("係り受け候補（述語の項の組）", ("DEP",)),
    ("持ち主つきノード", ("OWN",)),
]
ORIGIN_LABELS = {"LLMraw": "LLM", "E1": "LLM＋GiNZA補完", "G": "GiNZA", "GS": "分割GiNZA",
                 "MRG": "ノード結合", "DEP": "係り受け", "OWN": "持ち主つき"}
STATUS_COLORS = {PC.STATUS_ACCEPT: "#16a34a", PC.STATUS_REVIEW: "#d97706", PC.STATUS_REJECT: "#94a3b8"}
COMPANY_COLORS = {"三菱電機": "#dc2626", "富士電機": "#2563eb", "ローム": "#16a34a", "東芝": "#9333ea",
                  "その他": "#64748b"}

SAMPLE_CLAIM = (
    "第１方向に離隔して並んで設けられ、前記第１方向に交差する第２方向に延びて設けられる"
    "流路を有する複数の多穴管と、\n"
    "前記複数の多穴管のそれぞれの一方が接続される第１ヘッダと、\n"
    "前記複数の多穴管のそれぞれの他方が接続される第２ヘッダと、\n"
    "を備え、\n"
    "前記第１ヘッダ及び前記第２ヘッダのそれぞれは、前記第１方向及び前記第２方向に延在する"
    "壁部に、被設置対象に固定される固定部を有する、\n"
    "冷却器。"
)

for _key, _default in (
    ("reviews", {}), ("workspace", []), ("analysis", None), ("compare_result", None),
    ("patent_db", None), ("search_results", None), ("dependent_result", None), ("stats_df", None),
    ("eval_results", None), ("eval_summary", None), ("net_node", None), ("map_patent", None),
    ("review_pid", None),
):
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ---------------------------------------------------------------------------
# クラウドLLMバックエンド（公開デプロイ用）
# ---------------------------------------------------------------------------

def _get_openrouter_api_key():
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:  # noqa: BLE001
        key = None
    return key or os.environ.get("OPENROUTER_API_KEY") or None


def _make_cloud_chat(api_key):
    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def _cloud_chat(system_prompt, user_text, model=CLOUD_MODEL, host=None):
        use_model = model if (model and "/" in model) else CLOUD_MODEL
        response = client.chat.completions.create(
            model=use_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            temperature=0, max_tokens=ts._OLLAMA_MAX_OUTPUT_TOKENS,
        )
        content = response.choices[0].message.content or ""
        content = ts._THINK_BLOCK_RE.sub("", content).strip()
        return ts._NOTE_BLOCK_RE.sub("", content).strip()

    return _cloud_chat


def _install_cloud_backend_if_configured():
    api_key = _get_openrouter_api_key()
    if not api_key:
        return "local"
    try:
        ts._ollama_chat = _make_cloud_chat(api_key)
        return "cloud"
    except Exception:  # noqa: BLE001
        return "local"


BACKEND = _install_cloud_backend_if_configured()


# ---------------------------------------------------------------------------
# 読み込み（キャッシュ）
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="GiNZAパイプラインを読み込み中…（初回のみ、数十秒かかります）")
def load_pipeline():
    return ts._load_pipeline(str(APP_DIR))


@st.cache_resource(show_spinner="候補選別モデルを学習中…（初回のみ、数十秒かかります）")
def load_selector(module_name):
    return SELECTOR_MODULES[module_name].Selector()


@st.cache_data(show_spinner="532件の分析データを読み込み中…")
def load_corpus():
    if not PC.CORPUS_FILE.exists():
        return None
    return PC.load_corpus()


@st.cache_data(show_spinner=False)
def corpus_similarity(_corpus, reviews_key):
    return PC.similarity_matrix(_corpus, st.session_state.reviews)


def reviews_key():
    """人手確認の内容が変わったときだけ、類似度などの計算をやり直すためのキー。"""
    return json.dumps(st.session_state.reviews, ensure_ascii=False, sort_keys=True)


CORPUS = load_corpus()
PATENTS = {p["id"]: p for p in CORPUS["patents"]} if CORPUS else {}


def patent_label(pid):
    p = PATENTS[pid]
    return f"{pid}｜{p['title']}｜{p['company']}"


def need_corpus():
    if CORPUS is None:
        st.error("corpus_sao_532.json が見つかりません。app.py と同じフォルダに置いてください。")
        st.stop()


# ---------------------------------------------------------------------------
# 抽出（1件）
# ---------------------------------------------------------------------------

def extraction_kwargs(method):
    if method == M4:
        return dict(verify_risky_ginza=True, risk_threshold=ts._DEFAULT_RISK_THRESHOLD,
                    extra_risk_rules=ts._build_extra_risk_rules(ts._DEFAULT_CLAIM_TITLE_RISK_THRESHOLD),
                    filter_invalid_targets=True)
    return dict(verify_risky_ginza=False, risk_threshold=ts._DEFAULT_RISK_THRESHOLD,
                extra_risk_rules=None, filter_invalid_targets=False)


def method_bands(method, threshold):
    """採用／要確認／除外の帯。532件で較正した帯は、コーパスと同じ手法のときだけ使う。"""
    if CORPUS and CORPUS["meta"].get("method_key") == METHOD_MODULE.get(method):
        return CORPUS["bands"]
    return {"accept": threshold, "threshold": threshold, "review_low": max(0.05, threshold - 0.10)}


def origin_of(srcs):
    return PC.origin_label(srcs) or "選別"


def duplicate_flags(gpp, mod, info, keys):
    """選ばれた関係と同じ組で同義の関係（「有する」と「備える」など）の候補に印を付ける
    （人が確認するときに同じものが繰り返し出ないようにする）。"""
    from node_match_eval import rel_match

    n = gpp._normalize_node_text_lenient
    if hasattr(mod, "canon"):
        cm = mod.canon(info)
    else:
        cm = {X + "の" + Y: Y for Y, xs in info.get("merge_owners", {}).items() for X in xs}

    def key(c):
        return frozenset((n(cm.get(c["source"], c["source"])), n(cm.get(c["target"], c["target"]))))

    chosen = {}
    for c in info["cands"]:
        if (c["source"], c["relation"], c["target"]) in keys:
            chosen.setdefault(key(c), []).append(c["relation"])
    return [(c["source"], c["relation"], c["target"]) not in keys
            and any(rel_match(gpp, c["relation"], r) or rel_match(gpp, r, c["relation"]) for r in chosen.get(key(c), []))
            for c in info["cands"]]


@st.cache_data(show_spinner=False, max_entries=64)
def analyze(text, method, model, host):
    """1件の請求項を解析し、処理の各段階と全候補の判定を返す。"""
    gpp = load_pipeline()
    t0 = time.time()
    if method in METHOD_MODULE:
        mod = SELECTOR_MODULES[METHOD_MODULE[method]]
        selector = load_selector(METHOD_MODULE[method])
        info = mod.build_candidates(ts, gpp, text, model=model, host=host)
        prob = selector.predict(gpp, info)
        extra = [selector.max_per_pair] if hasattr(selector, "max_per_pair") else []
        chosen = mod.select(gpp, info, prob, selector.threshold, *extra)
        keys = {(r["source"], r["relation"], r["target"]) for r in chosen}
        bands = method_bands(method, selector.threshold)
        dup = duplicate_flags(gpp, mod, info, keys)
        cands = []
        for i, c in enumerate(info["cands"]):
            sel = (c["source"], c["relation"], c["target"]) in keys
            p = float(prob[i]) if len(prob) else 0.0
            cands.append({"source": c["source"], "relation": c["relation"], "target": c["target"],
                          "prob": round(p, 4), "selected": sel,
                          "status": PC.STATUS_REJECT if dup[i] else PC.classify(p, sel, bands),
                          "origin": origin_of(c.get("srcs", [])), "srcs": list(c.get("srcs", []))})
        cands.sort(key=lambda r: -r["prob"])
        steps = [("前処理・構成要素の抽出（GiNZA）", f"構成要素 {len(info['tags'])} 個／形式：{info.get('format', '―')}")]
        for label, prefixes in SRC_GROUPS:
            k = sum(1 for c in info["cands"] if any(s.startswith(prefixes) for s in c.get("srcs", [])))
            if k:
                steps.append((label, f"候補 {k} 件"))
        steps.append(("候補の統合（重複をまとめる）", f"候補 {len(info['cands'])} 件"))
        steps.append(("選別モデル（確率の算出）", f"しきい値 {selector.threshold:.3f}／選ばれた関係 {len(chosen)} 件"))
        return {"method": method, "tags": list(info["tags"]), "title": info.get("title"),
                "format": info.get("format", ""), "cands": cands, "steps": steps, "bands": bands,
                "elapsed": time.time() - t0}
    comps, rels = ts.analyze_claim_llm_direct(text, pp=gpp, model=model, host=host, **extraction_kwargs(method))
    cands = [{"source": r["source"], "relation": r["relation"], "target": r["target"], "prob": None,
              "selected": True, "status": PC.STATUS_ACCEPT, "origin": r.get("type", ""), "srcs": []}
             for r in rels]
    steps = [("前処理・構成要素の抽出（GiNZA）", f"構成要素 {len(comps)} 個"),
             ("LLM抽出＋GiNZA補完・検証", f"関係 {len(rels)} 件（確率なし：すべて採用として表示）")]
    return {"method": method, "tags": [c["text"] for c in comps], "title": None, "format": "", "cands": cands,
            "steps": steps, "bands": None, "elapsed": time.time() - t0}


def llm_extract(text):
    """比較・従属請求項・検索・精度検証の各ページ共通の抽出（推奨手法、選ばれた関係のみ）。"""
    res = analyze(text, RECOMMENDED, model_name, (ollama_host or "").strip() or None)
    rels = [{"source": c["source"], "relation": c["relation"], "target": c["target"], "type": c["origin"] or "selected"}
            for c in res["cands"] if c["selected"]]
    return [{"text": t} for t in res["tags"]], rels


def sao_graph(relations, height=None):
    if not relations:
        st.info("表示できるSAO関係がありません。")
        return
    if relations_to_nested_dot is not None:
        st.graphviz_chart(relations_to_nested_dot(relations), use_container_width=True)
        st.caption("箱の入れ子＝構成（備える・有する・含む）／矢印＝構成要素間の関係／点線の楕円＝どの構成要素にも属さない対象")
    else:
        lines = ['digraph SAO {rankdir="LR"; node [shape=box, style="rounded,filled", fillcolor="#f1f5f9"];']
        for r in relations:
            lines.append('"%s" -> "%s" [label="%s"];' % (html.escape(r["source"]), html.escape(r["target"]),
                                                         html.escape(r["relation"])))
        st.graphviz_chart("\n".join(lines + ["}"]), use_container_width=True)


def status_badges(counts):
    cols = st.columns(3)
    for col, s in zip(cols, PC.STATUS_ORDER):
        col.markdown(
            f"<div style='border-left:6px solid {STATUS_COLORS[s]};padding:6px 12px;background:rgba(148,163,184,.08);"
            f"border-radius:6px'><div style='font-size:.85rem;opacity:.8'>{s}</div>"
            f"<div style='font-size:1.6rem;font-weight:700'>{counts.get(s, 0):,}</div></div>",
            unsafe_allow_html=True)


def editor_config():
    return {
        "採用する": st.column_config.CheckboxColumn("採用する", help="チェックした関係だけが確定されます"),
        "確率": st.column_config.ProgressColumn("確率", min_value=0.0, max_value=1.0, format="%.2f"),
        "判定": st.column_config.TextColumn("AIの判定", disabled=True),
        "抽出元": st.column_config.TextColumn("抽出元", disabled=True),
    }


def plot_or_warn(fig, **kw):
    return st.plotly_chart(fig, use_container_width=True, **kw)


# ---------------------------------------------------------------------------
# サイドバー（共通設定）
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 抽出の設定")
    if BACKEND == "cloud":
        st.success("🌐 公開デモモード：OpenRouter経由の qwen2.5-7b-instruct を使用", icon="🌐")
        model_name = st.text_input("モデル名（OpenRouter）", value=CLOUD_MODEL)
        ollama_host = ""
    else:
        st.info("💻 ローカルモード：このPCのOllamaに接続します", icon="💻")
        model_name = st.text_input("Ollamaモデル名", value=ts.DEFAULT_MODEL)
        ollama_host = st.text_input("Ollamaホスト（空欄 = http://localhost:11434）", value="")
    st.caption(f"新しい請求項の解析には、推奨手法「{RECOMMENDED}」を使います（AI解析ページで切り替え可）。")


# ===========================================================================
# 🏠 ダッシュボード
# ===========================================================================

def page_dashboard():
    need_corpus()
    import plotly.express as px
    import plotly.graph_objects as go

    reviews = st.session_state.reviews
    st.title("🏠 ダッシュボード")
    st.caption(f"半導体関連特許 {len(CORPUS['patents'])} 件の請求項を、{CORPUS['meta']['method']} で解析した結果と、"
               "人による確認の進み具合。")
    c = PC.status_counts(CORPUS, reviews)
    m = st.columns(4)
    m[0].metric("特許件数", f"{c['特許件数']:,}")
    m[1].metric("AIが抽出したSAO", f"{c['抽出SAO']:,}", help="選別モデルが選んだ関係（採用＋要確認の一部）")
    m[2].metric("人手確認済みの特許", f"{c['人手確認済みの特許']:,} / {c['特許件数']:,}")
    m[3].metric("確定SAO（人手確認済み）", f"{c['確定SAO（人手確認済み）']:,}")
    st.progress(c["人手確認済みの特許"] / max(c["特許件数"], 1), text="人手確認の進み具合")

    st.markdown("#### AIの判定（全候補）")
    status_badges(c)
    b = CORPUS["bands"]
    st.caption(
        f"採用＝選別モデルが選び、確率 {b['accept']:.2f} 以上／要確認＝選ばれたが確率がそれ未満、"
        f"または選ばれなかったが確率 {b['review_low']:.2f} 以上／除外＝それ以外。")

    left, right = st.columns([3, 2])
    with left:
        rows = []
        for p in CORPUS["patents"]:
            cnt = Counter(r["status"] for r in p["relations"])
            for s in PC.STATUS_ORDER[:2]:
                rows.append({"企業": p["company"], "判定": s, "件数": cnt.get(s, 0)})
        df = pd.DataFrame(rows).groupby(["企業", "判定"], as_index=False)["件数"].sum()
        fig = px.bar(df, x="企業", y="件数", color="判定", color_discrete_map=STATUS_COLORS, barmode="stack",
                     title="企業別：採用・要確認の件数")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        plot_or_warn(fig)
    with right:
        s = b.get("stats", {})
        if s:
            fig = go.Figure(go.Funnel(
                y=["全候補", "要確認以上", "採用"],
                x=[s["採用"]["件数"] + s["要確認"]["件数"] + s["除外"]["件数"],
                   s["採用"]["件数"] + s["要確認"]["件数"], s["採用"]["件数"]],
                marker={"color": ["#94a3b8", "#d97706", "#16a34a"]}))
            fig.update_layout(title="候補の絞り込み", height=360, margin=dict(l=10, r=10, t=50, b=10))
            plot_or_warn(fig)

    with st.expander("📐 判定の根拠（5分割交差検証・正解データとの完全一致で推定）", expanded=False):
        s = b.get("stats", {})
        if s:
            t = pd.DataFrame([
                {"判定": "採用", "件数": s["採用"]["件数"], "そのうち正しい割合（推定）": f"{100 * s['採用']['精度']:.1f}%"},
                {"判定": "要確認", "件数": s["要確認"]["件数"], "そのうち正しい割合（推定）": f"{100 * s['要確認']['精度']:.1f}%"},
                {"判定": "除外", "件数": s["除外"]["件数"], "そのうち正しい割合（推定）": f"{100 * s['除外']['精度']:.1f}%"},
            ])
            st.dataframe(t, hide_index=True, use_container_width=True)
            loc = s.get("正解の所在", {})
            st.write("正しいSAOが、どの判定に入っているか：" + "／".join(
                f"{k} {100 * v:.1f}%" for k, v in sorted(loc.items(), key=lambda kv: -kv[1])))
            st.caption("確率は、その請求項を学習に使っていないモデルで求めたもの。「採用」は精度"
                       f"{100 * b.get('target_precision', 0.8):.0f}%以上になる確率の下限で区切っている。"
                       "要確認だけを人が見れば、効率よく誤りを直せる。")
        st.markdown("**研究の到達点（主指標：トリプル完全一致、532件）**")
        st.dataframe(pd.DataFrame(EXACT_HISTORY), hide_index=True, use_container_width=True)

    st.markdown("#### よく現れる構成要素（基本語）")
    cnt = Counter()
    for p in CORPUS["patents"]:
        cnt.update({PC.base_term(x) for r in PC.effective_relations(p, reviews) for x in (r["source"], r["target"])})
    top = pd.DataFrame(cnt.most_common(20), columns=["構成要素", "特許件数"])
    fig = px.bar(top[::-1], x="特許件数", y="構成要素", orientation="h", height=520)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    plot_or_warn(fig)

    queue = sorted(CORPUS["patents"], key=lambda p: -sum(r["status"] == PC.STATUS_REVIEW for r in p["relations"]))
    queue = [p for p in queue if p["id"] not in reviews][:10]
    if queue:
        st.markdown("#### 次に確認するとよい特許（要確認が多い順）")
        st.dataframe(pd.DataFrame([{
            "特許番号": p["id"], "発明の名称": p["title"], "企業": p["company"],
            "要確認": sum(r["status"] == PC.STATUS_REVIEW for r in p["relations"]),
            "採用": sum(r["status"] == PC.STATUS_ACCEPT for r in p["relations"])} for p in queue]),
            hide_index=True, use_container_width=True)


# ===========================================================================
# 🧪 AI解析（1件）
# ===========================================================================

def page_analyze():
    st.title("🧪 AI解析")
    st.caption("請求項を1件解析し、処理の各段階と、AIの判定（採用／要確認／除外）を表示します。"
               "結果は表で修正してから確定できます。")
    col1, col2 = st.columns([3, 1])
    with col2:
        method = st.selectbox("抽出手法", METHODS, index=METHODS.index(RECOMMENDED))
        if CORPUS:
            pick = st.selectbox("532件から読み込む（任意）", ["（使わない）"] + [p["id"] for p in CORPUS["patents"]],
                                format_func=lambda x: x if x == "（使わない）" else patent_label(x))
        else:
            pick = "（使わない）"
    with col1:
        default = PATENTS[pick]["text"] if pick != "（使わない）" else SAMPLE_CLAIM
        text = st.text_area("特許請求項テキスト", value=default, height=220, key=f"an_text_{pick}")
    if st.button("🧪 解析する", type="primary"):
        if not text.strip():
            st.warning("請求項テキストを入力してください。")
            st.stop()
        try:
            with st.spinner("解析中…（LLMの呼び出しを含みます）"):
                st.session_state.analysis = analyze(text, method, model_name, (ollama_host or "").strip() or None)
                st.session_state.analysis["text"] = text
                st.session_state.analysis["pick"] = pick
        except Exception as exc:  # noqa: BLE001
            st.error("解析中にエラーが発生しました。" + (
                "OpenRouterのレート制限に達した可能性があります。" if BACKEND == "cloud" else
                f"Ollamaが起動しているか（ollama serve）、モデルが取得済みか（ollama pull {model_name}）を確認してください。"))
            st.exception(exc)
            st.stop()

    res = st.session_state.analysis
    if not res:
        return
    st.success(f"解析完了（{res['elapsed']:.1f}秒）／手法：{res['method']}")

    st.markdown("#### ① 処理の流れ")
    cols = st.columns(len(res["steps"]))
    for i, (col, (name, detail)) in enumerate(zip(cols, res["steps"])):
        col.markdown(
            f"<div style='border:1px solid rgba(148,163,184,.5);border-radius:10px;padding:8px;min-height:110px'>"
            f"<div style='font-size:.75rem;opacity:.7'>STEP {i + 1}</div><div style='font-weight:700;font-size:.9rem'>"
            f"{html.escape(name)}</div><div style='font-size:.85rem;margin-top:4px'>{html.escape(detail)}</div></div>",
            unsafe_allow_html=True)

    st.markdown("#### ② AIの判定")
    counts = Counter(c["status"] for c in res["cands"])
    status_badges(counts)
    if res["bands"]:
        st.caption(f"採用：確率 {res['bands']['accept']:.2f} 以上で選ばれた関係／要確認：選ばれたがそれ未満、"
                   f"または確率 {res['bands']['review_low']:.2f} 以上／除外：それ以外（表では非表示）")

    st.markdown("#### ③ 確認・修正")
    st.caption("「採用する」のチェックを付け外しし、主語・関係・目的語は直接書き換えられます。"
               "表の一番下の行から関係を追加できます。")
    show = [c for c in res["cands"] if c["status"] != PC.STATUS_REJECT]
    df = pd.DataFrame([{
        "採用する": c["status"] == PC.STATUS_ACCEPT or (c["status"] == PC.STATUS_REVIEW and c["selected"]),
        "主語(S)": c["source"], "関係(A)": c["relation"], "目的語(O)": c["target"], "確率": c["prob"],
        "判定": c["status"], "抽出元": c["origin"]} for c in show],
        columns=["採用する", "主語(S)", "関係(A)", "目的語(O)", "確率", "判定", "抽出元"])
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True,
                            column_config=editor_config(), key=f"an_editor_{hash(res['text']) % 10**8}_{res['method']}")
    confirmed = [r for r in PC.table_to_review(edited) if r["keep"]]

    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown("**確定予定のSAO構造**")
        sao_graph(confirmed)
    with g2:
        st.markdown("**構成要素（GiNZA）**")
        st.write("、".join(res["tags"]) or "―")
        b1, b2 = st.columns(2)
        if b1.button("✅ 確定してワークスペースに保存", type="primary"):
            if res.get("pick") and res["pick"] != "（使わない）":
                st.session_state.reviews[res["pick"]] = PC.table_to_review(edited)
                st.success(f"{res['pick']} の人手確認結果として保存しました（ダッシュボード・分析ページに反映）。")
            else:
                st.session_state.workspace.append({"id": f"解析{len(st.session_state.workspace) + 1}",
                                                   "text": res["text"], "relations": confirmed,
                                                   "method": res["method"]})
                st.success("ワークスペースに保存しました（エクスポートページから書き出せます）。")
        b2.download_button("⬇️ CSVで保存", pd.DataFrame(confirmed).to_csv(index=False).encode("utf-8-sig"),
                           file_name="sao_confirmed.csv", mime="text/csv")
    with st.expander("除外した候補も含めた全候補（確率順）"):
        st.dataframe(pd.DataFrame([{k: c[k] for k in ("status", "prob", "source", "relation", "target", "origin")}
                                   for c in res["cands"]]).rename(columns={
            "status": "判定", "prob": "確率", "source": "主語", "relation": "関係", "target": "目的語", "origin": "抽出元"}),
            hide_index=True, use_container_width=True)


# ===========================================================================
# ✍️ 人手確認（532件）
# ===========================================================================

def page_review():
    need_corpus()
    reviews = st.session_state.reviews
    st.title("✍️ 人手確認")
    st.caption("AIの判定を確認・修正して確定します。確定した内容は、ダッシュボード・ネットワーク・類似性マップ・"
               "レーダーなどすべての分析に反映されます（エクスポートページで保存・読み込みできます）。")
    f1, f2, f3 = st.columns([2, 2, 1])
    companies = sorted({p["company"] for p in CORPUS["patents"]})
    comp = f1.multiselect("企業で絞り込む", companies)
    order = f2.selectbox("並び順", ["要確認が多い順", "特許番号順", "採用が少ない順"])
    only_open = f3.checkbox("未確認のみ", value=True)
    ps = [p for p in CORPUS["patents"] if (not comp or p["company"] in comp) and (not only_open or p["id"] not in reviews)]
    key = {"要確認が多い順": lambda p: -sum(r["status"] == PC.STATUS_REVIEW for r in p["relations"]),
           "特許番号順": lambda p: p["id"],
           "採用が少ない順": lambda p: sum(r["status"] == PC.STATUS_ACCEPT for r in p["relations"])}[order]
    ps = sorted(ps, key=key)
    st.progress(len(reviews) / len(CORPUS["patents"]), text=f"確認済み {len(reviews)} / {len(CORPUS['patents'])} 件")
    if not ps:
        st.info("条件に合う未確認の特許はありません。")
        return
    ids = [p["id"] for p in ps]
    default = st.session_state.review_pid if st.session_state.review_pid in ids else ids[0]
    pid = st.selectbox("確認する特許", ids, index=ids.index(default), format_func=patent_label)
    st.session_state.review_pid = pid
    p = PATENTS[pid]
    counts = Counter(r["status"] for r in p["relations"])
    status_badges(counts)

    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"**{p['title']}**　{p['applicant']}　出願日 {p.get('filing_date') or '―'}")
        if p.get("url"):
            st.markdown(f"[J-PlatPatで開く]({p['url']})")
        table = PC.review_table(p, reviews)
        words = set(table["主語(S)"]).union(table["目的語(O)"]) if len(table) else set()
        st.markdown(f"<div style='font-size:.92rem;line-height:1.8;border:1px solid rgba(148,163,184,.4);"
                    f"border-radius:8px;padding:10px;max-height:420px;overflow:auto'>{PC.highlight(p['text'], words)}</div>",
                    unsafe_allow_html=True)
    with right:
        edited = st.data_editor(table, num_rows="dynamic", use_container_width=True, hide_index=True,
                                column_config=editor_config(), key=f"rv_{pid}")
        b1, b2, b3 = st.columns(3)
        if b1.button("✅ 確定する", type="primary", key=f"ok_{pid}"):
            reviews[pid] = PC.table_to_review(edited)
            nxt = [i for i in ids if i != pid and i not in reviews]
            st.session_state.review_pid = nxt[0] if nxt else None
            st.rerun()
        if pid in reviews and b2.button("↩️ 確認を取り消す", key=f"undo_{pid}"):
            del reviews[pid]
            st.rerun()
        b3.caption("確定すると次の特許へ進みます")
    st.markdown("**確定後のSAO構造（プレビュー）**")
    sao_graph([r for r in PC.table_to_review(edited) if r["keep"]])


# ===========================================================================
# 🌍 Patent World（オズの世界）
# ===========================================================================

def page_world():
    st.title("🌍 Patent World（オズの世界）")
    tab1, tab2 = st.tabs(["🌌 オズの世界（技術ランドスケープ）", "🔎 SAOの特徴で色分け"])
    with tab1:
        st.caption("特許532件を「発明の名称＋FI」でベクトル化し、UMAPで3次元空間に配置した技術ランドスケープ。"
                   "ドラッグで回転、スクロールでズーム。")
        if OZ_WORLD_HTML_PATH.exists():
            components.html(OZ_WORLD_HTML_PATH.read_text(encoding="utf-8"), height=760, scrolling=False)
        else:
            st.error(f"{OZ_WORLD_HTML_PATH.name} が見つかりません。app.py と同じフォルダに置いてください。")
    with tab2:
        need_corpus()
        import plotly.express as px

        feats = PC.feature_table(CORPUS, st.session_state.reviews)
        pos = pd.DataFrame([{"特許番号": p["id"], "x": p["x"], "y": p["y"], "z": p["z"]} for p in CORPUS["patents"]])
        df = feats.merge(pos, on="特許番号").dropna(subset=["x"])
        color = st.selectbox("色分けに使う特徴", ["企業"] + PC.RADAR_AXES + ["請求項の文字数"])
        fig = px.scatter_3d(df, x="x", y="y", z="z", color=color,
                            color_discrete_map=COMPANY_COLORS if color == "企業" else None,
                            hover_name="発明の名称", hover_data={"特許番号": True, "企業": True, "SAO関係数": True,
                                                              "x": False, "y": False, "z": False}, height=640)
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        plot_or_warn(fig)
        st.caption("配置はオズの世界と同じ（発明の名称＋FI）。色でSAOの構造的特徴を重ねて、"
                   "技術的に近い特許どうしで請求項の書き方がどう違うかを見る。")
        pid = st.selectbox("詳しく見る特許", [p["id"] for p in CORPUS["patents"]], format_func=patent_label,
                           key="world_pid")
        show_patent_card(pid)


def show_patent_card(pid, extra_words=()):
    p = PATENTS[pid]
    rels = PC.effective_relations(p, st.session_state.reviews)
    st.markdown(f"**{p['title']}**（{pid}）　{p['applicant']}　FI: {p['fi']}")
    c1, c2 = st.columns([2, 3])
    with c1:
        words = {x for r in rels for x in (r["source"], r["target"])} | set(extra_words)
        st.markdown(f"<div style='font-size:.9rem;line-height:1.8;max-height:360px;overflow:auto;"
                    f"border:1px solid rgba(148,163,184,.4);border-radius:8px;padding:10px'>"
                    f"{PC.highlight(p['text'], words)}</div>", unsafe_allow_html=True)
    with c2:
        st.dataframe(pd.DataFrame([{"主語": r["source"], "関係": r["relation"], "目的語": r["target"],
                                    "判定": r.get("status", "")} for r in rels]),
                     hide_index=True, use_container_width=True, height=360)


# ===========================================================================
# 🕸️ SAOネットワーク
# ===========================================================================

def _clicked(event, field="customdata"):
    try:
        pts = event.selection.points if hasattr(event, "selection") else event["selection"]["points"]
    except Exception:  # noqa: BLE001
        return None
    for pt in pts or []:
        cd = pt.get(field) if isinstance(pt, dict) else getattr(pt, field, None)
        if cd is not None:
            return cd[0] if isinstance(cd, (list, tuple)) else cd
    return None


def page_network():
    need_corpus()
    import plotly.graph_objects as go

    st.title("🕸️ SAOネットワーク")
    st.caption("全特許のSAOを、番号や「前記」を除いた基本語（例：第１電極→電極）でまとめたネットワーク。"
               "丸の大きさ＝その構成要素が現れる特許の件数、線の太さ＝その関係が現れる特許の件数。"
               "**丸をクリックすると、その構成要素が出てくる請求項が下に表示されます。**")
    c1, c2, c3, c4, c5 = st.columns(5)
    companies = sorted({p["company"] for p in CORPUS["patents"]})
    comp = c1.multiselect("企業", companies)
    top_n = c2.slider("構成要素の数", 20, 150, 50, step=10)
    min_p = c3.slider("構成要素の最低出現特許数", 1, 20, 3)
    min_e = c4.slider("線を引く最低特許数", 1, 20, 3, help="その関係が何件の特許に現れたら線を引くか")
    kind = c5.selectbox("関係の種類", ["すべて", "構成（有する・備える）", "機能・配置（それ以外）"])
    ids = [p["id"] for p in CORPUS["patents"] if not comp or p["company"] in comp]
    nodes, edges = PC.build_network(CORPUS, ids, st.session_state.reviews, min_patents=min_p, top_n=top_n)
    edges = [e for e in edges if e["count"] >= min_e]
    if kind != "すべて":
        want = kind.startswith("構成")
        edges = [e for e in edges if PC.is_has(e["relation"]) == want]
    linked = {x for e in edges for x in (e["source"], e["target"])}
    nodes = [n for n in nodes if n["id"] in linked] or nodes
    if not nodes:
        st.info("条件に合う構成要素がありません。")
        return
    pos = PC.layout_network(nodes, edges)
    fig = go.Figure()
    maxc = max(e["count"] for e in edges) if edges else 1
    for e in edges:
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(width=0.5 + 4 * e["count"] / maxc,
                                           color="rgba(37,99,235,.35)" if PC.is_has(e["relation"]) else "rgba(234,88,12,.45)")))
    sel = st.session_state.net_node
    deg = Counter()
    for e in edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    xs = [pos[n["id"]][0] for n in nodes]
    ys = [pos[n["id"]][1] for n in nodes]
    maxn = max(n["count"] for n in nodes)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", textposition="top center",
        text=[n["id"] if (n["count"] >= sorted([m["count"] for m in nodes], reverse=True)[min(34, len(nodes) - 1)]
                          or n["id"] == sel) else "" for n in nodes],
        textfont=dict(size=11), customdata=[[n["id"]] for n in nodes],
        hovertext=[f"{n['id']}<br>特許 {n['count']} 件／つながり {deg[n['id']]}<br>表記例：{'、'.join(n['surfaces'][:4])}"
                   for n in nodes], hoverinfo="text", showlegend=False,
        marker=dict(size=[10 + 30 * (n["count"] / maxn) ** 0.5 for n in nodes],
                    color=["#f59e0b" if n["id"] == sel else "#0ea5e9" for n in nodes],
                    line=dict(width=1, color="#0f172a"))))
    fig.update_layout(height=680, margin=dict(l=10, r=10, t=10, b=10), dragmode="pan",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), clickmode="event+select")
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="net_chart")
    st.caption("青い線＝構成（有する・備える）／橙の線＝機能・配置（接続される・配置される等）")
    # クリックした直後の再実行では図を変えずに選択を受け取り、印を付け直すために再実行する
    # （図が変わると選択状態が消えるため、既定値は session_state に書き込まない）
    clicked = _clicked(event)
    if clicked and clicked != st.session_state.net_node:
        st.session_state.net_node = clicked
        st.rerun()
    names = [n["id"] for n in sorted(nodes, key=lambda n: -n["count"])]
    cur = st.session_state.net_node if st.session_state.net_node in names else names[0]
    term = st.selectbox("構成要素（クリックでも選べます）", names, index=names.index(cur))
    if term != cur:
        st.session_state.net_node = term
        st.rerun()
    hits = [h for h in PC.patents_with_node(CORPUS, term, st.session_state.reviews) if h["patent"]["id"] in ids]
    st.markdown(f"#### 「{term}」が出てくる請求項：{len(hits)} 件")
    nb = Counter()
    for e in edges:
        if term in (e["source"], e["target"]):
            nb[(e["source"], e["relation"], e["target"])] += e["count"]
    if nb:
        st.write("主なつながり：" + "／".join(f"{s}→{r}→{t}（{c}件）" for (s, r, t), c in nb.most_common(6)))
    for h in hits[:30]:
        p = h["patent"]
        with st.expander(f"{p['id']}｜{p['title']}｜{p['company']}（関係 {len(h['relations'])} 件）"):
            st.markdown(f"<div style='font-size:.9rem;line-height:1.8'>{PC.highlight(p['text'], h['surfaces'])}</div>",
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"主語": r["source"], "関係": r["relation"], "目的語": r["target"]}
                                       for r in h["relations"]]), hide_index=True, use_container_width=True)
    if len(hits) > 30:
        st.caption(f"ほか {len(hits) - 30} 件（企業で絞り込むと減らせます）")


# ===========================================================================
# 🗺️ 類似性マップ
# ===========================================================================

def page_similarity():
    need_corpus()
    import plotly.express as px

    st.title("🗺️ 類似性マップ")
    st.caption("各特許のSAO（基本語にしたトリプル・組・構成要素）をTF-IDFで数値化し、似ているものほど近くに"
               "配置した地図（UMAP）。 **点をクリックすると、似ている特許と共通するSAOが表示されます。** "
               "オズの世界（発明の名称＋FI）とは違い、請求項の構造の近さで並べている。")
    S = corpus_similarity(CORPUS, reviews_key())
    ids = [p["id"] for p in CORPUS["patents"]]
    df = pd.DataFrame([{"特許番号": p["id"], "発明の名称": p["title"], "企業": p["company"], "x": p.get("map_x"),
                        "y": p.get("map_y"), "SAO数": len(PC.effective_relations(p, st.session_state.reviews))}
                       for p in CORPUS["patents"]])
    color = st.radio("色分け", ["企業", "SAO数"], horizontal=True)
    fig = px.scatter(df, x="x", y="y", color=color, color_discrete_map=COMPANY_COLORS if color == "企業" else None,
                     hover_name="発明の名称", hover_data={"特許番号": True, "x": False, "y": False},
                     custom_data=["特許番号"], height=620)
    sel = st.session_state.map_patent
    if sel in ids:
        i = ids.index(sel)
        nbr = np.argsort(-S[i])[:5]
        fig.add_scatter(x=df.loc[nbr, "x"], y=df.loc[nbr, "y"], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=2, color="#f59e0b")))
        fig.add_scatter(x=[df.loc[i, "x"]], y=[df.loc[i, "y"]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=18, symbol="star", color="#f59e0b", line=dict(width=1, color="#000")))
    fig.update_traces(marker=dict(size=8), selector=dict(mode="markers", type="scatter", showlegend=True))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="map_chart")
    clicked = _clicked(event)
    if clicked and clicked != st.session_state.map_patent:
        st.session_state.map_patent = clicked
        st.rerun()
    cur = st.session_state.map_patent if st.session_state.map_patent in ids else ids[0]
    pid = st.selectbox("基準の特許（クリックでも選べます）", ids, index=ids.index(cur), format_func=patent_label)
    if pid != cur:
        st.session_state.map_patent = pid
        st.rerun()
    i = ids.index(pid)
    top_k = st.slider("表示する類似特許の数", 3, 15, 5)
    rows = []
    for j in np.argsort(-S[i])[:top_k]:
        rows.append({"特許番号": ids[j], "発明の名称": PATENTS[ids[j]]["title"], "企業": PATENTS[ids[j]]["company"],
                     "類似度": round(float(S[i, j]), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    other = st.selectbox("共通部分を見る特許", [r["特許番号"] for r in rows], format_func=patent_label)
    ra = PC.effective_relations(PATENTS[pid], st.session_state.reviews)
    rb = PC.effective_relations(PATENTS[other], st.session_state.reviews)
    ex = PC.similarity_explain(ra, rb)
    c1, c2, c3 = st.columns(3)
    for col, k in zip((c1, c2, c3), ("共通のSAO", "共通の組", "共通の構成要素")):
        col.markdown(f"**{k}（{len(ex[k])}）**")
        col.write("\n".join(f"- {x}" for x in ex[k][:25]) or "―")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**{pid}**")
        sao_graph(ra)
    with b:
        st.markdown(f"**{other}**")
        sao_graph(rb)


# ===========================================================================
# 🧭 構造レーダー
# ===========================================================================

def page_radar():
    need_corpus()
    import plotly.graph_objects as go

    st.title("🧭 構造レーダー")
    st.info("このレーダーチャートは、請求項の**書き方の構造**（構成要素の数・階層の深さ・機能や配置の記述の多さ等）"
            "を示すもので、特許の強さや価値を表すものではありません。各軸は532件の中での順位（パーセンタイル）。",
            icon="ℹ️")
    feats = PC.feature_table(CORPUS, st.session_state.reviews)
    pct = PC.percentile_scores(feats)
    mode = st.radio("比べるもの", ["企業の平均", "特許どうし"], horizontal=True)
    fig = go.Figure()
    axes = PC.RADAR_AXES
    if mode == "企業の平均":
        comps = st.multiselect("企業", sorted(pct["企業"].unique()), default=["三菱電機", "富士電機", "ローム", "東芝"])
        for c in comps:
            v = pct[pct["企業"] == c][axes].mean().tolist()
            fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], fill="toself", name=c,
                                          line=dict(color=COMPANY_COLORS.get(c))))
        raw = feats.groupby("企業")[axes + ["請求項の文字数"]].mean().round(2).loc[comps] if comps else None
    else:
        pids = st.multiselect("特許（最大4件）", feats["特許番号"].tolist(), default=feats["特許番号"].tolist()[:2],
                              max_selections=4, format_func=patent_label)
        for pid in pids:
            v = pct[pct["特許番号"] == pid][axes].iloc[0].tolist()
            fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], fill="toself",
                                          name=f"{pid} {PATENTS[pid]['title'][:12]}"))
        raw = feats[feats["特許番号"].isin(pids)].set_index("特許番号")[axes + ["請求項の文字数"]] if pids else None
    rng = [20, 80] if mode == "企業の平均" else [0, 100]
    fig.update_layout(polar=dict(radialaxis=dict(range=rng, ticksuffix="")), height=560,
                      margin=dict(l=40, r=40, t=30, b=30))
    plot_or_warn(fig)
    if raw is not None:
        st.markdown("**実際の値**")
        st.dataframe(raw, use_container_width=True)
    with st.expander("各軸の意味"):
        st.markdown(
            "- **構成要素数**：SAOに現れる構成要素（ノード）の数\n"
            "- **SAO関係数**：関係（トリプル）の数\n"
            "- **階層の深さ**：「有する・備える」でたどれる入れ子の最大の深さ\n"
            "- **分岐の多さ**：1つの構成要素が平均いくつの関係の主語になっているか\n"
            "- **関係の多様性**：関係の種類（接続される・配置される等）の数\n"
            "- **機能・配置の記述**：「有する・備える」以外の関係の割合\n"
            "- **数値限定**：本文中の数値・範囲の限定（〜以上、μm、℃等）の数")


# ===========================================================================
# 🫧 技術分布（バブル・ヒートマップ）
# ===========================================================================

def page_distribution():
    need_corpus()
    import plotly.express as px

    st.title("🫧 技術分布")
    tab1, tab2 = st.tabs(["🫧 出願年 × SAO数 × 請求項数", "🔥 企業 × 技術 ヒートマップ"])
    reviews = st.session_state.reviews
    with tab1:
        df = PC.company_year_bubble(CORPUS, reviews)
        ymetric = st.radio("縦軸", ["平均SAO数", "SAO数合計"], horizontal=True)
        fig = px.scatter(df, x="出願年", y=ymetric, size="請求項数", color="企業", color_discrete_map=COMPANY_COLORS,
                         hover_data={"請求項数": True, "平均SAO数": ":.1f", "SAO数合計": True}, size_max=48, height=560)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(dtick=1))
        plot_or_warn(fig)
        st.caption("横軸＝出願年、縦軸＝請求項あたりのSAO数（構造の細かさ）、バブルの大きさ＝その年の請求項（特許）の件数。")
    with tab2:
        c1, c2, c3 = st.columns(3)
        axis = c1.selectbox("技術の軸", ["FIサブクラス", "FIメイングループ", "主要構成要素（SAO）"])
        top = c2.slider("表示する技術の数", 5, 30, 15)
        norm = c3.checkbox("企業ごとの割合で表示", value=False)
        m = PC.company_tech_matrix(CORPUS, axis=axis, reviews=reviews, top_tech=top)
        if m.empty:
            st.info("データがありません。")
            return
        z = m.div(m.sum(axis=1), axis=0).round(3) if norm else m
        fig = px.imshow(z, text_auto=".0%" if norm else True, aspect="auto", color_continuous_scale="YlOrRd",
                        height=180 + 45 * len(z))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=axis, yaxis_title="企業")
        plot_or_warn(fig)
        a, b = st.columns(2)
        comp = a.selectbox("企業", m.index.tolist())
        tech = b.selectbox("技術", m.columns.tolist())
        hits = []
        for p in CORPUS["patents"]:
            if p["company"] != comp:
                continue
            if axis == "FIサブクラス":
                ok = tech in p["fi_sub"]
            elif axis == "FIメイングループ":
                ok = tech in p["fi_main"]
            else:
                ok = any(PC.base_term(x) == tech for r in PC.effective_relations(p, reviews) for x in (r["source"], r["target"]))
            if ok:
                hits.append({"特許番号": p["id"], "発明の名称": p["title"], "出願年": p["year"], "FI": p["fi"]})
        st.markdown(f"**{comp} × {tech}：{len(hits)} 件**")
        st.dataframe(pd.DataFrame(hits), hide_index=True, use_container_width=True)


# ===========================================================================
# 🐚 2つの請求項を比較
# ===========================================================================

def page_compare():
    st.title("🐚 2つの請求項を比較")
    st.caption(f"推奨手法（{RECOMMENDED}）で請求項A・Bを解析し、Jaccard類似度・構造の類似度・"
               "クレームの広さ狭さ・意味マッチングで比較します。")
    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area("請求項A", height=220, key="text_a")
    with col_b:
        text_b = st.text_area("請求項B", height=220, key="text_b")
    use_semantic = st.checkbox("②意味マッチングも使う（初回はモデルの読み込みに1分程度かかります）", value=False)
    if st.button("🐚 比較する", type="primary", key="compare_run"):
        if not text_a.strip() or not text_b.strip():
            st.warning("請求項A・Bの両方を入力してください。")
            st.session_state.compare_result = None
        else:
            with st.spinner("解析中..."):
                try:
                    _, relations_a = llm_extract(text_a)
                    _, relations_b = llm_extract(text_b)
                    jaccard_score, common, only_a, only_b = pp.jaccard_similarity(relations_a, relations_b)
                    structural_score, structural_detail = pp.structural_similarity(relations_a, relations_b)
                    semantic_score, semantic_matches = None, None
                    if use_semantic:
                        try:
                            semantic_score, semantic_matches = pp.semantic_similarity(relations_a, relations_b)
                        except Exception as e:  # noqa: BLE001
                            st.error(f"意味マッチングでエラーが発生しました: {e}")
                    st.session_state.compare_result = {
                        "jaccard_score": jaccard_score, "common": common, "only_a": only_a, "only_b": only_b,
                        "structural_score": structural_score, "structural_detail": structural_detail,
                        "semantic_score": semantic_score, "semantic_matches": semantic_matches,
                        "scope_a": pp.compute_claim_scope_score(relations_a),
                        "scope_b": pp.compute_claim_scope_score(relations_b),
                        "relations_a": relations_a, "relations_b": relations_b,
                    }
                except Exception as e:  # noqa: BLE001
                    st.error(f"解析中にエラーが発生しました: {e}")
                    st.session_state.compare_result = None
    res = st.session_state.compare_result
    if res is None:
        return
    st.markdown("### 🦪 診断結果")
    sc = st.columns(3)
    sc[0].metric("①Jaccard類似度（表記の一致）", f"{res['jaccard_score']:.3f}")
    sc[1].metric("②意味マッチング類似度", f"{res['semantic_score']:.3f}" if res["semantic_score"] is not None else "―（未使用）")
    sc[2].metric("③構造の類似度", f"{res['structural_score']:.3f}")
    with st.expander("③構造比較の内訳を見る"):
        d = res["structural_detail"]
        st.write("\n".join(f"- {k}: {v:.3f}" for k, v in d.items() if isinstance(v, (int, float))))
    st.markdown("### 🐙 クレームの広さ・狭さの比較")
    ca, cb = st.columns(2)
    for col, key, name in ((ca, "scope_a", "請求項A"), (cb, "scope_b", "請求項B")):
        narrowness, breadth, detail = res[key]
        with col:
            st.markdown(f"**{name}**")
            st.metric("広さスコア", f"{breadth:.3f}")
            st.caption(f"構成要素数: {detail['構成要素数']} / 数値スペック: {detail['数値スペックの数']} / "
                       f"階層の深さ: {detail['階層の深さ']}")
    st.caption("※ このスコアは絶対的な尺度ではなく、AとBを相対的に比べるための指標です。")
    ga, gb = st.columns(2)
    with ga:
        sao_graph(res["relations_a"])
    with gb:
        sao_graph(res["relations_b"])
    st.markdown("### 🧩 ①Jaccard：トリプルの一致・不一致")
    c1, c2, c3 = st.columns(3)
    for col, key, name in ((c1, "common", "共通トリプル"), (c2, "only_a", "Aだけにあるトリプル"), (c3, "only_b", "Bだけにあるトリプル")):
        with col:
            st.markdown(f"**{name}（{len(res[key])}件）**")
            for t in sorted(res[key]):
                st.write(t)
    if res["semantic_matches"] is not None:
        st.markdown("### 🫧 ②意味マッチング：対応付けの詳細")
        st.dataframe([{"類似度": round(sim, 2), "判定": "完全一致" if ta == tb else ("意味が近い" if sim >= 0.6 else "対応薄い"),
                       "トリプルA": " / ".join(ta), "トリプルB": " / ".join(tb)}
                      for ta, tb, sim in sorted(res["semantic_matches"], key=lambda x: -x[2])],
                     use_container_width=True, hide_index=True)


# ===========================================================================
# 🪼 従属請求項を展開
# ===========================================================================

def page_dependent():
    st.title("🪼 従属請求項を展開")
    st.caption("公報の書き方（【請求項１】【請求項２】…）のまま貼り付けると、「請求項１に記載の」等の引用関係を"
               f"展開してから、推奨手法（{RECOMMENDED}）で解析します。")
    claims_text = st.text_area("請求項群", height=280, key="claims_text",
                               placeholder="【請求項１】\n（請求項1の全文）\n【請求項２】\n（「請求項１に記載の」を含む全文）")
    parsed = pp.parse_claims_block(claims_text) if claims_text.strip() else {}
    if claims_text.strip() and not parsed:
        st.warning("請求項を認識できませんでした。テキストを確認してください。")
    elif parsed:
        st.success(f"✅ 請求項 {sorted(parsed.keys())} を認識しました。")
    target = st.selectbox("展開する請求項番号", sorted(parsed.keys())) if parsed else None
    if st.button("🔍 展開して解析する", disabled=not parsed):
        try:
            full_text = pp.resolve_dependent_claim(target, parsed)
            _, relations = llm_extract(full_text)
            st.session_state.dependent_result = {"full_text": full_text, "relations": relations}
        except Exception as e:  # noqa: BLE001
            st.error(f"展開・解析中にエラーが発生しました: {e}")
            st.session_state.dependent_result = None
    res = st.session_state.dependent_result
    if res is None:
        return
    st.markdown("#### 📖 展開後の完全な請求項テキスト")
    st.info(res["full_text"])
    rels = res["relations"]
    if not rels:
        st.info("関係が抽出できませんでした。")
        return
    c1, c2 = st.columns([3, 2])
    with c1:
        sao_graph(rels)
        narrowness, breadth, _ = pp.compute_claim_scope_score(rels)
        m = st.columns(2)
        m[0].metric("広さスコア", f"{breadth:.3f}")
        m[1].metric("狭さスコア", f"{narrowness:.3f}")
    with c2:
        st.dataframe([{"主語": r["source"], "関係": r["relation"], "目的語": r["target"]} for r in rels],
                     use_container_width=True, hide_index=True)


# ===========================================================================
# 🔦 まとめて検索
# ===========================================================================

def build_patent_database_llm(records, progress_callback=None):
    model = pp._get_embed_model()
    database = []
    for i, (rid, text) in enumerate(records):
        try:
            _, relations = llm_extract(text)
        except Exception:  # noqa: BLE001
            relations = []
        if progress_callback:
            progress_callback(i + 1, len(records))
        if not relations:
            continue
        triples = sorted(pp.relations_to_triple_set(relations, normalize_numbers=True))
        emb = model.encode([pp._triple_to_text(t) for t in triples], normalize_embeddings=True)
        v = np.mean(emb, axis=0)
        database.append({"id": rid, "text": text, "relations": relations, "doc_embedding": v / (np.linalg.norm(v) + 1e-8)})
    return database


def search_similar_claims_llm(query_text, database, top_k=10, rerank_k=5):
    _, q_rel = llm_extract(query_text)
    if not q_rel:
        return []
    model = pp._get_embed_model()
    triples = sorted(pp.relations_to_triple_set(q_rel, normalize_numbers=True))
    emb = model.encode([pp._triple_to_text(t) for t in triples], normalize_embeddings=True)
    q = np.mean(emb, axis=0)
    q = q / (np.linalg.norm(q) + 1e-8)
    scored = sorted(({"id": e["id"], "text": e["text"], "relations": e["relations"],
                      "fast_score": float(np.dot(q, e["doc_embedding"]))} for e in database), key=lambda x: -x["fast_score"])
    top = scored[:top_k]
    for e in top[:rerank_k]:
        e["precise_score"], e["matches"] = pp.semantic_similarity(q_rel, e["relations"])
    return sorted([e for e in top if "precise_score" in e], key=lambda x: -x["precise_score"]) + \
        [e for e in top if "precise_score" not in e]


def page_search():
    st.title("🔦 まとめて検索")
    target = st.radio("検索対象", ["532件のコーパス（事前抽出済み・高速）", "自分で登録した請求項"], horizontal=True)
    if target.startswith("532"):
        need_corpus()
        from sklearn.feature_extraction.text import TfidfVectorizer

        st.caption("検索したい請求項を推奨手法で解析し、532件のSAO（基本語のトリプル・組・構成要素）とTF-IDFの"
                   "コサイン類似度で比べます。")
        q = st.text_area("検索したい請求項テキスト", height=160, key="corpus_query")
        k = st.slider("表示件数", 3, 30, 10)
        if st.button("🔍 検索する", type="primary", key="corpus_search") and q.strip():
            with st.spinner("解析・検索中…"):
                _, q_rel = llm_extract(q)
                docs = [PC.sao_tokens(PC.effective_relations(p, st.session_state.reviews)) for p in CORPUS["patents"]]
                vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True).fit(docs)
                X = vec.transform(docs)
                v = vec.transform([PC.sao_tokens(q_rel)])
                sims = (X @ v.T).toarray().ravel()
            st.session_state.search_results = {"mode": "corpus", "q_rel": q_rel, "sims": sims, "k": k}
        res = st.session_state.search_results
        if res and res.get("mode") == "corpus":
            order = np.argsort(-res["sims"])[:res["k"]]
            for rank, j in enumerate(order, 1):
                p = CORPUS["patents"][j]
                ex = PC.similarity_explain(res["q_rel"], PC.effective_relations(p, st.session_state.reviews))
                with st.expander(f"{rank}位｜{p['id']}｜{p['title']}｜{p['company']}｜類似度 {res['sims'][j]:.3f}"):
                    st.write("共通のSAO：" + ("、".join(ex["共通のSAO"][:15]) or "―"))
                    st.write("共通の構成要素：" + ("、".join(ex["共通の構成要素"][:20]) or "―"))
                    st.markdown(PC.highlight(p["text"], set()), unsafe_allow_html=True)
        return

    st.caption("複数の請求項をデータベース化し、調べたい請求項に似ているものを検索します（推奨手法で解析→埋め込みベクトルで検索）。")
    uploaded_csv = st.file_uploader("CSVファイル（id, text の2列）", type=["csv"])
    bulk_text = st.text_area("またはここに、請求項を「-----」で区切って貼り付ける", height=180, key="bulk_text")
    if st.button("📚 データベースを構築する"):
        records = []
        if uploaded_csv is not None:
            reader = csv.DictReader(io.StringIO(uploaded_csv.getvalue().decode("utf-8-sig")))
            for row in reader:
                rtext = row.get("text") or row.get("本文") or ""
                if rtext.strip():
                    records.append((row.get("id") or row.get("番号") or f"行{len(records) + 1}", rtext.strip()))
        elif bulk_text.strip():
            records = [(f"請求項{i + 1}", p.strip()) for i, p in enumerate(bulk_text.split("-----")) if p.strip()]
        if not records:
            st.warning("CSVのアップロード、またはテキストの貼り付けのどちらかを行ってください。")
        else:
            bar = st.progress(0, text=f"0/{len(records)}件")
            try:
                st.session_state.patent_db = build_patent_database_llm(
                    records, progress_callback=lambda d, t: bar.progress(d / t, text=f"{d}/{t}件"))
                st.session_state.search_results = None
            except Exception as e:  # noqa: BLE001
                st.error(f"データベース構築中にエラーが発生しました: {e}")
            bar.empty()
    if st.session_state.patent_db is not None:
        st.success(f"✅ {len(st.session_state.patent_db)} 件を登録済みです。")
    query_text = st.text_area("検索したい請求項テキスト", height=160, key="query_text")
    c1, c2 = st.columns(2)
    top_k = c1.slider("粗い絞り込みで残す件数", 3, 30, 10)
    rerank_k = c2.slider("精密な再評価をする件数（上位から）", 1, 10, 5)
    if st.button("🔍 検索する", key="search_run"):
        if st.session_state.patent_db is None:
            st.warning("先にデータベースを構築してください。")
        elif query_text.strip():
            with st.spinner("検索中..."):
                st.session_state.search_results = {"mode": "db", "results": search_similar_claims_llm(
                    query_text, st.session_state.patent_db, top_k=top_k, rerank_k=rerank_k)}
    res = st.session_state.search_results
    if res and res.get("mode") == "db":
        for i, r in enumerate(res["results"]):
            label = f"精密スコア {r['precise_score']:.3f}" if "precise_score" in r else f"粗いスコア {r['fast_score']:.3f}"
            with st.expander(f"{i + 1}位　【{r['id']}】　{label}"):
                st.write(r["text"])
                if "matches" in r:
                    st.dataframe([{"類似度": round(sim, 2), "クエリ側": " / ".join(ta), "この請求項側": " / ".join(tb)}
                                  for ta, tb, sim in sorted(r["matches"], key=lambda x: -x[2])],
                                 use_container_width=True, hide_index=True)


# ===========================================================================
# 📊 特許統計分析（CSV）
# ===========================================================================

def _find_column(df, aliases):
    normalized = {str(c).strip().lower().replace(" ", "").replace("　", ""): c for c in df.columns}
    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "").replace("　", "")
        if key in normalized:
            return normalized[key]
    return None


def _split_multi_value(value):
    if pd.isna(value):
        return []
    s = re.sub(r"[；;、,\n\r]+", "|", str(value).strip())
    return [x.strip() for x in s.split("|") if x.strip()]


def _extract_year(value):
    if pd.isna(value):
        return None
    m = re.search(r"(19|20)\d{2}", str(value))
    return int(m.group(0)) if m else None


def _extract_fi_subclass(value):
    out = []
    for fi in _split_multi_value(value):
        m = re.match(r"^([A-HY][0-9]{2}[A-Z])", fi.strip().upper())
        out.append(m.group(1) if m else fi.strip().upper())
    return list(dict.fromkeys(out))


def page_stats():
    import plotly.express as px

    st.title("📊 特許統計分析（CSV）")
    st.caption("J-PlatPat等から出力したCSV（出願日・FI・出願人/権利者の列）を読み込み、書誌情報を集計します。"
               "532件のコーパスを使う場合は「532件のデータを使う」を選んでください。")
    src = st.radio("データ", ["532件のデータを使う", "CSVをアップロード／貼り付け"], horizontal=True)
    df = None
    if src.startswith("532"):
        need_corpus()
        df = pd.DataFrame([{"出願日": p.get("filing_date"), "FI": p["fi"], "出願人/権利者": p["applicant"]}
                           for p in CORPUS["patents"]])
    else:
        up = st.file_uploader("📁 統計分析用CSV", type=["csv"], key="stats_csv_upload")
        txt = st.text_area("またはCSV本文を貼り付け", height=150, key="stats_csv_text",
                           placeholder="出願日,FI,出願人/権利者\n2022-04-01,H01L 21/00,株式会社A")
        if st.button("📊 読み込む", type="primary"):
            try:
                content = up.getvalue().decode("utf-8-sig") if up is not None else txt
                st.session_state.stats_df = pd.read_csv(io.StringIO(content)) if content.strip() else None
            except Exception as e:  # noqa: BLE001
                st.error(f"CSVの読み込みに失敗しました: {e}")
        df = st.session_state.stats_df
    if df is None:
        return
    dcol = _find_column(df, ["出願日", "出願年月日", "application_date", "filing_date", "date"])
    fcol = _find_column(df, ["FI", "FI分類", "fi_code", "fi"])
    acol = _find_column(df, ["出願人/権利者", "出願人／権利者", "出願人", "applicant", "applicants"])
    if not (dcol and fcol and acol):
        st.error("「出願日」「FI」「出願人/権利者」の列が必要です。")
        return
    work = pd.DataFrame({"出願年": df[dcol].apply(_extract_year), "FI": df[fcol], "出願人": df[acol]}).dropna(subset=["出願年"])
    work["出願年"] = work["出願年"].astype(int)
    work["筆頭FIサブクラス"] = work["FI"].apply(lambda x: (_extract_fi_subclass(x) or [None])[0])
    work["筆頭出願人"] = work["出願人"].apply(lambda x: (_split_multi_value(x) or [None])[0])
    st.success(f"✅ {len(work):,} 件を集計しました。")
    top_n = st.number_input("ランキング表示件数", 5, 100, 15, step=5)
    y = work.groupby("出願年").size().rename("件数").reset_index()
    plot_or_warn(px.line(y, x="出願年", y="件数", markers=True, title="① 年別出願件数"))
    c1, c2 = st.columns(2)
    fi = work.groupby("筆頭FIサブクラス").size().sort_values(ascending=False).head(int(top_n)).rename("件数").reset_index()
    c1.plotly_chart(px.bar(fi[::-1], x="件数", y="筆頭FIサブクラス", orientation="h", title="② 筆頭FIサブクラス"),
                    use_container_width=True)
    ap = work.groupby("筆頭出願人").size().sort_values(ascending=False).head(int(top_n)).rename("件数").reset_index()
    c2.plotly_chart(px.bar(ap[::-1], x="件数", y="筆頭出願人", orientation="h", title="③ 筆頭出願人"),
                    use_container_width=True)
    rows = [{"出願人": a, "FIサブクラス": f} for _, r in work.iterrows()
            for a in _split_multi_value(r["出願人"]) for f in _extract_fi_subclass(r["FI"])]
    if rows:
        t = pd.DataFrame(rows).groupby(["出願人", "FIサブクラス"]).size().rename("件数").reset_index()
        ta = t.groupby("出願人")["件数"].sum().sort_values(ascending=False).head(10).index
        tf = t.groupby("FIサブクラス")["件数"].sum().sort_values(ascending=False).head(10).index
        t = t[t["出願人"].isin(ta) & t["FIサブクラス"].isin(tf)]
        plot_or_warn(px.scatter(t, x="FIサブクラス", y="出願人", size="件数", color="件数", size_max=40,
                                title="④ 出願人 × FIサブクラス（バブル）", height=520))


# ===========================================================================
# ✅ 精度検証（自動ヘルスチェック）
# ===========================================================================

def page_health():
    st.title("✅ 精度検証（自動ヘルスチェック）")
    st.caption("人手の正解データがない前提で、既知の不具合パターン（関係が1件も取れない・自己ループ・意味のない語が"
               "ノードになる・部品が孤立する 等）を自動チェックし、合格率を精度の代理指標として使います。"
               "件数分のLLM呼び出しが発生します。")
    up = st.file_uploader("📁 請求項リスト（.xlsx / .csv）", type=["xlsx", "csv"], key="eval_file_upload")
    c1, c2 = st.columns(2)
    text_col = c1.text_input("請求項本文の列名", value="請求項本文")
    id_col = c2.text_input("ID列名（無ければ空欄）", value="id")
    limit = st.number_input("検証する件数の上限（0で全件）", min_value=0, value=20, step=10)
    if st.button("🚀 精度検証を実行する", type="primary"):
        if up is None:
            st.warning("ファイルをアップロードしてください。")
        else:
            try:
                df = pd.read_excel(up) if up.name.lower().endswith(".xlsx") else pd.read_csv(up)
                if text_col not in df.columns:
                    st.error(f"列「{text_col}」が見つかりません。列一覧: {list(df.columns)}")
                else:
                    texts = df[text_col].fillna("").astype(str).tolist()
                    ids = df[id_col].astype(str).tolist() if id_col and id_col in df.columns else list(range(len(texts)))
                    records = [(i, t) for i, t in zip(ids, texts) if t.strip()]
                    if limit:
                        records = records[:int(limit)]
                    bar = st.progress(0, text=f"0/{len(records)}件")
                    results, summary = pp.evaluate_corpus_health(
                        records, analyze_fn=llm_extract, progress_callback=lambda d, t: bar.progress(d / t, text=f"{d}/{t}件"))
                    bar.empty()
                    st.session_state.eval_results, st.session_state.eval_summary = results, summary
            except Exception as e:  # noqa: BLE001
                st.error(f"検証中にエラーが発生しました: {e}")
    summary = st.session_state.eval_summary
    if summary is None:
        return
    m = st.columns(4)
    m[0].metric("合格率", f"{summary['pass_rate'] * 100:.1f}%")
    m[1].metric("検証件数", f"{summary['total']}件")
    m[2].metric("平均構成要素数", f"{summary['avg_components']:.1f}")
    m[3].metric("平均関係数", f"{summary['avg_relations']:.1f}")
    rdf = pd.DataFrame(st.session_state.eval_results)
    checks = [c for c in rdf.columns if c.startswith("check_")]
    if checks:
        st.dataframe((rdf[checks].mean().sort_values() * 100).round(1).rename("合格率(%)").reset_index(),
                     use_container_width=True, hide_index=True)
    only_failed = st.checkbox("不合格のものだけ表示", value=True)
    show = rdf[~rdf["passed"]] if only_failed else rdf
    st.dataframe(show[[c for c in ["id", "text", "passed", "n_components", "n_relations", "error"] + checks
                       if c in show.columns]], use_container_width=True, hide_index=True)


# ===========================================================================
# 📤 エクスポート
# ===========================================================================

def page_export():
    need_corpus()
    st.title("📤 エクスポート")
    reviews = st.session_state.reviews
    st.markdown("#### 分析結果の書き出し")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Excel（分析結果一式）**")
        st.caption("概要／特許一覧／SAO（AI判定）／SAO（人手確認済み）／構造的特徴／類似特許 のシート")
        if st.button("Excelファイルを作成"):
            with st.spinner("作成中…"):
                st.session_state["_xlsx"] = PC.export_excel(CORPUS, reviews, corpus_similarity(CORPUS, reviews_key()))
        if st.session_state.get("_xlsx"):
            st.download_button("⬇️ Excelをダウンロード", st.session_state["_xlsx"], file_name="patent_sao_analysis.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        st.markdown("**CSV（現在のSAO）**")
        st.caption("人手確認済みの特許は確認結果、それ以外はAIが選んだSAO")
        st.download_button("⬇️ CSVをダウンロード", PC.relations_csv(CORPUS, reviews), file_name="patent_sao.csv",
                           mime="text/csv")
    st.divider()
    st.markdown("#### 人手確認の保存と読み込み")
    st.caption("確認結果はブラウザを閉じると消えます。作業を続けるときは保存したCSVを読み込んでください。")
    d1, d2 = st.columns(2)
    d1.download_button(f"⬇️ 人手確認結果を保存（{len(reviews)}件）", PC.reviews_to_csv(reviews),
                       file_name="sao_reviews.csv", mime="text/csv", disabled=not reviews)
    up = d2.file_uploader("人手確認結果のCSVを読み込む", type=["csv"], key="reviews_upload")
    if up is not None and d2.button("読み込む"):
        try:
            loaded = PC.reviews_from_csv(up.getvalue())
            st.session_state.reviews.update({k: v for k, v in loaded.items() if k in PATENTS})
            st.success(f"{len(loaded)} 件の確認結果を読み込みました。")
        except Exception as e:  # noqa: BLE001
            st.error(f"読み込みに失敗しました: {e}")
    if st.session_state.workspace:
        st.divider()
        st.markdown(f"#### AI解析のワークスペース（{len(st.session_state.workspace)}件）")
        rows = [{"解析ID": w["id"], "主語(S)": r["source"], "関係(A)": r["relation"], "目的語(O)": r["target"],
                 "手法": w["method"]} for w in st.session_state.workspace for r in w["relations"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.download_button("⬇️ ワークスペースをCSVで保存", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                           file_name="sao_workspace.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# ナビゲーション
# ---------------------------------------------------------------------------
nav = st.navigation({
    "概要": [st.Page(page_dashboard, title="ダッシュボード", icon="🏠", default=True)],
    "抽出と確認": [st.Page(page_analyze, title="AI解析", icon="🧪", url_path="analyze"),
                 st.Page(page_review, title="人手確認（532件）", icon="✍️", url_path="review")],
    "可視化・分析": [st.Page(page_world, title="Patent World", icon="🌍", url_path="world"),
                   st.Page(page_network, title="SAOネットワーク", icon="🕸️", url_path="network"),
                   st.Page(page_similarity, title="類似性マップ", icon="🗺️", url_path="similarity"),
                   st.Page(page_radar, title="構造レーダー", icon="🧭", url_path="radar"),
                   st.Page(page_distribution, title="技術分布", icon="🫧", url_path="distribution")],
    "個別ツール": [st.Page(page_compare, title="2つの請求項を比較", icon="🐚", url_path="compare"),
                st.Page(page_dependent, title="従属請求項を展開", icon="🪼", url_path="dependent"),
                st.Page(page_search, title="まとめて検索", icon="🔦", url_path="search"),
                st.Page(page_stats, title="特許統計分析（CSV）", icon="📊", url_path="stats"),
                st.Page(page_health, title="精度検証", icon="✅", url_path="health")],
    "出力": [st.Page(page_export, title="エクスポート", icon="📤", url_path="export")],
})
nav.run()
