# -*- coding: utf-8 -*-
"""
特許分析プラットフォーム（卒業研究）
====================================

「AIを用いた日本語特許文献の構造分析に関する研究
 ―SAO構造を用いた半導体関連特許の類似性分析―」

任意の特許リスト（J-PlatPat などから出力した CSV／Excel）を読み込み、請求項から
SAO（主語―関係―目的語）構造を取り出して、人が確認・修正しながら分析する
「人とAIの協働型」の特許分析ツール。抽出方法は、532件の正解データで最も精度の
高かった実験13（係り受け候補＋区間の主役の候補＋2段階選別）に固定している。

ページ構成（サイドバーのナビゲーション）
  データ
    📥 データの読み込み       … 特許リストを読み込んで一括解析／解析済みデータを開く／サンプル（532件）
    🏠 ダッシュボード          … 抽出・確認の進み具合、判定の根拠、抽出の自動チェック
  抽出と確認
    🧪 AI解析                 … 1件の請求項を解析し、処理の各段階と採用／要確認／除外を表示
    ✍️ 人手確認               … AIの判定を人が確認・修正して確定する
  可視化・分析
    🌍 Patent World           … オズの世界（発明の名称＋FI の近さで3次元に配置）
    🕸️ SAOネットワーク         … 構成要素のつながり。ノードをクリックすると該当請求項へ
    🗺️ 類似性マップ            … SAOの近さで配置。点をクリックすると似た特許と共通部分
    🧭 FIレーダー              … 出願人（または出願年）ごとのFIの分布
    🫧 技術分布               … 出願年×FIのバブル、出願人×技術のヒートマップ、出願の推移とランキング
  個別ツール
    🐚 2つの請求項を比較 / 🪼 従属請求項を展開
  出力
    📤 エクスポート            … Excel／CSV／解析済みデータ（JSON）、人手確認結果の保存と読み込み

【ファイル構成】
プログラムは app.py（画面）と patent_pipeline.py（解析の処理すべて）の2つ。
同じフォルダに、次のデータファイルを置く。
  sao_selector13_train.npz … 実験13の選別モデルの学習データ（必須）
  oz_world_embed.html      … Patent World（オズの世界）の表示部品
  corpus_sao_532.json      … サンプルデータ（半導体関連特許532件の解析結果）

【実行方法（自分のPC・Ollama）】
    pip install -r requirements.txt
    ollama pull qwen2.5:7b
    streamlit run app.py

【Streamlit Community Cloud】
Secrets に OPENROUTER_API_KEY を設定すると、LLM呼び出し（patent_pipeline._ollama_chat）を
OpenRouter 経由の qwen2.5-7b-instruct に差し替えて動かす。
"""

import csv
import html
import io
import json
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import patent_pipeline as pp

ts = pp                                  # LLM抽出（旧 translate_sao.py）
PC = pp.platform_core                    # 分析・集計・書き出し（旧 platform_core.py）
relations_to_nested_dot = pp.relations_to_nested_dot  # 入れ子の構造図（旧 nested_graph.py）

st.set_page_config(page_title="特許分析プラットフォーム", layout="wide", page_icon="🔬")

APP_DIR = Path(__file__).resolve().parent
OZ_WORLD_HTML_PATH = APP_DIR / "oz_world_embed.html"
CLOUD_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SAMPLE_NAME = "サンプル：半導体関連特許 532件"

SRC_GROUPS = [
    ("LLM抽出（qwen2.5）", ("LLMraw", "E1:llm_direct")),
    ("GiNZA補完", ("E1:claim_title_ginza", "E1:ginza_has_fallback", "E1:attribute",
                 "E1:ginza_has_fallback_conflict", "E1:claim_title_ginza_conflict")),
    ("GiNZA単体（係り受け・位置・所有）", ("G:",)),
    ("請求項の分割（手がかり句）", ("GS:",)),
    ("ノード結合（XのY）", ("MRG",)),
    ("係り受け候補（述語の項の組）", ("DEP",)),
    ("区間の主役との組", ("SEG",)),
    ("題名が持つ構成要素", ("TITLE",)),
]
STATUS_COLORS = {PC.STATUS_ACCEPT: "#16a34a", PC.STATUS_REVIEW: "#d97706", PC.STATUS_REJECT: "#94a3b8"}

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
    ("review_pid", None), ("dataset", None), ("ingest", None),
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


@st.cache_resource(show_spinner="SAO選別モデル（実験13）を準備中…（初回のみ、1分ほどかかります）")
def load_selector():
    return pp.Selector13()


def tidy_candidates(cands):
    """関係語の頭に残った助詞を取り（「には有する」→「有する」）、同じ候補になったものは
    確率の高い方だけ残す（cands は確率の高い順）。"""
    out, seen = [], set()
    for c in cands:
        c = dict(c, relation=PC.clean_relation(c["relation"]))
        k = (c["source"], c["relation"], c["target"])
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


@st.cache_data(show_spinner="サンプルデータを読み込み中…")
def load_sample():
    f = PC.find_corpus_file()
    if f is None:
        return None
    try:
        data = PC.load_corpus(f)
        assert isinstance(data.get("patents"), list)
    except Exception as exc:  # noqa: BLE001
        # ファイルが空・途中までしかない・HTML（GitHubのページを保存したもの）などの場合でも、
        # アプリ全体は止めず、サンプルなしで起動する
        head = f.read_bytes()[:80]
        return {"_error": f"{f.name} を読み込めませんでした（{type(exc).__name__}、大きさ {f.stat().st_size:,} バイト、"
                          f"先頭 {head[:40]!r}）。"}
    data["meta"].update({"name": SAMPLE_NAME, "sample": True})
    for p in data["patents"]:
        p["relations"] = tidy_candidates(p["relations"])
    PC.assign_groups(data)
    return data


def set_dataset(data):
    """表示するデータセットを切り替える（人手確認などの作業状態はリセット）。"""
    data.setdefault("meta", {})["uid"] = data["meta"].get("uid") or uuid.uuid4().hex
    if not data.get("groups"):
        PC.assign_groups(data)
    st.session_state.dataset = data
    for k, v in (("reviews", {}), ("review_pid", None), ("net_node", None), ("map_patent", None),
                 ("search_results", None), ("_xlsx", None)):
        st.session_state[k] = v


SAMPLE_ERROR = None
if st.session_state.dataset is None:
    _sample = load_sample()
    if _sample is not None and "_error" in _sample:
        SAMPLE_ERROR = _sample["_error"]
    elif _sample is not None:
        set_dataset(json.loads(json.dumps(_sample)))

DATA = st.session_state.dataset
PATENTS = {p["id"]: p for p in DATA["patents"]} if DATA else {}
COLORS = PC.group_colors(DATA) if DATA else {}


def data_key():
    return DATA["meta"]["uid"] + "|" + json.dumps(st.session_state.reviews, ensure_ascii=False, sort_keys=True)


@st.cache_data(show_spinner=False, max_entries=8)
def corpus_similarity(_data, key):
    return PC.similarity_matrix(_data, st.session_state.reviews)


def patent_label(pid):
    p = PATENTS[pid]
    return f"{pid}｜{p['title'] or '（名称なし）'}｜{p['company']}"


def need_data():
    if SAMPLE_ERROR and not DATA:
        st.error("サンプルデータ：" + SAMPLE_ERROR + "GitHub の corpus_sao_532.json を、配布したファイルで"
                 "上げ直してください（中身が「{\"meta\": …」で始まるJSONのはずです）。")
    if not DATA or not DATA.get("patents"):
        st.warning("分析するデータがありません。「データの読み込み」ページで特許リストを読み込んでください。")
        st.stop()


# ---------------------------------------------------------------------------
# 抽出（実験13：係り受け候補＋区間の主役の候補＋2段階選別）
# ---------------------------------------------------------------------------

def duplicate_flags(gpp, info, keys):
    """選ばれた関係と同じ組で同義の関係（「有する」と「備える」など）の候補に印を付ける
    （人が確認するときに同じものが繰り返し出ないようにする）。"""
    n = gpp._normalize_node_text_lenient
    cm = pp.canon13(info)

    def key(c):
        return frozenset((n(cm.get(c["source"], c["source"])), n(cm.get(c["target"], c["target"]))))

    chosen = {}
    for c in info["cands"]:
        if (c["source"], c["relation"], c["target"]) in keys:
            chosen.setdefault(key(c), []).append(c["relation"])
    return [(c["source"], c["relation"], c["target"]) not in keys
            and any(pp.rel_match(gpp, c["relation"], r) or pp.rel_match(gpp, r, c["relation"])
                    for r in chosen.get(key(c), []))
            for c in info["cands"]]


@st.cache_data(show_spinner=False, max_entries=2000)
def analyze(text, model, host):
    """1件の請求項を解析し、処理の各段階と全候補の判定を返す。"""
    gpp = load_pipeline()
    selector = load_selector()
    t0 = time.time()
    info = pp.build_candidates13(ts, gpp, text, model=model, host=host)
    prob = selector.predict(gpp, info)
    chosen = pp.select13(gpp, info, prob, selector.threshold, selector.max_per_pair)
    keys = {(r["source"], r["relation"], r["target"]) for r in chosen}
    bands = dict(PC.DEFAULT_BANDS, threshold=selector.threshold)
    dup = duplicate_flags(gpp, info, keys)
    cands = []
    for i, c in enumerate(info["cands"]):
        sel = (c["source"], c["relation"], c["target"]) in keys
        p = float(prob[i]) if len(prob) else 0.0
        cands.append({"source": c["source"], "relation": c["relation"], "target": c["target"],
                      "prob": round(p, 4), "selected": sel,
                      "status": PC.STATUS_REJECT if dup[i] else PC.classify(p, sel, bands),
                      "origin": PC.origin_label(c.get("srcs", [])) or "選別", "srcs": list(c.get("srcs", []))})
    cands.sort(key=lambda r: -r["prob"])
    cands = tidy_candidates(cands)
    steps = [("前処理・構成要素の抽出（GiNZA）", f"構成要素 {len(info['tags'])} 個／形式：{info.get('format', '―')}")]
    for label, prefixes in SRC_GROUPS:
        k = sum(1 for c in info["cands"] if any(s.startswith(prefixes) for s in c.get("srcs", [])))
        if k:
            steps.append((label, f"候補 {k} 件"))
    steps.append(("候補の統合（重複をまとめる）", f"候補 {len(info['cands'])} 件"))
    steps.append(("選別モデル（確率の算出）", f"しきい値 {selector.threshold:.3f}／選ばれた関係 {len(chosen)} 件"))
    return {"tags": list(info["tags"]), "title": info.get("title"), "format": info.get("format", ""),
            "cands": cands, "steps": steps, "bands": bands, "elapsed": time.time() - t0}


def run_analyze(text):
    return analyze(text, model_name, (ollama_host or "").strip() or None)


def llm_extract(text):
    """比較・従属請求項・検索・精度検証の各ページ共通の抽出（選ばれた関係のみ）。"""
    res = run_analyze(text)
    rels = [{"source": c["source"], "relation": c["relation"], "target": c["target"], "type": c["origin"] or "selected"}
            for c in res["cands"] if c["selected"]]
    return [{"text": t} for t in res["tags"]], rels


def llm_error_message():
    return ("OpenRouterのレート制限に達した可能性があります。しばらく待って再試行してください。" if BACKEND == "cloud" else
            f"Ollamaが起動しているか（ollama serve）、モデルが取得済みか（ollama pull {model_name}）を確認してください。")


def sao_graph(relations):
    if not relations:
        st.info("表示できるSAO関係がありません。")
        return
    direction = "TB" if st.session_state.get("graph_dir", "").startswith("縦") else "LR"
    st.graphviz_chart(relations_to_nested_dot(PC.tidy_relations(relations), direction=direction),
                      use_container_width=True)
    st.markdown(
        "<div style='font-size:.8rem;opacity:.85'>箱の入れ子＝構成（備える・有する・含む）／点線の楕円＝どの構成要素にも"
        "属さない対象（方向・設置対象など）／矢印の色："
        "<span style='color:#2563eb'>■接続</span>　<span style='color:#16a34a'>■配置・位置</span>　"
        "<span style='color:#9333ea'>■覆う・収める</span>　<span style='color:#ea580c'>■動作・機能</span>　"
        "<span style='color:#475569'>■その他</span>（右上のボタンで全画面表示）</div>", unsafe_allow_html=True)


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


def plot(fig, **kw):
    return st.plotly_chart(fig, use_container_width=True, **kw)


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


# ---------------------------------------------------------------------------
# サイドバー（共通設定）
# ---------------------------------------------------------------------------
with st.sidebar:
    if DATA:
        n_done = sum(1 for p in DATA["patents"] if p.get("analyzed", True))
        st.markdown(f"**📂 {DATA['meta'].get('name', 'データセット')}**")
        st.caption(f"{len(DATA['patents']):,} 件（解析済み {n_done:,} 件）")
    st.markdown("### ⚙️ LLMの設定")
    if BACKEND == "cloud":
        st.success("🌐 公開デモモード：OpenRouter経由の qwen2.5-7b-instruct を使用", icon="🌐")
        model_name = st.text_input("モデル名（OpenRouter）", value=CLOUD_MODEL)
        ollama_host = ""
    else:
        st.info("💻 ローカルモード：このPCのOllamaに接続します", icon="💻")
        model_name = st.text_input("Ollamaモデル名", value=ts.DEFAULT_MODEL)
        ollama_host = st.text_input("Ollamaホスト（空欄 = http://localhost:11434）", value="")
    st.markdown("### 🧩 SAO構造図")
    graph_dir = st.radio("図の向き", ["横（左→右）", "縦（上→下）"], horizontal=True, key="graph_dir")
    st.caption(f"抽出方法：{PC.METHOD_NAME}。{PC.METHOD_SCORE}。")


# ===========================================================================
# 📥 データの読み込み
# ===========================================================================

def page_data():
    st.title("📥 データの読み込み")
    if DATA:
        st.info(f"現在のデータ：**{DATA['meta'].get('name')}**（{len(DATA['patents']):,} 件）", icon="📂")
    tab1, tab2, tab3 = st.tabs(["📄 特許リストを読み込んで解析", "💾 解析済みデータを開く", "🧪 サンプルデータ"])

    with tab1:
        st.markdown(
            "J-PlatPat などから出力した **CSV／Excel** を読み込み、各特許の請求項を実験13の方法で一括解析します。"
            "請求項の列は必須、それ以外（文献番号・発明の名称・出願人・FI・出願日）はあれば使います。"
            "請求項の欄に【請求項１】【請求項２】…と複数入っている場合は、請求項1だけを解析します。")
        up = st.file_uploader("特許リスト（.csv / .xlsx）", type=["csv", "xlsx", "xls"], key="ds_upload")
        pasted = st.text_area("または、請求項を「-----」で区切って貼り付け（書誌情報なしで解析）", height=120,
                              key="ds_paste")
        df = None
        if up is not None:
            try:
                df = PC.read_table(up.getvalue(), up.name)
            except Exception as e:  # noqa: BLE001
                st.error(f"ファイルを読み込めませんでした: {e}")
        elif pasted.strip():
            df = pd.DataFrame({"請求項": [t.strip() for t in pasted.split("-----") if t.strip()]})
        if df is not None:
            st.caption(f"{len(df):,} 行を読み込みました。列の対応を確認してください。")
            det = PC.detect_columns(df)
            labels = {"claim": "請求項（必須）", "id": "文献番号", "title": "発明の名称", "applicant": "出願人",
                      "fi": "FI", "date": "出願日", "url": "URL"}
            opts = ["（なし）"] + list(df.columns)
            cols = {}
            grid = st.columns(4)
            for i, (k, lab) in enumerate(labels.items()):
                cur = det.get(k)
                cols[k] = grid[i % 4].selectbox(lab, opts, index=opts.index(cur) if cur in opts else 0, key=f"col_{k}")
                cols[k] = None if cols[k] == "（なし）" else cols[k]
            cols["ipc"] = det.get("ipc")
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
            c1, c2 = st.columns([2, 1])
            name = c1.text_input("データセットの名前", value=Path(up.name).stem if up is not None else "貼り付けた請求項")
            limit = c2.number_input("解析する件数の上限（0で全件）", min_value=0, value=0, step=10)
            st.caption("1件あたり数秒〜1分ほどかかります（LLMの呼び出しを含む）。途中で止まっても、"
                       "同じボタンでもう一度押すと続きから解析します（解析済みの請求項は再計算しません）。"
                       + ("公開デモ（OpenRouter無料枠）は1日あたりの回数に制限があるため、少ない件数で試してください。"
                          if BACKEND == "cloud" else ""))
            if st.button("🚀 解析を開始（または続きから）", type="primary", disabled=not cols["claim"]):
                ing = st.session_state.ingest
                if not ing or ing.get("name") != name or ing.get("n_rows") != len(df):
                    ing = {"name": name, "n_rows": len(df),
                           "patents": PC.patents_from_table(df, cols, limit=int(limit) or None), "errors": {}}
                    st.session_state.ingest = ing
                run_ingest(ing)
        ing = st.session_state.ingest
        if ing and not all(p["analyzed"] for p in ing["patents"]):
            done = sum(p["analyzed"] for p in ing["patents"])
            st.warning(f"「{ing['name']}」は {done}/{len(ing['patents'])} 件まで解析済みです。"
                       "上のボタンで続きから解析するか、ここまでの結果で分析を始められます。")
            if st.button("ここまでの結果で分析を始める", disabled=done == 0):
                finish_ingest(ing, partial=True)

    with tab2:
        st.markdown("以前にこのアプリで解析して保存したデータ（エクスポートページの「解析済みデータ（JSON）」）を開きます。"
                    "LLMを呼ばずにすぐ分析を始められます。")
        upj = st.file_uploader("解析済みデータ（.json）", type=["json"], key="ds_json")
        if upj is not None and st.button("このデータを開く", type="primary"):
            try:
                data = json.loads(upj.getvalue().decode("utf-8"))
                assert isinstance(data.get("patents"), list)
                if any("x" not in p or "map_x" not in p for p in data["patents"]):
                    with st.spinner("配置を計算中…"):
                        PC.finalize_dataset(data)
                data.setdefault("bands", dict(PC.DEFAULT_BANDS))
                set_dataset(data)
                st.success(f"「{data['meta'].get('name')}」（{len(data['patents'])}件）を開きました。")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"読み込めませんでした（このアプリで保存したJSONか確認してください）: {e}")

    with tab3:
        st.markdown("卒業研究で使った半導体関連特許532件（三菱電機・富士電機・ローム・東芝など）の解析結果です。"
                    "各特許の判定は、その特許を学習に使っていない交差検証のモデルで求めています。")
        if st.button("サンプルデータを開く"):
            s = load_sample()
            if s is None or "_error" in s:
                st.error("サンプルデータを開けません。" + (s["_error"] if s else "corpus_sao_532.json が見つかりません。"))
            else:
                set_dataset(json.loads(json.dumps(s)))
                st.rerun()


def run_ingest(ing):
    todo = [p for p in ing["patents"] if not p["analyzed"]]
    if not todo:
        finish_ingest(ing)
        return
    try:
        with st.spinner("選別モデルを準備中…"):
            load_pipeline()
            load_selector()
    except Exception as exc:  # noqa: BLE001
        st.error("解析の準備に失敗しました。")
        st.exception(exc)
        return
    total = len(ing["patents"])
    bar = st.progress(0.0, text="解析中…")
    t0 = time.time()
    ok = 0
    for k, p in enumerate(todo):
        done = total - len(todo) + k
        bar.progress(done / total, text=f"{done}/{total} 件　{p['id']}　{(p['title'] or '')[:30]}")
        try:
            res = run_analyze(p["text"])
            PC.apply_analysis(p, res["cands"])
            ing["errors"].pop(p["id"], None)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            ing["errors"][p["id"]] = str(exc)[:200]
            if k >= 2 and ok == 0:
                bar.empty()
                st.error("最初の3件が続けて失敗したため止めました。" + llm_error_message())
                st.code(str(exc)[:500])
                return
    bar.progress(1.0, text=f"{total}/{total} 件（{time.time() - t0:.0f}秒）")
    if ing["errors"]:
        st.warning(f"{len(ing['errors'])} 件は解析に失敗しました（もう一度ボタンを押すと再試行します）。")
    finish_ingest(ing, partial=bool(ing["errors"]))


def finish_ingest(ing, partial=False):
    ps = [p for p in ing["patents"] if p["analyzed"]]
    if not ps:
        st.error("解析できた特許がありません。")
        return
    data = PC.new_dataset(ing["name"], [json.loads(json.dumps(p)) for p in ps])
    with st.spinner("オズの世界・類似性マップの配置を計算中…"):
        PC.finalize_dataset(data)
    set_dataset(data)
    if not partial:
        st.session_state.ingest = None
    st.success(f"「{ing['name']}」（{len(ps)}件）の解析が終わりました。ダッシュボードや各分析ページで見られます。"
               "エクスポートページから解析済みデータ（JSON）を保存しておくと、次回はすぐ開けます。")
    st.balloons()


# ===========================================================================
# 🏠 ダッシュボード
# ===========================================================================

def page_dashboard():
    need_data()
    import plotly.express as px
    import plotly.graph_objects as go

    reviews = st.session_state.reviews
    st.title("🏠 ダッシュボード")
    st.caption(f"「{DATA['meta'].get('name')}」の請求項を {PC.METHOD_NAME} で解析した結果と、人による確認の進み具合。")
    c = PC.status_counts(DATA, reviews)
    m = st.columns(4)
    m[0].metric("特許件数", f"{c['特許件数']:,}")
    m[1].metric("AIが抽出したSAO", f"{c['抽出SAO']:,}", help="選別モデルが選んだ関係（採用＋要確認の一部）")
    m[2].metric("人手確認済みの特許", f"{c['人手確認済みの特許']:,} / {c['特許件数']:,}")
    m[3].metric("確定SAO（人手確認済み）", f"{c['確定SAO（人手確認済み）']:,}")
    st.progress(c["人手確認済みの特許"] / max(c["特許件数"], 1), text="人手確認の進み具合")

    st.markdown("#### AIの判定（全候補）")
    status_badges(c)
    b = DATA.get("bands", PC.DEFAULT_BANDS)
    st.caption(f"採用＝選別モデルが選び、確率 {b['accept']:.2f} 以上／要確認＝選ばれたが確率がそれ未満、"
               f"または選ばれなかったが確率 {b['review_low']:.2f} 以上／除外＝それ以外。")

    left, right = st.columns([3, 2])
    with left:
        rows = []
        for p in DATA["patents"]:
            cnt = Counter(r["status"] for r in p["relations"])
            for s in PC.STATUS_ORDER[:2]:
                rows.append({"出願人": p["group"], "判定": s, "件数": cnt.get(s, 0)})
        df = pd.DataFrame(rows).groupby(["出願人", "判定"], as_index=False)["件数"].sum()
        fig = px.bar(df, x="出願人", y="件数", color="判定", color_discrete_map=STATUS_COLORS, barmode="stack",
                     title="出願人別：採用・要確認の件数")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        plot(fig)
    with right:
        fig = go.Figure(go.Funnel(y=["全候補", "要確認以上", "採用"],
                                  x=[c["採用"] + c["要確認"] + c["除外"], c["採用"] + c["要確認"], c["採用"]],
                                  marker={"color": ["#94a3b8", "#d97706", "#16a34a"]}))
        fig.update_layout(title="候補の絞り込み", height=360, margin=dict(l=10, r=10, t=50, b=10))
        plot(fig)

    with st.expander("📐 判定の根拠（532件の正解データ・5分割交差検証で推定）", expanded=False):
        s = PC.DEFAULT_BANDS["stats"]
        st.dataframe(pd.DataFrame([{"判定": k, "そのうち正しい割合（推定）": f"{100 * s[k]['精度']:.1f}%"}
                                   for k in ("採用", "要確認", "除外")]), hide_index=True, use_container_width=True)
        st.write("正しいSAOが、どの判定に入っているか：" + "／".join(
            f"{k} {100 * v:.1f}%" for k, v in sorted(s["正解の所在"].items(), key=lambda kv: -kv[1])))
        st.caption(f"抽出方法：{PC.METHOD_NAME}（{PC.METHOD_SCORE}）。「採用」は精度80%以上になる確率の下限で"
                   "区切っている。要確認だけを人が見れば、効率よく誤りを直せる。分野の違う特許では精度が変わりうる。")

    st.markdown("#### よく現れる構成要素（基本語）")
    cnt = Counter()
    for p in DATA["patents"]:
        cnt.update({PC.base_term(x) for r in PC.effective_relations(p, reviews) for x in (r["source"], r["target"])})
    top = pd.DataFrame(cnt.most_common(20), columns=["構成要素", "特許件数"])
    fig = px.bar(top[::-1], x="特許件数", y="構成要素", orientation="h", height=520)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    plot(fig)

    with st.expander("✅ 抽出の自動チェック（関係が0件・自己ループ・意味のない語・孤立した部品 などの不具合パターン）"):
        texts = {p["id"]: (p["text"], PC.effective_relations(p, reviews)) for p in DATA["patents"]}

        def _from_data(pid):
            rels = texts[pid][1]
            return [{"text": x} for x in {y for r in rels for y in (r["source"], r["target"])}], rels

        results, summary = pp.evaluate_corpus_health([(pid, pid) for pid in texts], analyze_fn=_from_data)
        m2 = st.columns(3)
        m2[0].metric("合格率", f"{summary['pass_rate'] * 100:.1f}%")
        m2[1].metric("平均構成要素数", f"{summary['avg_components']:.1f}")
        m2[2].metric("平均関係数", f"{summary['avg_relations']:.1f}")
        names = {"check_has_components": "構成要素が取れている", "check_has_relations": "関係が1件以上ある",
                 "check_no_self_loop": "自己ループがない", "check_no_bare_generic_node": "「面」「部」だけの曖昧なノードがない",
                 "check_ending_noun_present": "題名（末尾の名詞）がノードにある",
                 "check_orphan_ratio_ok": "どこにもつながらない構成要素が少ない",
                 "check_no_fan_in_anomaly": "1つの部品に持ち主が集中していない"}
        rdf = pd.DataFrame(results).rename(columns=names)
        checks = [c for c in rdf.columns if c.startswith("check_") or c in names.values()]
        if checks:
            st.dataframe((rdf[checks].mean().sort_values() * 100).round(1).rename("合格率(%)").reset_index()
                         .rename(columns={"index": "チェック項目"}), use_container_width=True, hide_index=True)
        bad = rdf[~rdf["passed"]]
        if len(bad):
            st.markdown(f"不合格の特許（{len(bad)}件）：人手確認で優先して見るとよい")
            st.dataframe(bad[["id"] + checks].rename(columns={"id": "特許番号"}), use_container_width=True,
                         hide_index=True)

    queue = sorted(DATA["patents"], key=lambda p: -sum(r["status"] == PC.STATUS_REVIEW for r in p["relations"]))
    queue = [p for p in queue if p["id"] not in reviews][:10]
    if queue:
        st.markdown("#### 次に確認するとよい特許（要確認が多い順）")
        st.dataframe(pd.DataFrame([{
            "特許番号": p["id"], "発明の名称": p["title"], "出願人": p["company"],
            "要確認": sum(r["status"] == PC.STATUS_REVIEW for r in p["relations"]),
            "採用": sum(r["status"] == PC.STATUS_ACCEPT for r in p["relations"])} for p in queue]),
            hide_index=True, use_container_width=True)


# ===========================================================================
# 🧪 AI解析（1件）
# ===========================================================================

def page_analyze():
    st.title("🧪 AI解析")
    st.caption(f"請求項を1件、{PC.METHOD_NAME}で解析し、処理の各段階とAIの判定（採用／要確認／除外）を表示します。"
               "結果は表で修正してから確定できます。")
    col1, col2 = st.columns([3, 1])
    with col2:
        opts = ["（使わない）"] + ([p["id"] for p in DATA["patents"]] if DATA else [])
        pick = st.selectbox("データセットから読み込む（任意）", opts,
                            format_func=lambda x: x if x == "（使わない）" else patent_label(x))
    with col1:
        default = PATENTS[pick]["text"] if pick != "（使わない）" else SAMPLE_CLAIM
        text = st.text_area("特許請求項テキスト", value=default, height=220, key=f"an_text_{pick}")
    if st.button("🧪 解析する", type="primary"):
        if not text.strip():
            st.warning("請求項テキストを入力してください。")
            st.stop()
        try:
            with st.spinner("解析中…（LLMの呼び出しを含みます）"):
                res = dict(run_analyze(PC.first_claim(text)))
                res.update(text=text, pick=pick)
                st.session_state.analysis = res
        except Exception as exc:  # noqa: BLE001
            st.error("解析中にエラーが発生しました。" + llm_error_message())
            st.exception(exc)
            st.stop()

    res = st.session_state.analysis
    if not res:
        return
    st.success(f"解析完了（{res['elapsed']:.1f}秒）")

    st.markdown("#### ① 処理の流れ")
    per_row = 6
    cols = []
    for k in range(0, len(res["steps"]), per_row):
        cols += st.columns(per_row)
    for i, (col, (name, detail)) in enumerate(zip(cols, res["steps"])):
        col.markdown(
            f"<div style='border:1px solid rgba(148,163,184,.5);border-radius:10px;padding:8px;min-height:110px'>"
            f"<div style='font-size:.75rem;opacity:.7'>STEP {i + 1}</div><div style='font-weight:700;font-size:.9rem'>"
            f"{html.escape(name)}</div><div style='font-size:.85rem;margin-top:4px'>{html.escape(detail)}</div></div>",
            unsafe_allow_html=True)

    st.markdown("#### ② AIの判定")
    status_badges(Counter(c["status"] for c in res["cands"]))
    st.caption(f"採用：確率 {res['bands']['accept']:.2f} 以上で選ばれた関係／要確認：選ばれたがそれ未満、"
               f"または確率 {res['bands']['review_low']:.2f} 以上／除外：それ以外（表では非表示）")

    st.markdown("#### ③ 確認・修正")
    st.caption("「採用する」のチェックを付け外しし、主語・関係・目的語は直接書き換えられます。表の一番下の行から関係を追加できます。")
    show = [c for c in res["cands"] if c["status"] != PC.STATUS_REJECT]
    df = pd.DataFrame([{
        "採用する": c["status"] == PC.STATUS_ACCEPT or (c["status"] == PC.STATUS_REVIEW and c["selected"]),
        "主語(S)": c["source"], "関係(A)": c["relation"], "目的語(O)": c["target"], "確率": c["prob"],
        "判定": c["status"], "抽出元": c["origin"]} for c in show],
        columns=["採用する", "主語(S)", "関係(A)", "目的語(O)", "確率", "判定", "抽出元"])
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True,
                            column_config=editor_config(), key=f"an_editor_{hash(res['text']) % 10**8}")
    confirmed = [r for r in PC.table_to_review(edited) if r["keep"]]

    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown("**確定予定のSAO構造**")
        sao_graph(confirmed)
    with g2:
        st.markdown("**構成要素（GiNZA）**")
        st.write("、".join(res["tags"]) or "―")
        b1, b2 = st.columns(2)
        if b1.button("✅ 確定して保存", type="primary"):
            if res.get("pick") in PATENTS:
                st.session_state.reviews[res["pick"]] = PC.table_to_review(edited)
                st.success(f"{res['pick']} の人手確認結果として保存しました（ダッシュボード・分析ページに反映）。")
            else:
                st.session_state.workspace.append({"id": f"解析{len(st.session_state.workspace) + 1}",
                                                   "text": res["text"], "relations": confirmed})
                st.success("ワークスペースに保存しました（エクスポートページから書き出せます）。")
        b2.download_button("⬇️ CSVで保存", pd.DataFrame(confirmed).to_csv(index=False).encode("utf-8-sig"),
                           file_name="sao_confirmed.csv", mime="text/csv")
    with st.expander("除外した候補も含めた全候補（確率順）"):
        st.dataframe(pd.DataFrame([{k: c[k] for k in ("status", "prob", "source", "relation", "target", "origin")}
                                   for c in res["cands"]]).rename(columns={
            "status": "判定", "prob": "確率", "source": "主語", "relation": "関係", "target": "目的語", "origin": "抽出元"}),
            hide_index=True, use_container_width=True)


# ===========================================================================
# ✍️ 人手確認
# ===========================================================================

def page_review():
    need_data()
    reviews = st.session_state.reviews
    st.title("✍️ 人手確認")
    st.caption("AIの判定を確認・修正して確定します。確定した内容は、ダッシュボード・ネットワーク・類似性マップなど"
               "すべての分析に反映されます（エクスポートページで保存・読み込みできます）。")
    f1, f2, f3 = st.columns([2, 2, 1])
    comp = f1.multiselect("出願人で絞り込む", sorted({p["company"] for p in DATA["patents"]}))
    order = f2.selectbox("並び順", ["要確認が多い順", "特許番号順", "採用が少ない順"])
    only_open = f3.checkbox("未確認のみ", value=True)
    ps = [p for p in DATA["patents"] if (not comp or p["company"] in comp) and (not only_open or p["id"] not in reviews)]
    key = {"要確認が多い順": lambda p: -sum(r["status"] == PC.STATUS_REVIEW for r in p["relations"]),
           "特許番号順": lambda p: p["id"],
           "採用が少ない順": lambda p: sum(r["status"] == PC.STATUS_ACCEPT for r in p["relations"])}[order]
    ps = sorted(ps, key=key)
    st.progress(len(reviews) / len(DATA["patents"]), text=f"確認済み {len(reviews)} / {len(DATA['patents'])} 件")
    if not ps:
        st.info("条件に合う未確認の特許はありません。")
        return
    ids = [p["id"] for p in ps]
    default = st.session_state.review_pid if st.session_state.review_pid in ids else ids[0]
    pid = st.selectbox("確認する特許", ids, index=ids.index(default), format_func=patent_label)
    st.session_state.review_pid = pid
    p = PATENTS[pid]
    counts = Counter(r["status"] for r in p["relations"])
    counts[PC.STATUS_REJECT] = p.get("n_rejected", 0)
    status_badges(counts)

    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"**{p['title']}**　{p['applicant']}　出願日 {p.get('filing_date') or '―'}")
        if p.get("url"):
            st.markdown(f"[公報を開く]({p['url']})")
        table = PC.review_table(p, reviews)
        words = set(table["主語(S)"]).union(table["目的語(O)"]) if len(table) else set()
        st.markdown(f"<div style='font-size:.92rem;line-height:1.8;border:1px solid rgba(148,163,184,.4);"
                    f"border-radius:8px;padding:10px;max-height:420px;overflow:auto'>{PC.highlight(p['text'], words)}</div>",
                    unsafe_allow_html=True)
    with right:
        edited = st.data_editor(table, num_rows="dynamic", use_container_width=True, hide_index=True,
                                column_config=editor_config(), key=f"rv_{DATA['meta']['uid']}_{pid}")
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

@st.cache_data(show_spinner=False, max_entries=4)
def world_html(_data, uid):
    template = OZ_WORLD_HTML_PATH.read_text(encoding="utf-8")
    if _data["meta"].get("sample"):
        return template  # サンプルは研究で作った配置（UMAP）をそのまま使う
    return PC.oz_world_html(_data, template)


def page_world():
    need_data()
    import plotly.express as px

    st.title("🌍 Patent World（オズの世界）")
    tab1, tab2 = st.tabs(["🌌 オズの世界（技術ランドスケープ）", "🔎 SAOの特徴で色分け"])
    with tab1:
        st.caption("各特許を「発明の名称＋FI」の近さで3次元空間に配置した技術ランドスケープ。"
                   "色は出願人、線は近い特許どうし。ドラッグで回転、スクロールでズーム。")
        if OZ_WORLD_HTML_PATH.exists():
            components.html(world_html(DATA, DATA["meta"]["uid"]), height=760, scrolling=False)
        else:
            st.error(f"{OZ_WORLD_HTML_PATH.name} が見つかりません。app.py と同じフォルダに置いてください。")
    with tab2:
        feats = PC.feature_table(DATA, st.session_state.reviews)
        pos = pd.DataFrame([{"特許番号": p["id"], "x": p.get("x"), "y": p.get("y"), "z": p.get("z"),
                             "出願人": p["group"]} for p in DATA["patents"]])
        df = feats.merge(pos, on="特許番号").dropna(subset=["x"])
        axes = ["構成要素数", "SAO関係数", "階層の深さ", "分岐の多さ", "関係の多様性", "機能・配置の記述", "数値限定"]
        color = st.selectbox("色分けに使う特徴", ["出願人"] + axes + ["請求項の文字数"])
        fig = px.scatter_3d(df, x="x", y="y", z="z", color=color,
                            color_discrete_map=COLORS if color == "出願人" else None,
                            hover_name="発明の名称", hover_data={"特許番号": True, "SAO関係数": True,
                                                              "x": False, "y": False, "z": False}, height=640)
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        plot(fig)
        pid = st.selectbox("詳しく見る特許", [p["id"] for p in DATA["patents"]], format_func=patent_label,
                           key="world_pid")
        show_patent_card(pid)


def show_patent_card(pid):
    p = PATENTS[pid]
    rels = PC.effective_relations(p, st.session_state.reviews)
    st.markdown(f"**{p['title']}**（{pid}）　{p['applicant']}　FI: {p['fi'] or '―'}")
    c1, c2 = st.columns([2, 3])
    with c1:
        words = {x for r in rels for x in (r["source"], r["target"])}
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

def page_network():
    need_data()
    import plotly.graph_objects as go

    st.title("🕸️ SAOネットワーク")
    st.caption("SAOを、番号や「前記」を除いた基本語（例：第１電極→電極）でまとめたネットワーク。"
               "丸の大きさ＝その構成要素が現れる特許の件数、線の太さ＝その関係が現れる特許の件数。"
               " **丸をクリックすると、その構成要素が出てくる請求項が下に表示されます。** ")
    small = len(DATA["patents"]) < 40
    c1, c2, c3, c4, c5 = st.columns(5)
    comp = c1.multiselect("出願人", sorted({p["company"] for p in DATA["patents"]}))
    top_n = c2.slider("構成要素の数", 10, 150, 50, step=10)
    min_p = c3.slider("構成要素の最低出現特許数", 1, 20, 1 if small else 3)
    min_e = c4.slider("線を引く最低特許数", 1, 20, 1 if small else 3, help="その関係が何件の特許に現れたら線を引くか")
    kind = c5.selectbox("関係の種類", ["すべて", "構成（有する・備える）", "機能・配置（それ以外）"])
    ids = [p["id"] for p in DATA["patents"] if not comp or p["company"] in comp]
    nodes, edges = PC.build_network(DATA, ids, st.session_state.reviews, min_patents=min_p, top_n=top_n)
    edges = [e for e in edges if e["count"] >= min_e]
    if kind != "すべて":
        want = kind.startswith("構成")
        edges = [e for e in edges if PC.is_has(e["relation"]) == want]
    linked = {x for e in edges for x in (e["source"], e["target"])}
    nodes = [n for n in nodes if n["id"] in linked] or nodes
    if not nodes:
        st.info("条件に合う構成要素がありません。スライダーの値を下げてください。")
        return
    pos = PC.layout_network(nodes, edges)
    fig = go.Figure()
    maxc = max([e["count"] for e in edges] or [1])
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
    maxn = max(n["count"] for n in nodes)
    label_min = sorted([m["count"] for m in nodes], reverse=True)[min(34, len(nodes) - 1)]
    fig.add_trace(go.Scatter(
        x=[pos[n["id"]][0] for n in nodes], y=[pos[n["id"]][1] for n in nodes], mode="markers+text",
        textposition="top center", textfont=dict(size=11),
        text=[n["id"] if (n["count"] >= label_min or n["id"] == sel) else "" for n in nodes],
        customdata=[[n["id"]] for n in nodes],
        hovertext=[f"{n['id']}<br>特許 {n['count']} 件／つながり {deg[n['id']]}<br>表記例：{'、'.join(n['surfaces'][:4])}"
                   for n in nodes], hoverinfo="text", showlegend=False,
        marker=dict(size=[10 + 30 * (n["count"] / maxn) ** 0.5 for n in nodes],
                    color=["#f59e0b" if n["id"] == sel else "#0ea5e9" for n in nodes],
                    line=dict(width=1, color="#0f172a"))))
    fig.update_layout(height=680, margin=dict(l=10, r=10, t=10, b=10), dragmode="pan",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), clickmode="event+select")
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="net_chart")
    st.caption("青い線＝構成（有する・備える）／橙の線＝機能・配置（接続される・配置される等）")
    # クリック直後の再実行では図を変えずに選択を受け取り、印を付け直すために再実行する
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
    hits = [h for h in PC.patents_with_node(DATA, term, st.session_state.reviews) if h["patent"]["id"] in ids]
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
        st.caption(f"ほか {len(hits) - 30} 件（出願人で絞り込むと減らせます）")


# ===========================================================================
# 🗺️ 類似性マップ
# ===========================================================================

def page_similarity():
    need_data()
    import plotly.express as px

    st.title("🗺️ 類似性マップ")
    st.caption("各特許のSAO（基本語にしたトリプル・組・構成要素）をTF-IDFで数値化し、似ているものほど近くに"
               "配置した地図。 **点をクリックすると、似ている特許と共通するSAOが表示されます。** "
               "オズの世界（発明の名称＋FI）とは違い、請求項の構造の近さで並べている。")
    if len(DATA["patents"]) < 2:
        st.info("2件以上の特許が必要です。")
        return
    S = corpus_similarity(DATA, data_key())
    ids = [p["id"] for p in DATA["patents"]]
    df = pd.DataFrame([{"特許番号": p["id"], "発明の名称": p["title"], "出願人": p["group"], "x": p.get("map_x", 0.0),
                        "y": p.get("map_y", 0.0), "SAO数": len(PC.effective_relations(p, st.session_state.reviews))}
                       for p in DATA["patents"]])
    color = st.radio("色分け", ["出願人", "SAO数"], horizontal=True)
    fig = px.scatter(df, x="x", y="y", color=color, color_discrete_map=COLORS if color == "出願人" else None,
                     hover_name="発明の名称", hover_data={"特許番号": True, "x": False, "y": False},
                     custom_data=["特許番号"], height=620)
    fig.update_traces(marker=dict(size=8))
    sel = st.session_state.map_patent
    if sel in ids:
        i = ids.index(sel)
        nbr = np.argsort(-S[i])[:5]
        fig.add_scatter(x=df.loc[nbr, "x"], y=df.loc[nbr, "y"], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=2, color="#f59e0b")))
        fig.add_scatter(x=[df.loc[i, "x"]], y=[df.loc[i, "y"]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=18, symbol="star", color="#f59e0b", line=dict(width=1, color="#000")))
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
    top_k = st.slider("表示する類似特許の数", 1, min(15, len(ids) - 1), min(5, len(ids) - 1))
    rows = [{"特許番号": ids[j], "発明の名称": PATENTS[ids[j]]["title"], "出願人": PATENTS[ids[j]]["company"],
             "類似度": round(float(S[i, j]), 3)} for j in np.argsort(-S[i])[:top_k]]
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
# 🧭 FIレーダー
# ===========================================================================

def page_radar():
    need_data()
    import plotly.graph_objects as go

    st.title("🧭 FIレーダー")
    st.caption("出願人（または出願年）ごとに、どのFI（技術分類）の特許をどれだけ持っているかを比べる。"
               "軸はデータ全体でよく使われているFI、値はそのグループの特許のうちそのFIが付いているものの割合。")
    if not any(p.get("fi") for p in DATA["patents"]):
        st.info("このデータセットにはFIの列がありません。FIを含む特許リストを読み込むと表示できます。")
        return
    c1, c2, c3, c4 = st.columns(4)
    by = c1.radio("比べるもの", ["出願人", "出願年"], horizontal=True)
    level_name = c2.selectbox("FIの細かさ", list(PC.FI_LEVELS))
    top_fi = c3.slider("軸にするFIの数", 3, 16, 8)
    share = c4.radio("値", ["割合（%）", "件数"], horizontal=True) == "割合（%）"
    level = PC.FI_LEVELS[level_name]
    if by == "出願人":
        cands = [c for c, _ in Counter(p["company"] for p in DATA["patents"]).most_common() if c != "不明"]
    else:
        cands = sorted({str(p["year"]) for p in DATA["patents"] if p.get("year")})
    default = cands[:4] if by == "出願人" else cands[-4:]
    groups = st.multiselect("比べる" + by + "（最大6）", cands, default=default, max_selections=6)
    if not groups:
        st.info(f"比べる{by}を選んでください。")
        return
    axes, vals, sizes = PC.fi_radar_data(DATA, by=by, level=level, groups=groups, top_fi=top_fi, share=share)
    if len(axes) < 3:
        st.info("レーダーチャートには3つ以上のFIが必要です。FIの細かさを変えるか、比べるグループを増やしてください。")
        return
    palette = PC.GROUP_PALETTE
    fig = go.Figure()
    for k, g in enumerate(groups):
        v = vals[g]
        color = COLORS.get(g) if by == "出願人" and g in COLORS and g != "その他" else palette[k % len(palette)]
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], fill="toself", opacity=0.55,
                                      name=f"{g}（{sizes[g]}件）", line=dict(color=color)))
    fig.update_layout(polar=dict(radialaxis=dict(ticksuffix="%" if share else "")), height=580,
                      margin=dict(l=60, r=60, t=30, b=30))
    plot(fig)
    st.markdown("**実際の値**")
    st.dataframe(pd.DataFrame({f"{g}（{sizes[g]}件）": vals[g] for g in groups}, index=axes),
                 use_container_width=True)
    st.caption("1件の特許に複数のFIが付いている場合は、それぞれに数える。")


# ===========================================================================
# 🫧 技術分布（バブル・ヒートマップ）
# ===========================================================================

def page_distribution():
    need_data()
    import plotly.express as px

    st.title("🫧 技術分布")
    tab1, tab2, tab3 = st.tabs(["🫧 出願年 × FI（バブル）", "🔥 出願人 × 技術（ヒートマップ）", "📈 出願の推移とランキング"])
    reviews = st.session_state.reviews
    with tab1:
        if not any(p.get("fi") for p in DATA["patents"]) or not any(p.get("year") for p in DATA["patents"]):
            st.info("出願日とFIの列があるデータで表示できます。")
        else:
            c1, c2, c3, c4 = st.columns(4)
            level_name = c1.selectbox("FIの細かさ", list(PC.FI_LEVELS)[:2], key="bub_level")
            top_fi = c2.slider("表示するFIの数", 5, 30, 12, key="bub_top")
            comps = c3.multiselect("出願人で絞り込む", sorted({p["company"] for p in DATA["patents"]}), key="bub_comp")
            color_by = c4.radio("色", ["平均SAO数", "出願人（最多）"], key="bub_color")
            level = PC.FI_LEVELS[level_name]
            rows = []
            for p in DATA["patents"]:
                if not p.get("year") or (comps and p["company"] not in comps):
                    continue
                n_sao = len(PC.effective_relations(p, reviews))
                for f in set(PC.fi_codes(p, level)):
                    rows.append({"出願年": p["year"], "FI": f, "SAO数": n_sao, "出願人": p["group"]})
            df = pd.DataFrame(rows)
            if df.empty:
                st.info("条件に合うデータがありません。")
            else:
                order = df.groupby("FI").size().sort_values(ascending=False).head(top_fi).index.tolist()
                df = df[df["FI"].isin(order)]
                agg = (df.groupby(["出願年", "FI"])
                       .agg(件数=("SAO数", "size"), 平均SAO数=("SAO数", "mean"),
                            出願人=("出願人", lambda x: x.value_counts().index[0])).reset_index())
                agg["平均SAO数"] = agg["平均SAO数"].round(1)
                kw = dict(color="平均SAO数", color_continuous_scale="Viridis") if color_by == "平均SAO数" else \
                    dict(color="出願人", color_discrete_map=COLORS)
                fig = px.scatter(agg, x="出願年", y="FI", size="件数", size_max=26, height=160 + 42 * len(order),
                                 category_orders={"FI": order}, hover_data={"件数": True, "平均SAO数": True,
                                                                             "出願人": True}, **kw)
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(dtick=1))
                plot(fig)
                st.caption("横軸＝出願年、縦軸＝FI（件数の多い順）、バブルの大きさ＝その年・そのFIの特許の件数。"
                           "色は、その特許の平均SAO数（請求項の構造の細かさ）か、いちばん多い出願人。"
                           "1件の特許に複数のFIがあれば、それぞれに数える。")
    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        axis = c1.selectbox("技術の軸", ["FIサブクラス", "FIメイングループ", "主要構成要素（SAO）"])
        top = c2.slider("表示する技術の数", 5, 30, 15)
        top_comp = c3.slider("表示する出願人の数", 3, 30, 12)
        norm = c4.checkbox("出願人ごとの割合で表示", value=False)
        m = PC.company_tech_matrix(DATA, axis=axis, reviews=reviews, top_tech=top, top_comp=top_comp)
        if m.empty:
            st.info("データがありません（FIの列がない場合は「主要構成要素（SAO）」を選んでください）。")
        else:
            z = m.div(m.sum(axis=1), axis=0).round(3) if norm else m
            fig = px.imshow(z, text_auto=".0%" if norm else True, aspect="auto", color_continuous_scale="YlOrRd",
                            height=180 + 40 * len(z))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=axis, yaxis_title="出願人")
            plot(fig)
            a, b = st.columns(2)
            comp = a.selectbox("出願人", m.index.tolist())
            tech = b.selectbox("技術", m.columns.tolist())
            hits = []
            for p in DATA["patents"]:
                if p["company"] != comp:
                    continue
                if axis == "FIサブクラス":
                    ok = tech in p["fi_sub"]
                elif axis == "FIメイングループ":
                    ok = tech in p["fi_main"]
                else:
                    ok = any(PC.base_term(x) == tech for r in PC.effective_relations(p, reviews)
                             for x in (r["source"], r["target"]))
                if ok:
                    hits.append({"特許番号": p["id"], "発明の名称": p["title"], "出願年": p["year"], "FI": p["fi"]})
            st.markdown(f"**{comp} × {tech}：{len(hits)} 件**")
            st.dataframe(pd.DataFrame(hits), hide_index=True, use_container_width=True)
    with tab3:
        top_n = st.slider("ランキングの表示件数", 5, 30, 15)
        years = pd.DataFrame([{"出願年": p["year"], "出願人": p["group"]} for p in DATA["patents"] if p.get("year")])
        if not years.empty:
            y = years.groupby(["出願年", "出願人"]).size().rename("件数").reset_index()
            fig = px.bar(y, x="出願年", y="件数", color="出願人", color_discrete_map=COLORS, title="年別出願件数",
                         height=380)
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis=dict(dtick=1))
            plot(fig)
        c1, c2 = st.columns(2)
        fi = pd.Series([(PC.fi_codes(p, 0) or [None])[0] for p in DATA["patents"]]).dropna().value_counts().head(top_n)
        if len(fi):
            c1.plotly_chart(px.bar(x=fi.values[::-1], y=fi.index[::-1], orientation="h", title="筆頭FIサブクラス",
                                   labels={"x": "件数", "y": "FI"}), use_container_width=True)
        ap = pd.Series([p["company"] for p in DATA["patents"]]).value_counts().head(top_n)
        c2.plotly_chart(px.bar(x=ap.values[::-1], y=ap.index[::-1], orientation="h", title="筆頭出願人",
                               labels={"x": "件数", "y": "出願人"}), use_container_width=True)


# ===========================================================================
# 🐚 2つの請求項を比較
# ===========================================================================

def page_compare():
    st.title("🐚 2つの請求項を比較")
    st.caption("請求項A・Bを解析し、Jaccard類似度・構造の類似度・クレームの広さ狭さ・意味マッチングで比較します。")
    col_a, col_b = st.columns(2)
    opts = ["（貼り付ける）"] + ([p["id"] for p in DATA["patents"]] if DATA else [])
    fmt = lambda x: x if x == "（貼り付ける）" else patent_label(x)  # noqa: E731
    with col_a:
        pa = st.selectbox("請求項A（データセットから選ぶか貼り付け）", opts, format_func=fmt, key="cmp_a")
        text_a = st.text_area("請求項A", value=PATENTS[pa]["text"] if pa in PATENTS else "", height=220, key=f"text_a_{pa}")
    with col_b:
        pb = st.selectbox("請求項B（データセットから選ぶか貼り付け）", opts, format_func=fmt, key="cmp_b")
        text_b = st.text_area("請求項B", value=PATENTS[pb]["text"] if pb in PATENTS else "", height=220, key=f"text_b_{pb}")
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
               "展開してから解析します。")
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
# 📤 エクスポート
# ===========================================================================

def page_export():
    need_data()
    st.title("📤 エクスポート")
    reviews = st.session_state.reviews
    st.markdown("#### 分析結果の書き出し")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**解析済みデータ（JSON）**")
        st.caption("このデータセットの解析結果一式。「データの読み込み」で開くと、次回はLLMを呼ばずにすぐ分析できる。")
        st.download_button("⬇️ 解析済みデータを保存", json.dumps(DATA, ensure_ascii=False).encode("utf-8"),
                           file_name=f"{DATA['meta'].get('name', 'dataset')}_sao.json", mime="application/json")
    with c2:
        st.markdown("**Excel（分析結果一式）**")
        st.caption("概要／特許一覧／SAO（AI判定）／SAO（人手確認済み）／構造的特徴／類似特許")
        if st.button("Excelファイルを作成"):
            with st.spinner("作成中…"):
                st.session_state["_xlsx"] = PC.export_excel(DATA, reviews, corpus_similarity(DATA, data_key()))
        if st.session_state.get("_xlsx"):
            st.download_button("⬇️ Excelをダウンロード", st.session_state["_xlsx"], file_name="patent_sao_analysis.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.markdown("**CSV（現在のSAO）**")
        st.caption("人手確認済みの特許は確認結果、それ以外はAIが選んだSAO")
        st.download_button("⬇️ CSVをダウンロード", PC.relations_csv(DATA, reviews), file_name="patent_sao.csv",
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
            hit = {k: v for k, v in loaded.items() if k in PATENTS}
            st.session_state.reviews.update(hit)
            st.success(f"{len(hit)} 件の確認結果を読み込みました（このデータセットに無い {len(loaded) - len(hit)} 件は除外）。")
        except Exception as e:  # noqa: BLE001
            st.error(f"読み込みに失敗しました: {e}")
    if st.session_state.workspace:
        st.divider()
        st.markdown(f"#### AI解析のワークスペース（{len(st.session_state.workspace)}件）")
        rows = [{"解析ID": w["id"], "主語(S)": r["source"], "関係(A)": r["relation"], "目的語(O)": r["target"]}
                for w in st.session_state.workspace for r in w["relations"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.download_button("⬇️ ワークスペースをCSVで保存", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                           file_name="sao_workspace.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# ナビゲーション
# ---------------------------------------------------------------------------
nav = st.navigation({
    "データ": [st.Page(page_data, title="データの読み込み", icon="📥", url_path="data"),
             st.Page(page_dashboard, title="ダッシュボード", icon="🏠", default=True)],
    "抽出と確認": [st.Page(page_analyze, title="AI解析", icon="🧪", url_path="analyze"),
                 st.Page(page_review, title="人手確認", icon="✍️", url_path="review")],
    "可視化・分析": [st.Page(page_world, title="Patent World", icon="🌍", url_path="world"),
                   st.Page(page_network, title="SAOネットワーク", icon="🕸️", url_path="network"),
                   st.Page(page_similarity, title="類似性マップ", icon="🗺️", url_path="similarity"),
                   st.Page(page_radar, title="FIレーダー", icon="🧭", url_path="radar"),
                   st.Page(page_distribution, title="技術分布", icon="🫧", url_path="distribution")],
    "個別ツール": [st.Page(page_compare, title="2つの請求項を比較", icon="🐚", url_path="compare"),
                st.Page(page_dependent, title="従属請求項を展開", icon="🪼", url_path="dependent")],
    "出力": [st.Page(page_export, title="エクスポート", icon="📤", url_path="export")],
}, expanded=True)
nav.run()
