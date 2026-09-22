# -*- coding: utf-8 -*-
"""
特許請求項SAO構造分析 統合デモ（卒業研究）
==========================================

「AIを用いた日本語特許文献の構造分析に関する研究
 ―SAO構造を用いた半導体関連特許の類似性分析―」

本アプリは、本研究で開発したLLM+GiNZAハイブリッドSAO抽出パイプライン
（translate_sao.py の analyze_claim_llm_direct、実験4相当の最終採用設定）
を中心に、請求項比較・従属請求項展開・複数請求項検索・特許統計分析・
精度検証・技術ランドスケープ可視化（オズの世界）をまとめた統合デモです。

【抽出方式について（重要）】
以前は、この統合アプリの原型（work/app.py）はGiNZA単体（LLM不使用）の
抽出方式（patent_pipeline.pyのanalyze_claim_ginza_only等）を使っていた。
しかし本研究の卒論最終稿では、実験1〜7を経てLLMハイブリッド方式
（translate_sao.py の analyze_claim_llm_direct、実験4の設定：検証機構＋
target語フィルタ）を正式な最終手法として採用している。そこで本アプリの
「SAO構造抽出デモ」以外の各ページ（請求項比較・従属請求項展開・まとめて
検索・精度検証）も、抽出処理はすべて同じ実験4相当のLLMハイブリッド方式
（下記 llm_extract()）に統一してある。これにより、どのページで見ても
卒論の最終採用手法と矛盾しない一貫した結果になる。

【バックエンドについて】
本研究の実験（実験1〜7）はすべて、ローカル環境で動作するOllama上の
qwen2.5:7bモデルを用いて行っている。これがこの研究の正式な実験系である。

一方、本アプリをStreamlit Community Cloud等、公開のWebサーバー上で
「誰でも使えるデモ」として動かす場合、そのサーバーからローカルのOllama
（localhost:11434）に到達することはできない。そこで本アプリは、
Streamlitの Secrets に OPENROUTER_API_KEY が設定されている場合のみ、
translate_sao.py の低レベルLLM呼び出し関数 `_ollama_chat` を、OpenRouter
経由で同じ qwen2.5-7b-instruct モデルを呼び出す関数にモンキーパッチする
（下記 `_install_cloud_backend_if_configured` を参照）。translate_sao.py
自体は一切変更しない。ローカルでOllamaを使って実行する場合（Secretsが
無い場合）は、これまで通りOllamaに接続する。

【実行方法：自分のPCでOllamaを使って動かす場合】
Ollama（qwen2.5:7bモデル）が動作しているPC上で、本ファイルを
translate_sao.py・patent_pipeline.py・en_relation_rules.py・
oz_world_embed.html と同じフォルダに置いて、以下を実行してください。

    pip install -r requirements.txt
    streamlit run app.py

（事前に `ollama serve` でOllamaを起動し、`ollama pull qwen2.5:7b` で
モデルを取得しておく必要があります。GiNZAパイプラインの読み込みに
数十秒かかることがありますが、これは初回のみです。「まとめて検索」
「2つの請求項を比較」の意味マッチングは、初回利用時に埋め込みモデル
（sentence-transformers、数百MB）のダウンロードが走ります。）

本アプリはサイドバーの「ページ」切り替えで、以下の7ページを持つ。
  1. 🔬 SAO構造抽出デモ … 1件の請求項をその場で抽出・可視化
  2. 🐚 2つの請求項を比較 … Jaccard・構造・広さ狭さ・意味マッチングで比較
  3. 🪼 従属請求項を展開 … 引用関係を展開してから解析
  4. 🔦 まとめて検索 … 複数の請求項をDB化し、類似請求項を検索
  5. 📊 特許統計分析 … CSVから出願件数・FI・出願人統計、バブル/レーダー図
  6. ✅ 精度検証 … 大量の請求項に対する自動ヘルスチェック
  7. 🌌 オズの世界 … 532件の技術ランドスケープ可視化（Three.js埋め込み）

【実行方法：Streamlit Community Cloudで公開する場合】
1. GitHubリポジトリに、本ファイル・translate_sao.py・patent_pipeline.py・
   en_relation_rules.py・oz_world_embed.html・requirements.txt を
   すべて一緒にコミットする（どれか1つでも欠けると import エラーに
   なるか、該当ページが表示されません）。
2. https://openrouter.ai で無料アカウントを作成し、APIキーを発行する。
3. Streamlit CloudアプリのSettings→Secretsに、以下の形式で追加する。

       OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxx"

   （このキーはSecretsにのみ記入し、GitHubには絶対にコミットしないこと。）
4. デプロイすると、本アプリは自動的にOpenRouter経由のクラウド版
   qwen2.5-7b-instructモデルを使用する。

【無料枠での注意】
「2つの請求項を比較」の意味マッチングと「まとめて検索」は、
sentence-transformers（多言語埋め込みモデル、数百MB）を読み込む。
Streamlit Community Cloudの無料枠はメモリに制限があるため、
GiNZA・LLM呼び出しと合わせて動かすとメモリ不足になる可能性がある。
その場合は「2つの請求項を比較」の意味マッチングのチェックボックスを
オフのまま使うか、ローカル実行を推奨する。
"""

import csv
import html
import io
import json
import os
import re
import time
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

import patent_pipeline as pp
import translate_sao as ts

st.set_page_config(page_title="特許請求項SAO構造分析", layout="wide", page_icon="🔬")

APP_DIR = Path(__file__).resolve().parent

CLOUD_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OZ_WORLD_HTML_PATH = APP_DIR / "oz_world_embed.html"

PAGE_SAO_DEMO = "🔬 SAO構造抽出デモ"
PAGE_COMPARE = "🐚 2つの請求項を比較"
PAGE_DEPENDENT = "🪼 従属請求項を展開"
PAGE_SEARCH = "🔦 まとめて検索"
PAGE_STATS = "📊 特許統計分析"
PAGE_HEALTH = "✅ 精度検証"
PAGE_OZ_WORLD = "🌌 オズの世界（特許技術ランドスケープ）"
PAGE_OPTIONS = [
    PAGE_SAO_DEMO, PAGE_COMPARE, PAGE_DEPENDENT, PAGE_SEARCH,
    PAGE_STATS, PAGE_HEALTH, PAGE_OZ_WORLD,
]

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

# セッション状態の初期化（新しく追加したページで使う）
for _key in (
    "compare_result", "patent_db", "search_results", "dependent_result",
    "stats_df", "eval_results", "eval_summary",
):
    if _key not in st.session_state:
        st.session_state[_key] = None

# ---------------------------------------------------------------------------
# これまでの実験の到達点（532件・Gold正解データ・MICRO平均・緩い評価基準）
# サイドバーに表示し、このデモが研究全体の「総結集」であることを示す。
# ---------------------------------------------------------------------------
EXPERIMENT_HISTORY = [
    {"実験": "実験1", "内容": "LLM直接抽出＋GiNZA補完（検証なし・ベースライン）",
     "Precision": "72.55%", "Recall": "87.05%", "F1": "79.14%"},
    {"実験": "実験2", "内容": "ginza_has_fallback|有する を条件付き検証（閾値16）",
     "Precision": "73.65%", "Recall": "86.99%", "F1": "79.77%"},
    {"実験": "実験3", "内容": "検証をclaim_title_ginza|有する等にも拡張",
     "Precision": "74.37%", "Recall": "86.88%", "F1": "80.14%"},
    {"実験": "実験4", "内容": "Goldに一件も出現しないtarget語を無条件フィルタ（最終採用）",
     "Precision": "75.25%", "Recall": "86.88%", "F1": "80.65%"},
]

TYPE_LABELS = {
    "llm_direct": "LLM直接抽出",
    "claim_title_ginza": "GiNZA補完（題名由来）",
    "claim_title_ginza_conflict": "GiNZA補完（題名由来・LLMと矛盾）",
    "ginza_has_fallback": "GiNZA補完（有する系）",
    "ginza_has_fallback_conflict": "GiNZA補完（有する系・LLMと矛盾）",
    "attribute": "出自関係（の）",
    "translate": "英訳経由抽出",
}
TYPE_COLORS = {
    "llm_direct": "#2563eb",
    "claim_title_ginza": "#16a34a",
    "claim_title_ginza_conflict": "#ca8a04",
    "ginza_has_fallback": "#16a34a",
    "ginza_has_fallback_conflict": "#ca8a04",
    "attribute": "#7c3aed",
    "translate": "#64748b",
}


# ---------------------------------------------------------------------------
# クラウドLLMバックエンド（公開デプロイ用）
#
# translate_sao.py はローカルのOllamaにしか対応していないが、その内部の
# LLM呼び出しはすべて _ollama_chat(system_prompt, user_text, model, host)
# という1つのモジュールグローバル関数を経由している（メイン抽出も、
# 検証機構の追加LLM呼び出しも同じ関数を通る）。そこで、Secretsに
# OpenRouterのAPIキーが設定されている場合だけ、起動時にこの関数を
# OpenRouter経由の実装に差し替える（モンキーパッチ）。translate_sao.py
# 自体には一切手を加えない。
# ---------------------------------------------------------------------------

def _get_openrouter_api_key():
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("OPENROUTER_API_KEY")
    return key or None


def _make_cloud_chat(api_key):
    """translate_sao._ollama_chat と同じシグネチャ・後処理を持つ、
    OpenRouter経由の置き換え関数を作る。"""
    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def _cloud_chat(system_prompt, user_text, model=CLOUD_MODEL, host=None):
        # hostはローカルOllama用の引数なのでクラウド版では無視する。
        # modelがローカル用のデフォルト（qwen2.5:7b等、Ollamaのタグ形式）
        # のままの場合は、OpenRouter向けのモデルIDに読み替える。
        use_model = model if (model and "/" in model) else CLOUD_MODEL
        response = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            max_tokens=ts._OLLAMA_MAX_OUTPUT_TOKENS,
        )
        content = response.choices[0].message.content or ""
        # _ollama_chatと同じ後処理（<think>ブロック・(note:...)注釈の除去）
        content = ts._THINK_BLOCK_RE.sub("", content).strip()
        content = ts._NOTE_BLOCK_RE.sub("", content).strip()
        return content

    return _cloud_chat


def _install_cloud_backend_if_configured():
    """OPENROUTER_API_KEYが設定されていれば、ts._ollama_chatをクラウド版に
    差し替える。戻り値はバックエンドの種類（"cloud" / "local"）。"""
    api_key = _get_openrouter_api_key()
    if not api_key:
        return "local"
    try:
        ts._ollama_chat = _make_cloud_chat(api_key)
        return "cloud"
    except Exception:
        st.warning(
            "OpenRouter用クライアントの初期化に失敗したため、ローカルOllama"
            "バックエンドにフォールバックします。requirements.txtに openai "
            "パッケージが含まれているか確認してください。"
        )
        st.code(traceback.format_exc())
        return "local"


BACKEND = _install_cloud_backend_if_configured()


@st.cache_resource(show_spinner="GiNZAパイプラインを読み込み中…（初回のみ、数十秒かかります）")
def load_pipeline():
    return ts._load_pipeline(str(APP_DIR))


def build_extraction_kwargs(preset, risk_threshold, claim_title_risk_threshold,
                             verify_risky_ginza, verify_extra_risky, filter_invalid_targets):
    """サイドバーの設定から analyze_claim_llm_direct の引数を組み立てる。"""
    if preset == "実験4相当（推奨・現行ベスト確定版）":
        return dict(
            verify_risky_ginza=True,
            risk_threshold=ts._DEFAULT_RISK_THRESHOLD,
            extra_risk_rules=ts._build_extra_risk_rules(ts._DEFAULT_CLAIM_TITLE_RISK_THRESHOLD),
            filter_invalid_targets=True,
        )
    if preset == "実験1相当（ベースライン・検証なし）":
        return dict(
            verify_risky_ginza=False,
            risk_threshold=ts._DEFAULT_RISK_THRESHOLD,
            extra_risk_rules=None,
            filter_invalid_targets=False,
        )
    # カスタム
    extra_risk_rules = None
    if verify_extra_risky:
        extra_risk_rules = ts._build_extra_risk_rules(claim_title_risk_threshold)
    return dict(
        verify_risky_ginza=verify_risky_ginza,
        risk_threshold=risk_threshold,
        extra_risk_rules=extra_risk_rules,
        filter_invalid_targets=filter_invalid_targets,
    )


# 実験4相当（最終採用設定）の固定kwargs。SAO抽出デモ以外の全ページは、
# プリセット選択UIを出さず、常にこの設定で抽出する（卒論の最終手法との
# 一貫性を保つため）。
_EXP4_KWARGS = build_extraction_kwargs(
    "実験4相当（推奨・現行ベスト確定版）", None, None, None, None, None)


def llm_extract(text):
    """SAO抽出デモ以外の各ページ（比較・従属請求項展開・まとめて検索・
    精度検証）が共通で使う抽出関数。実験4相当（最終採用設定）で
    analyze_claim_llm_direct を呼び出す。戻り値: (components, relations)。
    """
    return ts.analyze_claim_llm_direct(
        text,
        pp=load_pipeline(),
        model=model_name,
        host=(ollama_host.strip() or None) if ollama_host else None,
        **_EXP4_KWARGS,
    )


def relations_to_dot(relations):
    """SAO関係リストをGraphviz DOT形式に変換する（st.graphviz_chart用）。"""
    lines = [
        "digraph SAO {",
        'rankdir="LR";',
        'node [shape=box, style="rounded,filled", fillcolor="#f1f5f9", '
        'fontname="Noto Sans JP,Yu Gothic,sans-serif", color="#94a3b8"];',
        'edge [fontname="Noto Sans JP,Yu Gothic,sans-serif", fontsize=11];',
    ]
    nodes = set()
    for r in relations:
        nodes.add(r["source"])
        nodes.add(r["target"])
    for n in nodes:
        label = html.escape(n).replace('"', '\\"')
        lines.append(f'"{n}" [label="{label}"];')
    for r in relations:
        color = TYPE_COLORS.get(r["type"], "#334155")
        rel_label = html.escape(r["relation"])
        lines.append(
            f'"{r["source"]}" -> "{r["target"]}" '
            f'[label="{rel_label}", color="{color}", fontcolor="{color}"];'
        )
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 「まとめて検索」ページ用：build_patent_database / search_similar_claims の
# LLMハイブリッド版。patent_pipeline.py の同名関数は内部でanalyze_claim
# （GiNZA＋LLM補完の旧マージ方式）を呼ぶため、ここではllm_extract
# （実験4相当）を呼ぶ薄いラッパーとして再実装する（patent_pipeline.py
# 自体は変更しない）。
# ---------------------------------------------------------------------------

def build_patent_database_llm(records, progress_callback=None):
    import numpy as np

    model = pp._get_embed_model()
    database = []
    total = len(records)
    for i, (rid, text) in enumerate(records):
        try:
            _, relations = llm_extract(text)
        except Exception:
            relations = []
        if progress_callback:
            progress_callback(i + 1, total)
        if not relations:
            continue
        triples = sorted(pp.relations_to_triple_set(relations, normalize_numbers=True))
        texts = [pp._triple_to_text(t) for t in triples]
        embeddings = model.encode(texts, normalize_embeddings=True)
        doc_embedding = np.mean(embeddings, axis=0)
        doc_embedding = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-8)
        database.append({
            "id": rid, "text": text, "relations": relations,
            "doc_embedding": doc_embedding,
        })
    return database


def search_similar_claims_llm(query_text, database, top_k=10, rerank_k=5):
    import numpy as np

    _, query_relations = llm_extract(query_text)
    if not query_relations:
        return []

    model = pp._get_embed_model()
    triples = sorted(pp.relations_to_triple_set(query_relations, normalize_numbers=True))
    texts = [pp._triple_to_text(t) for t in triples]
    embeddings = model.encode(texts, normalize_embeddings=True)
    query_embedding = np.mean(embeddings, axis=0)
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

    scored = []
    for entry in database:
        fast_score = float(np.dot(query_embedding, entry["doc_embedding"]))
        scored.append({"id": entry["id"], "text": entry["text"],
                        "relations": entry["relations"], "fast_score": fast_score})
    scored.sort(key=lambda x: -x["fast_score"])
    top_candidates = scored[:top_k]

    for entry in top_candidates[:rerank_k]:
        precise_score, matches = pp.semantic_similarity(query_relations, entry["relations"])
        entry["precise_score"] = precise_score
        entry["matches"] = matches

    reranked = [e for e in top_candidates if "precise_score" in e]
    not_reranked = [e for e in top_candidates if "precise_score" not in e]
    reranked.sort(key=lambda x: -x["precise_score"])
    return reranked + not_reranked


# ---------------------------------------------------------------------------
# 「特許統計分析」ページ用ヘルパー（CSVの出願日・FI・出願人列を解析する。
# SAO抽出とは独立した書誌情報の集計なので、抽出方式には依存しない）
# ---------------------------------------------------------------------------

def _find_column(df, aliases):
    """CSVの列名が多少違っても自動認識する。"""
    normalized = {
        str(c).strip().lower().replace(" ", "").replace("　", ""): c
        for c in df.columns
    }
    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "").replace("　", "")
        if key in normalized:
            return normalized[key]
    return None


def _split_multi_value(value):
    """出願人/権利者・FIなどの複数値を汎用的に分割する。"""
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    s = re.sub(r"[；;、,\n\r]+", "|", s)
    return [x.strip() for x in s.split("|") if x.strip()]


def _extract_year(value):
    """日付、YYYY、YYYY-MM-DD、YYYY/MM/DD等から年を抽出。"""
    if pd.isna(value):
        return None
    m = re.search(r"(19|20)\d{2}", str(value).strip())
    return int(m.group(0)) if m else None


def _extract_fi_subclass(value):
    """
    FI/IPCからサブクラスまでを抽出する。
    例: H01L 21/00 -> H01L
        G06F 3/01  -> G06F
        H10B 12/00 -> H10B
    """
    subclasses = []
    for fi in _split_multi_value(value):
        fi = str(fi).strip().upper()
        m = re.match(r"^([A-HY][0-9]{2}[A-Z])", fi)
        if m:
            subclasses.append(m.group(1))
        elif fi:
            subclasses.append(fi)
    return list(dict.fromkeys(subclasses))


def _prepare_stats_df(df):
    """統計分析用の列を自動認識する。"""
    df = df.copy()

    date_col = _find_column(df, [
        "出願日", "出願年月日", "出願日付",
        "application_date", "filing_date", "filingdate", "date"
    ])
    fi_col = _find_column(df, ["FI", "FI分類", "FIコード", "fi_code", "fi"])
    applicant_col = _find_column(df, [
        "出願人/権利者", "出願人／権利者",
        "出願人", "出願人名", "出願人名称",
        "applicant", "applicants", "applicant_name"
    ])

    if date_col is None:
        raise ValueError("「出願日」列が見つかりません。")
    if fi_col is None:
        raise ValueError("「FI」列が見つかりません。")
    if applicant_col is None:
        raise ValueError("「出願人/権利者」列が見つかりません。")

    work = pd.DataFrame({
        "出願年": df[date_col].apply(_extract_year),
        "FI原文": df[fi_col],
        "出願人/権利者原文": df[applicant_col],
    })

    work["筆頭FI"] = work["FI原文"].apply(
        lambda x: _split_multi_value(x)[0] if _split_multi_value(x) else None
    )
    work["筆頭FIサブクラス"] = work["FI原文"].apply(
        lambda x: _extract_fi_subclass(x)[0] if _extract_fi_subclass(x) else None
    )
    work["出願人/権利者"] = work["出願人/権利者原文"].apply(
        lambda x: _split_multi_value(x)[0] if _split_multi_value(x) else None
    )

    work = work.dropna(subset=["出願年"]).copy()
    work["出願年"] = work["出願年"].astype(int)
    return work, date_col, fi_col, applicant_col


def _make_applicant_fi_table(work):
    """出願人/権利者ごとに、FIサブクラスを集計する。"""
    rows = []
    for _, row in work.iterrows():
        applicants = _split_multi_value(row["出願人/権利者原文"])
        subclasses = _extract_fi_subclass(row["FI原文"])
        if not applicants or not subclasses:
            continue
        for applicant in applicants:
            for subclass in subclasses:
                rows.append({"出願人/権利者": applicant, "FIサブクラス": subclass})

    if not rows:
        return pd.DataFrame(columns=["出願人/権利者", "FIサブクラス", "件数"])

    tmp = pd.DataFrame(rows)
    return (
        tmp.groupby(["出願人/権利者", "FIサブクラス"])
        .size()
        .reset_index(name="件数")
        .sort_values(["出願人/権利者", "件数", "FIサブクラス"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def _build_bubble_radar_database(work):
    """バブルチャート・レーダーチャート用に、work（_prepare_stats_dfの戻り値）
    を patent_pipeline.plot_applicant_fi_bubble / build_applicant_fi_radar_data
    が期待する [{"出願人": [...], "FI": [...]}, ...] 形式に変換する。"""
    database = []
    for _, row in work.iterrows():
        applicants = _split_multi_value(row["出願人/権利者原文"])
        fis = _split_multi_value(row["FI原文"])
        if not applicants or not fis:
            continue
        database.append({"出願人": applicants, "FI": fis})
    return database


def _show_patent_statistics(df):
    """4種類の特許統計＋バブルチャート・レーダーチャートを表示する。"""
    try:
        work, date_col, fi_col, applicant_col = _prepare_stats_df(df)
    except Exception as e:
        st.error(str(e))
        return

    if work.empty:
        st.warning("出願年を読み取れるデータがありません。")
        return

    st.success(
        f"✅ {len(work):,} 件を分析しました。"
        f"（出願日: {date_col} / FI: {fi_col} / 出願人/権利者: {applicant_col}）"
    )

    top_n = st.number_input(
        "ランキング表示件数", min_value=5, max_value=100, value=20, step=5,
        key="stats_top_n",
    )

    # ① 年別出願件数
    st.markdown("### 📈 ① 年別出願件数推移")
    yearly = work.groupby("出願年").size().rename("出願件数").sort_index()
    st.line_chart(yearly, x_label="出願年", y_label="出願件数")
    st.dataframe(yearly.reset_index(), use_container_width=True, hide_index=True)

    # ② 筆頭FIサブクラスランキング
    st.markdown("### 🏆 ② 筆頭FIサブクラスランキング")
    first_fi = (
        work.dropna(subset=["筆頭FIサブクラス"])
        .groupby("筆頭FIサブクラス").size().reset_index(name="出願件数")
        .sort_values(["出願件数", "筆頭FIサブクラス"], ascending=[False, True])
        .reset_index(drop=True)
    )
    first_fi.insert(0, "順位", range(1, len(first_fi) + 1))
    st.bar_chart(
        first_fi.head(int(top_n)).set_index("筆頭FIサブクラス")["出願件数"],
        horizontal=True, x_label="出願件数", y_label="FIサブクラス",
    )
    st.dataframe(first_fi.head(int(top_n)), use_container_width=True, hide_index=True)

    # ③ 筆頭出願人/権利者ランキング
    st.markdown("### 🏢 ③ 筆頭出願人/権利者ランキング")
    first_applicant = (
        work.dropna(subset=["出願人/権利者"])
        .groupby("出願人/権利者").size().reset_index(name="出願件数")
        .sort_values(["出願件数", "出願人/権利者"], ascending=[False, True])
        .reset_index(drop=True)
    )
    first_applicant.insert(0, "順位", range(1, len(first_applicant) + 1))
    st.bar_chart(
        first_applicant.head(int(top_n)).set_index("出願人/権利者")["出願件数"],
        horizontal=True, x_label="出願件数", y_label="出願人/権利者",
    )
    st.dataframe(first_applicant.head(int(top_n)), use_container_width=True, hide_index=True)

    # ④ 出願人/権利者別出願FIランキング
    st.markdown("### 🧩 ④ 出願人/権利者別出願FIサブクラスランキング")
    applicant_fi = _make_applicant_fi_table(work)
    if applicant_fi.empty:
        st.info("出願人/権利者別FIを集計できるデータがありません。")
    else:
        applicant_choices = sorted(applicant_fi["出願人/権利者"].unique())
        selected_applicant = st.selectbox(
            "詳しく見る出願人/権利者", applicant_choices, key="stats_selected_applicant")
        selected = applicant_fi[applicant_fi["出願人/権利者"] == selected_applicant].copy()
        selected.insert(0, "順位", range(1, len(selected) + 1))
        st.bar_chart(
            selected.head(int(top_n)).set_index("FIサブクラス")["件数"],
            horizontal=True, x_label="出願件数", y_label="FIサブクラス",
        )
        st.dataframe(selected.head(int(top_n)), use_container_width=True, hide_index=True)

    st.caption(
        "※ 筆頭FI・筆頭出願人/権利者は、CSVのセルに複数値がある場合、"
        "先頭に記載されたものを筆頭として集計します。"
        "出願人/権利者別出願FIは、同一出願に複数の出願人/権利者・FIがある場合、"
        "それぞれの組合せを1件として集計します。FIはサブクラス（例：H01L、G06F）"
        "単位で集計します。"
    )

    # ⑤⑥ バブルチャート・レーダーチャート（出願人×FIの組み合わせ傾向）
    st.divider()
    st.markdown("### 🫧 ⑤ 出願人×FI バブルチャート")
    st.caption("出願人とFIサブクラスの組み合わせごとの件数を、バブルの大きさで表す。")
    database = _build_bubble_radar_database(work)
    if not database:
        st.info("出願人・FIの組み合わせデータが見つかりません。")
        return

    bubble_col1, bubble_col2 = st.columns(2)
    with bubble_col1:
        bubble_top_applicants = st.slider("表示する出願人数", 3, 20, 10, key="bubble_top_applicants")
    with bubble_col2:
        bubble_top_fi = st.slider("表示するFI数", 3, 20, 10, key="bubble_top_fi")

    try:
        fig = pp.plot_applicant_fi_bubble(
            database, top_applicants=bubble_top_applicants, top_fi=bubble_top_fi)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"バブルチャートを作成できませんでした: {e}")

    st.markdown("### 🕸️ ⑥ 出願人別 FIレーダーチャート")
    st.caption("上位の出願人ごとに、よく使うFIサブクラスの分布を比較する。")
    radar_col1, radar_col2 = st.columns(2)
    with radar_col1:
        radar_top_applicants = st.slider("比較する出願人数", 2, 8, 5, key="radar_top_applicants")
    with radar_col2:
        radar_top_fi = st.slider("軸にするFI数", 3, 10, 6, key="radar_top_fi")

    try:
        profiles = pp.build_applicant_fi_radar_data(
            database, top_applicants=radar_top_applicants, top_fi=radar_top_fi)
        if profiles:
            fig = pp.plot_radar_chart(profiles, title="出願人別 FI分布レーダーチャート")
            st.pyplot(fig)
        else:
            st.info("レーダーチャートを作成するのに十分なデータがありません。")
    except Exception as e:
        st.warning(f"レーダーチャートを作成できませんでした: {e}")


# ---------------------------------------------------------------------------
# サイドバー（全ページ共通部分＋SAO抽出デモ専用の設定）
# ---------------------------------------------------------------------------
with st.sidebar:
    page = st.radio("ページ", PAGE_OPTIONS, index=0)
    st.divider()

    if BACKEND == "cloud":
        st.success(
            "🌐 公開デモモード：OpenRouter経由でクラウド版qwen2.5-7b-instruct"
            "を使用しています。",
            icon="🌐",
        )
        st.caption(
            "無料枠のレート制限（1日50回／1分20回程度）があるため、"
            "多人数が同時に試すと一時的に失敗することがあります。"
        )
    else:
        st.info(
            "💻 ローカルモード：このPC上のOllama（localhost）に接続します。",
            icon="💻",
        )

    st.divider()
    if BACKEND == "cloud":
        model_name = st.text_input("モデル名（OpenRouter）", value=CLOUD_MODEL)
        ollama_host = ""
    else:
        model_name = st.text_input("Ollamaモデル名", value=ts.DEFAULT_MODEL)
        ollama_host = st.text_input(
            "Ollamaホスト（空欄でデフォルト = http://localhost:11434）", value="")

    if page == PAGE_SAO_DEMO:
        st.divider()
        st.header("研究の到達点（実験1〜4）")
        st.caption("532件の半導体関連特許請求項・手動作成Gold正解データに対する、"
                   "MICRO平均・緩い評価基準でのスコア。")
        st.table(EXPERIMENT_HISTORY)

        st.divider()
        st.header("抽出設定")
        preset = st.radio(
            "設定プリセット",
            ["実験4相当（推奨・現行ベスト確定版）", "実験1相当（ベースライン・検証なし）", "カスタム"],
            index=0,
            help="実験1〜4はどれも同じベースのハイブリッド抽出（LLM直接抽出＋GiNZA補完）に、"
                 "検証機構や後処理フィルタを段階的に追加したもの。プリセットを切り替えて"
                 "同じ請求項を再抽出すると、各工夫がどこに効いているかを直接比較できます。",
        )

        verify_risky_ginza = False
        verify_extra_risky = False
        risk_threshold = ts._DEFAULT_RISK_THRESHOLD
        claim_title_risk_threshold = ts._DEFAULT_CLAIM_TITLE_RISK_THRESHOLD
        filter_invalid_targets = False

        if preset == "カスタム":
            verify_risky_ginza = st.checkbox(
                "実験2：ginza_has_fallback|有する を検証", value=True)
            risk_threshold = st.slider(
                "　└ 検証閾値（この件数以上で検証発火）", 1, 30, ts._DEFAULT_RISK_THRESHOLD)
            verify_extra_risky = st.checkbox(
                "実験3：claim_title_ginza|有する 等にも検証を拡張", value=True)
            claim_title_risk_threshold = st.slider(
                "　└ claim_title_ginza|有する の検証閾値", 1, 30,
                ts._DEFAULT_CLAIM_TITLE_RISK_THRESHOLD)
            filter_invalid_targets = st.checkbox(
                "実験4：Gold不出現target語を無条件フィルタ", value=True)
    else:
        # SAO抽出デモ以外のページでは、抽出設定は常に実験4相当固定。
        preset = "実験4相当（推奨・現行ベスト確定版）"
        verify_risky_ginza = True
        verify_extra_risky = True
        risk_threshold = ts._DEFAULT_RISK_THRESHOLD
        claim_title_risk_threshold = ts._DEFAULT_CLAIM_TITLE_RISK_THRESHOLD
        filter_invalid_targets = True
        if page != PAGE_OZ_WORLD:
            st.caption(
                "このページの抽出方式は、卒論の最終採用手法である"
                "実験4相当（検証機構＋target語フィルタ）に固定されています。"
            )


# ===========================================================================
# ① SAO構造抽出デモ
# ===========================================================================
if page == PAGE_SAO_DEMO:
    st.title("🔬 特許請求項 SAO構造抽出デモ")
    st.write(
        "特許の請求項テキストを貼り付けて実行すると、LLM+GiNZAハイブリッド"
        "パイプラインがSAO(主語-動作-目的語)構造をその場で抽出します。"
    )

    claim_text = st.text_area("特許請求項テキスト", value=SAMPLE_CLAIM, height=220)
    run = st.button("SAO構造を抽出する", type="primary")

    if run:
        if not claim_text.strip():
            st.warning("請求項テキストを入力してください。")
            st.stop()

        try:
            gpp = load_pipeline()
        except Exception:
            st.error(
                "GiNZAパイプラインの読み込みに失敗しました。GiNZA/spaCyが正しく"
                "インストールされているフォルダで実行しているか確認してください。"
            )
            st.code(traceback.format_exc())
            st.stop()

        kwargs = build_extraction_kwargs(
            preset, risk_threshold, claim_title_risk_threshold,
            verify_risky_ginza, verify_extra_risky, filter_invalid_targets,
        )

        spinner_text = (
            "クラウドLLM(OpenRouter)で抽出中…（検証機構が発火する場合、追加で"
            "LLM呼び出しが走ります）" if BACKEND == "cloud" else
            "Ollamaで抽出中…（検証機構が発火する場合、追加でLLM呼び出しが走ります）"
        )
        with st.spinner(spinner_text):
            t0 = time.time()
            try:
                components, relations = ts.analyze_claim_llm_direct(
                    claim_text, pp=gpp, model=model_name,
                    host=(ollama_host.strip() or None), **kwargs,
                )
            except Exception as exc:
                if BACKEND == "cloud":
                    st.error(
                        "抽出中にエラーが発生しました。OpenRouterの無料枠のレート"
                        "制限（1日50回／1分20回程度）に達した可能性があります。"
                        "しばらく待って再試行してください。"
                    )
                else:
                    st.error(
                        "抽出中にエラーが発生しました。Ollamaが起動しているか "
                        "（`ollama serve`）、指定したモデルが取得済みか "
                        f"（`ollama pull {model_name}`）を確認してください。"
                    )
                st.exception(exc)
                st.stop()
            elapsed = time.time() - t0

        st.success(f"抽出完了（{elapsed:.1f}秒） / 構成要素 {len(components)}件 "
                   f"/ SAO関係 {len(relations)}件")

        tab_graph, tab_table, tab_components, tab_json = st.tabs(
            ["SAO構造グラフ", "関係一覧", "検出された構成要素", "生データ(JSON)"])

        with tab_graph:
            if relations:
                st.graphviz_chart(relations_to_dot(relations), use_container_width=True)
                legend_items = sorted({r["type"] for r in relations})
                st.caption(
                    "色の凡例： " + " / ".join(
                        f":{'green' if TYPE_COLORS[t]=='#16a34a' else 'blue' if TYPE_COLORS[t]=='#2563eb' else 'orange' if TYPE_COLORS[t]=='#ca8a04' else 'violet'}[{TYPE_LABELS.get(t, t)}]"
                        for t in legend_items
                    )
                )
            else:
                st.info("SAO関係が抽出されませんでした。")

        with tab_table:
            if relations:
                table_rows = [
                    {
                        "主語(Subject)": r["source"],
                        "動作/関係(Action)": r["relation"],
                        "目的語(Object)": r["target"],
                        "抽出元": TYPE_LABELS.get(r["type"], r["type"]),
                    }
                    for r in relations
                ]
                st.dataframe(table_rows, use_container_width=True, hide_index=True)
            else:
                st.info("SAO関係が抽出されませんでした。")

        with tab_components:
            if components:
                st.write([c["text"] for c in components])
            else:
                st.info("構成要素が検出されませんでした。")

        with tab_json:
            result_json = json.dumps(
                {"components": components, "relations": relations},
                ensure_ascii=False, indent=2,
            )
            st.code(result_json, language="json")
            st.download_button(
                "JSONとしてダウンロード", data=result_json,
                file_name="sao_result.json", mime="application/json",
            )


# ===========================================================================
# ② 2つの請求項を比較
# ===========================================================================
elif page == PAGE_COMPARE:
    st.title("🐚 2つの請求項を比較")
    st.caption(
        "実験4相当のLLMハイブリッド方式で請求項A・Bを解析し、"
        "Jaccard類似度・構造の類似度・クレームの広さ狭さ・意味マッチングで比較します。"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area("請求項A", height=220, key="text_a")
    with col_b:
        text_b = st.text_area("請求項B", height=220, key="text_b")

    use_semantic = st.checkbox(
        "②意味マッチングも使う（初回はモデルの読み込みに1分程度かかります）",
        value=False,
    )

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
                    narrowness_a, breadth_a, scope_detail_a = pp.compute_claim_scope_score(relations_a)
                    narrowness_b, breadth_b, scope_detail_b = pp.compute_claim_scope_score(relations_b)

                    semantic_score, semantic_matches = None, None
                    if use_semantic:
                        with st.spinner("意味マッチングのモデルを読み込み中..."):
                            try:
                                semantic_score, semantic_matches = pp.semantic_similarity(relations_a, relations_b)
                            except Exception as e:
                                st.error(f"意味マッチングでエラーが発生しました: {e}")

                    st.session_state.compare_result = {
                        "jaccard_score": jaccard_score, "common": common,
                        "only_a": only_a, "only_b": only_b,
                        "structural_score": structural_score, "structural_detail": structural_detail,
                        "semantic_score": semantic_score, "semantic_matches": semantic_matches,
                        "scope_a": (narrowness_a, breadth_a, scope_detail_a),
                        "scope_b": (narrowness_b, breadth_b, scope_detail_b),
                    }
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {e}")
                    st.session_state.compare_result = None

    if st.session_state.compare_result is not None:
        res = st.session_state.compare_result

        st.markdown("### 🦪 診断結果")
        score_cols = st.columns(3)
        score_cols[0].metric("①Jaccard類似度（表記の一致）", f"{res['jaccard_score']:.3f}")
        if res["semantic_score"] is not None:
            score_cols[1].metric("②意味マッチング類似度", f"{res['semantic_score']:.3f}")
        else:
            score_cols[1].metric("②意味マッチング類似度", "―（未使用）")
        score_cols[2].metric("③構造の類似度", f"{res['structural_score']:.3f}")

        with st.expander("③構造比較の内訳を見る"):
            detail = res["structural_detail"]
            st.write(
                f"- 深さの類似度: {detail['深さの類似度']:.3f}\n"
                f"- 規模(ノード数)の類似度: {detail['規模(ノード数)の類似度']:.3f}\n"
                f"- 枝分かれパターンの類似度: {detail['枝分かれパターンの類似度']:.3f}\n"
                f"- 関係の種類の内訳の類似度: {detail['関係の種類の内訳の類似度']:.3f}"
            )

        st.markdown("### 🐙 クレームの広さ・狭さの比較")
        narrowness_a, breadth_a, scope_detail_a = res["scope_a"]
        narrowness_b, breadth_b, scope_detail_b = res["scope_b"]
        scope_col_a, scope_col_b = st.columns(2)
        with scope_col_a:
            st.markdown("**請求項A**")
            st.metric("広さスコア", f"{breadth_a:.3f}")
            st.caption(
                f"構成要素数: {scope_detail_a['構成要素数']} / "
                f"数値スペック: {scope_detail_a['数値スペックの数']} / "
                f"階層の深さ: {scope_detail_a['階層の深さ']}"
            )
        with scope_col_b:
            st.markdown("**請求項B**")
            st.metric("広さスコア", f"{breadth_b:.3f}")
            st.caption(
                f"構成要素数: {scope_detail_b['構成要素数']} / "
                f"数値スペック: {scope_detail_b['数値スペックの数']} / "
                f"階層の深さ: {scope_detail_b['階層の深さ']}"
            )
        st.caption("※ このスコアは絶対的な尺度ではなく、AとBを相対的に比べるための指標です。")

        st.markdown("### 🧩 ①Jaccard：トリプルの一致・不一致")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**共通トリプル（{len(res['common'])}件）**")
            for t in sorted(res["common"]):
                st.write(t)
        with col2:
            st.markdown(f"**Aだけにあるトリプル（{len(res['only_a'])}件）**")
            for t in sorted(res["only_a"]):
                st.write(t)
        with col3:
            st.markdown(f"**Bだけにあるトリプル（{len(res['only_b'])}件）**")
            for t in sorted(res["only_b"]):
                st.write(t)

        if res["semantic_matches"] is not None:
            st.markdown("### 🫧 ②意味マッチング：対応付けの詳細")
            matches_sorted = sorted(res["semantic_matches"], key=lambda x: -x[2])
            st.dataframe(
                [
                    {
                        "類似度": round(sim, 2),
                        "判定": "完全一致" if ta == tb else ("意味が近い" if sim >= 0.6 else "対応薄い"),
                        "トリプルA": " / ".join(ta), "トリプルB": " / ".join(tb),
                    }
                    for ta, tb, sim in matches_sorted
                ],
                use_container_width=True, hide_index=True,
            )


# ===========================================================================
# ③ 従属請求項を展開
# ===========================================================================
elif page == PAGE_DEPENDENT:
    st.title("🪼 従属請求項を展開")
    st.caption(
        "実際の公報の書き方（【請求項１】【請求項２】…）のまま貼り付けると、"
        "「請求項１に記載の」等の引用関係を自動展開してから、実験4相当の"
        "LLMハイブリッド方式で解析します。"
    )

    claims_text = st.text_area(
        "請求項群", height=280,
        placeholder="【請求項１】\n（請求項1の全文）\n【請求項２】\n（請求項2の全文。「請求項１に記載の」を含む）",
        key="claims_text",
    )

    parsed_claims = pp.parse_claims_block(claims_text) if claims_text.strip() else {}

    if claims_text.strip() and not parsed_claims:
        st.warning("請求項を認識できませんでした。テキストを確認してください。")
    elif parsed_claims:
        st.success(f"✅ 請求項 {sorted(parsed_claims.keys())} を認識しました。")
        with st.expander("認識結果を確認する"):
            st.dataframe(
                [{"番号": n, "本文（先頭60文字）": b[:60] + "..."} for n, b in sorted(parsed_claims.items())],
                use_container_width=True, hide_index=True,
            )

    st.divider()
    st.subheader("展開したい請求項を選ぶ")
    if parsed_claims:
        target_num = st.selectbox("展開する請求項番号", sorted(parsed_claims.keys()), key="target_claim_num")
    else:
        target_num = None
        st.info("先に請求項群を入力してください。")

    if st.button("🔍 展開して解析する", key="dependent_run", disabled=not parsed_claims):
        try:
            full_text = pp.resolve_dependent_claim(target_num, parsed_claims)
            components, relations = llm_extract(full_text)
            st.session_state.dependent_result = {"full_text": full_text, "relations": relations}
        except Exception as e:
            st.error(f"展開・解析中にエラーが発生しました: {e}")
            st.session_state.dependent_result = None

    if st.session_state.dependent_result is not None:
        res = st.session_state.dependent_result
        st.markdown("#### 📖 展開後の完全な請求項テキスト")
        st.info(res["full_text"])

        relations = res["relations"]
        if not relations:
            st.info("関係が抽出できませんでした。")
        else:
            st.success(f"🎉 {len(relations)} 件の関係を抽出しました！")
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("#### 🪸 構成要素間の関係図")
                st.graphviz_chart(relations_to_dot(relations), use_container_width=True)

                st.markdown("#### 🐙 クレームの広さ・狭さ")
                narrowness, breadth, scope_detail = pp.compute_claim_scope_score(relations)
                scope_cols = st.columns(2)
                scope_cols[0].metric("広さスコア", f"{breadth:.3f}")
                scope_cols[1].metric("狭さスコア", f"{narrowness:.3f}")
            with col2:
                st.markdown("#### 📋 抽出された関係（SAOトリプル）")
                st.dataframe(
                    [
                        {"主語": r["source"], "関係": r["relation"], "目的語": r["target"],
                         "種類": TYPE_LABELS.get(r["type"], r["type"])}
                        for r in relations
                    ],
                    use_container_width=True, hide_index=True,
                )


# ===========================================================================
# ④ まとめて検索
# ===========================================================================
elif page == PAGE_SEARCH:
    st.title("🔦 まとめて検索")
    st.caption(
        "複数の請求項をデータベース化しておき、調べたい請求項に似ているものを"
        "検索します（実験4相当のLLMハイブリッド方式で解析→埋め込みベクトルで検索）。"
    )

    st.subheader("🗄️ ステップ１：比較対象の請求項をまとめて登録する")
    st.caption(
        "CSVファイル（列名: id, text）をアップロードするか、"
        "下のテキストエリアに「-----」で区切って複数の請求項を貼り付けてください。"
    )

    uploaded_csv = st.file_uploader("CSVファイル（id, text の2列）", type=["csv"])
    bulk_text = st.text_area(
        "またはここに、請求項を「-----」で区切って貼り付ける", height=180,
        placeholder="1件目の請求項テキスト...\n-----\n2件目の請求項テキスト...\n-----\n3件目の請求項テキスト...",
        key="bulk_text",
    )

    if st.button("📚 データベースを構築する", key="build_db_run"):
        records = []
        if uploaded_csv is not None:
            content = uploaded_csv.getvalue().decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                rid = row.get("id") or row.get("番号") or f"行{len(records)+1}"
                rtext = row.get("text") or row.get("本文") or ""
                if rtext.strip():
                    records.append((rid, rtext.strip()))
        elif bulk_text.strip():
            parts = [p.strip() for p in bulk_text.split("-----") if p.strip()]
            records = [(f"請求項{i+1}", p) for i, p in enumerate(parts)]

        if not records:
            st.warning("CSVのアップロード、またはテキストの貼り付けのどちらかを行ってください。")
        else:
            progress_bar = st.progress(0, text=f"0/{len(records)}件")

            def _on_db_progress(done, total):
                progress_bar.progress(done / total if total else 0, text=f"{done}/{total}件")

            with st.spinner(
                f"{len(records)} 件を解析してデータベースを構築中…"
                "（初回は埋め込みモデルの読み込みに1分程度かかります）"
            ):
                try:
                    db = build_patent_database_llm(records, progress_callback=_on_db_progress)
                    st.session_state.patent_db = db
                    st.session_state.search_results = None
                except Exception as e:
                    st.error(f"データベース構築中にエラーが発生しました: {e}")
                    st.session_state.patent_db = None
            progress_bar.empty()

    if st.session_state.patent_db is not None:
        st.success(f"✅ {len(st.session_state.patent_db)} 件を登録済みです。")
        with st.expander("登録済みの一覧を見る"):
            st.dataframe(
                [{"id": e["id"], "本文（先頭50文字）": e["text"][:50] + "..."} for e in st.session_state.patent_db],
                use_container_width=True, hide_index=True,
            )

    st.divider()
    st.subheader("🔦 ステップ２：調べたい請求項を検索する")
    query_text = st.text_area("検索したい請求項テキスト", height=160, key="query_text")
    col_topk, col_rerank = st.columns(2)
    with col_topk:
        top_k = st.slider("粗い絞り込みで残す件数", min_value=3, max_value=30, value=10)
    with col_rerank:
        rerank_k = st.slider("精密な再評価をする件数（上位から）", min_value=1, max_value=10, value=5)

    if st.button("🔍 検索する", key="search_run"):
        if st.session_state.patent_db is None:
            st.warning("先にステップ１でデータベースを構築してください。")
        elif not query_text.strip():
            st.warning("検索したい請求項テキストを入力してください。")
        else:
            with st.spinner("検索中..."):
                try:
                    results = search_similar_claims_llm(
                        query_text, st.session_state.patent_db, top_k=top_k, rerank_k=rerank_k)
                    st.session_state.search_results = results
                except Exception as e:
                    st.error(f"検索中にエラーが発生しました: {e}")
                    st.session_state.search_results = None

    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        st.markdown(f"#### 🏆 検索結果（上位{len(results)}件）")
        for i, r in enumerate(results):
            has_precise = "precise_score" in r
            score_label = f"精密スコア {r['precise_score']:.3f}" if has_precise else f"粗いスコア {r['fast_score']:.3f}"
            with st.expander(f"{i+1}位　【{r['id']}】　{score_label}"):
                st.write(r["text"])
                st.caption(
                    f"粗いスコア: {r['fast_score']:.3f}"
                    + (f" ／ 精密スコア: {r['precise_score']:.3f}" if has_precise else "")
                )
                if has_precise:
                    matches_sorted = sorted(r["matches"], key=lambda x: -x[2])
                    st.dataframe(
                        [
                            {
                                "類似度": round(sim, 2),
                                "判定": "完全一致" if ta == tb else ("意味が近い" if sim >= 0.6 else "対応薄い"),
                                "クエリ側": " / ".join(ta), "この請求項側": " / ".join(tb),
                            }
                            for ta, tb, sim in matches_sorted
                        ],
                        use_container_width=True, hide_index=True,
                    )


# ===========================================================================
# ⑤ 特許統計分析
# ===========================================================================
elif page == PAGE_STATS:
    st.title("📊 特許統計分析")
    st.caption(
        "CSVをアップロードするか、CSV本文を貼り付けるだけで、年別出願件数・"
        "筆頭FI・筆頭出願人・出願人別FI・出願人×FIバブルチャート・"
        "出願人別FIレーダーチャートを表示します（SAO抽出とは独立した書誌情報の集計です）。"
    )

    st.markdown(
        """
        **推奨CSV列**
        - `出願日`：例 `2024-03-15`
        - `FI`：例 `H01L 21/00; H01L 29/00`
        - `出願人/権利者`：例 `株式会社A;株式会社B`

        英語列 `application_date / fi / applicant` も利用できます。
        既存のCSVに `id` や `text` など他の列があっても問題ありません。
        """
    )

    stats_uploaded_csv = st.file_uploader("📁 統計分析用CSVをアップロード", type=["csv"], key="stats_csv_upload")
    stats_csv_text = st.text_area(
        "またはCSV本文をここに貼り付け", height=180,
        placeholder=(
            "出願日,FI,出願人/権利者\n"
            "2022-04-01,H01L 21/00,株式会社A\n"
            "2023-06-12,H01L 29/00;H01L 21/00,株式会社B;株式会社C\n"
            "2024-01-20,H10B 12/00,株式会社A"
        ),
        key="stats_csv_text",
    )

    if st.button("📊 統計を分析する", type="primary", key="stats_run"):
        try:
            if stats_uploaded_csv is not None:
                content = stats_uploaded_csv.getvalue().decode("utf-8-sig")
            elif stats_csv_text.strip():
                content = stats_csv_text
            else:
                st.warning("CSVをアップロードするか、CSV本文を貼り付けてください。")
                content = None
            if content:
                st.session_state.stats_df = pd.read_csv(io.StringIO(content))
        except Exception as e:
            st.error(f"CSVの読み込みに失敗しました: {e}")
            st.session_state.stats_df = None

    if st.session_state.stats_df is not None:
        with st.expander("読み込んだCSVを確認する"):
            st.dataframe(st.session_state.stats_df, use_container_width=True, hide_index=True)
        _show_patent_statistics(st.session_state.stats_df)


# ===========================================================================
# ⑥ 精度検証（自動ヘルスチェック）
# ===========================================================================
elif page == PAGE_HEALTH:
    st.title("✅ 精度検証（自動ヘルスチェック）")
    st.caption(
        "人手のSAO正解データがない前提で、既知の不具合パターン"
        "（関係が1件も取れない・自己ループ・意味のない語がノードになる・"
        "部品が孤立する 等）を自動チェックし、その合格率を精度の代理指標として使います。"
        "抽出は実験4相当のLLMハイブリッド方式で行うため、実行にはLLM呼び出しが"
        "件数分発生します（時間がかかります）。"
    )

    eval_uploaded = st.file_uploader("📁 請求項リスト（.xlsx / .csv）をアップロード", type=["xlsx", "csv"], key="eval_file_upload")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        eval_text_col = st.text_input("請求項本文の列名", value="請求項本文", key="eval_text_col")
    with col_e2:
        eval_id_col = st.text_input("ID列名（無ければ空欄でOK）", value="id", key="eval_id_col")

    eval_limit = st.number_input(
        "検証する件数の上限（0で全件）", min_value=0, value=20, step=10, key="eval_limit",
        help="LLM呼び出しを伴うため、まず10〜30件程度で試すのがおすすめです。",
    )

    if st.button("🚀 精度検証を実行する", type="primary", key="eval_run"):
        if eval_uploaded is None:
            st.warning("ファイルをアップロードしてください。")
        else:
            try:
                if eval_uploaded.name.lower().endswith(".xlsx"):
                    eval_df = pd.read_excel(eval_uploaded)
                else:
                    eval_df = pd.read_csv(eval_uploaded)

                if eval_text_col not in eval_df.columns:
                    st.error(f"列「{eval_text_col}」が見つかりません。列一覧: {list(eval_df.columns)}")
                else:
                    texts = eval_df[eval_text_col].fillna("").astype(str).tolist()
                    if eval_id_col and eval_id_col in eval_df.columns:
                        ids = eval_df[eval_id_col].astype(str).tolist()
                    else:
                        ids = list(range(len(texts)))

                    records = [(i, t) for i, t in zip(ids, texts) if t.strip()]
                    if eval_limit and eval_limit > 0:
                        records = records[: int(eval_limit)]

                    progress_bar = st.progress(0, text=f"0/{len(records)}件")

                    def _on_eval_progress(done, total):
                        progress_bar.progress(done / total if total else 0, text=f"{done}/{total}件")

                    with st.spinner("解析＆ヘルスチェック中..."):
                        results, summary = pp.evaluate_corpus_health(
                            records, analyze_fn=llm_extract, progress_callback=_on_eval_progress)
                    progress_bar.empty()

                    st.session_state.eval_results = results
                    st.session_state.eval_summary = summary
            except Exception as e:
                st.error(f"検証中にエラーが発生しました: {e}")

    if st.session_state.eval_summary is not None:
        summary = st.session_state.eval_summary
        pass_rate_pct = summary["pass_rate"] * 100

        st.markdown("### 📈 結果サマリー")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("合格率", f"{pass_rate_pct:.1f}%")
        m2.metric("検証件数", f"{summary['total']}件")
        m3.metric("平均構成要素数", f"{summary['avg_components']:.1f}")
        m4.metric("平均関係数", f"{summary['avg_relations']:.1f}")

        if pass_rate_pct >= 90:
            st.success(f"🎉 目標の90%以上を達成しています（{pass_rate_pct:.1f}%）。")
        else:
            st.warning(
                f"⚠️ 目標の90%にはまだ届いていません（{pass_rate_pct:.1f}%）。"
                "下の「不合格の請求項」を確認して、どのチェック項目で"
                "落ちているものが多いか見てみてください。"
            )

        results_df = pd.DataFrame(st.session_state.eval_results)
        check_cols = [c for c in results_df.columns if c.startswith("check_")]
        if check_cols:
            st.markdown("### 🔍 チェック項目別の合格率")
            check_summary = (results_df[check_cols].mean().sort_values() * 100).round(1)
            st.dataframe(
                check_summary.rename("合格率(%)").reset_index().rename(columns={"index": "チェック項目"}),
                use_container_width=True, hide_index=True,
            )

        st.markdown("### 📋 詳細結果")
        show_only_failed = st.checkbox("不合格のものだけ表示", value=True, key="eval_show_failed")
        display_df = results_df[~results_df["passed"]] if show_only_failed else results_df
        cols = [c for c in ["id", "text", "passed", "n_components", "n_relations", "error"] + check_cols
                if c in display_df.columns]
        st.dataframe(
            display_df[cols] if len(display_df) > 0 else display_df,
            use_container_width=True, hide_index=True,
        )


# ===========================================================================
# ⑦ オズの世界（特許技術ランドスケープ）
# ===========================================================================
else:
    # 元々は独立したArtifact（https://claude.ai/artifact/4vpFUo1cqLbjT75ezJpYbZ）
    # として公開していたもの。532件の特許を「発明の名称＋FI」でベクトル化し、
    # SVD→UMAPで3次元に落とした技術ランドスケープをThree.jsで可視化する。
    # Three.js本体を含めて完全に自己完結した1つのHTMLファイル
    # （oz_world_embed.html）なので、そのままst.iframeで埋め込むだけで、
    # 追加のライブラリやネットワーク接続なしに動作する。
    st.title("🌌 オズの世界")
    st.caption(
        "特許532件を「発明の名称＋FI」でベクトル化し、UMAPで3次元空間に配置した"
        "技術ランドスケープ。ドラッグで回転、スクロールでズームできます。"
    )
    if not OZ_WORLD_HTML_PATH.exists():
        st.error(
            f"{OZ_WORLD_HTML_PATH.name} が見つかりません。app.pyと同じフォルダに"
            "配置してください。"
        )
    else:
        st.iframe(OZ_WORLD_HTML_PATH, height=800)
