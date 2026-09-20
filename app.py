# -*- coding: utf-8 -*-
"""
特許請求項SAO構造抽出 ライブデモ（卒業研究）
==========================================

「AIを用いた日本語特許文献の構造分析に関する研究
 ―SAO構造を用いた半導体関連特許の類似性分析―」

本アプリは、本研究で開発したLLM(Ollama)+GiNZAハイブリッドSAO抽出パイプライン
（translate_sao.py の analyze_claim_llm_direct）をそのまま呼び出し、
任意の特許請求項テキストを貼り付けるとSAO(Subject-Action-Object)構造を
その場で抽出・可視化するライブデモです。

【実行方法】
Ollama（qwen2.5:7bモデル）が動作しているPC上で、本ファイルを
translate_sao.py・patent_pipeline.py・en_relation_rules.py と
同じフォルダに置いて、以下を実行してください。

    pip install streamlit
    streamlit run app.py

（事前に `ollama serve` でOllamaを起動し、`ollama pull qwen2.5:7b` で
モデルを取得しておく必要があります。GiNZAパイプラインの読み込みに
数十秒かかることがありますが、これは初回のみです。）
"""

import html
import json
import time
import traceback
from pathlib import Path

import streamlit as st

import translate_sao as ts

st.set_page_config(page_title="特許請求項SAO構造抽出デモ", layout="wide", page_icon="🔬")

APP_DIR = Path(__file__).resolve().parent

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
    {"実験": "実験4", "内容": "Goldに一件も出現しないtarget語を無条件フィルタ",
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
# サイドバー
# ---------------------------------------------------------------------------
with st.sidebar:
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

    st.divider()
    model_name = st.text_input("Ollamaモデル名", value=ts.DEFAULT_MODEL)
    ollama_host = st.text_input(
        "Ollamaホスト（空欄でデフォルト = http://localhost:11434）", value="")

# ---------------------------------------------------------------------------
# メイン画面
# ---------------------------------------------------------------------------
st.title("🔬 特許請求項 SAO構造抽出デモ")
st.write(
    "特許の請求項テキストを貼り付けて実行すると、LLM(Ollama)+GiNZAハイブリッド"
    "パイプラインがSAO(主語-動作-目的語)構造をその場で抽出します。"
)

claim_text = st.text_area("特許請求項テキスト", value=SAMPLE_CLAIM, height=220)

run = st.button("SAO構造を抽出する", type="primary")

if run:
    if not claim_text.strip():
        st.warning("請求項テキストを入力してください。")
        st.stop()

    try:
        pp = load_pipeline()
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

    with st.spinner("Ollamaで抽出中…（検証機構が発火する場合、追加でLLM呼び出しが走ります）"):
        t0 = time.time()
        try:
            components, relations = ts.analyze_claim_llm_direct(
                claim_text,
                pp=pp,
                model=model_name,
                host=(ollama_host.strip() or None),
                **kwargs,
            )
        except Exception as exc:
            elapsed = time.time() - t0
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
