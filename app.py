import base64
import matplotlib.pyplot as plt
import pandas as pd
import io
import re
from collections import Counter
import streamlit as st

# Matplotlibの日本語文字化け対策
try:
    import japanize_matplotlib
except ImportError:
    pass

import patent_pipeline as pp

# ページ基本設定
st.set_page_config(page_title="日本語特許請求項SAO構造分析", page_icon="🪼", layout="wide")

# セッション状態の初期化
if "single_result" not in st.session_state:
    st.session_state.single_result = None
if "compare_result" not in st.session_state:
    st.session_state.compare_result = None
if "patent_db" not in st.session_state:
    st.session_state.patent_db = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "dependent_result" not in st.session_state:
    st.session_state.dependent_result = None
if "stats_df" not in st.session_state:
    st.session_state.stats_df = None


# --- 背景画像をBase64に変換してCSSに埋め込む関数 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# 背景CSSの設定
try:
    bin_str = get_base64_of_bin_file("deep_sea.jpg")
    bg_style = f"""
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 20, 40, 0.25), rgba(0, 10, 20, 0.45)), url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}
    """
except Exception:
    bg_style = """
    .stApp {
        background: linear-gradient(180deg, #08222f 0%, #071b28 14%, #051520 32%, #040f18 52%, #02090f 74%, #00050a 100%);
        background-attachment: fixed;
    }
    """

st.markdown(
    f"""
    <style>
    {bg_style}

    /* 文字色設定 */
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stCaption {{
        color: #d3e8f0;
    }}
    h1, h2, h3, h4 {{
        color: #aee0ee !important;
        text-shadow: 0 0 16px rgba(120, 200, 220, 0.35);
        font-weight: 800 !important;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: #86b3c4 !important;
    }}

    /* タブデザイン */
    button[data-baseweb="tab"] {{
        border-radius: 999px !important;
        padding: 0.4rem 1.2rem !important;
        margin-right: 0.4rem !important;
        background-color: rgba(10, 30, 40, 0.5) !important;
        border: 1px solid rgba(140, 200, 220, 0.2) !important;
        color: #bcdce8 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(80, 190, 210, 0.22), rgba(40, 120, 150, 0.22)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(120, 210, 230, 0.55) !important;
        font-weight: 700 !important;
    }}

    /* ボタンデザイン */
    .stButton > button {{
        border-radius: 999px !important;
        background: linear-gradient(135deg, #2fd7c4 0%, #1a8fb0 100%) !important;
        color: #021a24 !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 0.5rem 1.6rem !important;
    }}
    .stButton > button:hover {{
        filter: brightness(1.1);
    }}

    /* テキストエリア */
    .stTextArea textarea {{
        border-radius: 14px !important;
        border: 1.5px solid rgba(140, 200, 220, 0.25) !important;
        background: rgba(4, 18, 26, 0.65) !important;
        color: #e4f3f8 !important;
    }}
    .stTextArea textarea::placeholder {{
        color: #6b95a6 !important;
    }}

    /* Metricカード */
    div[data-testid="stMetric"] {{
        background: rgba(10, 30, 40, 0.45);
        border-radius: 16px;
        padding: 0.9rem;
        border: 1px solid rgba(140, 200, 220, 0.2);
    }}

    /* Matplotlib 画像表示コンテナ */
    div[data-testid="stImage"] {{
        background: rgba(248, 250, 250, 0.97);
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(140, 200, 220, 0.3);
    }}
    div[data-testid="stImage"] img {{
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 特許統計分析用ヘルパー
# ============================================================

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
    """出願人・FIなどの複数値を汎用的に分割する。"""
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    # 全角/半角の区切り文字、改行を統一
    s = re.sub(r"[；;、,\n\r]+", "|", s)
    parts = [x.strip() for x in s.split("|") if x.strip()]
    return parts


def _extract_year(value):
    """日付、YYYY、YYYY-MM-DD、YYYY/MM/DD等から年を抽出。"""
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None


def _prepare_stats_df(df):
    """
    統計分析に使う列を自動認識する。
    推奨列:
      出願日, FI, 出願人
    英語列にも対応:
      application_date/date, fi, applicant
    """
    df = df.copy()

    date_col = _find_column(df, [
        "出願日", "出願年月日", "出願日付", "application_date",
        "filing_date", "filingdate", "date"
    ])
    fi_col = _find_column(df, [
        "FI", "筆頭FI", "FI分類", "FIコード", "fi_code", "fi"
    ])
    applicant_col = _find_column(df, [
        "出願人", "出願人名", "出願人名称", "applicant",
        "applicants", "applicant_name"
    ])

    if date_col is None:
        raise ValueError("「出願日」列が見つかりません。例：出願日 / application_date")
    if fi_col is None:
        raise ValueError("「FI」列が見つかりません。例：FI / fi")
    if applicant_col is None:
        raise ValueError("「出願人」列が見つかりません。例：出願人 / applicant")

    work = pd.DataFrame({
        "出願年": df[date_col].apply(_extract_year),
        "FI原文": df[fi_col],
        "出願人原文": df[applicant_col],
    })

    work["筆頭FI"] = work["FI原文"].apply(
        lambda x: _split_multi_value(x)[0] if _split_multi_value(x) else None
    )
    work["筆頭出願人"] = work["出願人原文"].apply(
        lambda x: _split_multi_value(x)[0] if _split_multi_value(x) else None
    )

    work = work.dropna(subset=["出願年"])
    work["出願年"] = work["出願年"].astype(int)
    return work, date_col, fi_col, applicant_col


def _make_applicant_fi_table(work):
    """出願人ごとに、その出願に含まれるFIを集計する。"""
    rows = []
    for _, row in work.iterrows():
        applicants = _split_multi_value(row["出願人原文"])
        fis = _split_multi_value(row["FI原文"])
        if not applicants or not fis:
            continue
        for applicant in applicants:
            for fi in fis:
                rows.append({"出願人": applicant, "FI": fi})

    if not rows:
        return pd.DataFrame(columns=["出願人", "FI", "件数"])

    tmp = pd.DataFrame(rows)
    result = (
        tmp.groupby(["出願人", "FI"])
        .size()
        .reset_index(name="件数")
        .sort_values(["出願人", "件数", "FI"], ascending=[True, False, True])
    )
    return result


def _show_patent_statistics(df):
    """4種類の特許統計を表示する。"""
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
        f"（出願日: {date_col} / FI: {fi_col} / 出願人: {applicant_col}）"
    )

    # ① 年別出願件数
    st.markdown("### 📈 ① 年別出願件数推移")
    yearly = work.groupby("出願年").size().rename("出願件数").sort_index()
    st.line_chart(yearly, x_label="出願年", y_label="出願件数")
    st.dataframe(
        yearly.reset_index(),
        use_container_width=True,
        hide_index=True
    )

    # ② 筆頭FIランキング
    st.markdown("### 🏆 ② 筆頭FIランキング")
    first_fi = (
        work.dropna(subset=["筆頭FI"])
        .groupby("筆頭FI")
        .size()
        .reset_index(name="出願件数")
        .sort_values(["出願件数", "筆頭FI"], ascending=[False, True])
        .reset_index(drop=True)
    )
    first_fi.insert(0, "順位", range(1, len(first_fi) + 1))

    top_n = st.slider(
        "表示するランキング件数",
        min_value=5,
        max_value=30,
        value=10,
        key="stats_top_n"
    )
    st.bar_chart(
        first_fi.head(top_n).set_index("筆頭FI")["出願件数"],
        horizontal=True,
        x_label="出願件数",
        y_label="筆頭FI"
    )
    st.dataframe(first_fi.head(top_n), use_container_width=True, hide_index=True)

    # ③ 筆頭出願人ランキング
    st.markdown("### 🏢 ③ 筆頭出願人ランキング")
    first_applicant = (
        work.dropna(subset=["筆頭出願人"])
        .groupby("筆頭出願人")
        .size()
        .reset_index(name="出願件数")
        .sort_values(["出願件数", "筆頭出願人"], ascending=[False, True])
        .reset_index(drop=True)
    )
    first_applicant.insert(0, "順位", range(1, len(first_applicant) + 1))

    st.bar_chart(
        first_applicant.head(top_n).set_index("筆頭出願人")["出願件数"],
        horizontal=True,
        x_label="出願件数",
        y_label="筆頭出願人"
    )
    st.dataframe(
        first_applicant.head(top_n),
        use_container_width=True,
        hide_index=True
    )

    # ④ 出願人別出願FIランキング
    st.markdown("### 🧩 ④ 出願人別出願FIランキング")
    applicant_fi = _make_applicant_fi_table(work)

    if applicant_fi.empty:
        st.info("出願人別FIを集計できるデータがありません。")
        return

    # 出願人ごとに上位FIを表示
    applicant_choices = sorted(applicant_fi["出願人"].unique())
    selected_applicant = st.selectbox(
        "詳しく見る出願人",
        applicant_choices,
        key="stats_selected_applicant"
    )

    selected = applicant_fi[
        applicant_fi["出願人"] == selected_applicant
    ].copy()
    selected.insert(0, "順位", range(1, len(selected) + 1))

    st.bar_chart(
        selected.head(top_n).set_index("FI")["件数"],
        horizontal=True,
        x_label="出願件数",
        y_label="FI"
    )
    st.dataframe(
        selected.head(top_n),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "※「筆頭FI」「筆頭出願人」は、CSVの該当セルに複数値がある場合、"
        "先頭に記載されたものを筆頭として集計します。"
        "「出願人別出願FI」は、同一出願に複数の出願人・FIがある場合、"
        "それぞれの組合せを1件として集計します。"
    )


st.title("🪼 日本語特許請求項SAO構造分析")
st.caption("GiNZAで日本語特許請求項を「主語・動詞・目的語」に分解して、構成要素の関係を可視化します")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🪸 1つの請求項を解析",
    "🐚 2つの請求項を比較",
    "🔦 まとめて検索",
    "🪼 従属請求項を展開",
    "📊 特許統計分析",
])


# ============================================================
# タブ①：1つの請求項を解析して図を見る
# ============================================================
with tab1:
    st.subheader("📝 請求項を入力してください")
    text = st.text_area(
        "請求項テキスト",
        height=220,
        placeholder="例：第１の基板と、前記第１の基板上に設けられた第２の半導体層と、前記第２の半導体層に接続された電極と、を有する半導体装置。",
        key="single_text",
    )

    if st.button("✨ 解析する", type="primary", key="single_run"):
        if not text.strip():
            st.warning("請求項テキストを入力してください。")
            st.session_state.single_result = None
        else:
            with st.spinner("解析中..."):
                try:
                    components, relations = pp.analyze_claim(text)
                    st.session_state.single_result = {
                        "components": components,
                        "relations": relations,
                    }
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {e}")
                    st.session_state.single_result = None

    # 結果を表示
    if st.session_state.single_result is not None:
        relations = st.session_state.single_result["relations"]

        if not relations:
            st.info("関係が抽出できませんでした。文の書き方を見直してみてください。")
        else:
            st.success(f"🎉 {len(relations)} 件の関係を抽出しました！")

            col1, col2 = st.columns([3, 2])

            with col1:
                st.markdown("#### 🪸 構成要素間の関係図")
                graph = pp.build_graphviz(relations, title="構成要素間関係", theme="deepsea")
                st.graphviz_chart(graph, use_container_width=True)

                st.markdown("#### 🐙 クレームの広さ・狭さ")
                narrowness, breadth, scope_detail = pp.compute_claim_scope_score(relations)
                scope_cols = st.columns(2)
                scope_cols[0].metric("広さスコア（大きいほど抽象的）", f"{breadth:.3f}")
                scope_cols[1].metric("狭さスコア（大きいほど限定的）", f"{narrowness:.3f}")
                with st.expander("広さ・狭さの内訳を見る"):
                    st.write(
                        f"- 構成要素数: {scope_detail['構成要素数']}\n"
                        f"- 関係の総数: {scope_detail['関係の総数']}\n"
                        f"- 数値スペックの数: {scope_detail['数値スペックの数']}\n"
                        f"- 「有する」階層の深さ: {scope_detail['階層の深さ']}\n"
                        f"- 関係密度(関係数/構成要素数): {scope_detail['関係密度']}"
                    )
                    st.caption(
                        "※ このスコアは絶対的な尺度ではなく、他の請求項と相対的に比べるための指標です"
                        "（例：独立項と従属項の比較、改良前後のクレーム案の比較など）。"
                    )

            with col2:
                st.markdown("#### 📋 抽出された関係（SAOトリプル）")
                st.dataframe(
                    [
                        {
                            "主語": r["source"],
                            "関係": r["relation"],
                            "目的語": r["target"],
                            "種類": r["type"],
                        }
                        for r in relations
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# タブ②：2つの請求項を比較して類似度診断する
# ============================================================
with tab2:
    st.subheader("📝 2つの請求項を入力してください")

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
                    _, relations_a = pp.analyze_claim(text_a)
                    _, relations_b = pp.analyze_claim(text_b)

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
                        "jaccard_score": jaccard_score,
                        "common": common,
                        "only_a": only_a,
                        "only_b": only_b,
                        "structural_score": structural_score,
                        "structural_detail": structural_detail,
                        "semantic_score": semantic_score,
                        "semantic_matches": semantic_matches,
                        "scope_a": (narrowness_a, breadth_a, scope_detail_a),
                        "scope_b": (narrowness_b, breadth_b, scope_detail_b),
                    }
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {e}")
                    st.session_state.compare_result = None

    # 結果を表示
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
        st.caption(
            "※ このスコアは絶対的な尺度ではなく、AとBを相対的に比べるための指標です。"
        )

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
                        "トリプルA": " / ".join(ta),
                        "トリプルB": " / ".join(tb),
                    }
                    for ta, tb, sim in matches_sorted
                ],
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# タブ③：複数の請求項をデータベース化して、1件をまとめて検索する
# ============================================================
with tab3:
    st.subheader("🗄️ ステップ１：比較対象の請求項をまとめて登録する")
    st.caption(
        "CSVファイル（列名: id, text）をアップロードするか、"
        "下のテキストエリアに「-----」で区切って複数の請求項を貼り付けてください。"
    )

    uploaded_csv = st.file_uploader("CSVファイル（id, text の2列）", type=["csv"])
    bulk_text = st.text_area(
        "またはここに、請求項を「-----」で区切って貼り付ける",
        height=180,
        placeholder="1件目の請求項テキスト...\n-----\n2件目の請求項テキスト...\n-----\n3件目の請求項テキスト...",
        key="bulk_text",
    )

    if st.button("📚 データベースを構築する", key="build_db_run"):
        records = []
        if uploaded_csv is not None:
            import csv
            import io

            content = uploaded_csv.getvalue().decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                rid = row.get("id") or row.get("番号") or f"行{len(records)+1}"
                text = row.get("text") or row.get("本文") or ""
                if text.strip():
                    records.append((rid, text.strip()))
        elif bulk_text.strip():
            parts = [p.strip() for p in bulk_text.split("-----") if p.strip()]
            records = [(f"請求項{i+1}", p) for i, p in enumerate(parts)]

        if not records:
            st.warning("CSVのアップロード、またはテキストの貼り付けのどちらかを行ってください。")
        else:
            with st.spinner(f"{len(records)} 件を解析してデータベースを構築中...（初回は埋め込みモデルの読み込みに1分程度かかります）"):
                try:
                    db = pp.build_patent_database(records, show_progress=False)
                    st.session_state.patent_db = db
                    st.session_state.search_results = None
                except Exception as e:
                    st.error(f"データベース構築中にエラーが発生しました: {e}")
                    st.session_state.patent_db = None

    if st.session_state.patent_db is not None:
        st.success(f"✅ {len(st.session_state.patent_db)} 件を登録済みです。")
        with st.expander("登録済みの一覧を見る"):
            st.dataframe(
                [{"id": e["id"], "本文（先頭50文字）": e["text"][:50] + "..."} for e in st.session_state.patent_db],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    st.subheader("🔦 ステップ２：調べたい請求項を検索する")
    query_text = st.text_area(
        "検索したい請求項テキスト",
        height=160,
        key="query_text",
    )
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
                    results = pp.search_similar_claims(
                        query_text, st.session_state.patent_db, top_k=top_k, rerank_k=rerank_k
                    )
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
                st.caption(f"粗いスコア: {r['fast_score']:.3f}" + (f" ／ 精密スコア: {r['precise_score']:.3f}" if has_precise else ""))
                if has_precise:
                    matches_sorted = sorted(r["matches"], key=lambda x: -x[2])
                    st.dataframe(
                        [
                            {
                                "類似度": round(sim, 2),
                                "判定": "完全一致" if ta == tb else ("意味が近い" if sim >= 0.6 else "対応薄い"),
                                "クエリ側": " / ".join(ta),
                                "この請求項側": " / ".join(tb),
                            }
                            for ta, tb, sim in matches_sorted
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# タブ④：従属請求項を、親請求項の内容も含めて完全な形に展開する
# ============================================================
with tab4:
    st.subheader("📜 ステップ１：請求項群を貼り付ける")
    st.caption(
        "実際の公報の書き方（【請求項１】【請求項２】…）のまま、"
        "コピペで貼り付けてください。区切りの手作業は不要です。"
        "【請求項】の目印がなければ、全体を請求項１として扱います。"
    )
    claims_text = st.text_area(
        "請求項群",
        height=280,
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
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    st.subheader("🪼 ステップ２：展開したい請求項を選ぶ")
    if parsed_claims:
        target_num = st.selectbox("展開する請求項番号", sorted(parsed_claims.keys()), key="target_claim_num")
    else:
        target_num = None
        st.info("先にステップ１で請求項を入力してください。")

    if st.button("🔍 展開して解析する", key="dependent_run", disabled=not parsed_claims):
        try:
            (components, relations), full_text = pp.analyze_dependent_claim(target_num, parsed_claims)
            st.session_state.dependent_result = {
                "full_text": full_text,
                "relations": relations,
            }
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
                graph = pp.build_graphviz(relations, title="構成要素間関係（展開後）", theme="deepsea")
                st.graphviz_chart(graph, use_container_width=True)

                st.markdown("#### 🐙 クレームの広さ・狭さ")
                narrowness, breadth, scope_detail = pp.compute_claim_scope_score(relations)
                scope_cols = st.columns(2)
                scope_cols[0].metric("広さスコア", f"{breadth:.3f}")
                scope_cols[1].metric("狭さスコア", f"{narrowness:.3f}")

            with col2:
                st.markdown("#### 📋 抽出された関係（SAOトリプル）")
                st.dataframe(
                    [
                        {"主語": r["source"], "関係": r["relation"], "目的語": r["target"], "種類": r["type"]}
                        for r in relations
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# タブ⑤：CSVによる特許統計分析
# ============================================================
with tab5:
    st.subheader("📊 CSVから特許出願統計を分析")
    st.caption(
        "CSVをアップロードするか、CSV本文を貼り付けるだけで、"
        "年別出願件数・筆頭FI・筆頭出願人・出願人別FIを集計します。"
    )

    st.markdown(
        """
        **推奨CSV列**
        - `出願日`：例 `2024-03-15`
        - `FI`：例 `H01L 21/00; H01L 29/00`
        - `出願人`：例 `株式会社A;株式会社B`

        英語列 `application_date / fi / applicant` も利用できます。
        既存のCSVに `id` や `text` など他の列があっても問題ありません。
        """
    )

    stats_uploaded_csv = st.file_uploader(
        "📁 統計分析用CSVをアップロード",
        type=["csv"],
        key="stats_csv_upload"
    )

    stats_csv_text = st.text_area(
        "またはCSV本文をここに貼り付け",
        height=180,
        placeholder=(
            "出願日,FI,出願人\n"
            "2022-04-01,H01L 21/00,株式会社A\n"
            "2023-06-12,H01L 29/00;H01L 21/00,株式会社B;株式会社C\n"
            "2024-01-20,H10B 12/00,株式会社A"
        ),
        key="stats_csv_text"
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
                stats_df = pd.read_csv(io.StringIO(content))
                st.session_state.stats_df = stats_df
        except Exception as e:
            st.error(f"CSVの読み込みに失敗しました: {e}")
            st.session_state.stats_df = None

    if st.session_state.stats_df is not None:
        with st.expander("読み込んだCSVを確認する"):
            st.dataframe(
                st.session_state.stats_df,
                use_container_width=True,
                hide_index=True
            )

        _show_patent_statistics(st.session_state.stats_df)
