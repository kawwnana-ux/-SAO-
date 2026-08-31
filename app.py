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
    fi_col = _find_column(df, [
        "FI", "FI分類", "FIコード", "fi_code", "fi"
    ])
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
        lambda x: _split_multi_value(x)[0]
        if _split_multi_value(x) else None
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
                rows.append({
                    "出願人/権利者": applicant,
                    "FIサブクラス": subclass
                })

    if not rows:
        return pd.DataFrame(
            columns=["出願人/権利者", "FIサブクラス", "件数"]
        )

    tmp = pd.DataFrame(rows)
    return (
        tmp.groupby(["出願人/権利者", "FIサブクラス"])
        .size()
        .reset_index(name="件数")
        .sort_values(
            ["出願人/権利者", "件数", "FIサブクラス"],
            ascending=[True, False, True]
        )
        .reset_index(drop=True)
    )


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
        f"（出願日: {date_col} / FI: {fi_col} / "
        f"出願人/権利者: {applicant_col}）"
    )

    top_n = st.number_input(
        "ランキング表示件数",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        key="stats_top_n"
    )

    # ① 年別出願件数
    st.markdown("### 📈 ① 年別出願件数推移")
    yearly = (
        work.groupby("出願年")
        .size()
        .rename("出願件数")
        .sort_index()
    )
    st.line_chart(yearly, x_label="出願年", y_label="出願件数")
    st.dataframe(
        yearly.reset_index(),
        use_container_width=True,
        hide_index=True
    )

    # ② 筆頭FIサブクラスランキング
    st.markdown("### 🏆 ② 筆頭FIサブクラスランキング")
    first_fi = (
        work.dropna(subset=["筆頭FIサブクラス"])
        .groupby("筆頭FIサブクラス")
        .size()
        .reset_index(name="出願件数")
        .sort_values(
            ["出願件数", "筆頭FIサブクラス"],
            ascending=[False, True]
        )
        .reset_index(drop=True)
    )
    first_fi.insert(0, "順位", range(1, len(first_fi) + 1))

    st.bar_chart(
        first_fi.head(int(top_n)).set_index("筆頭FIサブクラス")["出願件数"],
        horizontal=True,
        x_label="出願件数",
        y_label="FIサブクラス"
    )
    st.dataframe(
        first_fi.head(int(top_n)),
        use_container_width=True,
        hide_index=True
    )

    # ③ 筆頭出願人/権利者ランキング
    st.markdown("### 🏢 ③ 筆頭出願人/権利者ランキング")
    first_applicant = (
        work.dropna(subset=["出願人/権利者"])
        .groupby("出願人/権利者")
        .size()
        .reset_index(name="出願件数")
        .sort_values(
            ["出願件数", "出願人/権利者"],
            ascending=[False, True]
        )
        .reset_index(drop=True)
    )
    first_applicant.insert(
        0, "順位", range(1, len(first_applicant) + 1)
    )

    st.bar_chart(
        first_applicant.head(int(top_n))
        .set_index("出願人/権利者")["出願件数"],
        horizontal=True,
        x_label="出願件数",
        y_label="出願人/権利者"
    )
    st.dataframe(
        first_applicant.head(int(top_n)),
        use_container_width=True,
        hide_index=True
    )

    # ④ 出願人/権利者別出願FIランキング
    st.markdown("### 🧩 ④ 出願人/権利者別出願FIサブクラスランキング")
    applicant_fi = _make_applicant_fi_table(work)

    if applicant_fi.empty:
        st.info("出願人/権利者別FIを集計できるデータがありません。")
        return

    applicant_choices = sorted(
        applicant_fi["出願人/権利者"].unique()
    )

    selected_applicant = st.selectbox(
        "詳しく見る出願人/権利者",
        applicant_choices,
        key="stats_selected_applicant"
    )

    selected = applicant_fi[
        applicant_fi["出願人/権利者"] == selected_applicant
    ].copy()

    selected.insert(0, "順位", range(1, len(selected) + 1))

    st.bar_chart(
        selected.head(int(top_n))
        .set_index("FIサブクラス")["件数"],
        horizontal=True,
        x_label="出願件数",
        y_label="FIサブクラス"
    )
    st.dataframe(
        selected.head(int(top_n)),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "※ 筆頭FI・筆頭出願人/権利者は、CSVのセルに複数値がある場合、"
        "先頭に記載されたものを筆頭として集計します。"
        "出願人/権利者別出願FIは、同一出願に複数の"
        "出願人/権利者・FIがある場合、それぞれの組合せを1件として集計します。"
        "FIはサブクラス（例：H01L、G06F）単位で集計します。"
    )
# ============================================================
# タブ⑤：CSVによる特許統計分析
# ============================================================
# ============================================================
# タブ⑤を作成
# ============================================================
tab4 = st.container()

# ============================================================
# タブ⑤：CSVによる特許統計分析
# ============================================================
with tab4:
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
        - `出願人/権利者`：例 `株式会社A;株式会社B`

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
            "出願日,FI,出願人/権利者\n"
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
