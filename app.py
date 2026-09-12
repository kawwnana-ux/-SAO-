import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
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
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_summary" not in st.session_state:
    st.session_state.eval_summary = None
if "gold_per_claim" not in st.session_state:
    st.session_state.gold_per_claim = None
if "gold_summary" not in st.session_state:
    st.session_state.gold_summary = None


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

st.title("🪼 日本語特許請求項SAO構造分析")
st.caption("SudachiPyで表記ゆれを正規化し、GiNZAの係り受け解析とルールベースの補正だけでSAO構造を抽出・可視化します（LLM不使用）")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🪸 1つの請求項を解析", "🐚 2つの請求項を比較", "🔦 まとめて検索",
    "🪼 従属請求項を展開", "📊 特許統計分析", "✅ 精度検証",
])


# ============================================================
# タブ①：1つの請求項を解析して図を見る
# ============================================================
with tab1:
    st.subheader("📝 請求項を入力してください")

    st.info(
        "🔧 SudachiPy（前処理）＋GiNZA（係り受け解析）＋ルールベース補正のみでSAO構造を抽出します。"
        "LLMは使用しません。"
    )
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
                    components, relations = pp.analyze_claim_ginza(text)
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

    st.caption("SudachiPy＋GiNZA＋ルールベース補正（LLM不使用）で請求項A・Bを解析して比較します。")

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
                    _, relations_a = pp.analyze_claim_ginza(text_a)
                    _, relations_b = pp.analyze_claim_ginza(text_b)

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


with tab6:
    st.subheader("✅ 大量の請求項に対する精度検証（自動ヘルスチェック）")
    st.caption(
        "人手のSAO正解データがない前提で、既知の不具合パターン"
        "（関係が1件も取れない・自己ループ・意味のない語がノードになる・"
        "動詞連鎖を遡りすぎて無関係な終端ノードに集まる・部品が孤立する 等）"
        "を自動チェックし、その合格率を精度の代理指標として使います。"
    )

    eval_uploaded = st.file_uploader(
        "📁 請求項リスト（.xlsx / .csv）をアップロード",
        type=["xlsx", "csv"],
        key="eval_file_upload",
    )

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        eval_text_col = st.text_input(
            "請求項本文の列名", value="請求項本文", key="eval_text_col"
        )
    with col_e2:
        eval_id_col = st.text_input(
            "ID列名（無ければ空欄でOK）", value="id", key="eval_id_col"
        )

    eval_limit = st.number_input(
        "検証する件数の上限（0で全件）", min_value=0, value=0, step=10,
        key="eval_limit",
        help="全532件だと時間がかかるので、まず50〜100件で試すのがおすすめです。",
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

                    def _on_progress(done, total):
                        progress_bar.progress(
                            done / total if total else 0, text=f"{done}/{total}件"
                        )

                    with st.spinner("解析＆ヘルスチェック中..."):
                        results, summary = pp.evaluate_corpus_health(
                            records, progress_callback=_on_progress
                        )
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
            check_summary = (
                results_df[check_cols].mean().sort_values() * 100
            ).round(1)
            st.dataframe(
                check_summary.rename("合格率(%)").reset_index().rename(
                    columns={"index": "チェック項目"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### 📋 詳細結果")
        show_only_failed = st.checkbox("不合格のものだけ表示", value=True, key="eval_show_failed")
        display_df = results_df[~results_df["passed"]] if show_only_failed else results_df
        st.dataframe(
            display_df[["id", "text", "passed", "n_components", "n_relations", "error"] + check_cols]
            if len(display_df) > 0 else display_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 詳細結果をCSVでダウンロード",
            data=csv_bytes,
            file_name="claim_health_check_results.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("### 🏅 正解SAO（手作業で作成した20件）との比較")
    st.caption(
        "532件全部の正解データは人手では作れないため、パターンの幅が"
        "出るように選んだ代表20件について、実際に読んで正解のSAOトリプルを"
        "手作業で作成しました。それと実際の抽出結果を突き合わせて、"
        "適合率(precision)・再現率(recall)・F値を計算します。"
    )

    gold_csv_upload = st.file_uploader(
        "📁 gold_sao.csv をアップロード", type=["csv"], key="gold_sao_upload"
    )
    gold_text_csv_upload = st.file_uploader(
        "📁 gold_sample_claims.csv をアップロード（正解データに対応する請求項本文）",
        type=["csv"],
        key="gold_text_upload",
    )

    if st.button("🏅 正解SAOと比較する", type="primary", key="gold_compare_run"):
        if gold_csv_upload is None or gold_text_csv_upload is None:
            st.warning("gold_sao.csv と gold_sample_claims.csv の両方をアップロードしてください。")
        else:
            try:
                gold_df = pd.read_csv(gold_csv_upload)
                text_df = pd.read_csv(gold_text_csv_upload)

                gold_by_claim = {}
                for _, row in gold_df.iterrows():
                    gold_by_claim.setdefault(row["claim_id"], []).append(
                        (row["source"], row["relation"], row["target"])
                    )
                texts_by_claim = dict(zip(text_df["claim_id"], text_df["請求項本文"]))

                gold_progress = st.progress(0, text="0/0件")

                def _on_gold_progress(done, total):
                    gold_progress.progress(done / total if total else 0, text=f"{done}/{total}件")

                with st.spinner("解析＆正解データとの突き合わせ中..."):
                    per_claim, gold_summary = pp.compare_with_gold(
                        gold_by_claim, texts_by_claim, progress_callback=_on_gold_progress
                    )
                gold_progress.empty()

                st.session_state.gold_per_claim = per_claim
                st.session_state.gold_summary = gold_summary
            except Exception as e:
                st.error(f"比較中にエラーが発生しました: {e}")

    if st.session_state.get("gold_summary") is not None:
        gsum = st.session_state.gold_summary
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("適合率 (Precision)", f"{gsum['precision']*100:.1f}%")
        g2.metric("再現率 (Recall)", f"{gsum['recall']*100:.1f}%")
        g3.metric("F値", f"{gsum['f1']*100:.1f}%")
        g4.metric("検証請求項数", f"{gsum['n_claims']}件")

        if gsum["f1"] * 100 >= 90:
            st.success(f"🎉 F値が目標の90%以上です（{gsum['f1']*100:.1f}%）。")
        else:
            st.warning(
                f"⚠️ F値はまだ90%に届いていません（{gsum['f1']*100:.1f}%）。"
                "下の請求項ごとの内訳で、どこで取りこぼしや誤抽出が"
                "多いか確認してください。"
            )

        gold_rows = []
        for r in st.session_state.gold_per_claim:
            gold_rows.append({
                "claim_id": r["claim_id"],
                "正解数": r["n_gold"],
                "抽出数": r["n_extracted"],
                "一致数": r["n_matched"],
                "precision": round(r["precision"], 3),
                "recall": round(r["recall"], 3),
                "f1": round(r["f1"], 3),
                "error": r.get("error"),
            })
        gold_result_df = pd.DataFrame(gold_rows)
        st.markdown("#### 請求項ごとの内訳")
        st.dataframe(gold_result_df, use_container_width=True, hide_index=True)

        claim_options = [r["claim_id"] for r in st.session_state.gold_per_claim]
        picked = st.selectbox("取りこぼし・誤抽出を見る請求項を選択", claim_options, key="gold_pick_claim")
        picked_row = next(r for r in st.session_state.gold_per_claim if r["claim_id"] == picked)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("**❌ 取りこぼした正解（missed_gold）**")
            st.dataframe(
                pd.DataFrame(picked_row.get("missed_gold", []), columns=["source", "relation", "target"]),
                use_container_width=True, hide_index=True,
            )
        with col_g2:
            st.markdown("**➕ 正解にない抽出（extra_extracted）**")
            st.dataframe(
                pd.DataFrame(picked_row.get("extra_extracted", []), columns=["source", "relation", "target"]),
                use_container_width=True, hide_index=True,
            )
