import base64
import matplotlib.pyplot as plt
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
if "abstract_db" not in st.session_state:
    st.session_state.abstract_db = None


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
st.caption("GiNZAで日本語特許請求項を「主語・動詞・目的語」に分解して、構成要素の関係を可視化します")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🪸 1つの請求項を解析", "🐚 2つの請求項を比較", "🔦 まとめて検索",
    "🪼 従属請求項を展開", "🔭 ポートフォリオ分析",
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
                st.graphviz_chart(graph, width='stretch')

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
                    width='stretch',
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
                width='stretch',
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
                width='stretch',
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
                        width='stretch',
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
                width='stretch',
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
                graph = pp.build_graphviz(relations, title="構成要素間関係（展開後）", theme="deepsea", color_by="claim")
                st.graphviz_chart(graph, width='stretch')

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
                    width='stretch',
                    hide_index=True,
                )


# ============================================================
# タブ⑤：ポートフォリオ分析（Explorer / Saturn V / CORE）
# ============================================================
with tab5:
    st.caption(
        "このタブでは、タブ「🔦 まとめて検索」で構築した請求項データベース、"
        "またはここで新しく作る「要約」データベースのどちらかを使えます。"
        "要約はJ-PlatPat等で一括ダウンロードしやすいため、大量の分析に向いています。"
    )

    db_source = st.radio(
        "使うデータベース",
        [
            "請求項データベース（🔦タブで構築済みのもの）",
            "要約データベース（ここで新しく作る）",
            "フルメタデータCSV（出願日・出願人・FI等、ATLAS用）",
        ],
        key="db_source_choice",
    )

    if db_source.startswith("フルメタデータ"):
        st.markdown("#### 🛰️ フルメタデータCSVを登録する")
        st.caption(
            "列名: 文献番号, 出願番号, 出願日, 公知日, 発明の名称, 出願人/権利者, "
            "FI, 要約, 公開番号, 公告番号, 登録番号, 審判番号, その他, ステージ, "
            "イベント詳細, 文献URL　を含むCSVをアップロードしてください。"
        )
        uploaded_meta_csv = st.file_uploader("CSVファイル", type=["csv"], key="meta_csv")
        if st.button("🛰️ データベースを構築する", key="build_full_db_run"):
            if uploaded_meta_csv is None:
                st.warning("CSVファイルをアップロードしてください。")
            else:
                with st.spinner("解析中...（初回は埋め込みモデルの読み込みに1分程度かかります）"):
                    try:
                        content = uploaded_meta_csv.getvalue().decode("utf-8-sig")
                        meta_records = pp.load_patent_metadata_csv(content)
                        st.session_state.abstract_db = pp.build_full_database(meta_records, show_progress=False)
                    except Exception as e:
                        st.error(f"データベース構築中にエラーが発生しました: {e}")
                        st.session_state.abstract_db = None
        if st.session_state.abstract_db is not None:
            st.success(f"✅ {len(st.session_state.abstract_db)} 件を登録済みです。")
        db = st.session_state.abstract_db

    elif db_source.startswith("要約"):
        st.markdown("#### 📄 要約をまとめて登録する")
        st.caption(
            "CSVファイル（列名: id, text）をアップロードするか、"
            "下のテキストエリアに「-----」で区切って複数の要約を貼り付けてください。"
            "【課題】【解決手段】等の見出しが入っていると、CORE分類の精度が上がります。"
        )
        uploaded_abs_csv = st.file_uploader("CSVファイル（id, text の2列）", type=["csv"], key="abs_csv")
        abs_bulk_text = st.text_area(
            "またはここに、要約を「-----」で区切って貼り付ける",
            height=180,
            placeholder="【課題】〜。【解決手段】〜。\n-----\n【課題】〜。【解決手段】〜。",
            key="abs_bulk_text",
        )
        if st.button("📚 要約データベースを構築する", key="build_abs_db_run"):
            abs_records = []
            if uploaded_abs_csv is not None:
                import csv
                import io

                content = uploaded_abs_csv.getvalue().decode("utf-8-sig")
                reader = csv.DictReader(io.StringIO(content))
                for row in reader:
                    rid = row.get("id") or row.get("番号") or f"行{len(abs_records)+1}"
                    text = row.get("text") or row.get("本文") or ""
                    if text.strip():
                        abs_records.append((rid, text.strip()))
            elif abs_bulk_text.strip():
                parts = [p.strip() for p in abs_bulk_text.split("-----") if p.strip()]
                abs_records = [(f"要約{i+1}", p) for i, p in enumerate(parts)]

            if not abs_records:
                st.warning("CSVのアップロード、またはテキストの貼り付けのどちらかを行ってください。")
            else:
                with st.spinner(f"{len(abs_records)} 件を解析中...（初回は埋め込みモデルの読み込みに1分程度かかります）"):
                    try:
                        st.session_state.abstract_db = pp.build_abstract_database(abs_records, show_progress=False)
                    except Exception as e:
                        st.error(f"データベース構築中にエラーが発生しました: {e}")
                        st.session_state.abstract_db = None

        if st.session_state.abstract_db is not None:
            st.success(f"✅ {len(st.session_state.abstract_db)} 件を登録済みです。")
            with st.expander("認識されたセクションを確認する"):
                st.dataframe(
                    [{"id": e["id"], "見出し": "、".join(e["sections"].keys())} for e in st.session_state.abstract_db],
                    width='stretch', hide_index=True,
                )
        db = st.session_state.abstract_db
    else:
        db = st.session_state.patent_db
        if db is None:
            st.info("先に「🔦 まとめて検索」タブでデータベースを構築してください。")

    if db is None:
        pass
    else:
        all_ids = [e["id"] for e in db]
        st.success(f"✅ {len(db)} 件のデータベースを利用します。")

        sub_tab_atlas, sub_tab_explorer, sub_tab_saturn, sub_tab_core, sub_tab_rank = st.tabs([
            "🗺️ 海図（ATLAS）",
            "🐡 群れ探査（Explorer）", "🌊 深海海流マップ（Saturn V）",
            "🪸 珊瑚礁分類（CORE）", "🐙 生態プロファイル",
        ])

        with sub_tab_atlas:
            st.caption("🧭 出願の海を一望する海図。件数の潮流・出願人という船団・FIという漁場を俯瞰します。")
            has_metadata = any(e.get("出願日") for e in db)
            if not has_metadata:
                st.info("このデータベースには出願日等のメタデータがありません。「フルメタデータCSV」を選んで構築してください。")
            else:
                freq_choice = st.radio("時系列の単位", ["年", "月"], horizontal=True, key="atlas_freq")
                if st.button("🌊 出願件数推移を見る", key="atlas_trend_run"):
                    fig = pp.plot_filing_trend(db, freq="Y" if freq_choice == "年" else "M")
                    st.pyplot(fig)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🏢 出願人ランキング", key="atlas_applicant_run"):
                        ranking = pp.rank_by_field(db, "出願人")
                        st.session_state.atlas_applicant_ranking = ranking
                    if st.session_state.get("atlas_applicant_ranking"):
                        fig = pp.plot_ranking_bar(st.session_state.atlas_applicant_ranking, title="出願人ランキング")
                        st.pyplot(fig)
                with col2:
                    fi_level = st.radio("FIの粒度", ["サブクラス", "メイングループ", "そのまま"], horizontal=True, key="atlas_fi_level")
                    if st.button("🔬 FIランキング", key="atlas_fi_run"):
                        ranking = pp.rank_fi(db, level=fi_level)
                        st.session_state.atlas_fi_ranking = ranking
                    if st.session_state.get("atlas_fi_ranking"):
                        fig = pp.plot_ranking_bar(st.session_state.atlas_fi_ranking, title=f"FIランキング（{fi_level}）")
                        st.pyplot(fig)

                if st.button("🫧 出願人×FI バブルチャート", key="atlas_bubble_run"):
                    try:
                        fig = pp.plot_applicant_fi_bubble(db, fi_level=fi_level)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"バブルチャート作成中にエラーが発生しました: {e}")

                st.divider()
                st.markdown("#### 🕸️ 出願人×FI レーダーチャート")
                top_applicants_n = st.slider("対象にする出願人数（出願件数が多い順）", min_value=2, max_value=10, value=3, key="atlas_radar_applicants")
                top_fi_n = st.slider("軸にするFIの数（出現件数が多い順）", min_value=3, max_value=10, value=5, key="atlas_radar_fi")
                if st.button("🕸️ レーダーチャートを作る", key="atlas_radar_run"):
                    try:
                        profiles = pp.build_applicant_fi_radar_data(
                            db, fi_level=fi_level, top_applicants=top_applicants_n, top_fi=top_fi_n
                        )
                        st.session_state.atlas_radar_result = profiles
                    except Exception as e:
                        st.error(f"レーダーチャート作成中にエラーが発生しました: {e}")
                if st.session_state.get("atlas_radar_result"):
                    fig = pp.plot_radar_chart(st.session_state.atlas_radar_result, title=f"出願人×FI（{fi_level}）")
                    st.pyplot(fig)

                st.divider()
                st.markdown("#### 🌪️ MEGA：動態分析（活動量×勢い）")
                st.caption("直近の出願件数（活動量）と、その伸び率（勢い）から、リーダー・新興・成熟・衰退のどのフェーズにあるかを診断します。")
                mega_group_by = st.radio("グループの単位", ["出願人", "FI"], horizontal=True, key="mega_group_by")
                col1, col2, col3 = st.columns(3)
                with col1:
                    mega_recent = st.slider("直近とみなす年数", min_value=1, max_value=5, value=3, key="mega_recent")
                with col2:
                    mega_compare = st.slider("比較対象の年数", min_value=1, max_value=5, value=3, key="mega_compare")
                with col3:
                    mega_top_n = st.slider("表示する件数", min_value=3, max_value=30, value=10, key="mega_top_n")
                if st.button("🌪️ MEGAを診断する", key="mega_run"):
                    try:
                        mega_data, latest_year = pp.compute_activity_momentum(
                            db, group_by=mega_group_by, recent_years=mega_recent, compare_years=mega_compare
                        )
                        if not mega_data:
                            st.warning("出願日のデータが見つかりませんでした。")
                        else:
                            st.session_state.mega_result = (mega_data, latest_year)
                    except Exception as e:
                        st.error(f"MEGA診断中にエラーが発生しました: {e}")
                if st.session_state.get("mega_result") is not None:
                    mega_data, latest_year = st.session_state.mega_result
                    st.caption(f"最新の出願年（{latest_year}年）を基準に計算しています。")
                    fig = pp.plot_mega_chart(mega_data, title=f"MEGA：{mega_group_by}別の活動量×勢い", top_n=mega_top_n)
                    st.pyplot(fig)

        # --- Explorer ---
        with sub_tab_explorer:
            st.caption("🐡 貼り付けたデータ全体のキーワードを泳ぎ回って探ります。出願人情報があれば、自動で上位の出願人同士を比較します。")
            kind = st.radio("対象にするキーワードの種類", ["構成要素＋動詞", "構成要素のみ", "動詞のみ"], horizontal=True, key="explorer_kind")
            kind_map = {"構成要素＋動詞": "both", "構成要素のみ": "component", "動詞のみ": "verb"}

            applicant_groups = pp.get_applicant_groups(db, top_n=5)

            if st.button("🐠 解析する", key="explorer_run"):
                if applicant_groups:
                    names = list(applicant_groups.keys())
                    result = pp.compare_keyword_groups(
                        db, applicant_groups[names[0]], applicant_groups[names[1]] if len(names) > 1 else [],
                        kind=kind_map[kind],
                    )
                    st.session_state.explorer_result = (result, names[0], names[1] if len(names) > 1 else None)
                else:
                    freq_all = pp.build_keyword_frequency(db, kind=kind_map[kind])
                    st.session_state.explorer_result = (freq_all, None, None)

            if st.session_state.get("explorer_result") is not None:
                payload, name_a, name_b = st.session_state.explorer_result
                if name_a is None:
                    # 出願人情報がない場合：データ全体をまとめて1つのワードクラウドにする
                    st.markdown("**全体のワードクラウド**")
                    fig = pp.plot_wordcloud(payload, title="キーワード頻度（全体）")
                    st.pyplot(fig)
                    st.dataframe(
                        [{"語": w, "件数": f} for w, f in payload.most_common(30)],
                        hide_index=True, width='stretch',
                    )
                else:
                    result = payload
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{name_a}のワードクラウド**")
                        fig = pp.plot_wordcloud(result["freq_a"], title=name_a)
                        st.pyplot(fig)
                    with col2:
                        if name_b:
                            st.markdown(f"**{name_b}のワードクラウド**")
                            fig = pp.plot_wordcloud(result["freq_b"], title=name_b)
                            st.pyplot(fig)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**共通する語（{len(result['common'])}件）**")
                        st.dataframe([{"語": w, name_a: a, name_b or "B": b} for w, a, b in result["common"]], hide_index=True, width='stretch')
                    with col2:
                        st.markdown(f"**{name_a}だけの語（{len(result['only_a'])}件）**")
                        st.dataframe([{"語": w, "件数": f} for w, f in result["only_a"]], hide_index=True, width='stretch')
                    with col3:
                        st.markdown(f"**{name_b or 'B'}だけの語（{len(result['only_b'])}件）**")
                        st.dataframe([{"語": w, "件数": f} for w, f in result["only_b"]], hide_index=True, width='stretch')

        # --- Saturn V ---
        with sub_tab_saturn:
            st.caption("🌊 意味の海流に乗せて、特許たちを漂わせます。似た内容の発明ほど、潮に流されて近くに寄り集まります。")
            if st.button("🐋 マップを作成する", key="saturn_run"):
                try:
                    points, var = pp.build_semantic_map(db)
                    applicant_groups = pp.get_applicant_groups(db, top_n=6)
                    groups = None
                    if applicant_groups:
                        groups = {}
                        for name, ids in applicant_groups.items():
                            for i in ids:
                                groups[i] = name
                    st.session_state.saturn_result = (points, var, groups)
                except Exception as e:
                    st.error(f"マップ作成中にエラーが発生しました: {e}")

            if st.session_state.get("saturn_result") is not None:
                points, var, groups = st.session_state.saturn_result
                fig = pp.plot_semantic_map(points, var, groups=groups, title="意味的俯瞰マップ")
                st.pyplot(fig)
                st.caption("意味的に近い特許同士が近くに配置されます。距離が近いほど、内容が似ていることを示します。")

            st.divider()
            st.markdown("#### 🌋 キーワード地形図")
            st.caption("よく出てくるキーワードを地図上に配置し、頻度を山の高さ（色）で表します。赤い山ほど、その技術用語が密集しています。")
            landscape_kind = st.radio("地形図の対象", ["構成要素", "動詞"], horizontal=True, key="landscape_kind")
            top_n = st.slider("表示するキーワード数", min_value=10, max_value=80, value=40, key="landscape_top_n")
            if st.button("🌋 地形図を作る", key="landscape_run"):
                try:
                    points_lc = pp.build_keyword_landscape(
                        db, top_n=top_n, kind="component" if landscape_kind == "構成要素" else "verb"
                    )
                    st.session_state.landscape_result = points_lc
                except Exception as e:
                    st.error(f"地形図作成中にエラーが発生しました: {e}")
            if st.session_state.get("landscape_result"):
                fig = pp.plot_keyword_landscape(st.session_state.landscape_result, title="キーワード地形図")
                st.pyplot(fig)

        # --- CORE ---
        with sub_tab_core:
            st.caption("🪸 発明の名称から自動抽出したキーワードと、FIサブクラスのマス目に特許を住まわせます。誰も住んでいない白いマスが、まだ誰も棲みついていない空白地帯です。")
            st.markdown("#### 🐚 自動ホワイトスペースマップ（発明の名称×FIサブクラス）")
            has_title_field = any(e.get("発明の名称") for e in db)
            if not has_title_field:
                st.info("このデータベースには「発明の名称」がありません。「フルメタデータCSV」で構築してください。")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    n_keywords = st.slider("縦軸のキーワード数", min_value=5, max_value=50, value=20, key="kwfi_n_keywords")
                with col2:
                    n_fi = st.slider("横軸のFI数", min_value=5, max_value=50, value=20, key="kwfi_n_fi")
                kwfi_fi_level = st.radio("FIの粒度", ["サブクラス", "メイングループ", "そのまま"], horizontal=True, key="kwfi_fi_level")
                if st.button("🐚 自動でマップを作る", key="kwfi_run"):
                    try:
                        matrix_kw, kw_list, fi_list = pp.build_keyword_fi_matrix(
                            db, top_keywords=n_keywords, top_fi=n_fi, fi_level=kwfi_fi_level
                        )
                        st.session_state.kwfi_result = (matrix_kw, kw_list, fi_list)
                    except Exception as e:
                        st.error(f"マップ作成中にエラーが発生しました: {e}")
                if st.session_state.get("kwfi_result") is not None:
                    matrix_kw, kw_list, fi_list = st.session_state.kwfi_result
                    fig = pp.plot_keyword_fi_heatmap(matrix_kw, kw_list, fi_list, title="発明の名称×FI ホワイトスペースマップ")
                    st.pyplot(fig)
                    st.caption("色が濃いマスほど出願件数が多く、白いマスがホワイトスペース候補です。")

        # --- 構成部位ランキング・件数分布・レーダーチャート ---
        with sub_tab_rank:
            st.caption("🐙 この海域によく生息する部位のランキング、群れの体格分布、そして群れ同士の生態比較です。")
            st.markdown("#### 🦑 構成部位ランキング（全体）")
            rank_kind = st.radio("ランキングの対象", ["構成要素", "動詞"], horizontal=True, key="rank_kind")
            if st.button("🦑 ランキングを作る", key="rank_run"):
                ranking = pp.rank_components(db, kind="component" if rank_kind == "構成要素" else "verb", top_n=20)
                st.session_state.rank_result = ranking
            if st.session_state.get("rank_result") is not None:
                fig = pp.plot_component_ranking(st.session_state.rank_result, title=f"{rank_kind}ランキング")
                st.pyplot(fig)

            st.divider()

            st.markdown("#### 📈 分布（広さ・狭さ、またはFI/キーワード数）")
            has_relations = any("relations" in e for e in db)
            if has_relations:
                st.caption("請求項データベースなので、クレームの広さ・狭さのスコアで分布を作ります。")
                if st.button("📈 分布を作る", key="dist_run"):
                    st.session_state.dist_result = ("scope", pp.compute_scope_distribution(db))
            else:
                st.caption("請求項データベースではないため、代わりにFIコード数（またはキーワード数）の分布を作ります。")
                dist_metric = st.radio("分布の対象", ["FIコード数", "キーワード数"], horizontal=True, key="dist_metric")
                if st.button("📈 分布を作る", key="dist_run"):
                    st.session_state.dist_result = (dist_metric, pp.compute_metadata_distribution(db, metric=dist_metric))
            if st.session_state.get("dist_result") is not None:
                metric, scores = st.session_state.dist_result
                if not scores:
                    st.warning("分布を作れるデータが見つかりませんでした。")
                elif metric == "scope":
                    fig = pp.plot_scope_distribution(scores)
                    st.pyplot(fig)
                else:
                    fig = pp.plot_metadata_distribution(scores, metric=metric)
                    st.pyplot(fig)

            st.divider()

            st.markdown("#### 🕸️ 出願人ごとの特徴レーダーチャート比較（自動）")
            st.caption("出願人情報があれば自動で上位を比較します。請求項データベースがあればクレームの特徴を、無ければFI/キーワードの特徴を使います。")
            if st.button("🕸️ レーダーチャートを作る", key="radar_run"):
                applicant_groups = pp.get_applicant_groups(db, top_n=5)
                profile_fn = pp.compute_group_profile if has_relations else pp.compute_metadata_group_profile
                if applicant_groups:
                    profiles = {name: profile_fn(db, ids) for name, ids in applicant_groups.items()}
                    profiles = {k: v for k, v in profiles.items() if v}
                else:
                    profiles = {"全体": profile_fn(db, [e["id"] for e in db])}
                if not profiles or not any(profiles.values()):
                    st.warning("レーダーチャートを作れるデータが見つかりませんでした。")
                else:
                    st.session_state.radar_result = profiles
            if st.session_state.get("radar_result") is not None:
                fig = pp.plot_radar_chart(st.session_state.radar_result)
                st.pyplot(fig)
