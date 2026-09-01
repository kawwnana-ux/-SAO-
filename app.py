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
if "group_a_ids" not in st.session_state:
    st.session_state.group_a_ids = ""
if "group_b_ids" not in st.session_state:
    st.session_state.group_b_ids = ""


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
# タブ⑤：ポートフォリオ分析（Explorer / Saturn V / CORE）
# ============================================================
with tab5:
    st.caption(
        "このタブは、タブ「🔦 まとめて検索」で構築したデータベースを再利用します。"
        "先にそちらでデータベースを作ってから使ってください。"
    )

    if st.session_state.patent_db is None:
        st.info("先に「🔦 まとめて検索」タブでデータベースを構築してください。")
    else:
        db = st.session_state.patent_db
        all_ids = [e["id"] for e in db]
        st.success(f"✅ {len(db)} 件のデータベースを利用します。")

        st.markdown("#### 🏷️ グループ分け（自社 / 競合など）")
        st.caption("id をカンマまたは改行区切りで入力してください。両方に入れなかったidは自動的にグループBに含まれません。")
        col_a, col_b = st.columns(2)
        with col_a:
            group_a_text = st.text_area("グループA（例：自社）のid", value=st.session_state.group_a_ids, height=80, key="group_a_input")
        with col_b:
            group_b_text = st.text_area("グループB（例：競合）のid", value=st.session_state.group_b_ids, height=80, key="group_b_input")

        def _parse_ids(text, fallback):
            ids = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]
            return ids if ids else fallback

        group_a_ids = _parse_ids(group_a_text, all_ids[: len(all_ids) // 2] or all_ids)
        group_b_ids = _parse_ids(group_b_text, [i for i in all_ids if i not in group_a_ids])

        sub_tab_explorer, sub_tab_saturn, sub_tab_core = st.tabs(["🔍 Explorer（キーワード探査）", "🪐 Saturn V（意味的俯瞰マップ）", "🧬 CORE（論理式分類）"])

        # --- Explorer ---
        with sub_tab_explorer:
            kind = st.radio("比較するキーワードの種類", ["構成要素＋動詞", "構成要素のみ", "動詞のみ"], horizontal=True, key="explorer_kind")
            kind_map = {"構成要素＋動詞": "both", "構成要素のみ": "component", "動詞のみ": "verb"}

            if st.button("🔍 比較する", key="explorer_run"):
                result = pp.compare_keyword_groups(db, group_a_ids, group_b_ids, kind=kind_map[kind])
                st.session_state.explorer_result = result

            if st.session_state.get("explorer_result") is not None:
                result = st.session_state.explorer_result
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**グループAのワードクラウド**")
                    fig = pp.plot_wordcloud(result["freq_a"], title="グループA")
                    st.pyplot(fig)
                with col2:
                    st.markdown("**グループBのワードクラウド**")
                    fig = pp.plot_wordcloud(result["freq_b"], title="グループB")
                    st.pyplot(fig)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**共通する語（{len(result['common'])}件）**")
                    st.dataframe([{"語": w, "A": a, "B": b} for w, a, b in result["common"]], hide_index=True, use_container_width=True)
                with col2:
                    st.markdown(f"**Aだけの語（{len(result['only_a'])}件）**")
                    st.dataframe([{"語": w, "件数": f} for w, f in result["only_a"]], hide_index=True, use_container_width=True)
                with col3:
                    st.markdown(f"**Bだけの語（{len(result['only_b'])}件）**")
                    st.dataframe([{"語": w, "件数": f} for w, f in result["only_b"]], hide_index=True, use_container_width=True)

        # --- Saturn V ---
        with sub_tab_saturn:
            if st.button("🪐 マップを作成する", key="saturn_run"):
                try:
                    points, var = pp.build_semantic_map(db)
                    groups = {i: "グループA" for i in group_a_ids}
                    groups.update({i: "グループB" for i in group_b_ids})
                    st.session_state.saturn_result = (points, var, groups)
                except Exception as e:
                    st.error(f"マップ作成中にエラーが発生しました: {e}")

            if st.session_state.get("saturn_result") is not None:
                points, var, groups = st.session_state.saturn_result
                fig = pp.plot_semantic_map(points, var, groups=groups, title="意味的俯瞰マップ")
                st.pyplot(fig)
                st.caption("意味的に近い特許同士が近くに配置されます。距離が近いほど、内容が似ていることを示します。")

        # --- CORE ---
        with sub_tab_core:
            st.caption(
                "各カテゴリを「カテゴリ名: キーワード1, キーワード2」の形で、1行に1カテゴリずつ入力してください。"
                "そのカテゴリに割り当てるには、キーワードのうち1つでも含まれていればOKです。"
            )
            col1, col2 = st.columns(2)
            with col1:
                axis1_text = st.text_area(
                    "縦軸のカテゴリ",
                    height=150,
                    placeholder="電気系: 電極, 温度センサ, 半導体層\n機構系: 車輪, アーム, ハンドル",
                    key="axis1_text",
                )
            with col2:
                axis2_text = st.text_area(
                    "横軸のカテゴリ",
                    height=150,
                    placeholder="接続あり: 接続\n取り付けあり: 取り付け",
                    key="axis2_text",
                )

            def _parse_axis(text):
                axis = {}
                for line in text.strip().split("\n"):
                    if ":" not in line:
                        continue
                    name, kws = line.split(":", 1)
                    keywords = [k.strip() for k in kws.split(",") if k.strip()]
                    if name.strip() and keywords:
                        axis[name.strip()] = {"any_of": keywords}
                return axis

            if st.button("🧬 分類する", key="core_run"):
                axis1_formulas = _parse_axis(axis1_text)
                axis2_formulas = _parse_axis(axis2_text)
                if not axis1_formulas or not axis2_formulas:
                    st.warning("縦軸・横軸の両方に、少なくとも1つのカテゴリを定義してください。")
                else:
                    matrix = pp.classify_patents(db, axis1_formulas, axis2_formulas)
                    st.session_state.core_result = (matrix, list(axis1_formulas.keys()) + ["(未分類)"], list(axis2_formulas.keys()) + ["(未分類)"])

            if st.session_state.get("core_result") is not None:
                matrix, axis1_names, axis2_names = st.session_state.core_result
                fig = pp.plot_classification_heatmap(matrix, axis1_names, axis2_names, title="論理式分類ヒートマップ")
                st.pyplot(fig)
                st.caption("枠で囲まれた0件のマスが、まだ組み合わせのない技術（ホワイトスペース）の候補です。")
