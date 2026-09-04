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
    "🪸 1つの請求項を解析", "🐚 2つの請求項を比較",
    "🪼 従属請求項を展開", "🔭 ポートフォリオ分析", "📋 記載チェック",
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

        confidence = pp.assess_claim_confidence(text)
        level_emoji = {"高": "📗", "中": "📙", "低": "📕"}
        st.markdown(f"#### {level_emoji[confidence['level']]} 信頼度：{confidence['level']}")
        if confidence["reasons"]:
            st.caption("以下のパターンが含まれているため、念のため本文と照らし合わせることをおすすめします。")
            for r in confidence["reasons"]:
                st.write(f"- {r}")
        else:
            st.caption("既知の弱点パターンには当てはまりませんでした。抽出結果をそのまま信頼しやすい請求項です。")

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
# タブ③：従属請求項を、親請求項の内容も含めて完全な形に展開する
# ============================================================
with tab3:
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
# タブ④：ポートフォリオ分析（Explorer / Saturn V / CORE）
# ============================================================
with tab4:
    st.caption(
        "このタブでは、「要約」データベース、または「フルメタデータCSV」のどちらかを使えます。"
        "どちらもJ-PlatPat等で一括ダウンロードしやすいため、手作業なしで大量の分析ができます。"
    )

    db_source = st.radio(
        "使うデータベース",
        [
            "要約データベース（ここで新しく作る）",
            "フルメタデータCSV（出願日・出願人・FI等、ATLAS用）",
            "請求項＋メタデータCSV（出願人・FI等、ホワイトスペース分析用）",
        ],
        key="db_source_choice",
    )

    if db_source.startswith("請求項＋"):
        st.markdown("#### 📜 請求項＋メタデータCSVを登録する")
        st.caption(
            "列名: id（省略可）, 出願人, FI（またはFI/IPC）, 請求項本文, "
            "発明の名称（任意）, 出願日（任意）, 特許番号（任意）　を含むCSVをアップロードしてください。"
        )
        uploaded_claims_csv = st.file_uploader("CSVファイル", type=["csv"], key="claims_csv")
        if st.button("📜 データベースを構築する", key="build_claims_db_run"):
            if uploaded_claims_csv is None:
                st.warning("CSVファイルをアップロードしてください。")
            else:
                with st.spinner("請求項をSAO解析中..."):
                    try:
                        content = uploaded_claims_csv.getvalue().decode("utf-8-sig")
                        claims_records = pp.load_claims_with_metadata_csv(content)
                        st.session_state.abstract_db = pp.build_claims_metadata_database(claims_records, show_progress=False)
                    except Exception as e:
                        st.error(f"データベース構築中にエラーが発生しました: {e}")
                        st.session_state.abstract_db = None
        if st.session_state.abstract_db is not None:
            st.success(f"✅ {len(st.session_state.abstract_db)} 件を登録済みです。")
        db = st.session_state.abstract_db

    elif db_source.startswith("フルメタデータ"):
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

    else:
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

    if db is None:
        pass
    else:
        all_ids = [e["id"] for e in db]
        st.success(f"✅ {len(db)} 件のデータベースを利用します。")

        has_applicant_field = any(e.get("出願人") for e in db)
        if has_applicant_field:
            with st.expander("🏷️ 出願人の名寄せ（グループ会社をまとめる）"):
                st.caption(
                    "「東芝デバイス＆ストレージ株式会社」のような子会社・関連会社を、"
                    "「東芝」のような親会社名にまとめます。カンマ区切りで、まとめたい語を指定してください。"
                )
                norm_keywords_text = st.text_input(
                    "まとめる語（カンマ区切り）",
                    value=", ".join(pp.DEFAULT_APPLICANT_GROUP_KEYWORDS),
                    key="norm_keywords",
                )
                if st.button("🏷️ 名寄せを適用する", key="apply_norm_run"):
                    keywords = [k.strip() for k in norm_keywords_text.split(",") if k.strip()]
                    db = pp.apply_applicant_normalization(db, group_keywords=keywords)
                    st.session_state.abstract_db = db
                    st.success("名寄せを適用しました。")

        sub_tab_atlas, sub_tab_explorer, sub_tab_saturn, sub_tab_core, sub_tab_rank = st.tabs([
            "📊 基礎統計",
            "🔍 キーワード分析", "🗺️ 類似度マップ",
            "🧩 分類マップ", "🏢 出願人比較",
        ])

        with sub_tab_atlas:
            st.caption("出願件数の推移、出願人ランキング、FIランキングなど、基本的な統計を確認できます。")
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

        # --- Explorer ---
        with sub_tab_explorer:
            st.caption("CSV全体をまとめたワードクラウド、または出願人ごとのワードクラウドを表示します。")
            kind = st.radio("対象にするキーワードの種類", ["構成要素＋動詞", "構成要素のみ", "動詞のみ"], horizontal=True, key="explorer_kind")
            kind_map = {"構成要素＋動詞": "both", "構成要素のみ": "component", "動詞のみ": "verb"}

            has_claim_field_ex = any("relations" in e for e in db)
            has_title_field_ex = any(e.get("発明の名称") for e in db)
            source_options_ex = ["自動"]
            if has_claim_field_ex:
                source_options_ex.append("請求項")
            if has_title_field_ex:
                source_options_ex.append("発明の名称")
            explorer_source = st.radio("キーワードの検索元", source_options_ex, horizontal=True, key="explorer_source")

            exclude_text = st.text_input("除外キーワード（カンマ区切り）", value="", key="explorer_exclude")
            exclude_list = [w.strip() for w in exclude_text.split(",") if w.strip()]

            applicant_groups = pp.get_applicant_groups(db, top_n=15)
            scope_options = ["全体"] + list(applicant_groups.keys())
            scope = st.selectbox("対象範囲", scope_options, key="explorer_scope")

            if st.button("🐠 解析する", key="explorer_run"):
                ids = applicant_groups[scope] if scope != "全体" else None
                freq = pp.build_keyword_frequency(
                    db, ids=ids, kind=kind_map[kind], source=explorer_source, exclude=exclude_list
                )
                st.session_state.explorer_result = (scope, freq)

            if st.session_state.get("explorer_result") is not None:
                scope_used, freq_all = st.session_state.explorer_result
                st.markdown(f"**{scope_used}のワードクラウド**")
                fig = pp.plot_wordcloud(freq_all, title=f"キーワード頻度（{scope_used}）")
                st.pyplot(fig)
                st.dataframe(
                    [{"語": w, "件数": f} for w, f in freq_all.most_common(30)],
                    hide_index=True, width='stretch',
                )

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

        # --- Saturn V ---
        with sub_tab_saturn:
            st.caption("特許同士の意味的な近さに基づいて地図上に配置します。似た内容の発明ほど近くに配置されます。")

            saturn_method_choice = st.radio(
                "次元圧縮の方法",
                ["PCA（軸に寄与率という意味を持たせられる）", "UMAP（クラスタ構造をよりはっきり分離しやすい）"],
                horizontal=True, key="saturn_method_choice",
            )
            saturn_method = "umap" if saturn_method_choice.startswith("UMAP") else "pca"

            if saturn_method == "umap":
                umap_col1, umap_col2 = st.columns(2)
                with umap_col1:
                    saturn_n_neighbors = st.slider(
                        "n_neighbors（近傍点の数。小さいほど局所的、大きいほど大域的な構造を重視）",
                        min_value=2, max_value=50, value=15, key="saturn_n_neighbors",
                    )
                with umap_col2:
                    saturn_min_dist = st.slider(
                        "min_dist（点同士に許容する最小距離。小さいほど密集したクラスタになる）",
                        min_value=0.0, max_value=0.99, value=0.1, step=0.01, key="saturn_min_dist",
                    )
            else:
                saturn_n_neighbors, saturn_min_dist = 15, 0.1

            if st.button("🐋 マップを作成する", key="saturn_run"):
                try:
                    points, var = pp.build_semantic_map(
                        db, method=saturn_method,
                        n_neighbors=saturn_n_neighbors, min_dist=saturn_min_dist,
                    )
                    applicant_groups = pp.get_applicant_groups(db, top_n=6)
                    groups = None
                    if applicant_groups:
                        groups = {}
                        for name, ids in applicant_groups.items():
                            for i in ids:
                                groups[i] = name
                    st.session_state.saturn_result = (points, var, groups, saturn_method)
                except Exception as e:
                    st.error(f"マップ作成中にエラーが発生しました: {e}")

            if st.session_state.get("saturn_result") is not None:
                points, var, groups, used_method = st.session_state.saturn_result
                map_title = "意味的俯瞰マップ（UMAP）" if used_method == "umap" else "意味的俯瞰マップ（PCA）"
                fig = pp.plot_semantic_map_interactive(points, var, groups=groups, title=map_title, method=used_method)
                st.plotly_chart(fig, width='content')
                st.caption("意味的に近い特許同士が近くに配置されます。点にカーソルを合わせると文献番号が表示されます。")

            st.divider()
            st.markdown("#### 🕸️ 類似度ネットワーク図")
            st.caption(
                "特許同士のコサイン類似度が閾値を超えたペアを線で結びます。"
                "上のマップが「全体としての配置・クラスタ傾向」を見るのに向いているのに対し、"
                "こちらは「どの特許とどの特許が具体的に似ているか」を直接確認するのに向いています。"
            )
            net_col1, net_col2 = st.columns(2)
            with net_col1:
                network_threshold = st.slider(
                    "類似度の閾値（これ以上似ていたら線を引く）",
                    min_value=0.50, max_value=0.99, value=0.75, step=0.01, key="network_threshold",
                )
            with net_col2:
                network_max_neighbors = st.slider(
                    "1件あたりの最大エッジ数",
                    min_value=1, max_value=20, value=5, key="network_max_neighbors",
                )
            if st.button("🕸️ ネットワーク図を作る", key="network_run"):
                try:
                    G = pp.build_similarity_network(
                        db, threshold=network_threshold, max_neighbors=network_max_neighbors,
                    )
                    applicant_groups = pp.get_applicant_groups(db, top_n=6)
                    groups = None
                    if applicant_groups:
                        groups = {}
                        for name, ids in applicant_groups.items():
                            for i in ids:
                                groups[i] = name
                    st.session_state.network_result = (G, groups)
                except Exception as e:
                    st.error(f"ネットワーク図作成中にエラーが発生しました: {e}")

            if st.session_state.get("network_result") is not None:
                G, groups = st.session_state.network_result
                if G.number_of_nodes() == 0:
                    st.warning("閾値を超えるペアが見つかりませんでした。閾値を下げてみてください。")
                else:
                    fig = pp.plot_similarity_network_interactive(G, groups=groups, title="特許類似度ネットワーク図")
                    st.plotly_chart(fig, width='content')
                    st.caption(
                        f"ノード数: {G.number_of_nodes()} / エッジ数: {G.number_of_edges()}"
                        "　※ ノードが大きいほど、似ている特許が多く繋がっています。"
                    )

        # --- CORE ---
        with sub_tab_core:
            st.caption("キーワードとFIサブクラスのマス目に特許を分類します。件数が0のマスが、まだ誰も出願していないホワイトスペース候補です。")
            st.markdown("#### 🐚 自動ホワイトスペースマップ")
            has_title_field = any(e.get("発明の名称") for e in db)
            has_claim_field = any("relations" in e for e in db)
            source_options = []
            if has_claim_field:
                source_options.append("請求項")
            if has_title_field:
                source_options.append("発明の名称")
            if not source_options:
                st.info("このデータベースには「発明の名称」も請求項もありません。「フルメタデータCSV」または「請求項＋メタデータCSV」で構築してください。")
            else:
                kwfi_source = st.radio("キーワードの抽出元", source_options, horizontal=True, key="kwfi_source")
                applicant_options = sorted({a for e in db for a in (e.get("出願人") or [])})
                kwfi_applicants = st.multiselect("出願人で絞り込む（空欄なら全件）", applicant_options, key="kwfi_applicants")
                col1, col2 = st.columns(2)
                with col1:
                    n_keywords = st.slider("縦軸のキーワード数", min_value=5, max_value=50, value=20, key="kwfi_n_keywords")
                with col2:
                    n_fi = st.slider("横軸のFI数", min_value=5, max_value=50, value=20, key="kwfi_n_fi")
                kwfi_fi_level = st.radio("FIの粒度", ["サブクラス", "メイングループ", "そのまま"], horizontal=True, key="kwfi_fi_level")
                if st.button("🐚 自動でマップを作る", key="kwfi_run"):
                    try:
                        matrix_kw, kw_list, fi_list = pp.build_keyword_fi_matrix(
                            db, top_keywords=n_keywords, top_fi=n_fi, fi_level=kwfi_fi_level,
                            source=kwfi_source, applicant_filter=kwfi_applicants or None,
                        )
                        st.session_state.kwfi_result = (matrix_kw, kw_list, fi_list)
                    except Exception as e:
                        st.error(f"マップ作成中にエラーが発生しました: {e}")
                if st.session_state.get("kwfi_result") is not None:
                    matrix_kw, kw_list, fi_list = st.session_state.kwfi_result
                    fig = pp.plot_keyword_fi_heatmap(matrix_kw, kw_list, fi_list, title=f"{kwfi_source}×FI ホワイトスペースマップ")
                    st.pyplot(fig)
                    st.caption("色が濃いマスほど出願件数が多く、白いマスがホワイトスペース候補です。")

            st.divider()
            st.markdown("#### 🔮 キーワード→FI推薦（母集団作成支援）")
            st.caption(
                "入力したキーワードを含む特許を実際のデータから検索し、その特許群でよく使われている"
                "FIを出現率つきで推薦します。J-PlatPatで検索条件（FI）を決める際の参考にしてください。"
            )
            fi_reco_keywords_text = st.text_input("キーワード（カンマ区切りで複数指定できます）", value="", key="fi_reco_keywords")
            fi_reco_match_mode = st.radio(
                "複数キーワードの扱い", ["いずれか（OR）", "すべて（AND）"], horizontal=True, key="fi_reco_match_mode"
            )
            fi_reco_fi_level = st.radio("FIの粒度", ["サブクラス", "メイングループ", "そのまま"], horizontal=True, key="fi_reco_fi_level")
            if st.button("🔮 FIを推薦する", key="fi_reco_run"):
                kws = [k.strip() for k in fi_reco_keywords_text.split(",") if k.strip()]
                if not kws:
                    st.warning("キーワードを1つ以上入力してください。")
                else:
                    result = pp.recommend_fi_for_keywords(
                        db, kws,
                        match_mode="すべて" if fi_reco_match_mode.startswith("すべて") else "いずれか",
                        fi_level=fi_reco_fi_level,
                    )
                    st.session_state.fi_reco_result = result
            if st.session_state.get("fi_reco_result") is not None:
                result = st.session_state.fi_reco_result
                if result["matched_count"] == 0:
                    st.warning("指定したキーワードを含む特許が見つかりませんでした。")
                else:
                    st.caption(f"キーワードにマッチした特許: {result['matched_count']}件")
                    fig = pp.plot_fi_recommendations(result, title="キーワード→FI推薦")
                    st.pyplot(fig)
                    st.dataframe(
                        [
                            {
                                "FI": r["FI"], "件数": r["件数"],
                                "出現率": f'{r["スコア"]*100:.1f}%',
                                "サンプルid": "、".join(r["サンプルid"]),
                            }
                            for r in result["recommendations"]
                        ],
                        hide_index=True, width='stretch',
                    )

        # --- 構成部位ランキング・件数分布・レーダーチャート ---
        with sub_tab_rank:
            st.caption("構成要素・動詞のランキング、出願人ごとの特徴比較（FIレーダーチャート）、動態分析（MEGA）を確認できます。")
            st.markdown("#### 🦑 構成部位ランキング（全体）")
            rank_kind = st.radio("ランキングの対象", ["構成要素", "動詞"], horizontal=True, key="rank_kind")
            if st.button("🦑 ランキングを作る", key="rank_run"):
                ranking = pp.rank_components(db, kind="component" if rank_kind == "構成要素" else "verb", top_n=20)
                st.session_state.rank_result = ranking
            if st.session_state.get("rank_result") is not None:
                fig = pp.plot_component_ranking(st.session_state.rank_result, title=f"{rank_kind}ランキング")
                st.pyplot(fig)

            st.divider()

            st.markdown("#### 🕸️ 出願人ごとのFIレーダーチャート比較（自動）")
            st.caption("出願人ごとに、よく使うFI（サブクラス等）の件数を軸にして比較します。出願人情報が必要です。")
            radar_fi_level = st.radio("FIの粒度", ["サブクラス", "メイングループ", "そのまま"], horizontal=True, key="rank_radar_fi_level")
            axis_selection_2 = st.radio("軸の選び方", ["複数社共通（おすすめ）", "全体件数順"], horizontal=True, key="rank_radar_axis_selection")
            col1, col2 = st.columns(2)
            with col1:
                radar_top_applicants = st.slider("対象にする出願人数", min_value=2, max_value=10, value=5, key="rank_radar_applicants")
            with col2:
                radar_top_fi = st.slider("軸にするFIの数", min_value=3, max_value=30, value=8, key="rank_radar_fi_n")
            if st.button("🕸️ レーダーチャートを作る", key="radar_run"):
                profiles = pp.build_applicant_fi_radar_data(
                    db, fi_level=radar_fi_level, top_applicants=radar_top_applicants, top_fi=radar_top_fi,
                    axis_selection="複数社共通" if axis_selection_2.startswith("複数社") else "全体件数順",
                )
                if not profiles:
                    st.warning("出願人情報が見つかりませんでした。フルメタデータCSVを使ってください。")
                else:
                    st.session_state.radar_result = profiles
            if st.session_state.get("radar_result") is not None:
                fig = pp.plot_radar_chart(st.session_state.radar_result, title=f"出願人×FI（{radar_fi_level}）")
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
                    mega_data, latest_year, used_recent, used_compare = pp.compute_activity_momentum(
                        db, group_by=mega_group_by, recent_years=mega_recent, compare_years=mega_compare
                    )
                    if not mega_data:
                        st.warning("出願日のデータが見つかりませんでした。")
                    else:
                        st.session_state.mega_result = (mega_data, latest_year, used_recent, used_compare)
                except Exception as e:
                    st.error(f"MEGA診断中にエラーが発生しました: {e}")
            if st.session_state.get("mega_result") is not None:
                mega_data, latest_year, used_recent, used_compare = st.session_state.mega_result
                st.caption(f"最新の出願年（{latest_year}年）を基準に計算しています。")
                if used_recent != mega_recent or used_compare != mega_compare:
                    st.caption(
                        f"⚠️ データの年範囲が指定した期間より狭かったため、"
                        f"直近{used_recent}年・比較{used_compare}年に自動調整しました。"
                    )
                fig = pp.plot_mega_chart_interactive(mega_data, title=f"MEGA：{mega_group_by}別の活動量×勢い", top_n=mega_top_n)
                st.plotly_chart(fig, width='stretch')

            st.divider()

            st.markdown("#### 📋 登録率分析")
            st.caption("グループ（出願人・FI・キーワード）ごとに、どのくらい登録に成功しているかを比較します。")
            reg_group_by = st.radio("グループの単位", ["出願人", "FI", "キーワード"], horizontal=True, key="reg_group_by")
            reg_fi_level = st.radio("FIの粒度（FI選択時のみ）", ["サブクラス", "メイングループ", "そのまま"], horizontal=True, key="reg_fi_level")
            if st.button("📋 登録率を見る", key="reg_run"):
                st.session_state.reg_result = pp.compute_registration_rate(db, group_by=reg_group_by, fi_level=reg_fi_level)
            if st.session_state.get("reg_result"):
                fig = pp.plot_registration_rate(st.session_state.reg_result, title=f"登録率（{reg_group_by}別）")
                st.pyplot(fig)

            st.divider()

            st.markdown("#### ⏱️ 権利化期間分析")
            st.caption("グループ（出願人・FI）ごとに、出願日から公知日までの期間（権利化にかかる期間の目安）を比較します。")
            ttp_group_by = st.radio("グループの単位", ["出願人", "FI"], horizontal=True, key="ttp_group_by")
            ttp_fi_level = st.radio("FIの粒度（FI選択時のみ）", ["サブクラス", "メイングループ", "そのまま"], horizontal=True, key="ttp_fi_level")
            if st.button("⏱️ 期間を見る", key="ttp_run"):
                st.session_state.ttp_result = pp.compute_time_to_publication(db, group_by=ttp_group_by, fi_level=ttp_fi_level)
            if st.session_state.get("ttp_result"):
                fig = pp.plot_time_to_publication(st.session_state.ttp_result, title=f"出願から公知までの期間（{ttp_group_by}別）")
                st.pyplot(fig)

            st.divider()

            st.markdown("#### 🏁 技術の「先願者」年表")
            st.caption("発明の名称のキーワード×FIの組み合わせごとに、最初に出願したのが誰・いつだったかを一覧にします。")
            has_title_field = any(e.get("発明の名称") for e in db)
            if not has_title_field:
                st.info("このデータベースには「発明の名称」がありません。「フルメタデータCSV」で構築してください。")
            else:
                first_filer_fi_level = st.radio("FIの粒度", ["サブクラス", "メイングループ", "そのまま"], horizontal=True, key="first_filer_fi_level")
                if st.button("🏁 年表を作る", key="first_filer_run"):
                    st.session_state.first_filer_result = pp.find_first_filers(db, fi_level=first_filer_fi_level)
                if st.session_state.get("first_filer_result"):
                    rows = st.session_state.first_filer_result
                    st.dataframe(
                        [
                            {
                                "キーワード": r["キーワード"], "FI": r["FI"],
                                "最初の出願人": r["最初の出願人"],
                                "最初の出願日": r["最初の出願日"].strftime("%Y-%m-%d") if r["最初の出願日"] else "",
                                "件数": r["件数"],
                            }
                            for r in rows
                        ],
                        hide_index=True, width='stretch',
                    )


# ============================================================
# タブ⑤：記載チェック（「前記」整合性・明確性リスク）
# ============================================================
with tab5:
    st.subheader("📋 記載チェック")
    st.caption(
        "SAO抽出の精度とは別に、実際の知財実務でチェックされている観点（「前記」参照の整合性、"
        "明確性要件違反になりやすい表現）を自動で検出します。"
    )

    check_mode = st.radio("チェックの種類", ["「前記」整合性チェック", "明確性リスクチェック"], horizontal=True, key="check_mode")

    if check_mode == "「前記」整合性チェック":
        st.markdown("#### 「前記Ｘ」「該Ｘ」が、従属先に一度も登場していないかを確認します")
        st.caption(
            "複数の請求項を「【請求項１】〜」の形式で貼り付けてください。"
            "指定した請求項番号について、その従属先（さらにその従属先）を遡って「前記」の整合性を確認します。"
        )
        claims_block_text = st.text_area(
            "請求項全文（【請求項１】〜の形式）",
            height=220,
            placeholder="【請求項1】\n…を備えることを特徴とする多機能ペン。\n【請求項2】\n前記…は、…であることを特徴とする請求項1に記載の多機能ペン。",
            key="zenki_claims_text",
        )
        target_claim_num = st.number_input("チェックしたい請求項番号", min_value=1, value=1, step=1, key="zenki_target")
        if st.button("📋 チェックする", key="zenki_check_run"):
            if not claims_block_text.strip():
                st.warning("請求項全文を入力してください。")
            else:
                try:
                    claims_dict = pp.parse_claims_block(claims_block_text)
                    if target_claim_num not in claims_dict:
                        st.error(f"請求項{target_claim_num}が見つかりませんでした。認識した請求項番号: {sorted(claims_dict.keys())}")
                    else:
                        warnings = pp.check_zenki_consistency(target_claim_num, claims_dict)
                        st.session_state.zenki_result = warnings
                except Exception as e:
                    st.error(f"チェック中にエラーが発生しました: {e}")
        if st.session_state.get("zenki_result") is not None:
            warnings = st.session_state.zenki_result
            if not warnings:
                st.success("✅ 「前記」の不整合は見つかりませんでした。")
            else:
                st.warning(f"⚠️ {len(warnings)} 件の疑わしい箇所が見つかりました。")
                for w in warnings:
                    st.write(f"- 請求項{w['claim_number']}で「前記{w['term']}」（または「該{w['term']}」）と記載されていますが、"
                             f"「{w['term']}」がそれより前（従属先を含む）に登場していません。")

    else:
        st.markdown("#### 明確性要件違反になりやすい表現を検出します")
        clarity_text = st.text_area(
            "請求項テキスト",
            height=180,
            placeholder="例：所定の条件を満たす場合に、適切な処理を行う制御手段を備える装置。",
            key="clarity_text",
        )
        if st.button("📋 チェックする", key="clarity_check_run"):
            if not clarity_text.strip():
                st.warning("請求項テキストを入力してください。")
            else:
                st.session_state.clarity_result = pp.check_clarity_risks(clarity_text)
        if st.session_state.get("clarity_result") is not None:
            findings = st.session_state.clarity_result
            if not findings:
                st.success("✅ 既知のリスクパターンは見つかりませんでした。")
            else:
                st.warning(f"⚠️ {len(findings)} 件の注意点が見つかりました。")
                for f in findings:
                    st.write(f"- **「{f['phrase']}」**：{f['reason']}")
