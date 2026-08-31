import streamlit as st
import matplotlib.pyplot as plt

import patent_pipeline as pp

st.set_page_config(page_title="🪼 特許SAOラボ", page_icon="🪼", layout="wide")

st.markdown(
    """
    <style>
    /* ============================================================
       深海テーマ：背景（光芒＋グラデーション）
    ============================================================ */
    .stApp {
        background:
            radial-gradient(ellipse 900px 500px at 50% -8%, rgba(150, 230, 255, 0.30), transparent 60%),
            radial-gradient(ellipse 500px 300px at 15% 15%, rgba(150, 230, 255, 0.12), transparent 55%),
            radial-gradient(circle at 85% 30%, rgba(199, 141, 255, 0.10), transparent 45%),
            radial-gradient(circle at 25% 85%, rgba(100, 255, 218, 0.08), transparent 45%),
            linear-gradient(180deg, #06304d 0%, #073a5c 18%, #052b47 40%, #03203a 62%, #01152a 82%, #000c1c 100%);
        background-attachment: fixed;
        position: relative;
        overflow-x: hidden;
    }

    /* 漂う泡 */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background-image:
            radial-gradient(circle, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.05) 60%, transparent 70%),
            radial-gradient(circle, rgba(255,255,255,0.45) 0%, rgba(255,255,255,0.04) 60%, transparent 70%),
            radial-gradient(circle, rgba(255,255,255,0.5) 0%, rgba(255,255,255,0.05) 60%, transparent 70%),
            radial-gradient(circle, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0.04) 60%, transparent 70%),
            radial-gradient(circle, rgba(255,255,255,0.45) 0%, rgba(255,255,255,0.04) 60%, transparent 70%),
            radial-gradient(circle, rgba(255,255,255,0.5) 0%, rgba(255,255,255,0.05) 60%, transparent 70%);
        background-size: 18px 18px, 26px 26px, 14px 14px, 22px 22px, 16px 16px, 30px 30px;
        background-position: 8% 100%, 22% 100%, 40% 100%, 63% 100%, 78% 100%, 92% 100%;
        background-repeat: no-repeat;
        animation: bubble-rise 9s linear infinite;
        opacity: 0.8;
    }
    @keyframes bubble-rise {
        0%   { background-position: 8% 105%, 22% 110%, 40% 100%, 63% 115%, 78% 108%, 92% 100%; opacity: 0; }
        10%  { opacity: 0.9; }
        90%  { opacity: 0.5; }
        100% { background-position: 6% -20%, 26% -25%, 44% -15%, 60% -30%, 82% -18%, 88% -22%; opacity: 0; }
    }

    /* 漂うクラゲ・お魚デコレーション */
    .sea-critters { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
    .sea-critters span {
        position: absolute;
        font-size: 1.8rem;
        opacity: 0.55;
        filter: drop-shadow(0 0 6px rgba(120, 220, 255, 0.5));
        animation: drift 22s ease-in-out infinite;
    }
    .sea-critters span:nth-child(1) { left: 6%;  top: 70%; font-size: 2.1rem; animation-duration: 26s; animation-delay: 0s; }
    .sea-critters span:nth-child(2) { left: 88%; top: 20%; font-size: 1.6rem; animation-duration: 20s; animation-delay: 2s; }
    .sea-critters span:nth-child(3) { left: 15%; top: 30%; font-size: 1.4rem; animation-duration: 24s; animation-delay: 4s; }
    .sea-critters span:nth-child(4) { left: 75%; top: 78%; font-size: 1.9rem; animation-duration: 28s; animation-delay: 1s; }
    .sea-critters span:nth-child(5) { left: 45%; top: 12%; font-size: 1.3rem; animation-duration: 18s; animation-delay: 6s; }
    @keyframes drift {
        0%   { transform: translate(0, 0) rotate(0deg); }
        50%  { transform: translate(24px, -30px) rotate(6deg); }
        100% { transform: translate(0, 0) rotate(0deg); }
    }

    /* 全体の文字色（深海の中で読みやすいライトカラーに） */
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stCaption {
        color: #dff3ff;
    }
    h1, h2, h3, h4 {
        color: #baf3ff !important;
        text-shadow: 0 0 14px rgba(120, 230, 255, 0.45);
        font-weight: 800 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #9fd4e8 !important;
    }

    /* タブ（貝殻っぽいガラスピル） */
    button[data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 0.4rem 1.2rem !important;
        margin-right: 0.4rem !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(150, 230, 255, 0.25) !important;
        color: #cdeeff !important;
        backdrop-filter: blur(6px);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(100,255,218,0.22), rgba(199,141,255,0.22)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(120, 230, 255, 0.6) !important;
        font-weight: 700 !important;
        box-shadow: 0 0 14px rgba(100, 255, 218, 0.35);
    }

    /* ボタン（発光クラゲ風） */
    .stButton > button {
        border-radius: 999px !important;
        background: linear-gradient(135deg, #64ffda 0%, #4fb3ff 45%, #c77dff 100%) !important;
        color: #02233a !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 0.5rem 1.6rem !important;
        box-shadow: 0 0 18px rgba(100, 255, 218, 0.45), 0 3px 10px rgba(0,0,0,0.35) !important;
    }
    .stButton > button:hover {
        filter: brightness(1.08);
        box-shadow: 0 0 26px rgba(100, 255, 218, 0.65), 0 3px 10px rgba(0,0,0,0.35) !important;
    }

    /* テキストエリア（水面のガラスカード） */
    .stTextArea textarea {
        border-radius: 18px !important;
        border: 1.5px solid rgba(150, 230, 255, 0.35) !important;
        background: rgba(6, 30, 48, 0.55) !important;
        color: #eaf9ff !important;
        backdrop-filter: blur(8px);
    }
    .stTextArea textarea::placeholder {
        color: #7fb8d4 !important;
    }
    .stTextInput input, .stTextArea label, .stTextInput label {
        color: #dff3ff !important;
    }

    /* metricカード（真珠貝カード） */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 0.9rem;
        border: 1px solid rgba(150, 230, 255, 0.25);
        box-shadow: 0 0 16px rgba(80, 200, 255, 0.15);
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] {
        color: #eaf9ff !important;
    }

    /* expander */
    .streamlit-expanderHeader {
        border-radius: 14px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        color: #dff3ff !important;
        border: 1px solid rgba(150, 230, 255, 0.2) !important;
    }
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 0 0 14px 14px !important;
    }

    /* checkbox・警告・成功・情報ボックス */
    .stCheckbox label { color: #dff3ff !important; }
    div[data-testid="stAlert"] {
        border-radius: 16px !important;
        backdrop-filter: blur(6px);
    }

    /* dataframe をガラスカードで囲む */
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(150, 230, 255, 0.25);
        box-shadow: 0 0 16px rgba(80, 200, 255, 0.12);
    }

    /* 関係図（matplotlib）は白背景のまま、深海に浮かぶ宝物カード風に額装 */
    div[data-testid="stImage"] {
        background: rgba(255, 250, 245, 0.97);
        border-radius: 22px;
        padding: 1rem;
        border: 1px solid rgba(150, 230, 255, 0.35);
        box-shadow: 0 0 24px rgba(80, 200, 255, 0.25), 0 6px 18px rgba(0,0,0,0.35);
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
    }
    </style>

    <div class="sea-critters">
        <span>🪼</span>
        <span>🐠</span>
        <span>🐙</span>
        <span>🫧</span>
        <span>🐡</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🪼 特許SAOラボ")
st.caption("GiNZAで日本語特許請求項を「主語・動詞・目的語」に分解して、構成要素の関係を可視化します🌊")

tab1, tab2 = st.tabs(["🪸 1つの請求項を解析", "🐚 2つの請求項を比較"])


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
        else:
            with st.spinner("解析中..."):
                try:
                    components, relations = pp.analyze_claim(text)
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {e}")
                    components, relations = None, None

            if relations is not None:
                if not relations:
                    st.info("関係が抽出できませんでした。文の書き方を見直してみてください。")
                else:
                    st.success(f"🎉 {len(relations)} 件の関係を抽出しました！")

                    col1, col2 = st.columns([3, 2])

                    with col1:
                        st.markdown("#### 🪸 構成要素間の関係図")
                        pp.visualize_relations(relations, title="構成要素間関係")
                        fig = plt.gcf()
                        st.pyplot(fig)
                        plt.close(fig)

                    with col2:
                        st.markdown("#### 📋 抽出された関係（SAOトリプル）")
                        st.dataframe(
                            [
                                {"主語": r["source"], "関係": r["relation"],
                                 "目的語": r["target"], "種類": r["type"]}
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
        else:
            with st.spinner("解析中..."):
                try:
                    _, relations_a = pp.analyze_claim(text_a)
                    _, relations_b = pp.analyze_claim(text_b)
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {e}")
                    relations_a, relations_b = None, None

            if relations_a is not None and relations_b is not None:
                jaccard_score, common, only_a, only_b = pp.jaccard_similarity(relations_a, relations_b)
                structural_score, structural_detail = pp.structural_similarity(relations_a, relations_b)

                semantic_score = None
                semantic_matches = None
                if use_semantic:
                    with st.spinner("意味マッチングのモデルを読み込み中..."):
                        try:
                            semantic_score, semantic_matches = pp.semantic_similarity(relations_a, relations_b)
                        except Exception as e:
                            st.error(f"意味マッチングでエラーが発生しました: {e}")

                st.markdown("### 🦪 診断結果")
                score_cols = st.columns(3)
                score_cols[0].metric("①Jaccard類似度（表記の一致）", f"{jaccard_score:.3f}")
                if semantic_score is not None:
                    score_cols[1].metric("②意味マッチング類似度", f"{semantic_score:.3f}")
                else:
                    score_cols[1].metric("②意味マッチング類似度", "―（未使用）")
                score_cols[2].metric("③構造の類似度", f"{structural_score:.3f}")

                with st.expander("③構造比較の内訳を見る"):
                    st.write(
                        f"- 深さの類似度: {structural_detail['深さの類似度']:.3f}\n"
                        f"- 規模(ノード数)の類似度: {structural_detail['規模(ノード数)の類似度']:.3f}\n"
                        f"- 枝分かれパターンの類似度: {structural_detail['枝分かれパターンの類似度']:.3f}\n"
                        f"- 関係の種類の内訳の類似度: {structural_detail['関係の種類の内訳の類似度']:.3f}"
                    )

                st.markdown("### 🧩 ①Jaccard：トリプルの一致・不一致")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**共通トリプル（{len(common)}件）**")
                    for t in sorted(common):
                        st.write(t)
                with col2:
                    st.markdown(f"**Aだけにあるトリプル（{len(only_a)}件）**")
                    for t in sorted(only_a):
                        st.write(t)
                with col3:
                    st.markdown(f"**Bだけにあるトリプル（{len(only_b)}件）**")
                    for t in sorted(only_b):
                        st.write(t)

                if semantic_matches is not None:
                    st.markdown("### 🫧 ②意味マッチング：対応付けの詳細")
                    matches_sorted = sorted(semantic_matches, key=lambda x: -x[2])
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
