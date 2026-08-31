import streamlit as st
import matplotlib.pyplot as plt

import patent_pipeline as pp

st.set_page_config(page_title="🌊 特許SAOラボ", page_icon="🌊", layout="wide")

st.markdown(
    """
    <style>
    /* ============================================================
       深海テーマ（リアル志向）：基調カラー
       派手な虹色をやめ、単色系（黒〜濃紺〜ティール）の
       ドキュメンタリー的な深海の色調に統一する。
    ============================================================ */
    .stApp {
        background:
            radial-gradient(ellipse 1100px 500px at 50% -12%, rgba(90, 170, 200, 0.16), transparent 62%),
            linear-gradient(180deg, #08222f 0%, #071b28 14%, #051520 32%, #040f18 52%, #02090f 74%, #00050a 100%);
        background-attachment: fixed;
        position: relative;
        overflow-x: hidden;
    }

    /* ------------------------------------------------------------
       ゴッドレイ（水面から差し込む光芒）
    ------------------------------------------------------------ */
    .godrays { position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; }
    .godrays span {
        position: absolute;
        top: -20%;
        width: 140px;
        height: 140%;
        background: linear-gradient(180deg, rgba(180, 225, 240, 0.16) 0%, rgba(150, 210, 230, 0.05) 45%, transparent 80%);
        filter: blur(18px);
        transform-origin: top center;
        animation: ray-sway 14s ease-in-out infinite;
    }
    .godrays span:nth-child(1) { left: 8%;  transform: rotate(8deg);  animation-delay: 0s; }
    .godrays span:nth-child(2) { left: 30%; transform: rotate(-4deg); width: 90px; animation-delay: 3s; }
    .godrays span:nth-child(3) { left: 55%; transform: rotate(6deg);  width: 180px; opacity: 0.7; animation-delay: 6s; }
    .godrays span:nth-child(4) { left: 78%; transform: rotate(-9deg); width: 110px; animation-delay: 1.5s; }
    @keyframes ray-sway {
        0%, 100% { transform: rotate(var(--r, 6deg)) translateX(0); opacity: 0.55; }
        50%      { transform: rotate(calc(var(--r, 6deg) * -1)) translateX(18px); opacity: 0.85; }
    }

    /* ------------------------------------------------------------
       コースティクス（水面の揺らめく光の網目模様）
       SVGフィルタ（feTurbulence）で有機的な揺らぎを再現
    ------------------------------------------------------------ */
    .caustics-layer {
        position: fixed;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        background: radial-gradient(ellipse 1200px 600px at 50% 0%, rgba(120, 190, 210, 0.32), transparent 65%);
        filter: url(#causticsFilter);
        mix-blend-mode: screen;
        opacity: 0.22;
    }

    /* ------------------------------------------------------------
       マリンスノー（ゆっくり降り積もる微粒子）
    ------------------------------------------------------------ */
    .marine-snow { position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; filter: blur(0.3px); }
    .marine-snow::before, .marine-snow::after {
        content: "";
        position: absolute;
        inset: -10% 0 0 0;
        background-image:
            radial-gradient(circle, rgba(210,235,245,0.55) 0%, transparent 70%),
            radial-gradient(circle, rgba(210,235,245,0.4) 0%, transparent 70%),
            radial-gradient(circle, rgba(210,235,245,0.5) 0%, transparent 70%),
            radial-gradient(circle, rgba(210,235,245,0.35) 0%, transparent 70%),
            radial-gradient(circle, rgba(210,235,245,0.45) 0%, transparent 70%),
            radial-gradient(circle, rgba(210,235,245,0.3) 0%, transparent 70%),
            radial-gradient(circle, rgba(210,235,245,0.5) 0%, transparent 70%);
        background-size: 3px 3px, 2px 2px, 4px 4px, 2px 2px, 3px 3px, 2px 2px, 3px 3px;
        background-position: 5% 0%, 18% 0%, 33% 0%, 47% 0%, 61% 0%, 76% 0%, 90% 0%;
        background-repeat: no-repeat;
        animation: snow-fall 26s linear infinite;
    }
    .marine-snow::after {
        animation-duration: 38s;
        animation-delay: -12s;
        background-size: 2px 2px, 3px 3px, 2px 2px, 3px 3px, 2px 2px, 4px 4px, 2px 2px;
        background-position: 12% 0%, 27% 0%, 42% 0%, 58% 0%, 70% 0%, 83% 0%, 95% 0%;
        opacity: 0.7;
    }
    @keyframes snow-fall {
        0%   { background-position: 5% -5%, 18% -8%, 33% -3%, 47% -10%, 61% -6%, 76% -4%, 90% -7%; opacity: 0; }
        8%   { opacity: 0.8; }
        92%  { opacity: 0.5; }
        100% { background-position: 9% 105%, 24% 108%, 37% 102%, 53% 110%, 66% 104%, 80% 106%, 94% 103%; opacity: 0; }
    }

    /* ------------------------------------------------------------
       ビネット（画面端を暗く落として奥行きを出す）
    ------------------------------------------------------------ */
    .vignette {
        position: fixed;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        background: radial-gradient(ellipse at 50% 40%, transparent 35%, rgba(0, 4, 8, 0.55) 100%);
    }

    /* ------------------------------------------------------------
       フィルムグレイン（微細なノイズで写真的な質感を足す）
    ------------------------------------------------------------ */
    .film-grain {
        position: fixed;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        opacity: 0.05;
        mix-blend-mode: overlay;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23g)'/%3E%3C/svg%3E");
        background-size: 220px 220px;
    }

    /* 全体の文字色（深海の中で読みやすいライトカラーに） */
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stCaption {
        color: #d3e8f0;
    }
    h1, h2, h3, h4 {
        color: #aee0ee !important;
        text-shadow: 0 0 16px rgba(120, 200, 220, 0.35);
        font-weight: 800 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #86b3c4 !important;
    }

    /* タブ（潜水艇のHUD風ガラスピル） */
    button[data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 0.4rem 1.2rem !important;
        margin-right: 0.4rem !important;
        background-color: rgba(10, 30, 40, 0.5) !important;
        border: 1px solid rgba(140, 200, 220, 0.2) !important;
        color: #bcdce8 !important;
        backdrop-filter: blur(6px);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(80, 190, 210, 0.22), rgba(40, 120, 150, 0.22)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(120, 210, 230, 0.55) !important;
        font-weight: 700 !important;
        box-shadow: 0 0 14px rgba(80, 190, 210, 0.3);
    }

    /* ボタン（発光生物のバイオルミネセンス） */
    .stButton > button {
        border-radius: 999px !important;
        background: linear-gradient(135deg, #2fd7c4 0%, #1a8fb0 100%) !important;
        color: #021a24 !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 0.5rem 1.6rem !important;
        box-shadow: 0 0 16px rgba(47, 215, 196, 0.4), 0 3px 10px rgba(0,0,0,0.4) !important;
    }
    .stButton > button:hover {
        filter: brightness(1.1);
        box-shadow: 0 0 24px rgba(47, 215, 196, 0.6), 0 3px 10px rgba(0,0,0,0.4) !important;
    }

    /* テキストエリア（潜水艇の計器パネル風） */
    .stTextArea textarea {
        border-radius: 14px !important;
        border: 1.5px solid rgba(140, 200, 220, 0.25) !important;
        background: rgba(4, 18, 26, 0.65) !important;
        color: #e4f3f8 !important;
        backdrop-filter: blur(8px);
    }
    .stTextArea textarea::placeholder {
        color: #6b95a6 !important;
    }
    .stTextInput input, .stTextArea label, .stTextInput label {
        color: #d3e8f0 !important;
    }

    /* metricカード */
    div[data-testid="stMetric"] {
        background: rgba(10, 30, 40, 0.45);
        border-radius: 16px;
        padding: 0.9rem;
        border: 1px solid rgba(140, 200, 220, 0.2);
        box-shadow: 0 0 14px rgba(60, 160, 190, 0.12);
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] {
        color: #e4f3f8 !important;
    }

    /* expander */
    .streamlit-expanderHeader {
        border-radius: 12px !important;
        background: rgba(10, 30, 40, 0.4) !important;
        color: #d3e8f0 !important;
        border: 1px solid rgba(140, 200, 220, 0.18) !important;
    }
    .streamlit-expanderContent {
        background: rgba(10, 30, 40, 0.25) !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* checkbox・警告・成功・情報ボックス */
    .stCheckbox label { color: #d3e8f0 !important; }
    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        backdrop-filter: blur(6px);
    }

    /* dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(140, 200, 220, 0.2);
        box-shadow: 0 0 14px rgba(60, 160, 190, 0.1);
    }

    /* 関係図（matplotlib）は白背景のまま、深海に沈む観測パネル風に額装 */
    div[data-testid="stImage"] {
        background: rgba(248, 250, 250, 0.97);
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(140, 200, 220, 0.3);
        box-shadow: 0 0 20px rgba(60, 160, 190, 0.2), 0 6px 18px rgba(0,0,0,0.45);
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
    }
    </style>

    <svg width="0" height="0" style="position:absolute">
        <filter id="causticsFilter">
            <feTurbulence type="fractalNoise" baseFrequency="0.012 0.05" numOctaves="2" seed="7" result="turb">
                <animate attributeName="baseFrequency" dur="22s" values="0.012 0.05;0.02 0.07;0.012 0.05" repeatCount="indefinite" />
            </feTurbulence>
            <feColorMatrix in="turb" type="matrix"
                values="0 0 0 0 0.55  0 0 0 0 0.85  0 0 0 0 0.95  0 0 0 1 0" result="tealTurb" />
            <feComponentTransfer in="tealTurb">
                <feFuncA type="gamma" exponent="7" amplitude="1.4" offset="0" />
            </feComponentTransfer>
            <feGaussianBlur stdDeviation="1.2" />
        </filter>
    </svg>

    <div class="godrays"><span></span><span></span><span></span><span></span></div>
    <div class="caustics-layer"></div>
    <div class="marine-snow"></div>
    <div class="vignette"></div>
    <div class="film-grain"></div>
    """,
    unsafe_allow_html=True,
)

st.title("🌊 特許SAOラボ")
st.caption("GiNZAで日本語特許請求項を「主語・動詞・目的語」に分解して、構成要素の関係を可視化します")

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
