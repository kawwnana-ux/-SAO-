import streamlit as st
import matplotlib.pyplot as plt

import patent_pipeline as pp

import patent_pipeline as pp
import inspect

_src = inspect.getsource(pp)
st.code(
    f"ファイルの行数: {len(_src.splitlines())}\n"
    f"extract_direct_relationsの定義回数: {_src.count('def extract_direct_relations')}\n"
    f"extract_has_relationsの定義回数: {_src.count('def extract_has_relations')}\n"
    f"「であって」を含むか(含むリスト機能): {'含む' if 'であって' in inspect.getsource(pp.extract_direct_relations) else '含まない'}\n"
)
st.set_page_config(page_title="特許請求項SAO解析", layout="wide")

st.title("特許請求項 SAO構造解析・類似度診断")
st.caption("GiNZAによる依存構造解析で、特許請求項を構成要素間の関係グラフに変換します。")

tab1, tab2 = st.tabs(["① 1つの請求項を解析", "② 2つの請求項を比較"])


# ============================================================
# タブ①：1つの請求項を解析して図を見る
# ============================================================
with tab1:
    st.subheader("請求項を入力してください")
    text = st.text_area(
        "請求項テキスト",
        height=220,
        placeholder="例：第１の基板と、前記第１の基板上に設けられた第２の半導体層と、前記第２の半導体層に接続された電極と、を有する半導体装置。",
        key="single_text",
    )

    if st.button("解析する", type="primary", key="single_run"):
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
                    st.success(f"{len(relations)} 件の関係を抽出しました。")

                    col1, col2 = st.columns([3, 2])

                    with col1:
                        st.markdown("#### 構成要素間の関係図")
                        pp.visualize_relations(relations, title="構成要素間関係")
                        fig = plt.gcf()
                        st.pyplot(fig)
                        plt.close(fig)

                    with col2:
                        st.markdown("#### 抽出された関係（SAOトリプル）")
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
    st.subheader("2つの請求項を入力してください")

    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area("請求項A", height=220, key="text_a")
    with col_b:
        text_b = st.text_area("請求項B", height=220, key="text_b")

    use_semantic = st.checkbox(
        "②意味マッチングも使う（初回はモデルの読み込みに1分程度かかります）",
        value=False,
    )

    if st.button("比較する", type="primary", key="compare_run"):
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

                st.markdown("### 診断結果")
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

                st.markdown("### ①Jaccard：トリプルの一致・不一致")
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
                    st.markdown("### ②意味マッチング：対応付けの詳細")
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
