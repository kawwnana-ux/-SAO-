import streamlit as st
import pandas as pd
import patent_pipeline_sao_preprocessed as pp

st.set_page_config(page_title='日本語特許SAO分析', page_icon='🪼', layout='wide')

st.title('🪼 日本語特許請求項SAO構造分析')
st.caption('SudachiPy → 読点挿入 → GiNZA係り受け解析 → 語順整序ヒント → SAO抽出・補正（LLM不使用）')

DEFAULT = '''放熱装置と、
前記放熱装置の主面に配置された少なくとも１つの取り付けフレームと、
スイッチング機能を有する少なくとも１つのパワー半導体モジュールと、
を備え、
前記パワー半導体モジュールは、正側電源入力端子、負側電源入力端子および出力端子を含み、
前記取り付けフレームは、少なくとも１つの開口部を有し、
前記パワー半導体モジュールは、前記開口部により前記取り付けフレームに対して位置決めされており、
前記取り付けフレームの一部は、前記正側電源入力端子、前記負側電源入力端子および前記出力端子と、前記放熱装置との間に位置する、
インテリジェントパワーモジュール。'''

with st.sidebar:
    st.header('前処理設定')
    insert_comma = st.checkbox('読点挿入', value=True)
    reorder = st.checkbox('語順整序ヒント', value=True)
    max_clause_chars = st.number_input('読点挿入の長さ閾値', min_value=10, max_value=100, value=45, step=5)
    st.info('語順整序は安全性を優先し、現在は原文を大きく書き換えず、GiNZAの係り受けから整序ヒントを作ります。')

text = st.text_area('請求項を入力', value=DEFAULT, height=300)

config = dict(pp.PREPROCESS_CONFIG)
config['insert_comma'] = insert_comma
config['reorder'] = reorder
config['max_clause_chars'] = int(max_clause_chars)

if st.button('解析する', type='primary'):
    with st.spinner('解析中…'):
        prep = pp.preprocess_for_sao(text, config)
        components, relations = pp.analyze_claim(text)

    st.subheader('1. 前処理結果')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**① SudachiPy正規化後**')
        st.code(prep['normalized'])
        st.markdown('**② 読点挿入後**')
        st.code(prep['punctuated'])
    with c2:
        st.markdown('**③ 語順整序後**')
        st.code(prep['reordered'])
        st.markdown('**④ 係り受け解析の整序ヒント**')
        if prep['order_hints']:
            st.dataframe(pd.DataFrame(prep['order_hints'], columns=['依存語index','親語index','依存語']), use_container_width=True)
        else:
            st.write('該当なし')

    st.subheader('2. 最終GiNZA係り受け解析')
    dep_rows = []
    for t in prep['doc']:
        dep_rows.append({
            'index': t.i,
            'text': t.text,
            '品詞': t.pos_,
            '原形': t.lemma_,
            '係り受け': t.dep_,
            '親index': t.head.i,
            '親': t.head.text,
        })
    st.dataframe(pd.DataFrame(dep_rows), use_container_width=True, height=420)

    st.subheader('3. 抽出された構成要素')
    st.write(components)

    st.subheader('4. 抽出されたSAO')
    rows = []
    for r in relations:
        rows.append({'Subject': r.get('source',''), 'Action': r.get('relation',''), 'Object': r.get('target','')})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning('SAO関係が抽出されませんでした。')
