import spacy
import ginza
import ja_ginza
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import os
import re
import sys
import glob
import json
import ollama

# デバッグ用（app.py の「関係抽出の分岐トレース」機能から参照される）。
# このパイプライン本体は現状トレースを書き込んでいないため常に空のまま。
DEBUG_MODE = False
DEBUG_TRACE = []

# ============================================================
# GiNZAモデルの読み込み
# ============================================================
# ja_ginza.__file__ から実際のインストール場所を直接調べることで、
# Colab（dist-packages）でも、Streamlit Cloud等（site-packages）でも
# 同じコードで動くようにする。

_ja_ginza_dir = os.path.dirname(ja_ginza.__file__)

model_candidates = glob.glob(os.path.join(_ja_ginza_dir, "ja_ginza-*"))
model_path = [p for p in model_candidates if not p.endswith(".cfg")][0]
config_path = os.path.join(model_path, "config.cfg")


def _fix_split_mode(cfg_path):
    """config.cfg の split_mode = null を "C" に書き換える。戻り値: 成功したか"""
    if not os.path.exists(cfg_path):
        return True
    with open(cfg_path, "r", encoding="utf-8") as f:
        config_text = f.read()
    if "split_mode = null" not in config_text:
        return True
    config_text = config_text.replace('split_mode = null', 'split_mode = "C"')
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(config_text)
    return True


try:
    _fix_split_mode(config_path)
except (OSError, PermissionError):
    # Streamlit Cloud等、インストール済みパッケージのファイルが
    # 読み取り専用になっている環境向けの保険。
    # 書き込み可能な場所（/tmp）にモデル一式をコピーしてから書き換える。
    import shutil
    import tempfile as _tempfile
    writable_model_path = os.path.join(_tempfile.gettempdir(), "ja_ginza_model_copy")
    if not os.path.exists(writable_model_path):
        shutil.copytree(model_path, writable_model_path)
    model_path = writable_model_path
    config_path = os.path.join(model_path, "config.cfg")
    _fix_split_mode(config_path)

nlp = spacy.load(model_path)
print("GiNZAの読み込みに成功しました！ モデル:", model_path)

# ============================================================
# 日本語フォント
# ============================================================

import tempfile
import urllib.request

# 「/tmp」はLinux専用のパスで、Windowsには存在しないためハードコードしない。
# tempfile.gettempdir()はOSに応じた正しい一時フォルダを返す
# （Windowsなら C:\Users\...\AppData\Local\Temp など）。
FONT_PATH = os.path.join(tempfile.gettempdir(), "NotoSansJP-Regular.ttf")
# ダウンロードに失敗した場合（ネットワーク制限のある環境等）でも
# アプリ自体が起動できなくなるのは困るので、失敗時はmatplotlibの
# デフォルトフォント（日本語グラフの一部が正しく表示されない可能性は
# あるが、起動は止めない）にフォールバックする。
FONT_PROP = fm.FontProperties()
try:
    if not os.path.exists(FONT_PATH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
            FONT_PATH,
        )
    fm.fontManager.addfont(FONT_PATH)
    FONT_PROP = fm.FontProperties(fname=FONT_PATH)
except Exception as _font_exc:  # noqa: BLE001
    print(f"日本語フォントの読み込みに失敗しました（グラフの文字表示に影響する可能性があります）: {_font_exc}")

RELATION_WORDS = {
    # 基本方向・位置
    "間", "側", "上", "下", "内部", "外部",
    "周囲", "近傍", "前", "後", "間隔",
    # 部位（上下前後・中央系）
    "上部", "下部", "底部", "前部", "後部", "左側", "右側",
    "頂部", "頂上", "中央部", "中心部", "側部", "隅部", "角部",
    "一側", "他側", "一端", "他端", "一方", "他方",
    # 端・先端系
    "上端", "下端", "先端", "基端", "端部", "末端", "終端",
    "左端", "右端", "前端", "後端", "頂点",
    # 面
    "外周面", "内周面", "外面", "内面", "表面", "裏面", "上面", "下面",
    "側面", "底面", "天面", "端面", "接触面", "対向面",
    # 周辺・中間
    "中央", "中間", "周辺", "周縁", "周辺部", "縁", "縁部",
    "外周", "内周", "内", "中", "どうし",
    # 方向
    "前方", "後方", "上方", "下方", "左方", "右方",
    "内側", "外側", "上側", "下側",
    "水平方向", "垂直方向", "長手方向", "幅方向", "厚さ方向", "径方向", "軸方向",
}

# 「有する」と同じ意味で使われる動詞（「Ａを備える」「Ａを具備する」等）
HAS_LEMMAS = {"有する", "備える", "具備する"}

# 「ことを特徴とする」のような決まり文句に出てくる、実在の構成要素ではない
# 一般的な語（構成要素としては登録しない）。
# 「仮」は、「仮固定された前記絶縁基板」のように「仮固定される」という
# 動詞（VERB、この受身形ではGiNZAが「固定」をVERBとしてタグ付けする）を
# 修飾する副詞的な語（dep_="obl"、「仮に」の意）として単独で出現することが
# あり、この場合_consume_noun_phraseが後続のVERBで名詞句の消費を止めるため、
# 「仮」だけが単独の（実在しない）構成要素として誤登録されてしまう
# （特開2025-174033で確認済み）。「仮固定工程」「仮基板」のように「仮」が
# 名詞の複合語の先頭として使われる場合は、後続もNOUN/PROPNなので複合語
# 全体（例：「仮固定工程」）としてまとめて登録され、この除外の影響を
# 受けない（除外は「仮」単独のフレーズにしか効かない）。532件の正解データ
# でも「仮」単独がノードとして使われている例は無いことを確認済み。
GENERIC_NOUNS = {
    "こと", "もの", "とき", "場合", "特徴", "ため", "下記", "上記", "所定", "方向", "単数",
    "仮",
    "以上", "以下", "未満", "超", "以内", "程度",
    # 特開2025-175400のLLM直接抽出で、量詞「それぞれ」が単独で構成要素
    # タグ化され、「電力変換装置 有する それぞれ」のような無意味なSAOが
    # 生成されることを確認。「それぞれ」は_QUANTIFIER_WORDS（所有者
    # プレフィックス除外用）には既に入っているが、それ自体が独立した
    # コンポーネントとして登録されるのは別問題のため、ここでも除外する。
    "それぞれ",
}


def _is_generic_relation_word_bigram(doc, i):
    """
    「外周面」（外周＋面）のように、GiNZAが2トークンに分割してしまう
    複合位置語を判定する。
    """
    if i + 1 >= len(doc):
        return False
    combined = doc[i].text + doc[i + 1].text
    if combined not in RELATION_WORDS:
        return False
    return doc[i + 1].head.pos_ == "VERB"


def _is_generic_relation_word(token):
    """
    「側」「内部」などが、一般的な位置関係の語（Ａの上に／Ａの間に、のように
    動詞に係る用法）として使われているか判定する。
    係り先が動詞であれば一般的な位置関係語（構成要素名からは除外する）。
    係り先が名詞であれば「円盤状カッター側」「破砕槽内部側」のように、
    どちらの面・方向かを表す複合語の一部（構成要素名に含める）とみなす。
    """
    if token.text not in RELATION_WORDS:
        return False
    return token.head.pos_ == "VERB"


def _is_counter_word(token):
    """
    「２枚」「３個」のような助数詞（数を数える単位語）かどうかを判定する。
    NUM（数詞）を直接の子に持つ短い名詞は、助数詞である可能性が高い。
    """
    if len(token.text) > 2:
        return False
    return any(child.pos_ == "NUM" and child.dep_ == "nummod" for child in token.children)


def _normalize_component_text(phrase):
    """
    「前記メタデータ生成部」のように、何らかの理由で「前記」「該」が
    スキップされずに構成要素名の先頭に残ってしまった場合の保険。
    「メタデータ生成部」（前記なし）の表記と食い違って、
    同じものが別ノードとして扱われてしまうのを防ぐため、
    先頭の「前記」「該」を取り除く。

    【検証済みだが不採用】「Ｃｕ層」→「Cu層」、「５０μｍ」→「50μm」の
    ように全角英数字を半角に統一する変換（unicodedata.normalize("NFKC", ...)
    版、および全角英数字だけを対象にした狭いstr.translate版の両方）を
    試した。532件の正解データのうち、"第"で始まる序数ラベルは常に全角
    （5869件）だが、それ以外のノードは全角・半角が混在している
    （"Ｌ１""Ｐ入力端子""２個""９０度"等の336件は全角のまま、
    "Cu層""50μm"等の47件だけが半角）ことが分かり、後者の混在を
    半角に統一しても一致率は上がらず、逆にMICRO/MACRO双方で
    precision・recallが悪化した（NFKC版: MICRO precision 0.41267→0.40835、
    recall 0.42584→0.42126／狭いtranslate版もほぼ同じ悪化）。
    全角/半角の使い分けは「"第"で始まるか」では説明できない、
    アノテーション時の表記ゆれ（一部のクレームでのみ半角入力された）
    に起因すると考えられるため、一般化した幅統一は不採用とした。
    """
    for prefix in ("前記", "該"):
        if phrase.startswith(prefix) and phrase != prefix:
            phrase = phrase[len(prefix):]
    return phrase


# ============================================================
# ① 構成要素抽出
# ============================================================

def _is_nominalized_adjective_start(doc, i):
    """形容詞語幹＋「さ」「み」による名詞化パターンを検出する（例: 厚さ、大きさ、重み）。
    GiNZA はこれを ADJ + PART(mark) として解析するため、通常の NOUN/PROPN 連結では
    「第１の厚さ」の「厚さ」部分が欠落してしまう。これを名詞句の一部として認識する。
    """
    if i + 1 >= len(doc):
        return False
    t, nxt = doc[i], doc[i + 1]
    return t.pos_ == "ADJ" and nxt.text in ("さ", "み") and nxt.dep_ == "mark"


def _consume_noun_phrase(doc, i, first_is_nominalized_adj=False, fresh_start=False):
    """i位置から名詞句トークン列を貪欲に消費し、(words, next_i) を返す共通ロジック。

    fresh_start=True のときは、先頭トークンを（品詞や関係語チェックに関わらず）
    無条件に採用する（＝呼び出し元で既に NOUN/PROPN・NUM+名詞・名詞化形容詞などの
    条件を確認済みで、新規にフレーズを開始する場合の従来挙動）。それ以外
    （fresh_start=False。「第１の」等の既存プレフィックスへの継続）では、
    先頭トークンも関係語チェックの対象にする（従来の while ループと同じ挙動を
    維持し、意図せず「端部」等の関係語をコンポーネント名に取り込んでしまう
    回帰を防ぐ）。
    """
    if first_is_nominalized_adj:
        return [doc[i].text, doc[i + 1].text], i + 2
    words = []
    first = True
    while i < len(doc):
        if first and fresh_start:
            words.append(doc[i].text)
            i += 1
            first = False
            continue
        if doc[i].text == "第" and i + 1 < len(doc) and doc[i + 1].pos_ == "NUM":
            # 「複数第４端子」のように、数量詞（「複数」等）の直後に「第」＋数字が
            # 「の」を挟まずそのまま続く場合、この汎用ループが「第」を先行する
            # 名詞句の続きとして取り込んでしまい、「第４端子」のうち「第」だけが
            # 消費されて「４端子」（「第」が欠落した形）になってしまう
            # （特開2022-047218で確認済み。「複数第」という余分な構成要素も
            # 誤って登録されてしまう）。「第」＋数字は専用の処理
            # （extract_patent_components_general内の「第」分岐）に譲るため、
            # ここでは消費を止めてメインループに「第」を再度フレッシュな開始
            # として処理させる。
            break
        if doc[i].pos_ in {"NOUN", "PROPN"}:
            # 「基板裏面を有し」のように、RELATION_WORDS内の語（「裏面」等）
            # 自身が動詞の直接の主語・目的語（obj/nsubj）になっている場合は、
            # 「Ｘの上に配置される」の「上」のような単なる位置関係の修飾語
            # ではなく、実体として名指しされている（正解データでも「基板」
            # とは別の独立した構成要素として扱われている）。この場合だけ、
            # 複合語の一部として取り込む対象から除外しない。
            if (
                (_is_generic_relation_word(doc[i]) and doc[i].dep_ not in ("obj", "nsubj", "nsubjpass"))
                or doc[i].text in ("前記", "該", "うち", "乃至")
                or doc[i].text in _NUMERIC_THRESHOLD_WORDS
                or not doc[i].text.strip()
            ):
                break
            words.append(doc[i].text)
            i += 1
            first = False
        elif _is_nominalized_adjective_start(doc, i):
            # 【修正】以前はここで無条件にbreakしていたため、「表面粗さ改善層」
            # のように名詞化形容詞（「粗さ」）が複合語の途中（表面+粗さ+改善+層、
            # すべて同じ head=層 に係る）に現れるケースで、「表面粗さ」で
            # 名詞句が途切れ、「改善層」が別コンポーネントとして分離されて
            # しまっていた（特開2022-177018で確認）。「第１の厚さ」のように
            # 名詞化形容詞がフレーズの末尾になる場合は、続くトークンが
            # NOUN/PROPNでも名詞化形容詞でもない（通常は格助詞等）ため、
            # このままループを継続してもすぐ下のelseで自然に停止する。
            # そのため無条件breakをやめ、ループを継続して後続のNOUN/PROPNも
            # 同じ複合語として取り込めるようにする。
            words.append(doc[i].text)
            words.append(doc[i + 1].text)
            i += 2
            first = False
        else:
            break
    return words, i



# 「一端」「他端」は、他のRELATION_WORDS（「間」「上面」「表面」等）とは違い、
# 「Ａの一端」のように所有者に係った上で、それ自体が動詞の主語・目的語になる
# （＝実体として指し示される）用法が非常に多い（「整流素子の陽極側の一端が、
# 〜に接続され」等）。他のRELATION_WORDSの語は、既存の owner 解決ロジック
# （_merged_modifier_name等）が「その語を除外して、係り先の名詞に辿り着く」
# 前提で正しく動いているため、同じ例外を広げるとかえって壊れる
# （試したところ「間」「上面」「表面」等では明確に悪化した）。そのため、
# この例外は実際にバグの原因だった「一端」「他端」の2語だけに絞る。
_RELATION_WORDS_ALLOWED_AS_ARGUMENT = {"一端", "他端", "一方", "他方"}


# 【検証済みだが不採用】「実装面と、（厚み方向において前記実装面と反対の）
# 裏面と、を有するモジュールベース」のように、RELATION_WORDS内の語
# （裏面/上面/側面/頂点等）が「有する」「備える」「具備する」「含む」の
# 直接の目的語（dep_="obj"）になっている場合だけ実引数として許可する
# （＝任意のVERBのobj/nsubjではなく、HAS系動詞のobjに限定する）という
# 狭いスコープの拡張を試した。個別のターゲット例（特許7766804の
# 「モジュールベース　有する　裏面」等）では確かに見逃しが解消したが、
# 532件回帰では add_TP=22 に対し add_FP=87（新たな誤抽出）が発生し、
# MICRO precision 0.41084→0.40997、MACRO precision 0.43216→0.43188と、
# 両方のprecisionが悪化した（f1はMICRO/MACROともわずかに上昇したが、
# 本プロジェクトの基準はprecision・recallとも非劣化が必須のため不採用）。
# 原因は、裏面/上面/側面/頂点などのRELATION_WORDSの語を一旦「構成要素」
# として登録すると、has_relations等の所有者解決ロジック（root_component
# フォールバック等）が、これらの語を"普通の構成要素"として扱って
# ドキュメント内の無関係な別の所有者候補と誤って結び付けてしまう
# ケースが多発したため（例：特許7101882で「頂点」が有する/接続され/介し
# 等、10件以上の無関係な誤関係を新たに生成した）。ローカルな目的語判定
# だけでは、こうした派生的な誤抽出の副作用を防げない。
#
# 「一端」「他端」の例外を全RELATION_WORDSに単純に広げる（＝任意のVERBの
# obj/nsubjなら実引数とみなす）と、位置関係の修飾語としての用法が大半を
# 占める語（間・上面・表面等）では悪化することも既に確認されている
# （直下のコメント参照）。そのため、この例外は「一端」「他端」の2語のみに
# 留める。


def _relation_word_is_real_argument(doc, start, end):
    """
    「一端」「他端」が、単なる位置関係の修飾語（「Ａの上に配置される」の
    「上」のような）ではなく、その動詞の主語・目的語として使われている
    （＝「整流素子の陽極側の一端が、…に接続され」のように、それ自体が
    実体として指されている）場合を判定する。

    この場合、その語を丸ごとRELATION_WORDSとして除外してしまうと、
    「一端」に係る「整流素子の」「陽極側の」という所有者情報も含めて
    構成要素が完全に失われてしまう（＝関係抽出全体が破綻する）。
    """
    if start != end:
        return False
    token = doc[start]
    if token.text not in _RELATION_WORDS_ALLOWED_AS_ARGUMENT:
        return False
    return token.dep_ in ("nsubj", "obj") and token.head.pos_ == "VERB"


import re as _re_symbolic_label
_SYMBOLIC_LATIN_LABEL_RE = _re_symbolic_label.compile(r"^[A-Za-zＡ-Ｚａ-ｚ]{1,4}$")
# 「（ａ）」「（ｂ）」のように、順次列挙形式のクレームで工程を列挙する際に
# 使われる、括弧＋1文字のアルファベットだけの工程ラベル。GiNZAはこれを
# 通常のNOUN（単独トークン）として解析するため、「ａ」単体が構成要素として
# 誤登録される（特開2019-079940で確認）。
#
# 【方針転換】以前はこの関数がGiNZA単体版の厳格評価
# （analyze_claim_ginza_only）と共有されているため、この修正はGiNZA単体版
# 532件回帰でprecisionがわずかに悪化する（MICRO 0.413315→0.413277、
# MACRO 0.437628→0.437503、recallは不変）ことを理由に不採用としていた。
# その後、ツールの方針を「LLM＋GiNZAの組み合わせ」に一本化し、GiNZA単体版の
# 厳格F1は今後の主指標としないことになったため、ここで採用する
# （LLM直接抽出方式のタグ付け精度向上を優先する）。
_STEP_ENUM_LABEL_RE = _re_symbolic_label.compile(r"^[A-Za-zａ-ｚＡ-Ｚ]$")


def _is_paren_step_label(doc, i):
    token = doc[i]
    if not _STEP_ENUM_LABEL_RE.match(token.text):
        return False
    prev_ok = i > 0 and doc[i - 1].text in ("（", "(")
    next_ok = i + 1 < len(doc) and doc[i + 1].text in ("）", ")")
    return prev_ok and next_ok


def extract_patent_components_general(doc):
    components = []
    i = 0
    while i < len(doc):
        token = doc[i]

        if token.text in ("前記", "該", "うち", "乃至") or not token.text.strip():
            i += 1
            continue

        if _is_paren_step_label(doc, i):
            i += 1
            continue

        if _is_generic_relation_word_bigram(doc, i):
            i += 2
            continue

        if (
            _SYMBOLIC_LATIN_LABEL_RE.match(token.text)
            and i + 1 < len(doc)
            and doc[i + 1].pos_ == "NUM"
        ):
            # 「Ｌ１」「Ｖ２」のような、アルファベット＋数字の記号的な名称
            # （寸法・電圧等を表す変数名）や、「ＳｉＣ」「ＣＯ２」のような
            # 化学式は、GiNZAのトークナイザで「Ｌ」（NOUN）と「１」（NUM）の
            # ように分割されてしまい、そのままでは正しい構成要素として
            # 認識されない。「第」＋数字のケースと同様に、直後がNUMなら
            # まとめて1つの構成要素にする。
            start = i
            words = [doc[i].text]
            i += 1
            while i < len(doc) and doc[i].pos_ == "NUM":
                words.append(doc[i].text)
                i += 1
            # 「Ａｌ２Ｏ３」「Ｓｉ３Ｎ４」のような複数元素の化学式は、
            # 「アルファベット＋数字」の組が読点等を挟まず連続して続く。
            # その場合は同じ構成要素として繋げて取り込む。
            while (
                i + 1 < len(doc)
                and _SYMBOLIC_LATIN_LABEL_RE.match(doc[i].text)
                and doc[i + 1].pos_ == "NUM"
            ):
                words.append(doc[i].text)
                i += 1
                while i < len(doc) and doc[i].pos_ == "NUM":
                    words.append(doc[i].text)
                    i += 1
            end = i - 1
            phrase = _normalize_component_text("".join(words))
            if phrase not in GENERIC_NOUNS:
                components.append({"text": phrase, "start": start, "end": end})
            continue

        if token.text == "第" and i + 1 < len(doc) and doc[i + 1].pos_ == "NUM":
            start = i
            words = [doc[i].text]
            i += 1
            words.append(doc[i].text)
            i += 1
            if i < len(doc) and doc[i].text == "の":
                words.append(doc[i].text)
                i += 1
            if i < len(doc) and (doc[i].pos_ in {"NOUN", "PROPN"} or _is_nominalized_adjective_start(doc, i)):
                more_words, i = _consume_noun_phrase(
                    doc, i, first_is_nominalized_adj=_is_nominalized_adjective_start(doc, i)
                )
                words.extend(more_words)
            end = i - 1
            phrase = _normalize_component_text("".join(words))
            if (
                phrase not in RELATION_WORDS or _relation_word_is_real_argument(doc, start, end)
            ) and phrase not in GENERIC_NOUNS:
                components.append({"text": phrase, "start": start, "end": end})
            continue

        if token.pos_ in {"NOUN", "PROPN"} or (
            token.pos_ == "NUM"
            and i + 1 < len(doc)
            and doc[i + 1].pos_ in {"NOUN", "PROPN"}
            and not _is_counter_word(doc[i + 1])
        ) or _is_nominalized_adjective_start(doc, i):
            # 「三次元」のように、数詞がそのまま名詞の一部になっている
            # 複合語（"第１の基板"のように間に"の"を挟まないもの）にも対応する。
            # ただし「２枚」「１種」のような「数字＋助数詞」（この後に
            # 「の」＋本当の名詞が続く）は、複合語の開始として扱わない
            # （helper _is_counter_word で判定）。
            if token.pos_ in {"NOUN", "PROPN"} and _is_counter_word(token):
                i += 1
                continue
            start = i
            words, i = _consume_noun_phrase(
                doc, i, first_is_nominalized_adj=_is_nominalized_adjective_start(doc, i), fresh_start=True
            )
            end = i - 1
            phrase = _normalize_component_text("".join(words))
            if (
                phrase not in RELATION_WORDS or _relation_word_is_real_argument(doc, start, end)
            ) and phrase not in GENERIC_NOUNS:
                components.append({"text": phrase, "start": start, "end": end})
            continue

        i += 1

    unique_components = []
    seen = set()
    for c in components:
        key = (c["start"], c["end"])
        if key in seen:
            continue
        seen.add(key)
        unique_components.append(c)
    return _disambiguate_components_by_name(doc, unique_components)


# 「複数の」「いくつかの」のような数量詞は、実体を区別する情報ではない
# （正解データでも名前に含めない傾向がある）ため、所有格プレフィックスの
# 候補からは除外する。
_QUANTIFIER_WORDS = {
    "複数", "いくつか", "各", "全て", "すべて", "一部", "少なくとも",
    "双方", "多く", "任意", "それぞれ", "幾つか",
}


def _has_zenki_prefix(doc, comp):
    """
    コンポーネントの直前トークンが「前記」「該」かどうかを見る。
    「前記」「該」が付いている場合、それは新規の導入ではなく、既に
    出てきた構成要素への後方参照である可能性が高い、という手がかりに使う。
    """
    idx = comp["start"] - 1
    while idx >= 0 and not doc[idx].text.strip():
        idx -= 1
    return idx >= 0 and doc[idx].text in ("前記", "該")


def _genitive_owner_candidate(doc, token, components):
    """
    tokenのnmod（「の」格）子から、所有格修飾語の候補を1つ探す
    （数量詞は候補から除外する）。
    """
    for child in token.children:
        if child.dep_ != "nmod":
            continue
        has_no = any(c.dep_ == "case" and c.text == "の" for c in child.children)
        if not has_no:
            continue
        if child.text in _QUANTIFIER_WORDS:
            continue
        child_comp = find_component_by_token(components, child.i)
        if child_comp is None:
            continue
        return child_comp
    return None


def _disambiguate_components_by_name(doc, components):
    """
    「スイッチング素子」のような同じ短い名前を持つ構成要素が請求項内に
    複数現れる場合、それらが本当に同じ実体（「前記」「該」による後方参照）
    なのか、たまたま同名なだけの別々の実体（例：「正極側のスイッチング素子」
    と「負極側のスイッチング素子」）なのかを判定し、後者の場合だけ、
    区別に必要な最小限の所有格修飾語を名前の先頭に付け足す。

    以前は所有格の連鎖を無条件・再帰的にすべて結合していたが、実データで
    検証したところ、(1) 数量詞（「複数の」等）まで巻き込んでしまう、
    (2) 正解データは関係の種類に応じて名前の粒度を変えており、一律結合とは
    噛み合わない、という2つの理由でかえってF1が悪化した。この反省を踏まえ、
    「本当に曖昧さがある場合だけ」最小限の修飾を加える、より保守的な
    アプローチに変更したもの。
    """
    groups = {}
    for idx, c in enumerate(components):
        groups.setdefault(c["text"], []).append(idx)

    new_components = [dict(c) for c in components]

    for name, idxs in groups.items():
        # 「一端」「他端」のように、それ単独では何を指すか分からない
        # 一般的すぎる語（_RELATION_WORDS_ALLOWED_AS_ARGUMENTで例外的に
        # 実体として認めている語）は、たとえ請求項内で重複していなくても、
        # 所有格修飾語があるなら常にそれを名前に含める
        # （正解データでも「整流素子の陽極側の一端」のように、常に
        #  所有格チェーンごと1つの構成要素名として扱われているため）。
        always_disambiguate = name in _RELATION_WORDS_ALLOWED_AS_ARGUMENT
        if always_disambiguate:
            fresh_idxs = [i for i in idxs if not _has_zenki_prefix(doc, components[i])]
        elif len(idxs) < 2:
            continue
        else:
            # 後方参照（前記/該）は「新規導入」ではないので、曖昧さ解消の対象から除く
            fresh_idxs = [i for i in idxs if not _has_zenki_prefix(doc, components[i])]
            if len(fresh_idxs) < 2:
                continue
        for i in fresh_idxs:
            comp = components[i]
            anchor = doc[comp["end"]]
            owner = _genitive_owner_candidate(doc, anchor, components)
            if owner is None:
                continue
            if (owner["start"], owner["end"]) == (comp["start"], comp["end"]):
                continue
            prefix = owner["text"] + "の"
            new_components[i]["text"] = _normalize_component_text(prefix + comp["text"])

    return new_components


# ============================================================
# ② 関係語（上・間 など）抽出
# ============================================================

def extract_relation_words_general(doc):
    results = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        for child in token.children:
            if child.pos_ != "NOUN":
                continue

            relation_word_text = None

            if child.text in RELATION_WORDS:
                relation_word_text = child.text
            elif child.i - 1 >= 0:
                prev = doc[child.i - 1]
                combined = prev.text + child.text
                if combined in RELATION_WORDS and prev.head.i == child.i:
                    # 「外周面」（外周＋面）のように2トークンに分割された
                    # 複合位置語。ラベルは結合した形にする。
                    relation_word_text = combined

            if relation_word_text is None:
                continue

            results.append({
                "relation_word": relation_word_text,
                "relation_index": child.i,
                "verb": token.text,
                "verb_index": token.i,
                "dependency": child.dep_,
            })
    return results


# ============================================================
# 共通ヘルパー
# ============================================================

def find_component_by_token(components, token_index):
    for c in components:
        if c["start"] <= token_index <= c["end"]:
            return c
    return None


def find_full_title_component(doc, components, comp):
    """
    請求項タイトル（クレーム末尾の、装置・方法全体を指す名詞句）を、
    直前の「Ｘの」所有格チェーンを含めた最大限の複合語として再構築する。

    正解データを532件検証した結果、「パワーモジュールの製造方法」
    「窒化珪素基板の製造方法」「半導体基板構造体の製造方法」のように、
    クレームタイトルを「Ｘの＜基本語＞」という所有格付きの複合語1つの
    ノードとして扱う例が非常に多いことを確認した（「の」を含むgoldノード
    のうち、extract_patent_components_generalの通常の名詞句消費では
    1つの構成要素として認識されないものが59.3%、gold全ノードで見ても
    厳密一致で一度もタグ化されないものが19.6%にも達する）。

    一方、_consume_noun_phraseは「の」で名詞句の消費を止める仕様に
    なっている。これは「Ｘの上面に配置される」のように「Ｘ」と「上面」を
    別ノードとして分離すべきケース（全体としてはこちらの方が多い）を
    壊さないための意図的な設計であり、_consume_noun_phrase自体を
    汎用的に変更するのは危険（過去に試して規模の大きい退行を起こして
    いる）。

    そこで、請求項タイトル（root_component／claim_title_comp。通常は
    クレーム中で一度しか出現せず、Ｘ単体が他の場所で構成要素として
    再利用されることがほぼ無い、安全に拡張できる特別な位置）に限り、
    直前の「Ｘの」チェーンを辿って複合語をまとめる。
    """
    start = comp["start"]
    while start - 1 >= 0 and doc[start - 1].text == "の":
        prev = find_component_by_token(components, start - 2)
        if prev is None or prev["end"] != start - 2:
            break
        start = prev["start"]
    if start == comp["start"]:
        return comp
    text = "".join(doc[i].text for i in range(start, comp["end"] + 1))
    return {"text": _normalize_component_text(text), "start": start, "end": comp["end"]}


def find_referenced_component(components, token):
    component = find_component_by_token(components, token.i)
    if component is not None:
        return component
    if token.pos_ not in {"NOUN", "PROPN"}:
        # 動詞などは、たまたま文字列が構成要素名と重なっていても
        # 参照とはみなさない（例：動詞「シール」と名詞「リングシール」）
        return None
    word = token.text
    for c in reversed(components):
        if word in c["text"] and c["end"] < token.i:
            return c
    return None


def find_previous_component_by_word(components, token):
    component = find_component_by_token(components, token.i)
    if component is not None:
        return component
    if token.pos_ not in {"NOUN", "PROPN"}:
        return None
    word = token.text
    for c in reversed(components):
        if c["end"] >= token.i:
            continue
        if word in c["text"]:
            return c
    return None


def find_target_component_from_verb(components, verb):
    targets = []
    current = verb
    visited = set()
    while True:
        if current.i in visited:
            break
        visited.add(current.i)
        component = find_component_by_token(components, current.i)
        if component is not None:
            if component not in targets:
                targets.append(component)
            break
        if current.head == current:
            break
        current = current.head
    return targets


def _find_outermost_component_from_verb(components, verb):
    """
    「Ｘを用いてＹに対応するＺを生成するＷ」のように、
    「用いる」の係り先を辿ると先に「Ｚ」（合成音声データ）に行き当たるが、
    本当の動作主はさらに奥にある「Ｗ」（合成音声データ生成部）である、
    というような何重にも入れ子になった文に対応する。

    find_target_component_from_verb は最初に見つかった構成要素で
    止まってしまうが、こちらは1つ目が見つかった後もさらに1段階だけ
    奥を探し、2つ目が見つかればそちらを採用する
    （そのまま際限なく奥まで辿ると、文書全体の最後の語＝請求項の
    タイトルに行き着いてしまうため、2つ目までで止める）。
    """
    found = []
    current = verb
    visited = set()
    while len(found) < 2:
        if current.i in visited:
            break
        visited.add(current.i)
        component = find_component_by_token(components, current.i)
        if component is not None and (not found or component["text"] != found[-1]["text"]):
            found.append(component)
        if current.head == current:
            break
        current = current.head
    if not found:
        return None
    return found[-1]


# ============================================================
# ③ 位置関係の抽出（「〜上に設けられた」等）
# ============================================================

def _is_locative_obl(token):
    """
    「破砕槽内壁面には固定刃を有し」のように、動詞の obl（斜格）が
    場所を表しているかどうかを判定する。
    「により」「によって」のような手段を表す格、「において」のような
    前提・状況を表す格（"より"/"おい"がfixedでついている場合）は
    場所ではないので除外する。

    「と」が格助詞の場合は、並列列挙（「Ａと、Ｂと、…」）や共同格
    （「Ａと接続される」）の目印であって場所ではないため、これも
    除外する（GiNZAが超長文で、直前の列挙項目の末尾（「〜端子と、」）を
    次の動詞のobl子として誤って結びつけてしまうことがあり、これを
    「場所」と誤判定すると全く無関係な語がsourceとして採用されてしまう）。
    """
    if token.dep_ != "obl":
        return False
    for child in token.children:
        if child.dep_ == "case":
            if child.text == "と":
                return False
            for grandchild in child.children:
                if grandchild.dep_ == "fixed" and grandchild.text in ("より", "よって", "おい"):
                    return False
    return True


def extract_has_location_relations(doc, components):
    """
    「Ａには／Ａに、Ｂを有し」のように、場所（RELATION_WORDSの
    固定リストにない語も含む）と「有する」の目的語との関係を抽出する。
    """
    relations = []
    for verb in doc:
        if verb.lemma_ not in HAS_LEMMAS or verb.pos_ != "VERB":
            continue

        obj_token = None
        for child in verb.children:
            if child.dep_ == "obj":
                obj_token = child
                break
        if obj_token is None:
            continue

        target = find_component_by_token(components, obj_token.i) or find_referenced_component(components, obj_token)
        if target is None:
            continue

        for child in verb.children:
            if not _is_locative_obl(child):
                continue
            source = find_component_by_token(components, child.i) or find_referenced_component(components, child)
            if source is None or source["text"] == target["text"]:
                continue
            relations.append({
                "source": source["text"],
                "relation": "には有する",
                "target": target["text"],
                "type": "positional",
            })
    return relations


def extract_installation_relations(doc, components):
    """
    「回転軸に設けた２枚のサイドプレート」のように、「設ける」の
    係り先が「間」のような位置語（構成要素として登録されていない語）
    になっている場合、その位置語の compound の子（＝実際に設置された
    構成要素）を探して関係先にする。
    """
    relations = []
    for verb in doc:
        if verb.lemma_ != "設ける" or verb.pos_ != "VERB":
            continue

        obl_child = None
        for child in verb.children:
            if child.dep_ == "obl":
                obl_child = child
                break
        if obl_child is None:
            continue

        source = find_component_by_token(components, obl_child.i) or find_referenced_component(components, obl_child)
        if source is None:
            continue

        head = verb.head
        target = find_component_by_token(components, head.i)
        if target is None:
            for child in head.children:
                if child.dep_ == "compound":
                    t = find_component_by_token(components, child.i)
                    if t is not None:
                        target = t
                        break
        if target is None:
            targets = find_target_component_from_verb(components, verb)
            target = targets[0] if targets else None

        if target is None or target["text"] == source["text"]:
            continue

        relations.append({
            "source": source["text"],
            "relation": "に設けた",
            "target": target["text"],
            "type": "positional",
        })
    return relations


def _merged_modifier_name(token, components):
    """
    「外側の面」のように、「の」で係る nmod の修飾語を語自体の前に
    くっつけた、より具体的な名前を作る。

    構成要素抽出の時点（_disambiguate_components_by_name）で、既に
    同じ所有格修飾語がbaseの先頭に組み込まれていることがある
    （「一端」→「陽極側の一端」等）。その場合にここでもう一度同じ
    修飾語を付けてしまうと「陽極側の陽極側の一端」のような二重付与に
    なってしまうため、baseが既にその修飾語で始まっている場合は
    スキップする。
    """
    comp = find_component_by_token(components, token.i)
    base = comp["text"] if comp is not None else token.text

    prefix = ""
    for child in token.children:
        if child.dep_ != "nmod":
            continue
        has_no = any(c.dep_ == "case" and c.text == "の" for c in child.children)
        if not has_no:
            continue
        child_comp = find_component_by_token(components, child.i)
        if child_comp is not None and comp is not None and (child_comp["start"], child_comp["end"]) == (comp["start"], comp["end"]):
            # child が既に base と同じ結合済みコンポーネント内（例:「第１の厚さ」）の場合は
            # 自己参照になってしまうため、所有格プレフィックスとして採用しない。
            continue
        candidate_prefix = (child_comp["text"] if child_comp is not None else child.text) + "の"
        if base.startswith(candidate_prefix):
            # 既に構成要素名の先頭にこの修飾語が組み込まれている（二重付与防止）
            continue
        prefix = candidate_prefix
        break

    return prefix + base


def _name_for_component(doc, comp, components):
    """コンポーネント辞書から、所有格プレフィックス（nmod「の」）を含めた表示名を作る。"""
    if comp is None:
        return None
    anchor = doc[comp["end"]]
    return _merged_modifier_name(anchor, components)


def _find_owner_via_acl(token, components):
    """
    「該サイドプレートの円盤状カッター側ではない外側の面」のように、
    否定の連体修飾（acl）を挟んで持ち主（例：サイドプレート）が
    係っている場合、それを辿って見つける。
    """
    for child in token.children:
        if child.dep_ != "acl":
            continue
        for grandchild in child.children:
            if grandchild.dep_ == "nmod":
                comp = find_component_by_token(components, grandchild.i) or find_referenced_component(components, grandchild)
                if comp is not None:
                    return comp
    return None


def extract_contact_relations(doc, components):
    """
    「〜に接するようにＸを接触させて」のように、「接する」節の主語が
    明示されていない場合、その節が係っている動詞（接触させて等）の
    目的語を主語として補う。

    また、接する対象（面など）は「外側の面」のように修飾語を含めた
    名前にし、さらにその面の持ち主（例：サイドプレート）が分かる場合は
    「持ち主 → 面 → 接するもの」という鎖にする
    （＝持ち主の直接の子として「接するもの」がぶら下がる形にするため）。
    """
    relations = []
    for verb in doc:
        if verb.lemma_ != "接する" or verb.pos_ != "VERB":
            continue

        obl_child = None
        for child in verb.children:
            if child.dep_ == "obl":
                obl_child = child
                break
        if obl_child is None:
            continue
        target_comp = find_component_by_token(components, obl_child.i) or find_referenced_component(components, obl_child)
        if target_comp is None:
            continue
        target_name = _merged_modifier_name(obl_child, components)

        parent_verb = verb.head
        agent = None
        if parent_verb.pos_ == "VERB":
            for child in parent_verb.children:
                if child.dep_ == "obj":
                    agent = find_component_by_token(components, child.i) or find_referenced_component(components, child)
                    if agent is not None:
                        break
        if agent is None or agent["text"] == target_name:
            continue

        owner = _find_owner_via_acl(obl_child, components)
        if owner is not None and owner["text"] != target_name:
            relations.append({
                "source": owner["text"],
                "relation": "有する",
                "target": target_name,
                "type": "has",
            })

        # 「面 が 接するもの に接する」という向きにして、
        # 持ち主 → 面 → 接するもの、という鎖になるようにする
        relations.append({
            "source": target_name,
            "relation": "に接する",
            "target": agent["text"],
            "type": "direct",
        })
    return relations


def extract_boundary_relations(doc, components):
    """
    「Ａの内部側とその外側のＢとの境界」のように、「境界」が
    複数のものの間にある場合、nmodの係り受けを辿って
    「境界」とその両側（Ａ・Ｂ）との関係を抽出する。

    「と」でつながっている語（＝対等に並んでいる境界の両側）だけを対象にし、
    「の」でつながっている語（＝単なる修飾語。例：外側の／サイドプレートの）
    は関係先にしない。
    """
    def has_to_marker(token):
        return any(c.dep_ == "case" and c.text == "と" for c in token.children)

    relations = []
    for c in components:
        if not c["text"].endswith("境界"):
            continue
        boundary_token = doc[c["end"]]

        stack = [boundary_token]
        seen = set()
        while stack:
            t = stack.pop()
            if t.i in seen:
                continue
            seen.add(t.i)

            for child in t.children:
                if child.dep_ != "nmod":
                    continue
                comp = find_component_by_token(components, child.i) or find_referenced_component(components, child)

                if has_to_marker(child):
                    if comp is not None and comp["text"] != c["text"]:
                        relations.append({
                            "source": c["text"],
                            "relation": "との境界",
                            "target": comp["text"],
                            "type": "positional",
                        })
                    # 「と」で繋がった語の中に、さらに入れ子で「と」の並列項が
                    # ある場合があるので、見つかった後も奥まで探索を続ける
                    stack.append(child)
                elif comp is None:
                    # まだ構成要素が見つかっていない場合だけ、さらに奥まで辿る
                    stack.append(child)
    return relations


def _find_nsubj_target_for_verb(doc, components, verb):
    """
    「前記取り付けフレームの一部は、Ａと、Ｂとの間に位置する」のように、
    位置関係の動詞（位置する等）が名詞を修飾する連体修飾節（acl）として
    使われている場合、GiNZAの長文解析でこの動詞自身の主語（「一部」）が、
    動詞にではなく、その動詞が係る先のさらに遠い名詞（節全体が最終的に
    かかる請求項全体の名前など）に直接の子（nsubj）として誤って
    結びついてしまうことがある（extract_has_relationsのsubj_token選択で
    対応した問題と同じ系統のバグ）。

    obj（目的語）が見つからない場合のフォールバックとして、まず動詞自身、
    次に動詞の係り先（head）の直接の子からnsubjを探し、動詞に一番近い
    ものを主語として採用する。所有格プレフィックス（「取り付けフレームの」
    等）も含めた名前を使う。
    """
    candidates = []
    seen_idx = set()
    for child in verb.children:
        if child.dep_ == "nsubj" and child.i not in seen_idx:
            candidates.append(child)
            seen_idx.add(child.i)
    if verb.head is not None and verb.head.i != verb.i:
        for child in verb.head.children:
            if child.dep_ == "nsubj" and child.i not in seen_idx:
                candidates.append(child)
                seen_idx.add(child.i)
    if not candidates:
        return None
    candidates.sort(key=lambda c: abs(c.i - verb.i))
    chosen = candidates[0]
    comp = find_component_by_token(components, chosen.i) or find_referenced_component(components, chosen)
    if comp is None:
        return None
    merged_text = _merged_modifier_name(chosen, components)
    return {"text": merged_text}


def extract_positional_relations(doc, components, relation_words):
    relations = []
    for relation in relation_words:
        relation_token = doc[relation["relation_index"]]

        source_components = []
        for child in relation_token.children:
            c = find_referenced_component(components, child)
            if c is not None and c not in source_components:
                source_components.append(c)

        verb = relation_token.head
        if verb.pos_ != "VERB":
            continue

        # 「ＡとＢとの間に」のように、「間」の基準となる複数の対象のうち
        # 一方（Ａ）が「間」自身の子（nmod）ではなく、動詞の別のobl引数
        # として並列に現れることがある（「Ａと、Ｂとの間に」で、Ａ側が
        # 動詞に直接係り、Ｂ側だけが「間」に係る場合）。そのような「と」
        # 付きの並列obl引数も基準側（source）として拾う。また、
        # 「Ａ、ＢおよびＣと」のように列挙されている場合は、nmodで
        # 連なる連鎖（Ｃ→Ｂ→Ａ）を辿って全項目を拾う。
        for sibling in verb.children:
            if sibling.i == relation_token.i or sibling.dep_ != "obl":
                continue
            if not any(gc.dep_ == "case" and gc.text == "と" for gc in sibling.children):
                continue
            chain_token = sibling
            visited_chain = set()
            while chain_token is not None and chain_token.i not in visited_chain:
                visited_chain.add(chain_token.i)
                c = find_referenced_component(components, chain_token)
                if c is not None and c not in source_components:
                    source_components.append(c)
                next_token = None
                for gc in chain_token.children:
                    if gc.dep_ == "nmod":
                        next_token = gc
                        break
                chain_token = next_token

        is_has_branch = verb.lemma_ in HAS_LEMMAS
        if is_has_branch:
            # 「Ａ間に、Ｂを有し」のように「有する」が使われている場合は、
            # 文全体の主語（根っこ）ではなく、「有する」の直接の目的語
            # （＝実際にそこに存在するもの）を関係先にする。
            target_components = []
            for child in verb.children:
                if child.dep_ == "obj":
                    t = find_component_by_token(components, child.i) or find_referenced_component(components, child)
                    if t is not None:
                        target_components.append(t)
                    break
            label = relation["relation_word"] + "に" + verb.text
        else:
            target_components = []
            for child in verb.children:
                if child.dep_ == "obj":
                    t = find_component_by_token(components, child.i) or find_referenced_component(components, child)
                    if t is not None:
                        target_components.append(t)
                    break
            if not target_components:
                # 動詞自身に直接の目的語(obj)がない場合、まず動詞（またはその
                # 係り先）のnsubjを探す。見つからない場合だけ、従来通り
                # 動詞連鎖を遡って構成要素を探す。
                nsubj_target = _find_nsubj_target_for_verb(doc, components, verb)
                if nsubj_target is not None:
                    target_components = [nsubj_target]
                else:
                    target_components = find_target_component_from_verb(components, verb)
            aux_texts = "".join(
                c.text for c in sorted(verb.children, key=lambda c: c.i)
                if c.pos_ == "AUX" and c.i > verb.i
            )
            label = relation["relation_word"] + "に" + verb.text + aux_texts

        for target in target_components:
            for source in source_components:
                if source["text"] == target["text"]:
                    continue
                if is_has_branch:
                    # 「Ａの間に、Ｂを有し」は「Ａには有する」と同じ
                    # 「場所（Ａ）→ そこにあるもの（Ｂ）」という向きなので、
                    # そのまま source=場所, target=中身 でよい。
                    relations.append({
                        "source": source["text"],
                        "relation": label,
                        "target": target["text"],
                        "type": "positional",
                    })
                else:
                    # 「ＡとＢとの間にＣが設けられる／位置する」等は、
                    # 意味的には「Ｃ（配置される主体）が、Ａ・Ｂ（基準・
                    # 境界となる相手）に対して間に位置する」であり、
                    # ＳＡＯとして素直に読めば source=Ｃ（配置される側）、
                    # target=Ａ・Ｂ（基準側）である。以前はここが逆
                    # （source=Ａ・Ｂ、target=Ｃ）になっており、正解データ
                    # （人手で作成したゴールドSAO）と方向が系統的に逆転して
                    # いた（直接関係の「に接続される」等で見つかった
                    # 系統的な向きの逆転と同じ性質の問題）。
                    relations.append({
                        "source": target["text"],
                        "relation": label,
                        "target": source["text"],
                        "type": "positional",
                    })
    return relations


# ============================================================
# ④ 直接関係の抽出（「Aに接続されたB」等）
# ============================================================

def _is_passive(verb):
    """動詞が受身形（〜られた／〜れた）かどうかを判定する"""
    return any(
        child.pos_ == "AUX" and child.lemma_ in ("れる", "られる")
        for child in verb.children
    )


def _is_negated(verb):
    """
    動詞が否定形（〜ない／〜ません等）かどうかを判定する。
    「〜が行われない」のような否定文から、肯定の関係として
    誤って抽出してしまうのを防ぐために使う。
    """
    return any(
        child.lemma_ in ("ない", "ず", "ぬ") and child.pos_ in ("AUX", "SCONJ")
        for child in verb.children
    )


def _is_instrumental_obl(token):
    """
    「回転カッター式破砕機により破砕する」のように、動詞の obl（斜格）が
    「により」「によって」で手段・道具を表しているかどうかを判定する。
    """
    if token.dep_ != "obl":
        return False
    for child in token.children:
        if child.dep_ == "case":
            for grandchild in child.children:
                if grandchild.dep_ == "fixed" and grandchild.text in ("より", "よって"):
                    return True
    return False


def _find_topic_in_verb_chain(doc, components, verb):
    """
    「前記テキスト翻訳部は、〜を翻訳して〜を生成し」のように、
    動詞が連鎖している場合、その連鎖（advcl/auxで繋がったVERB同士）の
    範囲内だけで「は」で明示された主題を探す。

    長い複文では、GiNZAが複数の「は」付き名詞を同じ動詞のnsubjとして
    （誤って）結びつけてしまうことがあるため、
      ① 対象の動詞より後ろに出てくる「は」は候補にしない
         （主語が動詞より後に来ることはないため）
      ② 複数見つかった場合は、動詞に一番近い（＝一番あとに出てくる）
         ものを採用する
    という2段階で絞り込む。

    「破砕槽内壁面には」のように「に」＋「は」が連続する場合は対象外にする。
    """
    original_verb_i = verb.i
    current = verb
    visited = set()
    candidates = []
    while current.i not in visited:
        visited.add(current.i)
        for child in current.children:
            if child.pos_ not in ("NOUN", "PROPN"):
                continue
            if child.i >= original_verb_i:
                continue
            has_bare_wa = False
            for cc in child.children:
                if cc.dep_ == "case" and cc.text == "は":
                    if cc.i > 0 and doc[cc.i - 1].pos_ == "ADP":
                        continue
                    has_bare_wa = True
                    break
            if has_bare_wa:
                comp = find_component_by_token(components, child.i) or find_referenced_component(components, child)
                if comp is not None:
                    candidates.append((child.i, comp))
        nxt = current.head
        if nxt.i == current.i or nxt.pos_ != "VERB":
            break
        current = nxt

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _find_nsubj_up_chain(verb):
    """
    「テキスト翻訳部は…翻訳して…生成し」のように、動詞が連鎖している場合、
    その動詞自身に主語(nsubj)がなくても、連鎖を遡った先の動詞
    （最終的にROOTに近い動詞）に本当の主語が付いていることが多い。
    それを探して返す。

    ただし、主語が動詞より後ろに来ることはないので、
    動詞より後ろにある nsubj は候補にしない
    （GiNZAが超長文で、無関係な後方の「は」付き名詞を
      同じ動詞のnsubjとして誤って結びつけてしまうことがあるため）。
    """
    original_verb_i = verb.i
    current = verb
    visited = set()
    while current.i not in visited:
        visited.add(current.i)
        for child in current.children:
            if child.dep_ == "nsubj" and child.i < original_verb_i:
                return child
        nxt = current.head
        if nxt.i == current.i or nxt.pos_ != "VERB":
            break
        current = nxt
    return None


def _find_nearest_topic_before_text(doc, components, verb):
    """
    最終手段：動詞連鎖・nsubj連鎖のどちらでも見つからない場合に、
    依存構造を無視して、テキスト上で動詞より手前にある一番近い
    裸の「は」を探す。

    GiNZAが超長文で「Ａは」を、途中の動詞を全部飛び越えて
    文末の語に直接結びつけてしまうことがあり（例：「メタデータ生成部は」が
    「通信端末」に直接nsubjとして付く）、動詞の連鎖を辿る方法では
    原理的に見つけられないため。

    句点（。）をまたいで探さない。「には」のような複合格助詞の
    「は」は対象外にする。また、候補と対象の動詞との間に
    別の「有する」系動詞（＝その節がすでに完結している印）が
    挟まっている場合は、その候補は別の節の主題とみなしてスキップする。
    """
    for i in range(verb.i - 1, -1, -1):
        t = doc[i]
        if t.text == "。":
            break
        if t.text == "は" and t.dep_ == "case" and t.head.pos_ in ("NOUN", "PROPN"):
            if i > 0 and doc[i - 1].pos_ == "ADP":
                continue
            has_boundary = any(
                doc[j].pos_ == "VERB" and doc[j].lemma_ in (HAS_LEMMAS | {"含む"})
                for j in range(i + 1, verb.i)
            )
            if has_boundary:
                continue
            # 候補（「は」の係り先の名詞）が、すでに明示的なnsubjとして
            # 別の動詞に係っている場合、その係り先からverbまでの経路を
            # 辿ってみて、途中で連体修飾（acl＝「〜する◯◯」のような、
            # 別の名詞を説明する節）を挟んでいれば、その候補は
            # 全く別の節（別の名詞句の説明）の主題とみなしてスキップする。
            # 経路が連用修飾（advcl等）だけで動詞から動詞へ直接繋がって
            # いる場合は、同じ節の一部とみなして使ってよい。
            if t.head.dep_ == "nsubj" and t.head.head.i != verb.i:
                cursor = t.head.head
                crosses_acl = False
                reached_verb = False
                seen_path = set()
                while cursor.i not in seen_path:
                    seen_path.add(cursor.i)
                    if cursor.i == verb.i:
                        reached_verb = True
                        break
                    if cursor.dep_ == "acl":
                        # aclを跨いだ先で結局この動詞（verb）自身に辿り着く場合
                        # （＝「Ｘは、〜する第１面と、〜する第２面と、を有し」の
                        # ように、aclが列挙項目自身を説明する連体修飾節で、
                        # その列挙項目自体がverbのobjになっている場合）は、
                        # 「別の名詞句の説明（＝別の節）」ではなく、verbの対象
                        # そのものの内部構造にすぎないため、跨いだだけで即座に
                        # 別節と判定せず、最後まで経路を追ってから判断する。
                        crosses_acl = True
                    if cursor.head.i == cursor.i:
                        break
                    cursor = cursor.head
                if crosses_acl and not reached_verb:
                    continue
            comp = find_component_by_token(components, t.head.i) or find_referenced_component(components, t.head)
            if comp is not None:
                return comp
    return None


def _find_following_capability_owner(doc, components, obj_token, limit_i):
    """
    「Ａを…受信可能な受信部」のように、目的語（Ａ）のすぐ後ろに
    「〜可能な◯◯部」という形が続いている場合、それを本当の持ち主として返す。

    GiNZAの解析では、こうした目的語が「受信部」を飛び越して
    外側の動詞（例：含む）に直接繋がってしまうことがあるため、
    テキストの並び順で「次に出てくる〜可能な部」を優先的に探す。
    limit_i より手前（次のリスト項目の区切りが来る前）までしか探さない。
    """
    for i in range(obj_token.i + 1, limit_i):
        t = doc[i]
        if t.text == "可能" and t.pos_ == "ADJ":
            head_noun = t.head
            if head_noun.pos_ in ("NOUN", "PROPN"):
                comp = find_component_by_token(components, head_noun.i) or find_referenced_component(components, head_noun)
                if comp is not None:
                    return comp
    return None


def extract_copula_relations(doc, components):
    """
    「前記第一検出部及び前記第二検出部は、それぞれ画像センサである」
    のように、「ＸはＹである」という言い切り文（コピュラ文）から
    「Ｘ →（である）→ Ｙ」という関係を抽出する。

    「外径が小である」のような形容詞的な述語（大きさの比較など）や、
    「Ｓａが０．０１２μｍ以下である」のような数値スペック
    （extract_attribute_relations の方で扱う）は対象外にする。
    """
    EXCLUDE_PREDICATES = {"小", "大", "同一", "同じ", "以下", "以上", "未満", "超", "程度", "以内"}
    relations = []
    for token in doc:
        if token.pos_ != "NOUN":
            continue
        if token.text in EXCLUDE_PREDICATES:
            continue
        has_cop = any(c.dep_ == "cop" for c in token.children)
        if not has_cop:
            continue

        nsubj_child = None
        for c in token.children:
            if c.dep_ == "nsubj":
                nsubj_child = c
                break
        if nsubj_child is None:
            continue

        # 「Ａ及びＢは」のように並列の主語になっている場合、
        # nmod連鎖を辿って全部集める
        subjects = []
        stack = [nsubj_child]
        seen = set()
        while stack:
            t = stack.pop()
            if t.i in seen:
                continue
            seen.add(t.i)
            comp = find_component_by_token(components, t.i) or find_referenced_component(components, t)
            if comp is not None and comp not in subjects:
                subjects.append(comp)
            for c in t.children:
                if c.dep_ == "nmod":
                    stack.append(c)
        if not subjects:
            continue

        target_comp = find_component_by_token(components, token.i)
        if target_comp is None:
            continue

        for subj in subjects:
            if subj["text"] == target_comp["text"]:
                continue
            relations.append({
                "source": subj["text"],
                "relation": "である",
                "target": target_comp["text"],
                "type": "direct",
            })
    return relations


def extract_comparison_relations(doc, components):
    """
    「前記第一検出部の焦点位置と前記第二検出部の焦点位置とは、
    …において互いに異なっている」のように、「ＡとＢとは異なる」という
    比較文から、比較されている項目同士の関係を抽出する。
    """
    def has_to_marker(t):
        return any(c.dep_ == "case" and c.text == "と" for c in t.children)

    relations = []
    for verb in doc:
        if verb.lemma_ != "異なる" or verb.pos_ != "VERB":
            continue

        obl_child = None
        for c in verb.children:
            if c.dep_ == "obl":
                obl_child = c
                break
        if obl_child is None:
            continue

        items = []
        comp0 = find_component_by_token(components, obl_child.i) or find_referenced_component(components, obl_child)
        if comp0 is not None:
            items.append(_merged_modifier_name(obl_child, components))

        stack = [c for c in obl_child.children if c.dep_ == "nmod" and has_to_marker(c)]
        seen = set()
        while stack:
            t = stack.pop()
            if t.i in seen:
                continue
            seen.add(t.i)
            comp = find_component_by_token(components, t.i) or find_referenced_component(components, t)
            if comp is not None:
                merged_name = _merged_modifier_name(t, components)
                if merged_name not in items:
                    items.append(merged_name)
            for gc in t.children:
                if gc.dep_ == "nmod" and has_to_marker(gc):
                    stack.append(gc)

        if len(items) < 2:
            continue

        for i in range(len(items) - 1):
            if items[i] == items[i + 1]:
                continue
            relations.append({
                "source": items[i],
                "relation": "とは異なる",
                "target": items[i + 1],
                "type": "direct",
            })

    # 「ＡはＢより小さい（大きい／高い／低い／長い／短い等）」のような
    # 比較表現。「より」で係る語（Ｂ）と、比較の対象（Ａ、通常は
    # nsubj）を抽出する。
    COMPARISON_ADJ = {
        "小さい", "大きい", "高い", "低い", "長い", "短い", "多い", "少ない", "広い", "狭い",
        "薄い", "厚い", "深い", "浅い", "硬い", "柔らかい", "重い", "軽い", "太い", "細い",
        "強い", "弱い", "遠い", "近い", "粗い", "濃い",
    }
    for adj in doc:
        # 「薄くて」「硬くて」のようなテ形（連用中止形）は、GiNZAでpos_が
        # ADJではなくVERBとして解析されることがある。lemma自体は元の
        # 形容詞（薄い／硬い等）のまま保たれるので、pos_をADJに限定せず
        # VERBも許容する（対象のlemma集合を絞っているため誤検出のリスクは
        # 低い）。
        if adj.pos_ not in ("ADJ", "VERB") or adj.lemma_ not in COMPARISON_ADJ:
            continue

        yori_child = None
        for c in adj.children:
            if c.dep_ == "obl" and any(cc.dep_ == "case" and cc.text == "より" for cc in c.children):
                yori_child = c
                break
        if yori_child is None:
            continue
        target_comp = find_component_by_token(components, yori_child.i) or find_referenced_component(components, yori_child)
        if target_comp is None:
            continue
        # 所有格プレフィックス込みの名前を使う（「熱膨張係数」等、持ち主違いの
        # 同名属性が多いため、bareな名前のままだと下のsource=target判定で
        # 別々の実体が誤って同一視され、関係が抽出されなくなってしまう）。
        target_name = _merged_modifier_name(yori_child, components)

        # 比較の対象（Ａ）は、このadj自身のnsubj、またはこのadjの
        # 係り先（複合語の頭、さらにその先の動詞や文全体の主語など）を
        # 辿った先にあるnsubjとして表れることが多い。
        subj_token = None
        for c in adj.children:
            if c.dep_ == "nsubj":
                subj_token = c
                break
        if subj_token is None:
            cursor = adj
            visited = set()
            while cursor.i not in visited:
                visited.add(cursor.i)
                for c in cursor.children:
                    if c.dep_ == "nsubj" and c.i < adj.i:
                        subj_token = c
                        break
                if subj_token is not None:
                    break
                if cursor.head.i == cursor.i:
                    break
                cursor = cursor.head
        if subj_token is None:
            continue
        source_comp = find_component_by_token(components, subj_token.i) or find_referenced_component(components, subj_token)
        if source_comp is None:
            continue
        source_name = _merged_modifier_name(subj_token, components)
        if source_name == target_name:
            continue

        relations.append({
            "source": source_name,
            "relation": f"より{adj.text}",
            "target": target_name,
            "type": "direct",
        })

    # 「ＡがＢを超える（上回る／下回る／満たす）」のような、
    # 動詞による比較・条件表現。ADJの場合と同じく、この動詞自身に
    # nsubjが付いていないことが多く（「〜を超えた場合に〜する」の
    # ように連体修飾として使われるため）、離れた場所にある別の動詞に
    # 主語が誤って直結してしまっていることがあるので、
    # head連鎖を遡ってnsubjを探す。
    COMPARISON_VERBS = {"超える", "上回る", "下回る", "満たす"}
    for verb in doc:
        if verb.pos_ != "VERB" or verb.lemma_ not in COMPARISON_VERBS:
            continue

        obj_token = None
        for c in verb.children:
            if c.dep_ == "obj":
                obj_token = c
                break
        if obj_token is None:
            continue
        target_comp = find_component_by_token(components, obj_token.i) or find_referenced_component(components, obj_token)
        if target_comp is None:
            continue
        target_name = _merged_modifier_name(obj_token, components)

        subj_token = None
        for c in verb.children:
            if c.dep_ == "nsubj":
                subj_token = c
                break
        if subj_token is None:
            cursor = verb
            visited = set()
            while cursor.i not in visited:
                visited.add(cursor.i)
                for c in cursor.children:
                    if c.dep_ == "nsubj" and c.i < verb.i:
                        subj_token = c
                        break
                if subj_token is not None:
                    break
                if cursor.head.i == cursor.i:
                    break
                cursor = cursor.head
        if subj_token is None:
            continue
        source_comp = find_component_by_token(components, subj_token.i) or find_referenced_component(components, subj_token)
        if source_comp is None:
            continue
        source_name = _merged_modifier_name(subj_token, components)
        if source_name == target_name:
            continue

        relations.append({
            "source": source_name,
            "relation": _relation_label(verb),
            "target": target_name,
            "type": "direct",
        })

    return relations


def extract_attribute_relations(doc, components):
    """
    「前記粘着剤層の…表面の算術平均粗さＳａが０．０１２μｍ以下である」
    のように、構成要素の数値スペック（属性）を表す文から
    「持ち主 →（属性名）→ 数値」という関係を抽出する。
    """
    COMPARISON_WORDS = {"以下", "以上", "未満", "超", "程度", "以内"}
    relations = []
    for token in doc:
        if token.text not in COMPARISON_WORDS or token.pos_ != "NOUN":
            continue

        nsubj_token = None
        number_token = None
        unit_token = None
        for child in token.children:
            if child.dep_ == "nsubj":
                nsubj_token = child
            elif child.dep_ == "advmod" and any(ch.isdigit() or ch in "．.０１２３４５６７８９" for ch in child.text):
                number_token = child
            elif child.dep_ == "compound":
                unit_token = child
        if nsubj_token is None or number_token is None:
            continue

        # 属性名（例：算術平均粗さＳａ）を、nsubj自身に直接くっついている
        # 修飾語（compound/amod等）を集めて組み立てる
        attr_words = []
        for c in sorted(nsubj_token.children, key=lambda c: c.i):
            if c.dep_ in ("compound", "amod") or c.pos_ == "PART":
                attr_words.append(c.text)
        attr_words.append(nsubj_token.text)
        attribute_name = "".join(attr_words)

        # 持ち主を、nsubjから「の」（nmod）や「から」（acl→obl）で繋がる
        # 連鎖を辿って探す。「算術平均粗さＳａ」→「表面」→「側」→（遠い）→
        # 「基材」→「粘着剤層」のように、途中に比較のための参照点
        # （基材など）を挟んでいることがあるため、見つかった後も
        # さらに奥（「の」で係る本当の持ち主）がないか探し続け、
        # 最後に見つかったものを採用する。
        owner = None
        current = nsubj_token
        visited = set()
        while current.i not in visited:
            visited.add(current.i)
            next_token = None
            for child in current.children:
                if child.dep_ == "nmod":
                    next_token = child
                    break
                if child.dep_ == "acl" and child.pos_ == "ADJ":
                    for gc in child.children:
                        if gc.dep_ == "obl":
                            next_token = gc
                            break
                    if next_token is not None:
                        break
            if next_token is None:
                break
            comp = find_component_by_token(components, next_token.i) or find_referenced_component(components, next_token)
            if comp is not None:
                owner = comp
            current = next_token
        if owner is None:
            continue

        value_text = number_token.text + (unit_token.text if unit_token is not None else "") + token.text

        relations.append({
            "source": owner["text"],
            "relation": attribute_name,
            "target": value_text,
            "type": "attribute",
        })
    return relations


def extract_composition_relations(doc, components):
    """
    「金属からなる導電部」「群から選択される金属」のように、
    「〜から」＋「なる／選択される／選ばれる」で材料・由来を表す
    パターンから関係を抽出する（マーカッシュ形式でよく使われる）。
    """
    COMPOSITION_LEMMAS = {"なる", "選択", "選ぶ"}
    relations = []
    for verb in doc:
        if verb.lemma_ not in COMPOSITION_LEMMAS or verb.pos_ != "VERB":
            continue

        from_child = None
        for child in verb.children:
            if child.dep_ != "obl":
                continue
            has_kara = any(
                c.dep_ == "case" and c.text == "から" for c in child.children
            )
            if has_kara:
                from_child = child
                break
        if from_child is None:
            continue

        source_comps = []
        stack = [from_child]
        seen = set()
        while stack:
            t = stack.pop()
            if t.i in seen:
                continue
            seen.add(t.i)
            comp = find_component_by_token(components, t.i) or find_referenced_component(components, t)
            if comp is not None and comp not in source_comps:
                source_comps.append(comp)
            for child in t.children:
                if child.dep_ == "nmod":
                    stack.append(child)
        if not source_comps:
            continue

        target_comp = None
        nsubj_child = None
        for child in verb.children:
            if child.dep_ == "nsubj":
                nsubj_child = child
                break
        if nsubj_child is not None:
            target_comp = (
                find_component_by_token(components, nsubj_child.i)
                or find_referenced_component(components, nsubj_child)
            )

        head_noun = verb.head
        if target_comp is None and head_noun.i != verb.i:
            target_comp = find_component_by_token(components, head_noun.i)
            if target_comp is None:
                fallback = find_target_component_from_verb(components, verb)
                target_comp = fallback[0] if fallback else None
        if target_comp is None:
            # GiNZAがこの動詞を誤って文全体の根っこ（head=自分自身）だと
            # 解析してしまっている場合の保険。この動詞は連体修飾
            # （〜される◯◯）として使われていることが多いので、
            # すぐ後ろに出てくる構成要素を係り先とみなす。
            for i in range(verb.i + 1, min(verb.i + 8, len(doc))):
                comp = find_component_by_token(components, i)
                if comp is not None:
                    target_comp = comp
                    break
        if target_comp is None:
            continue

        label = "からなる" if verb.lemma_ == "なる" else "から選択される"
        for source_comp in source_comps:
            if target_comp["text"] == source_comp["text"]:
                continue
            relations.append({
                "source": target_comp["text"],
                "relation": label,
                "target": source_comp["text"],
                "type": "direct",
            })
    return relations


def extract_capability_relations(doc, components):
    """
    「音を出力可能な音出力部」のように、動詞ではなく「〜可能な」という
    形容詞の形で能力を表すパターンから関係を抽出する。

    「Ｘ可能」（ＡＤＪ）が名詞Ｙ（例：音出力部）を修飾している場合、
    その目的語（例：音、ＸのＮＯＵＮ compound「出力」が動詞的働きをする）は
    Ｙ自身の直接の子（obj）としてGiNZAに解析されることが多いため、
    Ｙの子から探す。
    """
    relations = []
    for adj in doc:
        if adj.text != "可能" or adj.pos_ != "ADJ":
            continue

        verb_stem = None
        for child in adj.children:
            if child.dep_ == "compound":
                verb_stem = child.text
                break
        if verb_stem is None:
            continue

        head_noun = adj.head
        if head_noun.pos_ not in ("NOUN", "PROPN"):
            # 「前記所定値は、ユーザにより設定可能である」のように、
            # 「可能」が名詞を修飾する連体形ではなく、文の述語
            # そのものとして使われている場合（＝「可能」自身が
            # 係り先を持たない、またはAUX等に係る場合）に対応する。
            # 「により」で示される動作主から、nsubj（〜は）への
            # 関係として捉える。
            nsubj_token = None
            agent_token = None
            for child in adj.children:
                if child.dep_ == "nsubj":
                    nsubj_token = child
                elif child.dep_ == "obl":
                    has_niyori = any(
                        c.dep_ == "case" and c.text == "に" for c in child.children
                    ) and any(
                        gc.dep_ == "fixed" and gc.text == "より"
                        for c in child.children for gc in c.children
                    )
                    if has_niyori:
                        agent_token = child
            if nsubj_token is None or agent_token is None:
                continue
            nsubj_comp = find_component_by_token(components, nsubj_token.i) or find_referenced_component(components, nsubj_token)
            agent_comp = find_component_by_token(components, agent_token.i) or find_referenced_component(components, agent_token)
            if nsubj_comp is None or agent_comp is None or nsubj_comp["text"] == agent_comp["text"]:
                continue
            relations.append({
                "source": agent_comp["text"],
                "relation": verb_stem,
                "target": nsubj_comp["text"],
                "type": "direct",
            })
            continue

        source = find_component_by_token(components, head_noun.i) or find_referenced_component(components, head_noun)
        if source is None:
            continue

        obj_token = None
        for child in head_noun.children:
            if child.dep_ == "obj":
                obj_token = child
                break
        if obj_token is None:
            for child in adj.children:
                if child.dep_ == "obj":
                    obj_token = child
                    break
        if obj_token is None:
            continue

        target = find_component_by_token(components, obj_token.i) or find_referenced_component(components, obj_token)
        if target is None or target["text"] == source["text"]:
            continue

        relations.append({
            "source": source["text"],
            "relation": verb_stem,
            "target": target["text"],
            "type": "direct",
        })
    return relations


_QUANTIFIER_ONLY_WORDS = {
    "複数", "一部", "全部", "一つ", "ひとつ", "いくつか", "一種", "全て", "すべて",
    "少なくとも一部", "少なくとも一つ", "各々", "それぞれ",
}

# head 自体が「Xの◯◯」という属性・数値的な名詞（熱膨張係数、寸法、厚さ等）で
# 終わる場合に限り、acl の意味上の対象を「Xの」側へ差し替える。
# 「第１封止部分」のような、それ自体で完結した部材名（属性名詞で終わらない）を
# 誤って所有格側に差し替えてしまう事故を防ぐための一般的なガード。
_ATTRIBUTE_HEAD_SUFFIXES = (
    "係数", "値", "率", "量", "数", "径", "幅", "厚さ", "高さ", "寸法", "面積", "体積",
    "温度", "圧力", "速度", "強度", "硬度", "密度", "濃度", "長さ", "大きさ", "重さ",
    "深さ", "広さ", "距離", "角度", "比率", "割合",
)


_SURFACE_LOCATION_WORDS = {
    "主面", "上面", "下面", "表面", "裏面", "側面", "底面", "天面",
    "端面", "内面", "外面", "接触面", "対向面", "外周面", "内周面",
}


def _topic_owner(doc, token, components):
    """
    「Ｘは、…（Ｙに関する部分）…を含み」のように、「一部」「部分」等の
    裸の部分名詞に「Ｘの」という明示的な所有格が付いていない場合でも、
    その名詞が実際には現在の節の主題（トピック）Ｘに属していることが多い。
    tokenより手前で最後に「は」により主題化された構成要素を、その節の
    主題とみなす（新しい「Ｘは、」が現れれば、それ以降は主題が切り替わる
    ため、常に「直前で最後に主題化されたもの」を使えば、節の境界を
    明示的に区切らなくても自然に対応できる）。
    """
    topic = None
    for t in doc:
        if t.i >= token.i:
            break
        if t.dep_ == "case" and t.text == "は":
            comp = (
                find_component_by_token(components, t.head.i)
                or find_referenced_component(components, t.head)
            )
            if comp is not None:
                topic = comp
    return topic


def _genitive_owner(token, components):
    """
    「Ｘの主面」のように、tokenに「の」で係る所有格の名詞（Ｘ）があれば
    それを返す。無ければNone。
    """
    for child in token.children:
        if child.dep_ != "nmod":
            continue
        if any(c.dep_ == "case" and c.text == "の" for c in child.children):
            comp = find_component_by_token(components, child.i) or find_referenced_component(components, child)
            if comp is not None:
                return comp
    return None


def _acl_semantic_target(verb, components):
    """連体修飾節(acl)が構文上かかる名詞(head)ではなく、意味上その節が説明している
    名詞を求める。例:「(基板に)含まれる銅の熱膨張係数」では、GiNZA上は head が
    「熱膨張係数」になるが、「含まれる」が実際に説明しているのは「銅」である。
    head の子に「Xの」という nmod（所有格）があれば、そちらを意味上の対象として
    優先する。特定の請求項に依存しない一般的な構文パターン。

    ただし「複数の」「少なくとも一部の」のような数量詞は実体を指す名詞ではないため
    候補から除外する（例:「半導体層に形成された複数のトランジスタセル」では、
    head の「トランジスタセル」を意味上の対象のままにし、数量詞「複数」に
    差し替えてはならない）。

    さらに、head 自体が「熱膨張係数」のような属性・数値的な名詞で終わる場合に限り
    差し替えを行う。「第１封止部分」のようにそれ自体で完結した部材名の場合は、
    たまたま近くにある別の名詞句（無関係な並列句の一部等）を意味上の対象と
    誤認しないよう、差し替えを行わない。
    """
    if verb.dep_ != "acl":
        return None
    head = verb.head
    if not head.text.endswith(_ATTRIBUTE_HEAD_SUFFIXES):
        return None
    candidates = []
    for child in head.children:
        if child.i == verb.i or child.dep_ != "nmod":
            continue
        if any(c.dep_ == "case" and c.text == "の" for c in child.children):
            comp = find_component_by_token(components, child.i) or find_referenced_component(components, child)
            if comp is not None and comp["text"] not in _QUANTIFIER_ONLY_WORDS:
                candidates.append(child)
    if not candidates:
        return None
    # head直前（＝最も直接的な所有格）を優先する。
    nearest = max(candidates, key=lambda c: c.i)
    return find_component_by_token(components, nearest.i) or find_referenced_component(components, nearest)


def _relation_label(verb):
    """
    動詞トークンから、受身の助動詞（さ/れ/られ等）や過去の「た」を含む
    自然な形の関係ラベルを組み立てる。

    verb.textだけを使うと、「配置された」の「さ」「れ」「た」のように
    助動詞がGiNZAでは別トークン（dep_="aux"）として切り離されている
    ため、関係ラベルが「配置」のように動詞の語幹だけになってしまう
    （「配置される」等に比べて読みにくく、意味も伝わりにくい）。
    また、「位置決めされる」のように、動詞の語幹自体が「位置」
    「決め」の2トークンに分かれ、advclで連結されている場合もある
    （この場合、前半の語幹トークンは主語・目的語を持たない）。

    「て」「おり」のような継続を表す部分は、関係ラベルとしては
    冗長なので含めない。
    """
    prefix = ""
    for c in verb.children:
        if (
            c.dep_ == "advcl"
            and c.i == verb.i - 1
            and not any(gc.dep_ in ("nsubj", "obj", "obl") for gc in c.children)
        ):
            prefix = c.text
            break
    aux_tokens = sorted((c for c in verb.children if c.dep_ == "aux"), key=lambda c: c.i)
    suffix = "".join(a.text for a in aux_tokens)
    return f"{prefix}{verb.text}{suffix}"


def extract_direct_relations(doc, components):
    """
    「Ａに接続されたＢ」（受身）と「Ｂを破砕するＡ」（能動）の
    両方に対応する。受身なら修飾先の名詞(head)が動作の受け手＝target、
    能動なら修飾先の名詞(head)が動作の主体＝sourceになる。

    能動の場合、優先順位は次の通り：
      1) 動詞連鎖を遡って見つかる本当の主語（nsubj）
         例：「テキスト翻訳部は…翻訳して…生成し」の「テキスト翻訳部」
      2) 「により／によって」で明示された手段・道具
         例：「回転カッター式破砕機により破砕する」の「回転カッター式破砕機」
      3) どちらもなければ、修飾先の名詞(head)
    """
    relations = []
    for verb in doc:
        if verb.pos_ != "VERB":
            continue
        if verb.lemma_ in HAS_LEMMAS:
            # 「有する」「備える」「具備する」は extract_has_relations /
            # extract_has_location_relations / extract_positional_relations の
            # 方で別途処理しているのでここでは扱わない
            continue
        if verb.lemma_ in ("超える", "上回る", "下回る", "満たす"):
            # extract_comparison_relations の方で、head連鎖を遡って
            # 正しい主語を探す専用の処理をしているので、ここでは扱わない
            # （そのまま扱うと、誤ったheadを拾って重複した関係になる）
            continue
        if _is_negated(verb):
            # 「〜が行われない」のように否定されている場合、肯定の関係として
            # 抽出してしまうと意味が逆になるため、この動詞からは抽出しない。
            continue
        if verb.lemma_ == "接触" and any(
            child.dep_ == "advcl" and child.lemma_ == "接する" for child in verb.children
        ):
            # 「〜に接するように…接触させて」は extract_contact_relations の方で
            # 別途処理しているのでここでは扱わない（重複防止）
            continue

        # 「ＡはＢに対して〜される」のような、動詞が「に対して」格を
        # 持つ受身文は、その「に対して」格こそが本当のtarget（何に対して
        # 行われるか）であり、nsubj（受け手として書かれている方）が
        # sourceになる。これは通常の受身（head=target）とは逆のパターン
        # なので、他の処理より先に、専用のロジックで処理する。
        if _is_passive(verb):
            taisite_obl = None
            for child in verb.children:
                if child.dep_ == "obl":
                    for c2 in child.children:
                        if c2.dep_ == "case" and any(
                            gc.dep_ == "fixed" and gc.text in ("対し", "対して")
                            for gc in c2.children
                        ):
                            taisite_obl = child
                            break
                if taisite_obl is not None:
                    break
            if taisite_obl is not None:
                nsubj_tok = None
                for child in verb.children:
                    if child.dep_ == "nsubj":
                        nsubj_tok = child
                        break
                if nsubj_tok is not None:
                    source_c = (
                        find_component_by_token(components, nsubj_tok.i)
                        or find_referenced_component(components, nsubj_tok)
                    )
                    target_c = (
                        find_component_by_token(components, taisite_obl.i)
                        or find_referenced_component(components, taisite_obl)
                    )
                    if source_c is not None and target_c is not None and source_c["text"] != target_c["text"]:
                        relations.append({
                            "source": source_c["text"],
                            "relation": _relation_label(verb),
                            "target": target_c["text"],
                            "type": "direct",
                        })
                        continue

        # 「ドレインが、前記第１電極に接続され、ゲートが、…」のように、
        # 受身の動詞連鎖（advcl）が複数連なって最終的に1つの名詞（例：
        # 「第１トランジスタ」）にかかる構文がある。この場合、各動詞
        # 自身がnsubj（ドレイン等）とobl（電極等）を両方直接の子として
        # 持っており、それ自体で完結した（誰が何に接続されるか）関係を
        # 表している。ところが、この動詞のverb.head自身は名詞ではなく
        # 次の動詞（連鎖の続き）であるため、下のhead_component解決
        # ロジックは「係り先が構成要素でない」と判断し、連鎖をずっと
        # 遡って最終的な名詞（第１トランジスタ）まで辿ってしまい、
        # 本来の対象（電極）ではなく連鎖の最終到達点を誤ってtargetに
        # してしまう（結果「ドレイン→接続→第１トランジスタ」のような
        # 誤った関係になり、しかも連鎖中の全ノードが同じ誤ったtargetに
        # 集約されてしまう）。
        # nsubjとoblが両方その動詞自身の直接の子として存在する場合は、
        # 外側の名詞を探しにいく必要が無い、自己完結した関係なので、
        # そちらを最優先で使う。
        if _is_passive(verb):
            self_nsubj = next((c for c in verb.children if c.dep_ == "nsubj"), None)
            self_obl = next(
                (
                    c for c in verb.children
                    if c.dep_ == "obl"
                    and not any(
                        gc.dep_ == "fixed" and gc.text in ("より", "対し", "対して")
                        for gc in c.children
                    )
                ),
                None,
            )
            if self_nsubj is not None and self_obl is not None:
                source_c = (
                    find_component_by_token(components, self_nsubj.i)
                    or find_referenced_component(components, self_nsubj)
                )
                target_c = (
                    find_component_by_token(components, self_obl.i)
                    or find_referenced_component(components, self_obl)
                )
                if source_c is not None and target_c is not None and source_c["text"] != target_c["text"]:
                    relations.append({
                        "source": source_c["text"],
                        "relation": _relation_label(verb),
                        "target": target_c["text"],
                        "type": "direct",
                    })
                    continue

        head_component = find_component_by_token(components, verb.head.i)
        used_fallback = head_component is None
        if head_component is None:
            # 係り先が構成要素でない場合（別の動詞に連なっている等）は、
            # さらに上まで遡って構成要素を探す
            fallback = find_target_component_from_verb(components, verb)
            head_component = fallback[0] if fallback else None
        if head_component is None:
            # それでも見つからない場合、「Ｘは、…を検出し、…を算出する」の
            # ように、この動詞自体は次の動詞（算出）に連なっているだけで
            # 直接の相手を持たないように見えても、動詞自身の目的語（obj）が
            # あれば、それをtargetとして使う。
            own_obj = None
            for child in verb.children:
                if child.dep_ == "obj":
                    own_obj = child
                    break
            if own_obj is not None:
                head_component = (
                    find_component_by_token(components, own_obj.i)
                    or find_referenced_component(components, own_obj)
                )
        if head_component is None:
            continue

        # 「(基板に)含まれる銅の熱膨張係数」のように、連体修飾節(acl)が
        # 構文上かかる名詞(head)が「Xの◯◯」という属性・数値的な名詞
        # （熱膨張係数、寸法、厚さ等）である場合、節が実際に説明しているのは
        # head自身ではなく所有格側（銅）であることが多い。これを優先する。
        acl_target = _acl_semantic_target(verb, components)
        if acl_target is not None and acl_target["text"] != head_component["text"]:
            head_component = acl_target

        if _is_passive(verb):
            # 受身：通常はhead（動詞の係り先）が受け手（target）だが、
            # 「Ｘは、Ｙと〜接続され」のように、動詞のheadがGiNZAの
            # 長文誤解析で見当違いの場所（請求項タイトル等）を指して
            # しまっている場合（＝head_componentがフォールバックでしか
            # 見つからなかった場合）、「は」で明示的にマークされたobl子
            # （実質的な主語＝本当の受け手）の方が正しいtargetであることが
            # 多いので、そちらを優先する。
            passive_target = head_component
            head_is_claim_title = False
            topic_obl = None
            for child in verb.children:
                if (
                    child.dep_ == "obl"
                    and child.text not in RELATION_WORDS
                    and any(c.dep_ == "case" and c.text == "は" for c in child.children)
                ):
                    topic_obl = child
                    break
            if topic_obl is not None:
                topic_comp = (
                    find_component_by_token(components, topic_obl.i)
                    or find_referenced_component(components, topic_obl)
                )
                if topic_comp is not None:
                    passive_target = topic_comp
            else:
                # head_componentが「文書の最後の語＝請求項タイトル」の場合、
                # それは動詞から見て本当に近い係り先ではなく、連体修飾の
                # 連鎖を辿った結果たまたま行き着いただけの可能性が高い。
                # 「Ｘは、Ｙに隣接して配置され」のように、本当のtarget（Ｙ）が
                # 「隣接して」という連用修飾語（advcl）自身のobl子として、
                # 動詞から見て1段階深いところに埋め込まれていることがあるので、
                # そちらを優先する。
                last_real_token_i = len(doc) - 1
                while last_real_token_i > 0 and doc[last_real_token_i].pos_ == "PUNCT":
                    last_real_token_i -= 1
                head_is_claim_title = head_component["end"] == last_real_token_i
                if head_is_claim_title:
                    for child in verb.children:
                        if child.dep_ != "advcl":
                            continue
                        for gc in child.children:
                            if gc.dep_ == "obl":
                                gc_comp = (
                                    find_component_by_token(components, gc.i)
                                    or find_referenced_component(components, gc)
                                )
                                if gc_comp is not None:
                                    passive_target = gc_comp
                                    break
                        if passive_target is not head_component:
                            break

            # 「ＡはＢによりＣに対して位置決めされる」のような、手段格
            # （により）と対象格（に対して）を同時に持つ受身文に対応する。
            # 「により」は動作の手段であって、動作主でも受け手でもないので
            # source/targetの候補から完全に除外する。
            # 「に対して」は動作の対象（＝実質的な受け手）を表すので、
            # 見つかった場合はそちらをtargetとして最優先で使う。
            means_oblique_idx = set()
            taishite_target = None
            for child in verb.children:
                if child.dep_ != "obl":
                    continue
                fixed_texts = {gc.text for gc in child.children if gc.dep_ == "fixed"}
                if "より" in fixed_texts:
                    means_oblique_idx.add(child.i)
                elif "対し" in fixed_texts:
                    means_oblique_idx.add(child.i)
                    if taishite_target is None:
                        comp = (
                            find_component_by_token(components, child.i)
                            or find_referenced_component(components, child)
                        )
                        if comp is not None:
                            taishite_target = comp
            if taishite_target is not None:
                passive_target = taishite_target

            # 動詞に「本物のnsubj」（真の文法上の主語）が直接の子として
            # あるかどうかで、source/targetの向きを決める。
            #   ・nsubjが無い場合（連体修飾節：「Ａに接続されるＢ」のＢの
            #     ように、head_component（またはtopic_obl）が動作の
            #     受け手＝真の主語の代役になっているケース）は、
            #     source=passive_target（主語役）、target=source候補
            #     （に格の相手）が正しい向き。
            #   ・nsubjが直接ある場合（「Ｂは…Ａに配置され」のような
            #     通常の主語付き文）は、そのnsubj自身がsource_candidates
            #     に入ってきており、既にsource=nsubj、target=passive_target
            #     （head_componentのフォールバックで見つかる、位置・相手側の
            #     語）という正しい向きになっている。
            # 以前はnsubjの有無に関わらず常にsource=obl候補
            # （またはnsubj候補）／target=passive_targetという1通りの
            # 向きで固定していたため、nsubjが無いケース（連体修飾節）で
            # 正解データ（人手で作成したゴールドSAO）と方向が系統的に
            # 逆転していた。gold標準との比較でこれが確認されたため、
            # nsubjの有無で場合分けするよう修正する。
            has_real_nsubj_child = any(c.dep_ == "nsubj" for c in verb.children)

            # 動詞自身にnsubjが無く、passive_targetも結局
            # head_component（＝請求項タイトル等へのフォールバック）から
            # 動いていない場合、GiNZAが本当の主語（例：「突起部」）を
            # 動詞ではなくさらに遠いheadの子として誤って結びつけている
            # 可能性がある（extract_positional_relationsで対応した
            # 長距離nsubj誤結合と同種の問題）。_find_nsubj_target_for_verbで
            # 動詞・headの両方の子からnsubjを探し直し、見つかればそちらを
            # 真の主語として優先する。
            resolved_nsubj = None
            if not has_real_nsubj_child and head_is_claim_title and passive_target["text"] == head_component["text"]:
                nsubj_target = _find_nsubj_target_for_verb(doc, components, verb)
                if nsubj_target is not None and nsubj_target["text"] != passive_target["text"]:
                    resolved_nsubj = nsubj_target

            source_candidates = []
            for child in verb.children:
                if topic_obl is not None and child.i == topic_obl.i:
                    # targetとして使った語は、sourceの候補には入れない
                    continue
                if child.i in means_oblique_idx:
                    # 「により」「に対して」で係る語は、上ですでに処理済み
                    # なので、通常のsource候補としては扱わない
                    continue
                if child.dep_ in ("obl", "nsubj"):
                    source_candidates.append(child)
                elif child.dep_ == "advcl":
                    # 「〜と電気的に接続され」のように、本当の動作主（配線等）が
                    # 「電気的に」という副詞句のnmod修飾語として、動詞から見て
                    # 1段階深いところに埋め込まれていることがある。
                    for gc in child.children:
                        if gc.dep_ == "nmod":
                            source_candidates.append(gc)
            for child in source_candidates:
                if child.text == "場合":
                    # 「場合」は条件節の目印であって、動作主ではないので除外する
                    # （GiNZAが超長文でここに主語を誤って結びつけることがある）
                    continue
                source = (
                    find_previous_component_by_word(components, child)
                    or find_referenced_component(components, child)
                )
                if source is None or source["text"] == passive_target["text"]:
                    continue
                if has_real_nsubj_child:
                    relations.append({
                        "source": source["text"],
                        "relation": _relation_label(verb),
                        "target": passive_target["text"],
                        "type": "direct",
                    })
                else:
                    # 「前記放熱装置の主面に配置された取り付けフレーム」のように、
                    # 「に」格の相手（source）が「主面」「上面」のような
                    # 汎用的な面・位置の名詞で、かつ「Ｘの」という所有格の
                    # 相手（放熱装置）を持つ場合は、その面自体は実体のある
                    # 構成要素ではないので関係先から外し、所有格の相手を
                    # 直接の関係先にする。この場合、意味的には「放熱装置に
                    # フレームが配置される」なので、向きも
                    # source=放熱装置（面の持ち主）、target=passive_target
                    # （配置される側）に入れ替える。
                    if resolved_nsubj is not None:
                        relations.append({
                            "source": resolved_nsubj["text"],
                            "relation": _relation_label(verb),
                            "target": source["text"],
                            "type": "direct",
                        })
                        continue
                    owner = None
                    if child.text in _SURFACE_LOCATION_WORDS:
                        owner = _genitive_owner(child, components)
                    if owner is not None and owner["text"] != passive_target["text"]:
                        # 「前記放熱装置の主面に配置された取り付けフレーム」は、
                        # 正しくは「取り付けフレームが（放熱装置の主面に）
                        # 配置される」であり、source=passive_target
                        # （取り付けフレーム）、target=owner+の+主面
                        # （放熱装置の主面）であるべきことを実際の正解データ
                        # （特開2025-188284）との比較で確認した。以前は
                        # source/targetを丸ごと入れ替えていたが、それは逆
                        # だった。丸ごと入れ替えずにこの向きのまま
                        # 「owner+の+主面」をtargetにすると、532件回帰では
                        # 一時的にprecisionがわずかに悪化したが、それは
                        # 「Ｘの主面」を単独ノードとして残していたことが
                        # 原因だった（本人の指示で「主面」等はＸ自体として
                        # 統合する方針に確定し、_merge_surface_location_nodesが
                        # このowner+の+主面を自動的にownerへ統合するようになった
                        # ため、この向きのままで問題なくなった）。
                        relations.append({
                            "source": passive_target["text"],
                            "relation": _relation_label(verb),
                            "target": f"{owner['text']}の{source['text']}",
                            "type": "direct",
                        })
                    else:
                        relations.append({
                            "source": passive_target["text"],
                            "relation": _relation_label(verb),
                            "target": source["text"],
                            "type": "direct",
                        })
        else:
            # 能動：headが直接の係り先として構成要素そのものであれば、それを
            # 主語(source)として使う（例：「Ｘを表すＹ」のＹ＝head）。
            # headがfallback（動詞連鎖を遡って）でしか見つからなかった場合、
            # または head が「文書の最後の語＝請求項タイトル」で、かつ
            # 動詞自身に場所(obl)がある場合（順次列挙形式で複数の動詞が
            # みな最後の装置名にacl接続されてしまうケース）だけ、
            # ①「は」で明示された主題 → ②動詞連鎖を遡った主語(nsubj)
            # → ③動詞自身（またはその1つ先の連なった動詞）の「Ｘに」
            #   （場所を表すobl） → ④テキスト上の直前の「は」
            #   → ⑤「により」の道具、の優先順位で本当の主語を探し直す。
            effective_source = head_component

            last_real_token_i = len(doc) - 1
            while last_real_token_i > 0 and doc[last_real_token_i].pos_ == "PUNCT":
                last_real_token_i -= 1
            head_is_claim_title = head_component["end"] == last_real_token_i

            if used_fallback or head_is_claim_title:
                topic = _find_topic_in_verb_chain(doc, components, verb)
                if topic is not None:
                    effective_source = topic
                else:
                    nsubj_token = _find_nsubj_up_chain(verb)
                    if nsubj_token is not None:
                        subj = (
                            find_component_by_token(components, nsubj_token.i)
                            or find_referenced_component(components, nsubj_token)
                        )
                        if subj is not None:
                            effective_source = subj
                    else:
                        text_topic = _find_nearest_topic_before_text(doc, components, verb)
                        if text_topic is not None:
                            effective_source = text_topic
                        else:
                            own_locative = None
                            if verb.lemma_ != "用いる":
                                # 「用いる」は下の専用フォールバックの方で
                                # 別途処理しているのでここでは対象外にする（重複防止）。
                                # 動詞自身の子、または（それで見つからなければ）
                                # 1つだけ先の連なった動詞の子から場所(obl)を探す
                                # （例：「取り付け」の場所「部」は、共通の親「形成」の
                                #  子になっていて、取り付け自身の直接の子ではないため）。
                                # 遡りすぎるとGiNZAの誤解析を拾ってしまうので1段階まで。
                                # また、既にテキスト上の「は」探しで見つからなかった
                                # 場合の最終手段の1つとして使う（優先度を低くする）。
                                for candidate_v in (verb, verb.head if verb.head.pos_ == "VERB" else None):
                                    if candidate_v is None or own_locative is not None:
                                        continue
                                    for child in candidate_v.children:
                                        if _is_locative_obl(child):
                                            comp = (
                                                find_component_by_token(components, child.i)
                                                or find_referenced_component(components, child)
                                            )
                                            if comp is None:
                                                # 「先端」のように、oblの語自体が位置語として
                                                # 構成要素から除外されている場合、その「の」
                                                # 修飾語（例：「アーム」の先端の「アーム」）を見る
                                                for gc in child.children:
                                                    if gc.dep_ == "nmod":
                                                        comp = (
                                                            find_component_by_token(components, gc.i)
                                                            or find_referenced_component(components, gc)
                                                        )
                                                        if comp is not None:
                                                            break
                                            if comp is not None:
                                                own_locative = comp
                                                break
                            if own_locative is not None:
                                effective_source = own_locative
                            elif verb.lemma_ == "用いる":
                                # 「Ｘを用いて〜する部」のように何重にも入れ子に
                                # なっている場合、鎖の一番奥まで辿り直す
                                outer = _find_outermost_component_from_verb(components, verb)
                                if outer is not None:
                                    effective_source = outer

            if effective_source is head_component:
                for child in verb.children:
                    if _is_instrumental_obl(child):
                        tool = (
                            find_component_by_token(components, child.i)
                            or find_referenced_component(components, child)
                        )
                        if tool is not None:
                            effective_source = tool
                        break

            for child in verb.children:
                if child.dep_ != "obj":
                    continue

                # 「Ａ、Ｂ、Ｃ及びＤを含み」のように、objの前に「及び」等で
                # 繋がれた項目が複数ある場合、それはGiNZAの解析上、
                # 「及び」に近い項目からobjに向かって、nmod修飾語の連鎖
                # として表れる（Ａ→Ｂ→Ｃ→Ｄのように、途中には「及び」が
                # 付かないことが多い）。objの直接の子に「及び」「又は」
                # 等のcc（等位接続）子、または特許請求項で最も多用される
                # 「Ａと、Ｂとを有する」のような格助詞「と」（＝cc扱いに
                # ならない並列）が見つかったら、これは列挙だと判断し、
                # そこから続くnmodの連鎖を全部たどる（532件の正解データを
                # 調査したところ、objが2個以上の列挙になっている1163件中
                # 769件で少なくとも1項目が欠落しており、その大半がこの
                # 「と」による列挙を拾えていないことが原因だった）。
                obj_candidates = [child]
                has_enum_cc = any(
                    gc.dep_ == "cc" or (gc.dep_ == "case" and gc.text in ("と", "や"))
                    for gc in child.children
                )
                if has_enum_cc:
                    cursor = child
                    seen_ids = {cursor.i}
                    while True:
                        nmod_child = None
                        for nm in cursor.children:
                            if nm.dep_ == "nmod" and nm.i < cursor.i and nm.i not in seen_ids:
                                nmod_child = nm
                                break
                        if nmod_child is None:
                            break
                        obj_candidates.append(nmod_child)
                        seen_ids.add(nmod_child.i)
                        cursor = nmod_child

                for obj_cand in obj_candidates:
                    target = (
                        find_component_by_token(components, obj_cand.i)
                        or find_referenced_component(components, obj_cand)
                    )
                    if target is None or target["text"] == effective_source["text"]:
                        continue

                    real_owner = effective_source
                    real_relation = verb.text
                    if verb.lemma_ == "含む":
                        # 「Ａを…受信可能な受信部」のように、目的語の直後に
                        # 「〜可能な部」が続く場合は、そちらを本当の持ち主にする
                        cap_owner = _find_following_capability_owner(doc, components, obj_cand, verb.i)
                        if cap_owner is not None and cap_owner["text"] != target["text"]:
                            real_owner = cap_owner
                    if real_owner["text"] == target["text"]:
                        continue
                    relations.append({
                        "source": real_owner["text"],
                        "relation": real_relation,
                        "target": target["text"],
                        "type": "direct",
                    })

            # 「Ａを送信可能な送信部と、Ｂを受信可能な受信部とを含む通信部」
            # のように、「含む」の対象が複数のリスト項目になっている場合、
            # 単一のobjだけでなく、直前の「であって」区切りから動詞までの
            # 範囲にあるリスト項目もすべて対象にする。
            # 「であって」の区切りが実際に見つかった場合だけ適用する
            # （見つからない場合に文書の先頭まで遡って無関係な項目まで
            #  拾ってしまうのを防ぐため）。
            if verb.lemma_ == "含む":
                scope_start = None
                for j in range(verb.i - 1, -1, -1):
                    if doc[j].text == "。":
                        break
                    if doc[j].text == "て" and doc[j].dep_ == "mark":
                        head_noun = doc[j].head
                        is_de_atte = False
                        for hc in head_noun.children:
                            if hc.dep_ == "cop" and hc.text == "で":
                                if any(gc.dep_ == "fixed" and gc.text in ("あっ", "あり") for gc in hc.children):
                                    is_de_atte = True
                                break
                        if is_de_atte:
                            # 「であって」節の主語が今の所有者と同じものを
                            # 指している場合だけ、この区切りを採用する
                            # （例：「通信部であって」の通信部と、
                            #  「含む」の所有者である通信部が一致する場合）。
                            head_comp = (
                                find_component_by_token(components, head_noun.i)
                                or find_referenced_component(components, head_noun)
                            )
                            if head_comp is not None and head_comp["text"] == effective_source["text"]:
                                scope_start = j + 1
                            break
                if scope_start is not None:
                    for comp in components:
                        if not (scope_start <= comp["end"] < verb.i):
                            continue
                        if not _is_list_item_component(doc, comp, components):
                            continue
                        if comp["text"] == effective_source["text"]:
                            continue
                        relations.append({
                            "source": effective_source["text"],
                            "relation": _relation_label(verb),
                            "target": comp["text"],
                            "type": "direct",
                        })

    unique = []
    seen = set()
    for r in relations:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# ============================================================
# ⑤ 「有する」関係の抽出
# ============================================================

def _is_list_item_component(doc, comp, components=None):
    """
    「Ａと、Ｂと、Ｃと、…を有する（備える）」のような並列列挙で、
    その構成要素の直後に「と、」（区切りの格助詞＋読点）が来ているかどうかを
    判定する。「Ａと通信する」のような、単に「と」で係る場合（直後が
    読点でない）は対象外にする。

    「第１のトランジスタから第６のトランジスタと、…を有し」のような、
    範囲の始点側（「から」が直後に来る場合）も対象に含める
    （「乃至」は解析前に「から」へ正規化されるため、同じ扱いになる）。

    （注：「と」の直後が読点でなく次の構成要素の先頭に直接続く場合
    （「第１電極と第２電極とを有する」等）まで対象に広げることも試したが、
    532件回帰でrecallは上がったもののprecisionが下がり、他の抽出箇所との
    組み合わせでむしろ全体のrecallも下がる逆効果が確認されたため見送った。
    同じ問題は_extract_direct_relationsの列挙連鎖検出側で、より安全な形で
    改善済み。）
    """
    end = comp["end"]
    nxt = end + 1
    if (
        nxt + 1 < len(doc)
        and doc[nxt].text == "と"
        and doc[nxt].dep_ == "case"
        and doc[nxt + 1].text in ("、", "を")
    ):
        return True
    if (
        nxt < len(doc)
        and doc[nxt].text == "から"
        and doc[nxt].dep_ == "case"
        and comp["text"].startswith("第")
    ):
        return True
    return False


def _enumeration_scope_start(doc, components, owner):
    """
    「Ａと、Ｂと、Ｃと、…を有する」の兄弟項目Ｃ自身が、さらにその内部で
    「（Ｃの一部である）ｘと、ｙを有するＣ」のように別の列挙を持つ場合
    （入れ子の列挙）、Ｃの内部列挙の対象を集めるときに、Ｃより前にある
    兄弟項目（ＡやＢ）まで「直前の列挙」として誤って拾ってしまうことが
    ある（ＡやＢもそれぞれ「〜と、」で終わる、独立したリスト項目に
    見えてしまうため）。

    ownerが列挙項目そのもの（is_list_item_component）である場合は、
    ownerより前にある「ownerの直前の兄弟列挙項目」の終わりの位置を
    返し、それ以前のリスト項目は対象候補から除外できるようにする。
    ownerが列挙項目でない場合（根や、列挙と無関係な語）は、制限
    なし（-1）を返す。
    """
    if owner is None or not _is_list_item_component(doc, owner, components):
        return -1
    best = -1
    for c in components:
        if c["end"] < owner["start"] and _is_list_item_component(doc, c, components):
            if c["end"] > best:
                best = c["end"]
    return best


# 【検証済みだが不採用】「単数又は複数の半導体チップ、前記半導体チップに
# 接続された第一電極端子、及び前記半導体チップに接続された第二電極端子を
# 有した…」のように、列挙の各項目自体が関係節（「前記Ｘに接続された」）で
# 修飾されている入れ子の列挙を、その関係節のobl項（「Ｘ」）経由でさらに
# nmod連鎖をたどって回収する _acl_obl_backbone / _has_enum_marker /
# _walk_comma_enum_siblings を実装し、狙った2件
# （特開2025-175400「パワーモジュール 有する 第一電極端子/半導体チップ」、
#  特開2025-187080「第２端子 含む 導電部」）のうち前者は正しく回収できたが、
# 532件回帰で検証した結果、MICRO f1 0.4172→0.4163、MACRO f1 0.4162→0.4151と
# precision・recallともに悪化した（他のクレームで、無関係な語への誤った
# nmod連鎖を「列挙の兄弟」と誤認する副作用が、狙った2件の改善を上回った）。
# 非劣化の原則に反するため不採用。実装（helper関数3つ＋呼び出し箇所）は
# このコミットでは削除し、以前の単純なhas_enum_cc判定に戻した。


# 【検証済みだが効果なしのため不採用】「手がかり句を用いた特許請求項の
# 構造解析」（新森ら, 2004）のCOMPOSE_CUE直前パターン（表3: 「(名詞|記号)
# と(、|,|)?」の繰り返し）に基づき、GiNZAの係り受け木（nmod/acl）を
# 一切経由せず、verb（を有する/備える/含む等）の直前のトークン列だけを
# 表層的に後方へ読んで列挙項目を回収する _scan_cue_enum_targets_backward
# を実装し、既存のnmod連鎖歩き（上記）に追加的に（既存結果を壊さない
# 形で）組み込んで532件回帰を実施した。結果はMICRO/MACRO f1とも
# 1桁も変化なし（既存のnmod連鎖歩きが「と」で終わる単純な列挙は
# 既に全て正しく拾えており、この表層スキャンで新たに拾えた項目は
# 532件中0件だった）。「単数又は複数のＸ、（Ｘを参照する関係節）を
# 含むＹ、及び…を有する」のように、列挙の各項目自体が関係節で修飾
# されている（今回残っている）パターンは、目的語の直前ではなく、
# 目的語を修飾する関係節全体の開始位置の直前に「及び」「、」が来るため、
# この単純な後方スキャンでは検出できない（関係節の左端＝構成要素境界を
# 別途特定する必要があり、GiNZAの依存構造に頼らずに安全にそれを行う
# 方法は未解決）。効果が無いため、コードは追加せず元のnmod連鎖歩きのみ
# に留めた。


def extract_has_relations(doc, components):
    """
    「Ａを有する」「Ａは〜を有し」のような文から (所有者, 有する, 対象) を抽出する。

    3段階で所有者(owner)を判定する:
      1) 明示的な主語（〜は/〜が）がある節 → その主語
      2) 「〜を有する＜名詞＞」のように＜名詞＞を修飾する形（属格用法） →
         その＜名詞＞。対象は、それより前に出てきた構成要素すべて
         （「Ａと、Ｂと、Ｃと、を有するＸ」という並列列挙のパターン用）
      3) どちらでもない（連用形で他の動詞に連なっている場合など） →
         文全体の主語（＝依存構造上のROOTが属する構成要素）
    """
    root_token = None
    for t in doc:
        if t.head == t:
            root_token = t
            break
    root_component = (
        find_component_by_token(components, root_token.i) if root_token is not None else None
    )
    if root_component is not None:
        # 「パワーモジュールの製造方法」のように、クレームタイトルが
        # 「Ｘの＜基本語＞」という所有格付き複合語になっている場合、
        # 直前の「Ｘの」チェーンを含めた最大限の複合語に拡張する
        # （find_full_title_component参照。root_componentが実際に
        # ownerとして使われるのは、他の優先分岐が全部当たらなかった
        # 最後のフォールバックの場合だけなので、この拡張が既存の
        # 他の関係抽出に影響することはない）。
        root_component = find_full_title_component(doc, components, root_component)

    relations = []
    for verb in doc:
        if verb.lemma_ not in HAS_LEMMAS or verb.pos_ != "VERB":
            continue

        subj_token = None
        obj_token = None
        for child in verb.children:
            if child.dep_ == "nsubj":
                # 「Ｘは、Ａを含み、Ｙは、Ｂを有し、Ｚは、…」のように「は」で
                # 区切られた節が連なる長文では、GiNZAが各節自身の主語を
                # その節の動詞ではなく、鎖の先にあるもっと後方の動詞に
                # 誤って直接の子（nsubj）として付けてしまうことがある。
                # その結果、1つの動詞に複数のnsubj候補（本来は別の節の主語）が
                # 付くことがあるが、その動詞に一番近い（＝直前の）ものが、
                # その動詞自身の節の主語である可能性が最も高いため、
                # 最初に見つかったものではなく、動詞に最も近いものを採用する。
                if subj_token is None or child.i > subj_token.i:
                    subj_token = child
            if child.dep_ == "obj" and obj_token is None:
                obj_token = child

        head_component = find_component_by_token(components, verb.head.i)
        if head_component is None and verb.head.pos_ in ("NOUN", "PROPN"):
            # 「〜を有した単数又は複数のパワーモジュール」のように、動詞が
            # 直接かかる先が「単数」「複数」のような、それ自体は構成要素
            # ではない修飾語（nmodで、さらに奥の本当の名詞にかかる）である
            # ことがある。この場合、係り先を1段階で諦めずにさらに奥まで
            # 辿って本当の構成要素（パワーモジュール）を探す。
            # （辿らずに諦めると、後段でroot_component（請求項全体の名称）
            #  にフォールバックしてしまい、本来もっと内側の構成要素が
            #  持つべき部品が、誤って一番外側の構成要素の直下に
            #  ぶら下がってしまう）。
            # ただし、verb.headがVERB（別の節の動詞への長距離誤爬取）の
            # 場合にまで辿ってしまうと、本来root_componentへの
            # フォールバックが正しいケースまで誤って上書きしてしまう
            # ことが判明した（regressionで確認済み）ため、verb.headが
            # NOUN/PROPNの場合だけに限定する。
            _fallback_targets = find_target_component_from_verb(components, verb.head)
            if _fallback_targets:
                head_component = _fallback_targets[0]

        if (
            head_component is not None
            and root_token is not None
            and head_component["end"] == root_token.i
        ):
            # head_componentが、たまたまroot_component（＝クレーム全体の
            # タイトル、依存構造上のROOT）と同じ位置で終わっている場合
            # （「…を備える、パワーモジュールの製造方法。」のように、
            # 「を備える」がacl修飾として直接タイトル名詞にかかる場合）、
            # find_full_title_componentと同じ「Ｘの」拡張を適用する
            # （そうしないと、root_component側だけ拡張しても、この
            # head_component経由の分岐では拡張前の短いタイトルのままに
            # なってしまう）。
            head_component = find_full_title_component(doc, components, head_component)

        targets = []
        owner = None
        # 「Ａと、Ｂと、Ｃと、…を有する（備える）」の並列列挙から得られた
        # ターゲットは、請求項がその動詞で明示的に列挙した最も信頼度の高い
        # 構成要素なので、(start end)を記録しておき、後段（_simplify_hierarchy）
        # で「より具体的な鎖がある」という理由だけで安易に消されないようにする。
        enum_target_keys = set()

        # 「Ａと、Ｂと、Ｃと、…を有する（備える）」のような並列列挙が
        # 2件以上見つかる場合は、それを最優先で使う（「は」探しに
        # 惑わされないようにするため。特に超長文で、無関係な節の
        # 「Ｘは」を拾ってしまう問題を避けられる）。
        # ただし、本当にこの動詞の直前で終わる列挙でなければ
        # 意味がないので、一番近い列挙項目が動詞からそれほど
        # 離れていない場合だけ「最優先」として採用する
        # （そうしないと、文中の別の場所にあるたまたま「と」付きの
        # 語まで全部拾ってしまう）。
        # なお、この距離判定は①の優先分岐だけに使い、後段の
        # 通常のリスト列挙判定（③）には影響させない
        # （変数を分けて持つ）。
        all_list_targets = [
            c for c in components
            if c["end"] < verb.i and _is_list_item_component(doc, c, components)
        ]
        early_list_targets = list(all_list_targets)
        if early_list_targets:
            nearest_end = max(c["end"] for c in early_list_targets)
            if verb.i - nearest_end > 15:
                early_list_targets = []

        acl_owner = None
        acl_targets = []
        if verb.dep_ == "acl" and subj_token is None and obj_token is not None and head_component is not None:
            # 連体修飾節（acl）として名詞にかかる「Ｘを有する／備える／含むＹ」は、
            # 構文上Ｙ（修飾される名詞）がＸを持つ、という意味が一意に決まる
            # （例：「スイッチング機能を有するパワー半導体モジュール」→
            #  パワー半導体モジュールがスイッチング機能を有する）。
            # このパターンは節の主語を持たないため、後段の「周囲の並列列挙を
            # 拾う」ヒューリスティック（②③④）にそのまま処理させると、
            # たまたま近くにある無関係な列挙項目（「Ａと、Ｂと、…を備え」等）を
            # 対象として誤って拾ってしまうことがある。obj自体が明確に対象を
            # 示しているので、その誤りを避けるために最優先で処理する。
            cand = (
                find_component_by_token(components, obj_token.i)
                or find_referenced_component(components, obj_token)
            )
            # ただし、「Ａと、Ｂと、Ｃと、…を備える、Ｘ。」のように、この動詞自体が
            # 列挙を締めくくる（＝Ｘという名詞にかかる）トップレベルのacl節で、
            # かつobj自身も「Ｃと、」の形で他の兄弟項目と並列列挙されている場合は、
            # 話が別である。この場合はobjだけでなく列挙全体（Ａ、Ｂ、Ｃ）がＸの
            # 対象になるべきなので、ここでは処理せず後段の列挙ヒューリスティックに
            # 任せる（objがそれ自体「列挙項目」であるかどうかで判定する）。
            # 【検証済みだが不採用】「第１電極と第２電極とを有する半導体
            # チップ」のように、objが「と、」で終わる典型的な列挙項目
            # ではない場合（コンマが無い）でも、cand自体が
            # _is_list_item_component相当（「と」+「を」で終わる）なら
            # ここでhas_enum_cc＋nmod連鎖を辿って回収する案を試したが、
            # 532件回帰でmicro/macro共にrecallが悪化した（後段の列挙
            # ヒューリスティック②③④に任せた方が広い範囲を正しく拾える
            # ケースの方が多く、ここで早期に横取りすると却って範囲が
            # 狭くなってしまうらしい）。非劣化の原則に反するため不採用。
            if cand is not None and not _is_list_item_component(doc, cand, components):
                if cand["text"] != head_component["text"]:
                    acl_owner = head_component
                    acl_targets = [cand]
                    # 「基板主面及び基板裏面を有し」のように、obj自体は
                    # 「と、」で終わる典型的な列挙項目ではないが、objの
                    # 直接の子に等位接続（cc＝「及び」等）や「と」「や」の
                    # 格助詞が見つかる場合がある。この場合、
                    # _extract_direct_relationsのhas_enum_cc修正と同じ
                    # 考え方で、そこから続くnmodの連鎖を辿って列挙項目を
                    # 全部拾う（そうしないと、objだけが対象になり、それより
                    # 前の兄弟列挙項目＝基板主面が抜け落ちてしまう）。
                    # （入れ子の関係節を経由した拡張案は不採用。上のコメント
                    #  参照）
                    has_enum_cc = any(
                        gc.dep_ == "cc" or (gc.dep_ == "case" and gc.text in ("と", "や"))
                        for gc in obj_token.children
                    )
                    if has_enum_cc:
                        cursor = obj_token
                        seen_ids = {cursor.i}
                        while True:
                            nmod_child = None
                            for nm in cursor.children:
                                if nm.dep_ == "nmod" and nm.i < cursor.i and nm.i not in seen_ids:
                                    nmod_child = nm
                                    break
                            if nmod_child is None:
                                break
                            nm_comp = (
                                find_component_by_token(components, nmod_child.i)
                                or find_referenced_component(components, nmod_child)
                            )
                            if nm_comp is not None and nm_comp["text"] != head_component["text"]:
                                acl_targets.append(nm_comp)
                            seen_ids.add(nmod_child.i)
                            cursor = nmod_child

        if acl_owner is not None:
            owner = acl_owner
            targets = acl_targets
        elif subj_token is not None:
            owner = (
                find_component_by_token(components, subj_token.i)
                or find_referenced_component(components, subj_token)
            )
            t = None
            if obj_token is not None:
                t = (
                    find_component_by_token(components, obj_token.i)
                    or find_referenced_component(components, obj_token)
                )
            # 「ケースは、第１面と、第２面と、を有する」のように、明示的な主語
            # （nsubj）がある節でも、目的語（obj）自体が「〜と、」で終わる
            # 並列列挙の最後の項目である場合がある。この場合、obj_token
            # だけを対象にすると、それより前の兄弟列挙項目（第１面）が
            # 完全に抜け落ちてしまう（第１面はnmodでobjに係っているだけで、
            # 動詞の直接の子ではないため）。objが列挙項目だと判定できる
            # ときは、head_component分岐と同じ考え方で、この節の範囲内の
            # 列挙項目をすべて対象にする。
            scoped_list_targets = []
            if t is not None and owner is not None and _is_list_item_component(doc, t, components):
                scope_start = _enumeration_scope_start(doc, components, owner)
                scoped_list_targets = [
                    c for c in all_list_targets
                    if c["start"] > scope_start and c["text"] != owner["text"]
                ]
                # 「第１電極と第２電極とを有する」のように、最後の項目の直前の
                # 兄弟項目が読点なしで直接次の項目へ続く場合（コンマが無い）、
                # _is_list_item_component（直後が「、」であることを要求）に
                # 引っかからずall_list_targetsに入らない。この場合でも、
                # objからnmodの連鎖をたどって前の兄弟に行き着け、かつその
                # 兄弟自身が「と」「や」の格助詞で終わっている（＝列挙の
                # 一員である）と確認できる場合に限り、対象へ追加する
                # （nmod連鎖なら何でも対象にすると無関係な「Ｘの」修飾まで
                # 拾ってしまうため、「と／や」で終わることの確認を必須にする）。
                cursor = obj_token
                seen_ids = {cursor.i}
                while True:
                    nmod_child = None
                    for nm in cursor.children:
                        if nm.dep_ == "nmod" and nm.i < cursor.i and nm.i not in seen_ids:
                            nmod_child = nm
                            break
                    if nmod_child is None:
                        break
                    seen_ids.add(nmod_child.i)
                    is_enum_sibling = any(
                        gc.dep_ == "case" and gc.text in ("と", "や")
                        for gc in nmod_child.children
                    )
                    if not is_enum_sibling:
                        break
                    nm_comp = (
                        find_component_by_token(components, nmod_child.i)
                        or find_referenced_component(components, nmod_child)
                    )
                    if nm_comp is not None and nm_comp["text"] != owner["text"] and not any(
                        c["start"] == nm_comp["start"] and c["end"] == nm_comp["end"]
                        for c in scoped_list_targets
                    ):
                        scoped_list_targets.append(nm_comp)
                    cursor = nmod_child
            if scoped_list_targets:
                targets = scoped_list_targets
                enum_target_keys.update((c["start"], c["end"]) for c in targets)
            elif t is not None:
                targets.append(t)
        elif len(early_list_targets) >= 2 and (head_component is not None or root_component is not None):
            # head_component（動詞の係り先）がGiNZAの長文誤解析で
            # 見当違いの場所（例：後続の別の節）を指してしまっている
            # ことがあるため、その場合はテキスト上の直前の「Ｘは、」を
            # 優先的に所有者として使う。ただし、head_componentが
            # そもそも見つからず（＝「こと」等でroot_componentに
            # フォールバックする場合）は、topicの方が誤検出のリスクが
            # 高いため使わない。
            if head_component is not None:
                nearby_topic = _find_nearest_topic_before_text(doc, components, verb)
                if nearby_topic is not None:
                    owner = nearby_topic
                elif (
                    root_component is not None
                    and len({c["text"] for c in early_list_targets}) < len(early_list_targets)
                ):
                    # 「〜する工程と、〜する工程と、…を備え」のように、
                    # 列挙項目が同じ語（「工程」等）の繰り返しになっている
                    # 場合、それは方法クレーム特有の並列列挙であり、
                    # 所有者は請求項全体のタイトル（root_component）である
                    # 可能性が高い。head_componentが列挙とは無関係な語を
                    # 誤って指してしまっていることがあるため、
                    # この場合はroot_componentを優先する。
                    owner = root_component
                else:
                    owner = head_component
                targets = [c for c in early_list_targets if c["text"] != owner["text"]]
            else:
                # head_componentが見つからない（＝動詞の係り先が「こと」等で
                # 実在の構成要素ではない）場合でも、「ケースは、第１面と、
                # 第２面と、を有し」のように、テキスト上に近い明示的な主題
                # 「Ｘは、」があるなら、無条件にroot_componentへフォールバック
                # するより、そちらを所有者として優先する方が正確である。
                # ただし、この場合のtargetsは、topicより前にある全く別の
                # 列挙グループ（例：もっと前の「基板と、ケースと、端子と、
                # ワイヤと、を備え」）まで拾ってしまわないよう、topicの
                # 出現位置より後ろにある列挙項目だけに絞る
                # （head_componentが見つかっている場合はこの絞り込みをしない。
                #  head_component自体が動詞の直接の係り先という強い制約に
                #  なっているため、以前からの挙動を変えないようにする）。
                nearby_topic = _find_nearest_topic_before_text(doc, components, verb)
                if nearby_topic is not None:
                    owner = nearby_topic
                    targets = [
                        c for c in early_list_targets
                        if c["text"] != owner["text"] and c["start"] > owner["end"]
                    ]
                else:
                    owner = root_component
                    targets = [c for c in early_list_targets if c["text"] != owner["text"]]
            enum_target_keys.update((c["start"], c["end"]) for c in targets)
        elif _find_nearest_topic_before_text(doc, components, verb) is not None:
            # 明示的なnsubjが見つからなくても、テキスト上に「Ｘは、」という
            # 主題が近くにあれば、そちらを所有者として優先する
            # （GiNZAが超長文で、離れた場所にある「Ｘは」を正しく
            #  この動詞のnsubjとして結びつけられないことがあるため）。
            owner = _find_nearest_topic_before_text(doc, components, verb)
            list_targets = [
                c for c in components
                if c["end"] < verb.i and _is_list_item_component(doc, c, components) and c["text"] != owner["text"]
            ]
            targets = list_targets
            enum_target_keys.update((c["start"], c["end"]) for c in targets)
            if obj_token is not None:
                t = (
                    find_component_by_token(components, obj_token.i)
                    or find_referenced_component(components, obj_token)
                )
                if t is not None and t not in targets and t["text"] != owner["text"]:
                    targets.append(t)
        elif head_component is not None:
            owner = head_component
            # 「Ａと、Ｂと、Ｃと、…を有する」のような並列列挙のパターン用。
            # 直後に「と」が付くリスト項目だけを対象にする
            # （そうしないと、文中の無関係な名詞まで全部拾ってしまうため）。
            # 列挙が見つからない場合（＝単に「Ｘを備える」という単数の
            # 目的語だけの場合）は、動詞自身の目的語（obj）だけを使う
            # （以前は「それより前の構成要素を全部」という広すぎる
            #  フォールバックになっており、無関係な語まで拾っていた）。
            scope_start = _enumeration_scope_start(doc, components, owner)
            scoped_list_targets = [c for c in all_list_targets if c["start"] > scope_start]
            # 「第１電極と第２電極とを有する半導体チップ」のように、この
            # 動詞の目的語（obj_token＝第２電極）自体が列挙項目である
            # 場合でも、_enumeration_scope_start自身が「第２電極」を
            # （owner＝半導体チップの）外側の列挙における直前の兄弟項目と
            # 誤認し、scope_startとして「第２電極」のend自体を返してしまう
            # ことがある（第２電極はこのacl節の内側の目的語であって、
            # 半導体チップと同じ外側の列挙の兄弟ではないのに、テキスト上は
            # 両方とも「〜と」で終わる列挙項目に見えるため区別できない）。
            # その結果scoped_list_targetsが空になり、目的語自身（第２電極）
            # まで対象から漏れてしまう。scope_start由来の絞り込みで
            # 何も残らなかった場合は、目的語自体が列挙項目であるかどうかを
            # 別途確認し、そうであれば目的語自身を起点に、そこから
            # nmod連鎖をたどって読点なしで続く直前の兄弟項目
            # （extract_has_relationsのsubj_token分岐で532件回帰により
            #  検証済みの同じロジック）も回収する。
            if not scoped_list_targets and obj_token is not None:
                obj_comp = (
                    find_component_by_token(components, obj_token.i)
                    or find_referenced_component(components, obj_token)
                )
                if obj_comp is not None and _is_list_item_component(doc, obj_comp, components):
                    scoped_list_targets = [obj_comp]
                    cursor = obj_token
                    seen_ids = {cursor.i}
                    while True:
                        nmod_child = None
                        for nm in cursor.children:
                            if nm.dep_ == "nmod" and nm.i < cursor.i and nm.i not in seen_ids:
                                nmod_child = nm
                                break
                        if nmod_child is None:
                            break
                        seen_ids.add(nmod_child.i)
                        is_enum_sibling = any(
                            gc.dep_ == "case" and gc.text in ("と", "や")
                            for gc in nmod_child.children
                        )
                        if not is_enum_sibling:
                            break
                        nm_comp = (
                            find_component_by_token(components, nmod_child.i)
                            or find_referenced_component(components, nmod_child)
                        )
                        if nm_comp is not None and nm_comp["text"] != owner["text"] and not any(
                            c["start"] == nm_comp["start"] and c["end"] == nm_comp["end"]
                            for c in scoped_list_targets
                        ):
                            scoped_list_targets.append(nm_comp)
                        cursor = nmod_child
            if scoped_list_targets:
                targets = scoped_list_targets
                enum_target_keys.update((c["start"], c["end"]) for c in targets)
            else:
                t = (
                    find_component_by_token(components, obj_token.i)
                    or find_referenced_component(components, obj_token)
                ) if obj_token is not None else None
                targets = [t] if t is not None else []
        else:
            owner = root_component
            if owner is not None:
                # 分岐②と同じく、「Ａと、Ｂと、Ｃと、…を備える」のような
                # 並列列挙のパターンに対応する（「…ことを特徴とする」のように
                # 係り先が「こと」等でhead_componentが見つからない場合に
                # 特によく起きる）。
                scope_start = _enumeration_scope_start(doc, components, owner)
                targets = [
                    c for c in all_list_targets
                    if c["text"] != owner["text"] and c["start"] > scope_start
                ]
                enum_target_keys.update((c["start"], c["end"]) for c in targets)
            if obj_token is not None:
                t = (
                    find_component_by_token(components, obj_token.i)
                    or find_referenced_component(components, obj_token)
                )
                if t is not None and t not in targets and (owner is None or t["text"] != owner["text"]):
                    targets.append(t)

        if owner is None:
            continue

        for target in targets:
            if target is None or target["text"] == owner["text"]:
                continue
            relations.append({
                "source": owner["text"],
                "relation": "有する",
                "target": target["text"],
                "type": "has",
                "from_enumeration": (target["start"], target["end"]) in enum_target_keys,
            })

    unique = []
    seen = set()
    for r in relations:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# ============================================================
# ⑥ 全関係を統合
# ============================================================

def combine_all_relations(positional, direct, has):
    all_relations = list(positional) + list(direct) + list(has)
    unique = []
    seen = set()
    for r in all_relations:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def _simplify_hierarchy(relations, doc=None, components=None):
    """
    根（root）から全ノードへ直接「有する」で繋ぐのではなく、
    より具体的な位置関係・直接関係の鎖（例：破砕槽→内壁面→固定刃）が
    既にある場合は、そちらを優先して根からの重複した「有する」を消す。
    また、位置関係の起点になっているが誰からも指されていないノード
    （例：サイドプレート）は、根の直接の子として補って繋ぎ直す。

    「有する」「備える」が1つも使われていない請求項（順次列挙形式など）
    では has_edges が空になるが、その場合は文末の語（＝発明の名称、
    例：照明装置）を根とみなし、他のどこからも指されていないノードを
    その直接の子として補う。
    """
    has_edges = [r for r in relations if r["type"] == "has"]

    if not has_edges:
        if doc is None or components is None:
            return relations
        last_i = len(doc) - 1
        while last_i > 0 and doc[last_i].pos_ == "PUNCT":
            last_i -= 1
        claim_title = find_component_by_token(components, last_i)
        if claim_title is None:
            return relations

        all_targets = set(r["target"] for r in relations)
        extra = []
        added = set()
        for r in relations:
            if r["type"] not in ("positional", "direct"):
                continue
            s = r["source"]
            if s == claim_title["text"] or s in all_targets or s in added:
                continue
            extra.append({
                "source": claim_title["text"],
                "relation": "有する",
                "target": s,
                "type": "has",
                "claim_number": r.get("claim_number"),
            })
            added.add(s)
        return relations + extra

    owners = set(r["source"] for r in has_edges)
    all_targets = set(r["target"] for r in relations)
    roots = [o for o in owners if o not in all_targets]

    # 根の候補が「複数」または「０個」の場合に、文末の語（＝発明の名称）に
    # 一致するものを優先する（例：「基板」と「ロードポート」の両方が候補に
    # なってしまっても、実際の根は文末の「ロードポート」であるため）。
    # ０個になるのは、他の関係抽出の誤りで「有する」の所有者自身が
    # どこかのtargetにもなってしまっている場合で、まれに起こりうる。
    claim_title = None
    if len(roots) != 1 and doc is not None and components is not None:
        last_i = len(doc) - 1
        while last_i > 0 and doc[last_i].pos_ == "PUNCT":
            last_i -= 1
        claim_title = find_component_by_token(components, last_i)

    if len(roots) > 1:
        if claim_title is not None and claim_title["text"] in roots:
            roots = [claim_title["text"]]
        else:
            # setの反復順（ハッシュ値に依存し実行のたびに変わりうる）に
            # 頼ると、同じ入力でも実行ごとに異なる根が選ばれてしまうため、
            # 文字列として決定的な順序（sorted）にする。
            roots = sorted(roots)
    elif not roots:
        if claim_title is not None and claim_title["text"] in owners:
            roots = [claim_title["text"]]
        elif owners:
            # 同上の理由で、setの反復順ではなくsorted順で決定的に選ぶ。
            roots = [sorted(owners)[0]]

    root = roots[0] if roots else sorted(owners)[0]

    incoming = {}
    for r in relations:
        incoming.setdefault(r["target"], []).append(r)

    # ① 根からの「有する」より具体的な鎖がある場合は、根からの分を消す。
    #    ただし「には有する」「間に有し」「含む」のような“そこに含まれる”系
    #    の関係だけを対象にする（「上に設けられた」「接続」のような
    #    単なる並び関係は対象にしない＝半導体装置の例を壊さないため）。
    to_remove = []
    for r in has_edges:
        if r["source"] != root:
            continue
        if r.get("from_enumeration"):
            # 根が「Ａと、Ｂと、Ｃと、…を備える」のように明示的に列挙した
            # 構成要素は、請求項本文で最も明確に述べられている関係なので、
            # 他の節でたまたま「含む」等の語で触れられているというだけの
            # 理由で消してはいけない（例：「…を備え」で全体の部品として
            # 列挙されているのに、別の文で「Ａ、Ｂを含む被封止部材」のように
            # 副次的に言及されているせいで、備える側の関係が消えてしまう
            # 問題への対応）。
            continue
        n = r["target"]
        more_specific = [
            x for x in incoming.get(n, [])
            if x is not r
            and x["type"] in ("positional", "direct")
            and x["source"] != root
            and ("有" in x["relation"] or "含" in x["relation"] or "からなる" in x["relation"] or "選択される" in x["relation"])
        ]
        if more_specific:
            to_remove.append(r)

    simplified = [r for r in relations if r not in to_remove]

    # ② 位置関係・直接関係の起点になっているのに、誰からも指されていない
    #    ノードは、根の直接の子として補って繋ぐ。
    #    ただし「Ｘに設けられる（設置される）Ｙ」のような、ジェプソン形式の
    #    前提装置（Ｘ）は、根の部品ではなく外側の文脈にすぎないので、
    #    「設け」系の関係の中でしか登場しない語は補完の対象外にする。
    incoming2 = {}
    for r in simplified:
        incoming2.setdefault(r["target"], []).append(r)

    outgoing = {}
    for r in simplified:
        outgoing.setdefault(r["source"], []).append(r)

    # 「Ｘの一部」のように、ある構成要素Ｘの一部分を表す名前は、Ｘ自体が
    # 既に根や他の関係と繋がっている（＝孤立していない）なら、この時点では
    # まだ Ｘ→の→Ｘの一部 という関係が無くても、後段の
    # `_add_genitive_provenance_relations` で必ず繋がる。そのため、
    # ここで「誰からも指されていない孤立ノード」と誤判定して
    # 根から直接「有する」を張ってしまう（＝Ｘの一部があたかも根の
    # 直接の構成要素であるかのような、二重・不正確な関係になる）のを防ぐ。
    known_nodes = set()
    for r in simplified:
        known_nodes.add(r["source"])
        known_nodes.add(r["target"])

    def _has_connected_genitive_owner(node_text):
        idx = node_text.find("の")
        while idx != -1:
            owner_candidate = node_text[:idx]
            if owner_candidate and owner_candidate != node_text and owner_candidate in known_nodes:
                if owner_candidate == root or owner_candidate in incoming2:
                    return True
            idx = node_text.find("の", idx + 1)
        return False

    extra = []
    added = set()
    for r in simplified:
        if r["type"] not in ("positional", "direct"):
            continue
        s = r["source"]
        if s == root or s in added:
            continue
        if s in incoming2:
            continue
        own_relations = outgoing.get(s, [])
        if own_relations and all("設け" in x["relation"] for x in own_relations):
            continue
        if _has_connected_genitive_owner(s):
            continue
        extra.append({"source": root, "relation": "有する", "target": s, "type": "has", "claim_number": r.get("claim_number")})
        added.add(s)

    return simplified + extra


# ============================================================
# ⑦ パイプライン本体：請求項テキスト → 構成要素・関係（単文形式）
# ============================================================

def _clean_claim_text(text):
    """
    請求項テキストの前処理。前後の余分な空白だけを取り除く。

    以前は改行・空白を全部取り除いていたが、改行そのものが
    GiNZAにとって長い請求項を正しく区切って解析するための
    手がかりになっていることが分かったため、内部の改行・空白は
    そのまま残す。「\\n  前記」のように改行・空白が「前記」と
    同じ1つのトークンにくっついてしまう問題は、
    extract_patent_components_general 側で「前記」「該」「うち」を
    途中に出てきても区切るようにして対応済み。

    「乃至」（〜から〜まで、と同じ意味）は、GiNZAの辞書には
    あまり登録されていないらしく、名詞や動詞に誤ってタグ付け
    されてしまうことがある（文全体の構造解析が丸ごと崩れる
    原因になる）。意味が同じで、GiNZAが安定して解析できる
    「から」に置き換えることで回避する。
    """
    text = text.strip()
    text = text.replace("乃至", "から")
    return text


_NUMERIC_THRESHOLD_WORDS = {"以上", "以下", "未満", "超", "以内", "程度"}


def extract_numeric_threshold_relations(doc, components):
    """
    「銅板の硬度が４５Ｈｖ以上であり」「反り量が２μｍ／ｍｍ以下である」の
    ように、名詞化された比較語（以上/以下/未満/超/以内/程度）がnsubjで
    属性名（硬度、反り量等）を取り、その直接の子として数値・単位を
    持つ構文を扱う。

    正解データでは、この種の文を
        (「所有者の属性名」, "以上である"/"以下である"等, 数値＋単位)
    という形（属性名の複合語自体を1つのsourceにし、比較語をrelationに、
    数値だけをtargetにする）で表現していることが多い。
    extract_attribute_relations は同じ構文から別の分解
    （所有者だけをsourceにし、属性名をrelationに、数値＋比較語をまとめて
    targetにする）で関係を作るため、そちらを変更せず、正解データの
    慣習に合わせた形を新たに追加する（既存の抽出結果への影響を避ける）。
    """
    relations = []
    for token in doc:
        if token.text not in _NUMERIC_THRESHOLD_WORDS or token.pos_ != "NOUN":
            continue

        nsubj_token = None
        for child in token.children:
            if child.dep_ == "nsubj":
                nsubj_token = child
                break
        if nsubj_token is None:
            continue

        # 数値＋単位（「４５Ｈｖ」「２μｍ／ｍｍ」等）は、GiNZAの解析上
        # nsubj（属性名）と比較語（以上/以下等）の間に、compound一段だけ
        # とは限らない構造（nummod→nmodの入れ子等）で挟まっていることが
        # あるため、依存木を辿るのではなく、「が/は」の直後から比較語の
        # 直前までのテキスト範囲をそのまま数値として扱う（位置ベース）。
        case_child = None
        for c in nsubj_token.children:
            if c.dep_ == "case":
                case_child = c
        if case_child is None:
            continue
        value_start = case_child.i + 1
        value_end = token.i
        if value_start >= value_end:
            continue
        value_text = "".join(doc[i].text for i in range(value_start, value_end))
        if not value_text or not (
            value_text[0].isdigit() or value_text[0] in "．.０１２３４５６７８９"
        ):
            continue

        attr_words = [
            c.text for c in sorted(nsubj_token.children, key=lambda c: c.i)
            if c.dep_ in ("compound", "amod")
        ]
        attr_words.append(nsubj_token.text)
        attribute_name = "".join(attr_words)

        owner = _genitive_owner_candidate(doc, nsubj_token, components)
        if owner is None:
            continue

        full_name = _normalize_component_text(owner["text"] + "の" + attribute_name)

        relations.append({
            "source": full_name,
            "relation": token.text + "である",
            "target": value_text,
            "type": "attribute",
        })
    return relations


def _extract_raw_relations(text):
    """
    「有する」木構造の階層整理（_simplify_hierarchy）をかける前の、
    生の抽出結果を返す。1つの完全な請求項ではなく、従属請求項の
    追加限定文のような「断片」を解析するときに使う
    （断片だけを見て孤立ノードを無理に根に繋げてしまうのを防ぐため）。
    """
    text = _clean_claim_text(text)
    doc = nlp(text)
    components = extract_patent_components_general(doc)
    relation_words = extract_relation_words_general(doc)

    positional = extract_positional_relations(doc, components, relation_words)
    location = extract_has_location_relations(doc, components)
    installation = extract_installation_relations(doc, components)
    contact = extract_contact_relations(doc, components)
    boundary = extract_boundary_relations(doc, components)
    capability = extract_capability_relations(doc, components)
    composition = extract_composition_relations(doc, components)
    attribute = extract_attribute_relations(doc, components)
    numeric_threshold = extract_numeric_threshold_relations(doc, components)
    copula = extract_copula_relations(doc, components)
    comparison = extract_comparison_relations(doc, components)
    direct = extract_direct_relations(doc, components)
    has = extract_has_relations(doc, components)

    final_relations = combine_all_relations(
        positional + location + installation + boundary,
        direct + contact + capability + composition + attribute + numeric_threshold + copula + comparison,
        has,
    )
    return components, final_relations, doc


_GENITIVE_LINK_EXCLUDE_HEADS = (
    "一方", "他方", "双方", "反対側", "反対面", "反対", "両側", "一端", "他端",
    "一部", "各々", "それぞれ", "夫々", "全て", "全部", "一つ", "1つ", "１つ",
)
# 【検証済みだが不採用】「主面」「部分」等の面・部位の名詞も"一部"と同様に
# 常に除外すべきかと思い、_SURFACE_LOCATION_WORDSと"部分"を丸ごと
# 追加してみたが、532件の正解データを実際に確認すると「第１炭素層の上面」
# 「絶縁基板の裏面」等、"Ｘの◯◯"という面・部位の複合語に対して
# (Ｘ, の, Ｘの◯◯)という出自関係が付与されている例が48件も存在した
# （"主端子の一部"／"主端子の他の部分"のように、一部/部分でも複数の
# 部分参照がある場合はやはり付与されている）。ここで丸ごと除外すると、
# 532件回帰でmicro recallが悪化した（非劣化の原則に反するため不採用）。
# 「放熱装置の主面」（特開2025-188284）のように出自関係が無い方が正しい
# 例も確かにあるが、それを一般語句だけで判別する安全な基準は
# 見つかっていない。


def _add_genitive_provenance_relations(relations):
    """
    「Ｘの深さ」「Ｘの側面」「Ｘのゲート」のように、構成要素名Ｘに「の」を
    介して付いた複合語（＝Ｘの属性・部位・値を表す名詞句）が、他の関係の
    source/targetとして単独のエンティティのように使われている場合、
    正解データでは、その複合語自体に加えて「Ｘは、（複合語）を持つ」という
    出自を示す (Ｘ, "の", Ｘの◯◯) という関係が別途、明示的に付与されている
    ことが非常に多い（例：「凹部の深さ」→(凹部, の, 凹部の深さ)）。

    これは特定のクレームに依存しない一般的なパターンなので、最終的な関係
    リストに登場する全エンティティ名を走査し、「Ｘの◯◯」の形をしていて、
    かつＸ自身も同じクレーム内で独立したエンティティとして使われている
    場合に、この出自関係を機械的に補完する。

    ただし「Ｘの一方」「Ｘの他端」のように、「の」の直後が相対的・指示的な
    語（一方/他方/反対側など）の場合は、それ自体が独立した属性値ではなく
    単なる位置的な言い回しであることが多く、正解データでもこの関係が
    付与されていないことが多いため、除外する。
    """
    entity_texts = set()
    for r in relations:
        entity_texts.add(r["source"])
        entity_texts.add(r["target"])

    new_relations = []
    seen = {(r["source"], r["relation"], r["target"]) for r in relations}
    for e in entity_texts:
        idx = e.find("の")
        if idx <= 0:
            continue
        owner = e[:idx]
        rest = e[idx + 1:]
        if not rest or owner == e:
            continue
        if owner not in entity_texts:
            continue
        if rest.startswith(_GENITIVE_LINK_EXCLUDE_HEADS):
            continue
        if owner.endswith(("正極側", "負極側", "一方側", "他方側", "表側", "裏側")):
            continue
        key = (owner, "の", e)
        if key in seen:
            continue
        seen.add(key)
        new_relations.append({"source": owner, "relation": "の", "target": e, "type": "attribute"})

    return relations + new_relations


def _merge_surface_location_nodes(relations):
    """
    「Ｘの主面」「Ｘの裏面」「Ｘ主面」のように、実体を持たない面・部位の
    名詞（_SURFACE_LOCATION_WORDS）でＸを修飾しただけの複合語は、常にＸ
    そのものとして扱う（本人の指示：「構成要素の一部を表す言葉は、その
    構成要素として扱う」）。

    「一部」「部分」（_merge_partitive_nodes）とは異なり、同じＸに対して
    複数の面（例：Ｘの上面とＸの下面）が区別されている場合でも、常に
    統合する（本人が明示的に確認済み：上面/下面の区別が失われても構わない）。

    「ベース板の第１主面」「ベース板の第２主面」のように、「の」と面の
    名詞の間に順序語等が挟まっている場合も、同じ請求項内に実在する
    構成要素（ここでは「ベース板」）が複合語の前方一致で見つかれば、
    それを所有者として統合する。
    """
    all_nodes = set()
    for r in relations:
        all_nodes.add(r["source"])
        all_nodes.add(r["target"])

    rename_map = {}
    for node in all_nodes:
        for word in sorted(_SURFACE_LOCATION_WORDS, key=len, reverse=True):
            if node == word or not node.endswith(word):
                continue
            candidates = [o for o in all_nodes if o != node and node.startswith(o)]
            if not candidates:
                continue
            owner = max(candidates, key=len)
            rename_map[node] = owner
            break

    if not rename_map:
        return relations

    merged = []
    seen = set()
    for r in relations:
        new_source = rename_map.get(r["source"], r["source"])
        new_target = rename_map.get(r["target"], r["target"])
        if new_source == new_target:
            continue
        new_r = dict(r)
        new_r["source"] = new_source
        new_r["target"] = new_target
        key = (new_source, new_r.get("relation"), new_target)
        if key in seen:
            continue
        seen.add(key)
        merged.append(new_r)
    return merged


_PARTITIVE_SUFFIX_RE = re.compile(r"^(.+?)の(一部|部分)$")


_BARE_PARTITIVE_WORDS = {"一部", "部分"}


def _partitive_siblings(owner_text, all_nodes):
    """
    ownerに対して「ownerの＜…＞一部/部分」という形のノードが、同じ請求項内
    に他にいくつあるかを返す（ownerそのものは含まない）。
    """
    prefix = owner_text + "の"
    return [
        n for n in all_nodes
        if n != owner_text and n.startswith(prefix) and (n.endswith("一部") or n.endswith("部分"))
    ]


def _merge_partitive_nodes(relations, doc=None, components=None):
    """
    「Ｘの一部」「Ｘの部分」のようなノードが、同じ請求項内に「Ｘ」自体も
    独立したノードとして存在する場合、部分と全体を別ノードとして分けて
    表示する意味は薄いので、「Ｘの一部」を「Ｘ」に統合する
    （「Ｘ」がその請求項に登場しない場合は、区別して残す）。

    ただし、同じＸに対して「Ｘの一部」と「Ｘの他の部分」のように複数の
    部分参照がある場合や、「Ｘの内側部分」「Ｘの先端部分」のように
    修飾語で区別されている場合は、それぞれ別の実体を指しているので
    統合しない（532件の正解データを実際に確認し、この基準で「統合すべき
    なのに区別されている」14件と「区別するのが妥当」17件を判別できた
    ことから、正解データ側の作り方とパイプライン側の統合ルールを
    揃えている）。
    """
    all_nodes = set()
    for r in relations:
        all_nodes.add(r["source"])
        all_nodes.add(r["target"])

    rename_map = {}
    for node in all_nodes:
        m = _PARTITIVE_SUFFIX_RE.match(node)
        if m:
            base = m.group(1)
            if base not in all_nodes or base == node:
                continue
            siblings = _partitive_siblings(base, all_nodes)
            if len(siblings) > 1:
                continue  # 「一部」と「他の部分」のように複数の部分参照がある→区別する
            middle = node[len(base) + 1: -2]  # 「Ｘの」と末尾「一部/部分」の間
            if middle:
                continue  # 「内側部分」「先端部分」等、修飾語で区別されている→区別する
            rename_map[node] = base

    # 「Ｘの一部」の「Ｘの」が抽出処理の途中で既に落とされ、ノードの
    # テキストが単に「一部」「部分」だけになっている場合。doc/componentsが
    # 渡されていれば、所有格（「Ｘの」）の相手を辿って同じように統合する。
    if doc is not None and components is not None:
        for node in all_nodes:
            if node in rename_map or node not in _BARE_PARTITIVE_WORDS:
                continue
            for comp in components:
                if comp["text"] != node:
                    continue
                owner = _genitive_owner(doc[comp["end"]], components)
                if owner is None:
                    # 「Ｘの」という明示的な所有格が無い場合、節の主題を
                    # 暗黙の所有者として使う（例:「第１端子は、…部分に
                    # 設けられた貫通穴を含み」→「部分」は第１端子のもの）。
                    owner = _topic_owner(doc, doc[comp["start"]], components)
                if owner is None or owner["text"] not in all_nodes or owner["text"] == node:
                    continue
                if _partitive_siblings(owner["text"], all_nodes):
                    # 明示的な「ownerの＜…＞部分」参照が他にもある請求項では、
                    # 曖昧な裸の「一部/部分」を安易にownerへ統合しない
                    break
                rename_map[node] = owner["text"]
                break

    if not rename_map:
        return relations

    merged = []
    seen = set()
    for r in relations:
        new_source = rename_map.get(r["source"], r["source"])
        new_target = rename_map.get(r["target"], r["target"])
        if new_source == new_target:
            continue
        new_r = dict(r)
        new_r["source"] = new_source
        new_r["target"] = new_target
        key = (new_source, new_r.get("relation"), new_target)
        if key in seen:
            continue
        seen.add(key)
        merged.append(new_r)
    return merged


# ============================================================
# 請求項の表現形式分類
# ============================================================
# 「手がかり句を用いた特許請求項の構造解析」（新森ら, 2004）の分類
# （順次列挙形式／構成要素列挙形式／ジェプソン的形式）をもとに、
# 532件の正解データで検証した結果、順次列挙形式（方法クレーム）だけが
# 他の2形式（構成要素列挙0.439 / ジェプソン的0.413）に比べてF1が
# 極端に低い（0.178）ことを確認済み。方法クレームの中でもさらに、
# 「〜工程」という名詞の繰り返し列挙（既存のroot_componentフォールバック
# ヒューリスティックがそこそこ機能する）と、「〜すること」という動詞の
# 名詞化節による列挙（構成要素として認識されないため、has_relationsの
# 「同じ語の繰り返し列挙→root_componentへ」ヒューリスティックが、別々の
# 節に偶然出てくる同じ実体名を誤って「列挙の兄弟」と誤認し、無関係な
# 誤抽出を多発させる）の2系統があり、後者は特に深刻（F1=0.000の例が
# 複数）。

_METHOD_TITLE_RE = _re_symbolic_label.compile(r"(製造方法|検査方法|測定方法|評価方法|加工方法)。?\s*$")


def classify_claim_format(text):
    """
    請求項テキストを表現形式で分類する。

    戻り値は次のいずれか:
      "順次列挙形式"     … 方法クレーム（「〜工程」の繰り返し列挙、または
                            「〜すること」の動詞名詞化節による列挙）。
                            末尾が「製造方法」等の方法名で終わる、または
                            本文に「工程」が2回以上出現する場合に判定する。
      "ジェプソン的形式"   … 「〜において、」「〜であって、」による前半部/
                            後半部の分割、または「〜を特徴とする」で
                            終わる形式。
      "構成要素列挙形式"   … 上記のいずれにも当たらない、通常の装置クレーム
                            （「Ａと、Ｂと、を備える」等）。

    3形式は本来（新森らの論文でも）排他的ではないが、本パイプラインでは
    「タグ付け（構成要素境界の認識）ルールをどれに切り替えるか」という
    ただ1つの決定のために使うので、優先順位を付けて単一の値を返す
    （順次列挙形式の判定を最優先にするのは、方法クレームが
    「…において、」等のジェプソン的な言い回しを併用していても、
    タグ付けの観点では方法クレーム特有の扱いが優先されるべきため）。
    """
    if _METHOD_TITLE_RE.search(text) or text.count("工程") >= 2:
        return "順次列挙形式"
    if ("を特徴とする" in text) or ("であって、" in text) or ("において、" in text):
        return "ジェプソン的形式"
    return "構成要素列挙形式"


def analyze_claim_ginza_only(text):
    """
    単文形式の請求項テキストを渡すと (構成要素リスト, 関係リスト) を返す。
    Ollamaを一切使わない、GiNZAルールベースのみの解析（再現性が必要な
    厳格F1評価等はこちらを使う）。
    """
    components, final_relations, doc = _extract_raw_relations(text)
    final_relations = _simplify_hierarchy(final_relations, doc, components)
    final_relations = _merge_surface_location_nodes(final_relations)
    final_relations = _add_genitive_provenance_relations(final_relations)
    final_relations = _merge_partitive_nodes(final_relations, doc, components)
    return components, final_relations


def extract_sao_with_local_llm(claim, ginza_sao):
    """
    GiNZAが抽出したSAO候補をローカルLLMに渡し、「抜け（見落とし）」が
    ないかだけを確認してもらう。

    GiNZA候補は既に高精度に調整済みなので、ここではLLMに削除・書き換えの
    権限を与えない（削る役ではなく足す役）。小型ローカルLLMは複雑な列挙
    構文などで不安定なため、GiNZAが既に正しく取れている関係を上書き・
    消去させないことが重要。
    """
    prompt = f"""
あなたは日本語特許請求項のSAO構造をチェックする専門家です。

SAO：
S = Subject（主体・技術要素）
A = Action / Relation（動作・関係）
O = Object（対象・技術要素）

【特許請求項】
{claim}

【GiNZAが抽出したSAO候補（これは正しいものとして扱ってください）】
{json.dumps(ginza_sao, ensure_ascii=False, indent=2)}

あなたの役割は「削除・修正」ではなく「見落としの補完」だけです。

1. GiNZA候補にある関係は、内容が明らかな誤り（実際には存在しない関係）
   でない限り、削除・変更せずそのまま出力に含める。
2. 請求項本文を読んで、GiNZA候補に含まれていない関係のうち、
   本文に明記されている技術的な関係があれば追加する。
3. 1つのSubjectが複数のObjectと関係する場合
   （例：「Ａ、ＢおよびＣとの間に位置する」のような列挙）は、
   Objectごとに1件ずつ、別々のSAOとして出力する。
   まとめて1件にしたり、一部のObjectだけを残したりしない。
4. 「少なくとも」「主に」「さらに」「前記」などを
   単独のSubject/Objectにしない。
5. 「前記○○」は、対応する既出の技術要素名に置き換える。
6. 同じSAOは重複させない。
7. 請求項に書かれていない関係を勝手に作らない（ハルシネーション禁止）。
8. 出力は「GiNZA候補＋あなたが追加した関係」の全件とする。
   一部だけを出力する（間引く）のは禁止。

必ずJSONのみを出力してください。

[
  {{
    "subject": "主体",
    "relation": "関係",
    "object": "対象"
  }}
]
"""

    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
        )
        result = response["message"]["content"]
        match = re.search(r"\[.*\]", result, re.DOTALL)
        if not match:
            return []
        sao = json.loads(match.group())
        valid = []
        for item in sao:
            if not isinstance(item, dict):
                continue
            if not all(key in item for key in ["subject", "relation", "object"]):
                continue
            if not all(
                isinstance(item[key], str) and item[key].strip()
                for key in ["subject", "relation", "object"]
            ):
                continue
            valid.append(item)
        return valid
    except Exception as e:
        print("Local LLM error:", e)
        return []


def _normalize_for_dedup(s):
    """LLM補完とGiNZA結果の重複判定用に、空白だけ除去した比較キーを作る。"""
    return re.sub(r"[\s　]", "", str(s))


def analyze_claim(text):
    """
    請求項をGiNZAで解析し、そのSAO候補をローカルLLMで「補完」する。

    設計方針（重要）：
    GiNZAルールベースの抽出結果は、これまでの精度検証・回帰テストで
    磨き込んだ「信頼できる基準」として扱い、絶対に上書き・削除しない。
    ローカルLLM（Ollama）は、GiNZAが見落とした関係を追加するためだけに
    使う。小型ローカルLLMは複雑な列挙構文などで不安定（関係を削ったり、
    複数Objectのうち一部しか返さなかったりする）ため、GiNZA結果を必ず
    全件保持し、LLMの提案のうちGiNZA結果と重複しないものだけを追加する。
    """
    components, final_relations = analyze_claim_ginza_only(text)

    ginza_sao = [
        {"subject": r["source"], "relation": r["relation"], "object": r["target"]}
        for r in final_relations
    ]

    local_sao = extract_sao_with_local_llm(text, ginza_sao)

    merged_relations = list(final_relations)
    existing_keys = {
        (_normalize_for_dedup(r["source"]), _normalize_for_dedup(r["target"]))
        for r in final_relations
    }

    for rel in local_sao:
        if not isinstance(rel, dict):
            continue
        source = str(rel.get("source", rel.get("subject", ""))).strip()
        relation = str(rel.get("relation", "")).strip()
        target = str(rel.get("target", rel.get("object", ""))).strip()
        if not source or not relation or not target:
            continue
        key = (_normalize_for_dedup(source), _normalize_for_dedup(target))
        if key in existing_keys:
            continue
        merged_relations.append({
            "source": source,
            "relation": relation,
            "target": target,
            "type": "LLM補完",
        })
        existing_keys.add(key)

    return components, merged_relations


# 「LLM不使用」で解析したい場合（app.py タブ2の請求項比較等で使用）。
analyze_claim_ginza = analyze_claim_ginza_only
analyze_claim_with_ginza = analyze_claim_ginza_only


# ============================================================
# ⑧ 自動レイアウト（マインドマップ風：左→右の階層配置）
# ============================================================

from matplotlib.path import Path


def _box_size(text):
    """ノードのラベル文字列から、四角い箱の幅・高さを見積もる"""
    lines = text.split("\n")
    w = max(len(l) for l in lines) * 0.32 + 0.6
    h = 0.5 * len(lines) + 0.5
    return w, h


def _wrap_label(text, max_chars=6):
    """長いノード名は2行に折り返す"""
    if len(text) <= max_chars:
        return text
    mid = len(text) // 2
    return text[:mid] + "\n" + text[mid:]


def _reorder_layers_by_barycenter(layers, G, max_depth, iterations=4):
    """
    同じ階層（列）内のノードの並び順を、隣接する列にある繋がり先の
    平均位置（重心）に合わせて並べ替える。これにより、線同士の交差を
    大きく減らせる（グラフ描画で標準的に使われる重心法）。
    """
    order = {depth: list(nodes) for depth, nodes in layers.items()}
    pos_in_layer = {
        depth: {n: i for i, n in enumerate(nodes)} for depth, nodes in order.items()
    }

    def neighbors_at(node, depth):
        result = []
        for nb in G.predecessors(node):
            if nb in pos_in_layer.get(depth, {}):
                result.append(pos_in_layer[depth][nb])
        for nb in G.successors(node):
            if nb in pos_in_layer.get(depth, {}):
                result.append(pos_in_layer[depth][nb])
        return result

    for _ in range(iterations):
        for depth in range(1, max_depth + 1):
            if depth not in order:
                continue
            scores = {}
            for n in order[depth]:
                idxs = neighbors_at(n, depth - 1)
                scores[n] = sum(idxs) / len(idxs) if idxs else pos_in_layer[depth][n]
            order[depth].sort(key=lambda n: scores[n])
            pos_in_layer[depth] = {n: i for i, n in enumerate(order[depth])}
        for depth in range(max_depth - 1, -1, -1):
            if depth not in order:
                continue
            scores = {}
            for n in order[depth]:
                idxs = neighbors_at(n, depth + 1)
                scores[n] = sum(idxs) / len(idxs) if idxs else pos_in_layer[depth][n]
            order[depth].sort(key=lambda n: scores[n])
            pos_in_layer[depth] = {n: i for i, n in enumerate(order[depth])}

    return order


def compute_layout(G):
    """
    「有する」関係を軸にした、左→右のマインドマップ風レイアウト。
    根（root）を一番左に置き、階層が深くなるほど右に配置する。
    グラフが複数の孤立したグループ（連結成分）に分かれている場合は、
    それぞれを別グループとして縦に並べて配置する。
    同じ階層内のノードは、重心法で並べ替えて線の交差を減らす。
    """
    undirected = G.to_undirected()
    pos = {}
    x_gap = 4.5
    y_gap = 1.0
    y_cursor = 0.0

    for component_nodes in nx.connected_components(undirected):
        subG = G.subgraph(component_nodes)

        has_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get("type") == "has"]
        if has_edges:
            owners = set(u for u, v in has_edges)
            all_targets = set(v for u, v, d in subG.edges(data=True))
            roots = [n for n in owners if n not in all_targets]
            root = roots[0] if roots else next(iter(owners))
        else:
            in_deg = dict(subG.in_degree())
            no_incoming = [n for n in component_nodes if in_deg.get(n, 0) == 0]
            root = no_incoming[0] if no_incoming else next(iter(component_nodes))

        lengths = nx.single_source_shortest_path_length(subG.to_undirected(), root)
        layers = {}
        for node, depth in lengths.items():
            layers.setdefault(depth, []).append(node)

        max_depth = max(layers.keys())
        layers = _reorder_layers_by_barycenter(layers, subG, max_depth)

        # x位置：各深さ（列）ごとに、その列で一番幅の広い箱に合わせて
        # 次の列の開始位置をずらしていく
        depth_x = {}
        cx = 0.0
        for depth in range(max_depth + 1):
            nodes = layers.get(depth, [])
            if not nodes:
                continue
            max_w = max(_box_size(_wrap_label(n))[0] for n in nodes)
            depth_x[depth] = cx
            cx += max_w + x_gap

        comp_pos = {}
        comp_top = 0.0
        for depth, nodes in layers.items():
            heights = [_box_size(_wrap_label(n))[1] for n in nodes]
            total_h = sum(heights) + y_gap * (len(nodes) - 1)
            y = total_h / 2
            for node, h in zip(nodes, heights):
                comp_pos[node] = (depth_x[depth], y - h / 2)
                y -= h + y_gap
            comp_top = max(comp_top, total_h / 2)

        for node, (x, y) in comp_pos.items():
            pos[node] = (x, y + y_cursor)

        y_cursor -= (comp_top * 2 + 3.0)

    for node in G.nodes():
        if node not in pos:
            pos[node] = (0, y_cursor)
            y_cursor -= 3.0

    return pos


# ============================================================
# ⑨ 可視化（マインドマップ風：四角ノード＋曲線）
# ============================================================

TYPE_STYLE = {
    "has":        {"color": "#4C87C6", "label": "階層関係（有する）"},
    "positional": {"color": "#1f77b4", "label": "位置関係"},
    "direct":     {"color": "#2ca02c", "label": "直接関係（接続など）"},
    "attribute":  {"color": "#d18a1a", "label": "属性（数値スペック）"},
}


def _bezier_point_at(verts, t):
    """3次ベジェ曲線（verts=4点）上の、パラメータtの位置を計算する"""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = verts
    mt = 1 - t
    x = mt**3 * x0 + 3 * mt**2 * t * x1 + 3 * mt * t**2 * x2 + t**3 * x3
    y = mt**3 * y0 + 3 * mt**2 * t * y1 + 3 * mt * t**2 * y2 + t**3 * y3
    return x, y


_BRANCH_PALETTE = [
    {"fill": "#EAF2FF", "edge": "#5B8DEF", "line": "#8FB2F5"},   # 青
    {"fill": "#EAFBF1", "edge": "#34A870", "line": "#8FD6B2"},   # 緑
    {"fill": "#FFF3E6", "edge": "#E08A2E", "line": "#F0BE8C"},   # オレンジ
    {"fill": "#FCEAF5", "edge": "#D4529C", "line": "#EBA6CE"},   # ピンク
    {"fill": "#F0EBFF", "edge": "#8A63D2", "line": "#C2ACEE"},   # 紫
    {"fill": "#E9FBFF", "edge": "#2FA3B8", "line": "#8FD6E4"},   # 水色
    {"fill": "#FFF9E0", "edge": "#C9A400", "line": "#E8D480"},   # 黄
    {"fill": "#FDECEA", "edge": "#D9534F", "line": "#EDA6A3"},   # 赤
]
_ROOT_STYLE = {"fill": "#3B4252", "edge": "#3B4252", "line": "#B0B7C6", "text": "#FFFFFF"}


def _assign_branch_colors(G):
    """
    ルートから見て、どの大枝（rootの直接の子）に属するかを求め、
    枝ごとに違う色を割り当てる。NotebookLMのマインドマップのように、
    同じ枝の中は同じ色系統になる。
    「有する」だけでなく、直接関係・位置関係の辺もたどって、
    枝の色を子孫まできちんと引き継ぐ。
    """
    has_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "has"]
    owners = set(u for u, v in has_edges)
    all_targets = set(v for u, v, d in G.edges(data=True))
    roots = [n for n in owners if n not in all_targets]
    root = roots[0] if roots else (next(iter(G.nodes())) if G.nodes() else None)

    # 種類を問わず、隣接ノードを辿れるようにしておく
    neighbors = {}
    for u, v in G.edges():
        neighbors.setdefault(u, []).append(v)
        neighbors.setdefault(v, []).append(u)

    has_children = {}
    for u, v in has_edges:
        has_children.setdefault(u, []).append(v)

    branch_of = {}
    if root is not None:
        branch_of[root] = None
        top_children = has_children.get(root, [])
        for i, c in enumerate(top_children):
            branch_of[c] = i % len(_BRANCH_PALETTE)

        # rootの直接の子（各大枝の起点）から、辺の種類を問わず
        # 全方向にBFSで色を広げていく
        from collections import deque
        queue = deque(top_children)
        while queue:
            node = queue.popleft()
            b = branch_of.get(node)
            for nb in neighbors.get(node, []):
                if nb not in branch_of and nb != root:
                    branch_of[nb] = b
                    queue.append(nb)

    styles = {}
    for n in G.nodes():
        if n == root:
            styles[n] = _ROOT_STYLE
        else:
            b = branch_of.get(n)
            styles[n] = _BRANCH_PALETTE[b] if b is not None else _BRANCH_PALETTE[0]
    return styles, root


# ============================================================
# ⑨.5 Graphviz（dotエンジン）による階層DAGレイアウトでの可視化
# ============================================================
# 「有する」の階層構造に、それをまたぐ直接関係・位置関係の矢印が
# 乗っている今回のデータは、木ではなく「階層型の有向非巡回グラフ
# （DAG）」である。これはSugiyamaフレームワークと呼ばれる、
# 階層型グラフ描画のための確立された理論があり、Graphvizのdot
# エンジンがこれを実装している。自前でレイアウトを計算する
# matplotlib版（visualize_relations）よりも、階層の割り当て・
# 交差の最小化・矢印の迂回を自動でうまくやってくれる。

# 深海テーマの発光パレット（枝ごとに色分け）
_DEEPSEA_PALETTE = [
    {"fill": "#0E3A52", "border": "#5FD4E0", "font": "#E8FBFF"},   # シアン系の発光
    {"fill": "#123A2E", "border": "#4FE0A8", "font": "#E9FFF6"},   # 緑系の発光
    {"fill": "#3A2A4A", "border": "#B98CF0", "font": "#F5EEFF"},   # 紫系の発光
    {"fill": "#4A2438", "border": "#FF8AC0", "font": "#FFEBF4"},   # ピンク系の発光
    {"fill": "#3E3418", "border": "#F0D25C", "font": "#FFF9E0"},   # 黄系の発光
    {"fill": "#123C48", "border": "#6FE8E0", "font": "#EAFFFD"},   # ターコイズ
    {"fill": "#2E1F4A", "border": "#9C7CF0", "font": "#F0EBFF"},   # 藤色
    {"fill": "#0F2A44", "border": "#7CAEFF", "font": "#EAF2FF"},   # 青
]
_DEEPSEA_ROOT = {"fill": "#04121C", "border": "#8FE0F0", "font": "#FFFFFF"}


def _assign_claim_colors(final_relations):
    """
    analyze_dependent_claim() が付与した claim_number を使って、
    どのノードがどの請求項に由来するかを求め、請求項番号ごとに
    違う色を割り当てる（同じ請求項の追加分は同じ色になる）。

    ノードが複数の関係に登場する場合、一番小さい請求項番号
    （＝一番早く登場した請求項）を採用する。
    """
    node_claim = {}
    for r in final_relations:
        num = r.get("claim_number")
        if num is None:
            continue
        for n in (r["source"], r["target"]):
            if n not in node_claim or num < node_claim[n]:
                node_claim[n] = num

    claim_numbers = sorted(set(node_claim.values()))
    color_of_claim = {num: i % len(_BRANCH_PALETTE) for i, num in enumerate(claim_numbers)}

    styles = {}
    for n, num in node_claim.items():
        styles[n] = _BRANCH_PALETTE[color_of_claim[num]]
    return styles, node_claim


def build_graphviz(final_relations, title=None, theme="deepsea", color_by="branch"):
    """
    analyze_claim()等が返した関係リストを、Graphvizのdotエンジンで
    階層型に自動レイアウトしたグラフとして組み立てる。

    color_by: "branch"（既定。根から見た大枝ごとに色分け）、
              "claim"（analyze_dependent_claim()の結果専用。
              どの請求項番号に由来するノードかで色分けする）

    戻り値は graphviz.Digraph オブジェクト。
    Jupyter/Colabではそのまま表示でき、Streamlitでは
    st.graphviz_chart(戻り値) でそのまま描画できる。
    """
    import graphviz

    G = nx.DiGraph()
    for r in final_relations:
        G.add_node(r["source"])
        G.add_node(r["target"])
        G.add_edge(r["source"], r["target"], relation=r["relation"], type=r["type"])

    g = graphviz.Digraph(engine="dot")
    g.attr(
        rankdir="LR", splines="spline", nodesep="0.25", ranksep="0.85",
        bgcolor="transparent",
    )
    if title:
        g.attr(label=title, labelloc="t", fontsize="20",
               fontname="IPAexGothic",
               fontcolor="#E8FBFF" if theme == "deepsea" else "#233044")

    if len(G.nodes()) == 0:
        return g

    node_claim = {}
    if color_by == "claim":
        node_styles, node_claim = _assign_claim_colors(final_relations)
        # claim_numberが分からないノード（孤立ノード補完の前に消えた等）は
        # 通常の大枝ベースの色分けで補う
        branch_styles, _root = _assign_branch_colors(G)
        for n in G.nodes():
            if n not in node_styles:
                node_styles[n] = branch_styles.get(n, _BRANCH_PALETTE[0])
    else:
        node_styles, _root = _assign_branch_colors(G)

    palette = _DEEPSEA_PALETTE if theme == "deepsea" else _BRANCH_PALETTE
    root_style = _DEEPSEA_ROOT if theme == "deepsea" else _ROOT_STYLE

    def _style_for(n):
        s = node_styles.get(n)
        if s is _ROOT_STYLE:
            return root_style
        if s in _BRANCH_PALETTE:
            return palette[_BRANCH_PALETTE.index(s)]
        return palette[0]

    g.attr("node", shape="box", style="rounded,filled", fontname="IPAexGothic",
           fontsize="12", margin="0.18,0.1", penwidth="1.8")
    g.attr("edge", fontname="IPAexGothic", fontsize="10", penwidth="1.6")

    added = set()
    for n in G.nodes():
        style = _style_for(n)
        label = n
        if color_by == "claim" and n in node_claim:
            label = f"【請求項{node_claim[n]}】\n{n}"
        g.node(
            n,
            label=label,
            fillcolor=style["fill"],
            color=style["border"],
            fontcolor=style["font"],
        )
        added.add(n)

    for u, v, d in G.edges(data=True):
        line_style = _style_for(v)
        g.edge(u, v, label="→ " + d["relation"], color=line_style["border"],
               fontcolor=line_style["border"] if theme == "deepsea" else "#445566")

    return g


def visualize_relations(final_relations, title="特許請求項の構成要素間関係"):
    """analyze_claim()等が返した関係リストを渡すと、マインドマップ風の図を描画する"""
    G = nx.DiGraph()
    for r in final_relations:
        G.add_node(r["source"])
        G.add_node(r["target"])
        G.add_edge(r["source"], r["target"], relation=r["relation"], type=r["type"])

    if len(G.nodes()) == 0:
        print("関係が抽出できませんでした。構成要素や依存構造を確認してください。")
        return

    node_styles, _root_node = _assign_branch_colors(G)
    labels = {n: _wrap_label(n) for n in G.nodes()}
    pos = compute_layout(G)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    fig_w = max(14, (max(xs) - min(xs)) * 1.5)
    fig_h = max(6, (max(ys) - min(ys)) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    used_types = set(nx.get_edge_attributes(G, "type").values())

    # ノードの右端・左端に複数の線がつながる場合、全部を中心点に
    # 集中させず、相手ノードのy座標順に少しずつ上下にずらして
    # 接続することで、線同士の重なり・交差を目立たなくする。
    edge_list = list(G.edges(data=True))

    def _edge_dir(u, v):
        xu = pos[u][0]
        xv = pos[v][0]
        return "R" if xv >= xu else "L"

    # (node, "R"/"L") -> [(相手ノード, 辺の向き"out"/"in"), ...]（相手のyでソート済み）
    groups = {}
    for u, v, d in edge_list:
        du = _edge_dir(u, v)
        dv = _edge_dir(v, u)
        groups.setdefault((u, du), []).append(("out", v))
        groups.setdefault((v, dv), []).append(("in", u))

    edge_offset = {}  # (u, v, 'out'/'in') -> オフセット量
    for (node, direction), items in groups.items():
        items_sorted = sorted(items, key=lambda item: -pos[item[1]][1])
        n = len(items_sorted)
        h = _box_size(labels[node])[1]
        span = h * 0.75
        for idx, (kind, other) in enumerate(items_sorted):
            offset = 0.0 if n <= 1 else (idx / (n - 1) - 0.5) * span
            edge_offset[(node, other, kind)] = offset

    # 同じノードから出る辺が複数ある場合、ラベルの位置(t)を少しずつ
    # ずらして重ならないようにするための連番を振る
    source_seen = {}
    source_total = {}
    for u, v, d in edge_list:
        source_total[u] = source_total.get(u, 0) + 1

    # ------------------------------------------------------
    # 辺（ベジェ曲線）を先に描く
    # ------------------------------------------------------
    for u, v, d in edge_list:
        branch_style = node_styles.get(v, node_styles.get(u, _BRANCH_PALETTE[0]))
        line_color = branch_style["line"] if branch_style is not _ROOT_STYLE else "#B0B7C6"
        edge_color = branch_style["edge"] if branch_style is not _ROOT_STYLE else "#8A93A6"
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w0, h0 = _box_size(labels[u])
        w1, h1 = _box_size(labels[v])

        off_u = edge_offset.get((u, v, "out"), 0.0)
        off_v = edge_offset.get((v, u, "in"), 0.0)

        if abs(x1 - x0) < 0.01:
            # 同じ列（兄弟ノード）同士の接続：右側に迂回する縦方向の曲線にする
            sx, sy = x0 + w0 / 2, y0 + off_u
            tx, ty = x1 + w1 / 2, y1 + off_v
            bulge = 0.7 + abs(y1 - y0) * 0.12
            verts = [(sx, sy), (sx + bulge, sy), (tx + bulge, ty), (tx, ty)]
            path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        elif x1 >= x0:
            sx, sy = x0 + w0 / 2, y0 + off_u
            tx, ty = x1 - w1 / 2, y1 + off_v
            dx = (tx - sx) * 0.5
            verts = [(sx, sy), (sx + dx, sy), (tx - dx, ty), (tx, ty)]
            path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        else:
            sx, sy = x0 - w0 / 2, y0 + off_u
            tx, ty = x1 + w1 / 2, y1 + off_v
            dx = (tx - sx) * 0.5
            verts = [(sx, sy), (sx + dx, sy), (tx - dx, ty), (tx, ty)]
            path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])

        patch = mpatches.PathPatch(path, facecolor="none", edgecolor=line_color, lw=2.4, zorder=1,
                                    capstyle="round")
        ax.add_patch(patch)

        arrow_dx = 0.15 if tx > sx else -0.15
        ax.annotate("", xy=(tx, ty), xytext=(tx - arrow_dx, ty),
                    arrowprops=dict(arrowstyle="-|>", color=edge_color, lw=2.0))

        # ラベル位置：同じsourceから複数の辺が出ている場合、
        # tを0.3〜0.7の範囲でずらして重なりを減らす
        n_from_source = source_total.get(u, 1)
        idx = source_seen.get(u, 0)
        source_seen[u] = idx + 1
        if n_from_source > 1:
            t = 0.28 + 0.44 * (idx / (n_from_source - 1))
        else:
            t = 0.5
        mx, my = _bezier_point_at(verts, t)
        # 曲線の接線方向（微小区間の差分）を使って法線オフセットを求める
        px, py = _bezier_point_at(verts, min(t + 0.05, 1.0))
        ddx, ddy = px - mx, py - my
        dd = (ddx ** 2 + ddy ** 2) ** 0.5
        if dd > 0:
            off_x, off_y = -ddy / dd * 0.22, ddx / dd * 0.22
        else:
            off_x, off_y = 0, 0
        ax.text(mx + off_x, my + off_y, "→ " + d["relation"], fontsize=9, fontproperties=FONT_PROP,
                ha="center", va="center", color=edge_color, fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=line_color,
                          linewidth=1.1, alpha=0.95), zorder=3)

    # ------------------------------------------------------
    # ノード（角丸四角、枝ごとに色分け）
    # ------------------------------------------------------
    for n in G.nodes():
        x, y = pos[n]
        w, h = _box_size(labels[n])
        style = node_styles.get(n, _BRANCH_PALETTE[0])
        text_color = style.get("text", "#233044")
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.22",
            linewidth=1.8, edgecolor=style["edge"], facecolor=style["fill"], zorder=4,
        )
        ax.add_patch(box)
        ax.text(x, y, labels[n], fontsize=10.5, fontproperties=FONT_PROP,
                ha="center", va="center", zorder=5, color=text_color, fontweight="bold")

    ax.set_title(title, fontproperties=FONT_PROP, fontsize=18, pad=25)
    ax.set_xlim(min(xs) - 3, max(xs) + 3)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# ⑩ 箇条書き形式（「識別子：説明文。」の並び）に対応した解析
# ============================================================

import re

BULLET_LINE_RE = re.compile(r'^\s*([^\s：:。]{1,10})[：:]\s*(.+?)\s*$')


def _split_bullets(text):
    """
    テキストをヘッダー行と箇条書き行（識別子：説明文）に分ける。
    箇条書きが2つ未満なら None を返す（＝箇条書き形式ではない）。
    """
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]

    header_lines = []
    bullets = []

    for line in lines:
        m = BULLET_LINE_RE.match(line)
        if m:
            identifier, desc = m.group(1), m.group(2)
            if not desc.endswith("。"):
                desc += "。"
            bullets.append((identifier, desc))
        else:
            header_lines.append(line)

    if len(bullets) < 2:
        return None, None

    return "".join(header_lines), bullets


def _main_noun_of(text):
    """
    文の主要な語（＝その文が定義している対象）を返す。
    「〜の破砕槽。」のような名詞述語文では、依存構造上のROOTが
    その定義対象になっていることを利用する。
    （文中に同じ文字列が別の場所で先に出てきていても、
      ROOTベースで判定するので誤って重複除去されない）
    """
    doc = nlp(text)

    root = None
    for t in doc:
        if t.head == t:
            root = t
            break

    if root is not None and root.pos_ in {"NOUN", "PROPN"}:
        start = root.i
        i = root.i - 1
        while i >= 0 and doc[i].dep_ == "compound" and doc[i].head.i == start:
            start = i
            i -= 1
        return "".join(tok.text for tok in doc[start:root.i + 1])

    # ROOTが名詞でない場合のフォールバック
    components = extract_patent_components_general(doc)
    if components:
        return components[-1]["text"]
    nouns = [t.text for t in doc if t.pos_ in {"NOUN", "PROPN"}]
    return nouns[-1] if nouns else text.strip("。")


def _find_reference_relation(identifier, desc):
    """
    説明文の中に他の識別子（例：装置Ｂ）への言及があれば、
    そこで使われている動詞（得た、得られた等）を関係名として返す。
    """
    m = re.search(re.escape(identifier) + r'(?:で|から|より)?(得られた|得た)', desc)
    if m:
        return m.group(1)
    if identifier in desc:
        return "由来"
    return None


def analyze_claim_with_bullets(text, include_internal_detail=False):
    """
    箇条書き形式（識別子：説明文）にも対応した解析。
    箇条書きが見つからない場合は、通常の analyze_claim() にフォールバックする。
    """
    header_text, bullets = _split_bullets(text)

    if bullets is None:
        return analyze_claim(text)

    # ------------------------------------------------------
    # ① ヘッダー文から全体を表す装置名（コンテナ）を特定
    # ------------------------------------------------------
    container = _main_noun_of(header_text) if header_text.strip() else "全体"

    # ------------------------------------------------------
    # ② 各箇条書きの「実質的な名前（エイリアス）」を決定
    # ------------------------------------------------------
    alias = {}
    for identifier, desc in bullets:
        alias[identifier] = _main_noun_of(desc)

    relations = []

    # container --有する--> 各項目
    for identifier, _ in bullets:
        relations.append({
            "source": container,
            "relation": "有する",
            "target": alias[identifier],
            "type": "has",
        })

    # ------------------------------------------------------
    # ③ 項目間の参照関係（例：装置Ｃは装置Ｂ由来）
    # ------------------------------------------------------
    for identifier, desc in bullets:
        for other_id, _ in bullets:
            if other_id == identifier:
                continue
            if other_id in desc:
                rel_label = _find_reference_relation(other_id, desc) or "由来"
                relations.append({
                    "source": alias[other_id],
                    "relation": rel_label,
                    "target": alias[identifier],
                    "type": "direct",
                })

    # ------------------------------------------------------
    # ④ （オプション）各項目の内部構造も展開する場合
    # ------------------------------------------------------
    components = [{"text": container, "start": -1, "end": -1}]
    for identifier, desc in bullets:
        components.append({"text": alias[identifier], "start": -1, "end": -1})

        if include_internal_detail:
            sub_components, sub_relations = analyze_claim(desc)
            other_ids = [oid for oid, _ in bullets if oid != identifier]
            for sr in sub_relations:
                # 他の項目識別子（装置Ｂなど）への言及に由来する関係は、
                # ③ですでに扱っているのでここでは除外する
                if sr["source"] in other_ids or sr["target"] in other_ids:
                    continue
                relations.append(sr)
            components.extend(sub_components)

    # 重複整理
    unique_relations = []
    seen = set()
    for r in relations:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique_relations.append(r)

    return components, unique_relations


# ============================================================
# ⑪ 箇条書きの中の1項目だけを詳細展開する
# ============================================================

def get_bullet_detail(text, identifier):
    """
    箇条書き形式の請求項から、指定した識別子（例："装置Ｂ"）の
    説明文だけを取り出して、その内部構造（構成要素・関係）を返す。

    戻り値: (components, relations, alias)
        alias … その項目の実質的な名前（例:"破砕槽"）
    """
    _, bullets = _split_bullets(text)
    if bullets is None:
        raise ValueError("箇条書き形式が見つかりませんでした。")

    other_ids = [oid for oid, _ in bullets if oid != identifier]

    target_desc = None
    for oid, desc in bullets:
        if oid == identifier:
            target_desc = desc
            break

    if target_desc is None:
        raise ValueError(f"識別子 '{identifier}' が見つかりませんでした。")

    alias = _main_noun_of(target_desc)
    sub_components, sub_relations = analyze_claim(target_desc)

    # 他の項目識別子への言及に由来する関係は除外（この図には不要なので）
    filtered_relations = [
        r for r in sub_relations
        if r["source"] not in other_ids and r["target"] not in other_ids
    ]

    return sub_components, filtered_relations, alias


# ============================================================
# ⑫ 汎用エントリーポイント：どんな請求項でも自動で全体像＋各項目の詳細を出す
# ============================================================

def _has_internal_structure(desc, min_has_count=2):
    """
    説明文の中に「有する」（活用形含む）が複数回出てくる場合、
    内部にさらに構成要素があるとみなす。
    """
    doc = nlp(desc)
    count = sum(1 for t in doc if t.lemma_ in HAS_LEMMAS and t.pos_ == "VERB")
    return count >= min_has_count


def analyze_and_visualize(text, min_has_count=2):
    """
    どんな請求項テキストを渡しても対応する、汎用のエントリーポイント。

    - 箇条書き形式でなければ：1枚の図をそのまま描画する。
    - 箇条書き形式であれば：
        ① まず全体像（各項目間の「有する」「得た」等の関係）を1枚描画し、
        ② 内部に構成要素をさらに持っていそうな項目（"有する"が複数回出る説明文）
           を自動検出し、それぞれについて内部構造の詳細図をもう1枚ずつ描画する。

    戻り値: 描画したタイトルのリスト（確認用）
    """
    _, bullets = _split_bullets(text)
    titles = []

    if bullets is None:
        # 単文形式：今まで通り1枚
        components, relations = analyze_claim(text)
        visualize_relations(relations, title="構成要素間関係")
        titles.append("構成要素間関係")
        return titles

    # ① 全体像
    components, relations = analyze_claim_with_bullets(text, include_internal_detail=False)
    visualize_relations(relations, title="全体構成")
    titles.append("全体構成")

    # ② 内部構造を持っていそうな項目を自動検出して、それぞれ詳細図を描画
    for identifier, desc in bullets:
        if _has_internal_structure(desc, min_has_count=min_has_count):
            try:
                detail_components, detail_relations, alias = get_bullet_detail(text, identifier)
            except ValueError:
                continue
            if not detail_relations:
                continue
            title = f"{alias}（{identifier}）の内部構造"
            visualize_relations(detail_relations, title=title)
            titles.append(title)

    return titles


# ============================================================
# ⑬ 手動で親子関係を付け替える（自動判定できない場合の補正用）
# ============================================================

def reparent_nodes(relations, mapping):
    """
    指定したノードの「有する」による親を、別のノードに付け替える。
    元々あった「有する」の親は取り除き、指定した新しい親からの
    「有する」を必ず追加する（位置関係など他の種類の辺はそのまま残す）。

    mapping: {子ノード名: 新しい親ノード名} の辞書
    例: reparent_nodes(relations, {"回転軸": "回転カッター式破砕機"})
    """
    filtered = [
        r for r in relations
        if not (r["type"] == "has" and r["target"] in mapping)
    ]
    for child, new_parent in mapping.items():
        filtered.append({
            "source": new_parent,
            "relation": "有する",
            "target": child,
            "type": "has",
        })
    return filtered


def merge_nodes(relations, merges):
    """
    「実は同じもの」を指す2つのノード名を1つに統合する。
    Ａ＝Ｂだと判断した場合、Ｂ側のすべての出現をＡに書き換える。

    merges: {統合して消したい名前: 残す方の名前} の辞書
    例: merge_nodes(relations, {"回転カッター式破砕機": "破砕槽"})
        → 「回転カッター式破砕機」という表記をすべて「破砕槽」に統一する
    """
    def rename(name):
        return merges.get(name, name)

    renamed = []
    for r in relations:
        renamed.append({
            "source": rename(r["source"]),
            "relation": r["relation"],
            "target": rename(r["target"]),
            "type": r["type"],
        })

    # 統合した結果、自分自身への矢印（Ａ→Ａ）や重複は取り除く
    unique = []
    seen = set()
    for r in renamed:
        if r["source"] == r["target"]:
            continue
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# ============================================================
# ⑮ 特許類似度診断：①素朴なJaccard係数
# ============================================================

def relations_to_triple_set(relations, normalize_numbers=False):
    """
    analyze_claim() 等が返した関係リストを、比較しやすい
    (source, relation, target) の集合に変換する。

    normalize_numbers=True にすると、「第１」「第２」のような
    請求項ごとに変わりうる番号付けを取り除いて比較する
    （別々の特許同士を比べるときに、番号の違いだけで
    一致しなくなるのを防ぐため）。
    """
    import re as _re

    def _norm(text):
        if normalize_numbers:
            # 「第１」「第２」等の番号を取り除く
            text = _re.sub(r"第[０-９0-9一二三四五六七八九十]+", "", text)
        return text

    triples = set()
    for r in relations:
        triples.add((_norm(r["source"]), _norm(r["relation"]), _norm(r["target"])))
    return triples


def jaccard_similarity(relations_a, relations_b, normalize_numbers=True):
    """
    2つの請求項（analyze_claim()の関係リスト）から、
    素朴なJaccard係数（共通するSAOトリプルの割合）を計算する。

    戻り値: (類似度スコア(0〜1), 共通トリプルの集合,
             Aだけのトリプルの集合, Bだけのトリプルの集合)
    """
    set_a = relations_to_triple_set(relations_a, normalize_numbers)
    set_b = relations_to_triple_set(relations_b, normalize_numbers)

    common = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    union = set_a | set_b

    score = len(common) / len(union) if union else 0.0
    return score, common, only_a, only_b


def print_jaccard_report(relations_a, relations_b, name_a="請求項A", name_b="請求項B", normalize_numbers=True):
    """jaccard_similarity() の結果を、人が読みやすい形で表示する"""
    score, common, only_a, only_b = jaccard_similarity(relations_a, relations_b, normalize_numbers)

    print(f"=== {name_a} vs {name_b} ===")
    print(f"Jaccard類似度: {score:.3f}")
    print(f"共通トリプル数: {len(common)}")
    print(f"{name_a}のみ: {len(only_a)}件")
    print(f"{name_b}のみ: {len(only_b)}件")
    print()
    print("--- 共通するトリプル ---")
    for t in sorted(common):
        print(" ", t)
    print()
    print(f"--- {name_a}だけにあるトリプル ---")
    for t in sorted(only_a):
        print(" ", t)
    print()
    print(f"--- {name_b}だけにあるトリプル ---")
    for t in sorted(only_b):
        print(" ", t)

    return score


# ============================================================
# ⑮.5 特許類似度診断：②意味マッチング（埋め込み＋ハンガリアン法）
# ============================================================
# sentence-transformers は重いライブラリなので、実際に②を使う瞬間まで
# 読み込まない（起動を遅くしないため）。

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        # 多言語対応の定番モデル（日本語も含む。ライブラリとの互換性が良い）
        _embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    return _embed_model


def _triple_to_text(triple):
    source, relation, target = triple
    return f"{source}が{target}を{relation}"


def semantic_similarity(relations_a, relations_b, normalize_numbers=True):
    """
    埋め込み＋ハンガリアン法による、意味を考慮した類似度診断。
    表記が違っても、意味が近ければ高いスコアで対応付けられる。

    戻り値: (類似度スコア(0〜1), マッチしたペアのリスト
             [(トリプルA, トリプルB, 類似度), ...])
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    triples_a = sorted(relations_to_triple_set(relations_a, normalize_numbers))
    triples_b = sorted(relations_to_triple_set(relations_b, normalize_numbers))

    if not triples_a or not triples_b:
        return 0.0, []

    model = _get_embed_model()
    texts_a = [_triple_to_text(t) for t in triples_a]
    texts_b = [_triple_to_text(t) for t in triples_b]

    emb_a = model.encode(texts_a, normalize_embeddings=True)
    emb_b = model.encode(texts_b, normalize_embeddings=True)

    sim_matrix = emb_a @ emb_b.T

    n, m = sim_matrix.shape
    size = max(n, m)
    cost = np.ones((size, size))
    cost[:n, :m] = 1 - sim_matrix

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m:
            matches.append((triples_a[r], triples_b[c], float(sim_matrix[r, c])))

    total = sum(sim for _, _, sim in matches)
    score = total / size

    return score, matches


def print_semantic_report(relations_a, relations_b, name_a="請求項A", name_b="請求項B",
                           normalize_numbers=True, near_match_threshold=0.6):
    """semantic_similarity() の結果を、人が読みやすい形で表示する"""
    score, matches = semantic_similarity(relations_a, relations_b, normalize_numbers)
    matches_sorted = sorted(matches, key=lambda x: -x[2])

    print(f"=== {name_a} vs {name_b}（意味マッチング） ===")
    print(f"類似度スコア: {score:.3f}")
    print()
    print("--- マッチしたペア（類似度が高い順） ---")
    for ta, tb, sim in matches_sorted:
        if ta == tb:
            marker = "="
        elif sim >= near_match_threshold:
            marker = "≒"
        else:
            marker = "×"
        print(f"  [{sim:.2f}] {marker}  {ta}   /   {tb}")

    return score


# ============================================================
# ⑯ 特許類似度診断：③グラフ構造の比較
# ============================================================

def _has_tree_profile(relations):
    """
    「有する」の木構造の「形」を数値化する。
    構成要素の名前（意味）は一切見ず、木の深さ・枝分かれの仕方
    ・関係の種類の内訳だけを見るので、単語が全く違う特許同士でも
    「組み立て方が似ているか」を比較できる。
    """
    has_edges = [r for r in relations if r["type"] == "has"]

    children = {}
    for r in has_edges:
        children.setdefault(r["source"], []).append(r["target"])

    owners = set(r["source"] for r in has_edges)
    all_targets = set(r["target"] for r in relations)
    roots = [o for o in owners if o not in all_targets]
    root = roots[0] if roots else (next(iter(owners)) if owners else None)

    # 深さ・枝分かれ数（各ノードの子の数）をBFSで求める
    depths = {}
    branching = []
    if root is not None:
        from collections import deque
        depths[root] = 0
        queue = deque([root])
        while queue:
            node = queue.popleft()
            kids = children.get(node, [])
            if kids:
                branching.append(len(kids))
            for k in kids:
                if k not in depths:
                    depths[k] = depths[node] + 1
                    queue.append(k)

    max_depth = max(depths.values()) if depths else 0
    num_nodes = len(set(r["source"] for r in relations) | set(r["target"] for r in relations))

    type_counts = {}
    for r in relations:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1

    return {
        "max_depth": max_depth,
        "num_nodes": num_nodes,
        "branching": sorted(branching, reverse=True),
        "type_counts": type_counts,
    }


def _cosine_of_dicts(dict_a, dict_b):
    """2つの {キー: 個数} 辞書を、共通のキー空間のベクトルとみなしてコサイン類似度を計算する"""
    keys = set(dict_a) | set(dict_b)
    if not keys:
        return 1.0
    vec_a = [dict_a.get(k, 0) for k in keys]
    vec_b = [dict_b.get(k, 0) for k in keys]
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _list_similarity(list_a, list_b):
    """2つの数値リスト（枝分かれ数のリストなど）を、長さを揃えてコサイン類似度で比較する"""
    n = max(len(list_a), len(list_b))
    if n == 0:
        return 1.0
    a = list(list_a) + [0] * (n - len(list_a))
    b = list(list_b) + [0] * (n - len(list_b))
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def structural_similarity(relations_a, relations_b):
    """
    グラフの「形」だけを比較した類似度診断（③）。
    単語の意味は一切見ないので、①②（内容の類似度）と組み合わせて
    使うことで、「内容も構造も似ている」「内容は似ているが
    構成の仕方が違う」といった、より詳しい診断ができる。

    戻り値: (総合スコア(0〜1), 内訳のdict)
    """
    prof_a = _has_tree_profile(relations_a)
    prof_b = _has_tree_profile(relations_b)

    depth_a, depth_b = prof_a["max_depth"], prof_b["max_depth"]
    depth_sim = 1.0 - abs(depth_a - depth_b) / max(depth_a, depth_b, 1)

    size_a, size_b = prof_a["num_nodes"], prof_b["num_nodes"]
    size_sim = 1.0 - abs(size_a - size_b) / max(size_a, size_b, 1)

    branching_sim = _list_similarity(prof_a["branching"], prof_b["branching"])
    type_sim = _cosine_of_dicts(prof_a["type_counts"], prof_b["type_counts"])

    # 4つの指標の単純平均を総合スコアにする
    overall = (depth_sim + size_sim + branching_sim + type_sim) / 4

    detail = {
        "深さの類似度": depth_sim,
        "規模(ノード数)の類似度": size_sim,
        "枝分かれパターンの類似度": branching_sim,
        "関係の種類の内訳の類似度": type_sim,
        "Aの深さ/ノード数/枝分かれ": (depth_a, size_a, prof_a["branching"]),
        "Bの深さ/ノード数/枝分かれ": (depth_b, size_b, prof_b["branching"]),
    }
    return overall, detail


def print_structural_report(relations_a, relations_b, name_a="請求項A", name_b="請求項B"):
    """structural_similarity() の結果を、人が読みやすい形で表示する"""
    overall, detail = structural_similarity(relations_a, relations_b)

    print(f"=== {name_a} vs {name_b}（構造の比較） ===")
    print(f"総合スコア: {overall:.3f}")
    print(f"  深さの類似度: {detail['深さの類似度']:.3f}")
    print(f"  規模(ノード数)の類似度: {detail['規模(ノード数)の類似度']:.3f}")
    print(f"  枝分かれパターンの類似度: {detail['枝分かれパターンの類似度']:.3f}")
    print(f"  関係の種類の内訳の類似度: {detail['関係の種類の内訳の類似度']:.3f}")
    print()
    d_a, s_a, b_a = detail["Aの深さ/ノード数/枝分かれ"]
    d_b, s_b, b_b = detail["Bの深さ/ノード数/枝分かれ"]
    print(f"{name_a}: 深さ={d_a}, ノード数={s_a}, 枝分かれ={b_a}")
    print(f"{name_b}: 深さ={d_b}, ノード数={s_b}, 枝分かれ={b_b}")

    return overall


def print_full_diagnosis(relations_a, relations_b, name_a="請求項A", name_b="請求項B",
                          use_semantic=True, weights=(0.3, 0.4, 0.3)):
    """
    ①Jaccard・②意味マッチング・③構造比較の3つをまとめて実行し、
    総合診断結果を表示する。
    weights: (Jaccardの重み, 意味マッチングの重み, 構造比較の重み)
    """
    jaccard_score, _, _, _ = jaccard_similarity(relations_a, relations_b)

    if use_semantic:
        semantic_score, _ = semantic_similarity(relations_a, relations_b)
    else:
        semantic_score = None

    structural_score, _ = structural_similarity(relations_a, relations_b)

    print(f"########## {name_a} vs {name_b}：総合診断 ##########")
    print(f"①Jaccard類似度　　　: {jaccard_score:.3f}")
    if semantic_score is not None:
        print(f"②意味マッチング類似度: {semantic_score:.3f}")
    print(f"③構造の類似度　　　　: {structural_score:.3f}")

    if semantic_score is not None:
        w1, w2, w3 = weights
        total = w1 * jaccard_score + w2 * semantic_score + w3 * structural_score
    else:
        w1, w3 = weights[0], weights[2]
        total = (w1 * jaccard_score + w3 * structural_score) / (w1 + w3)

    print(f"---")
    print(f"総合類似度スコア: {total:.3f}")
    return total



# ============================================================
# ⑰ クレームの広さ・狭さのスコア化
# ============================================================
# SAOグラフの構造だけから、「このクレームはどれくらい抽象的
# （広い）か、具体的（狭い）か」を数値化する。
# 絶対的な尺度ではなく、複数のクレーム同士を相対的に比べるための
# 指標であることに注意（例：独立項と従属項の比較、改良前後の
# クレーム案の比較など）。

def compute_claim_scope_score(relations):
    """
    次の4つの観点から「狭さスコア」を計算する（各0〜1に正規化して平均）。
    ・構成要素の数が多いほど → 限定要素が多い → 狭い
    ・数値スペック（属性）の数が多いほど → 強く限定される → 狭い
      （数値限定は権利範囲を大きく狭める典型的な手法のため、重めに見る）
    ・「有する」階層の深さが深いほど → 細部まで規定されている → 狭い
    ・関係の密度（関係の総数 ÷ 構成要素数）が高いほど →
      構成要素同士の制約が多い → 狭い

    戻り値: (狭さスコア(0〜1、大きいほど狭い), 内訳のdict)
    """
    nodes = set()
    for r in relations:
        nodes.add(r["source"])
        nodes.add(r["target"])
    num_nodes = len(nodes)
    num_relations = len(relations)

    type_counts = {}
    for r in relations:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
    attribute_count = type_counts.get("attribute", 0)

    prof = _has_tree_profile(relations)
    max_depth = prof["max_depth"]

    density = num_relations / num_nodes if num_nodes else 0.0

    # 各要素を0〜1程度にならす（上限を決めてクリップする）
    node_score = min(num_nodes / 20, 1.0)
    attr_score = min(attribute_count / 3, 1.0)
    depth_score = min(max_depth / 4, 1.0)
    density_score = min(density / 2, 1.0)

    narrowness = round(node_score * 0.3 + attr_score * 0.3 + depth_score * 0.2 + density_score * 0.2, 3)
    breadth = round(1 - narrowness, 3)

    detail = {
        "構成要素数": num_nodes,
        "関係の総数": num_relations,
        "数値スペックの数": attribute_count,
        "階層の深さ": max_depth,
        "関係密度": round(density, 2),
        "内訳スコア": {
            "構成要素数": round(node_score, 2),
            "数値スペック": round(attr_score, 2),
            "階層の深さ": round(depth_score, 2),
            "関係密度": round(density_score, 2),
        },
    }
    return narrowness, breadth, detail


def print_scope_report(relations, name="請求項"):
    """compute_claim_scope_score() の結果を、人が読みやすい形で表示する"""
    narrowness, breadth, detail = compute_claim_scope_score(relations)

    print(f"=== {name}：クレームの広さ・狭さ ===")
    print(f"狭さスコア: {narrowness:.3f}　（広さスコア: {breadth:.3f}）")
    print(f"  構成要素数: {detail['構成要素数']}")
    print(f"  関係の総数: {detail['関係の総数']}")
    print(f"  数値スペックの数: {detail['数値スペックの数']}")
    print(f"  「有する」階層の深さ: {detail['階層の深さ']}")
    print(f"  関係密度(関係数/構成要素数): {detail['関係密度']}")
    print(f"  内訳スコア: {detail['内訳スコア']}")

    return narrowness


def compare_claim_scope(relations_list, names=None):
    """
    複数の請求項の狭さスコアを一括で計算し、狭い順に並べて表示する。
    relations_list: [relations, relations, ...]（analyze_claim()の戻り値の2番目）
    """
    if names is None:
        names = [f"請求項{i+1}" for i in range(len(relations_list))]

    rows = []
    for name, relations in zip(names, relations_list):
        narrowness, breadth, detail = compute_claim_scope_score(relations)
        rows.append((name, narrowness, breadth, detail))

    rows.sort(key=lambda x: -x[1])

    print(f"{'請求項':25s} {'狭さ':>8s} {'広さ':>8s} {'要素数':>6s} {'数値':>6s} {'深さ':>6s} {'密度':>6s}")
    for name, narrowness, breadth, detail in rows:
        print(f"{name:25s} {narrowness:>8.3f} {breadth:>8.3f} "
              f"{detail['構成要素数']:>6d} {detail['数値スペックの数']:>6d} "
              f"{detail['階層の深さ']:>6d} {detail['関係密度']:>6.2f}")

    return rows


# ============================================================
# ⑱ バッチ処理：1件を大量の既存請求項と比較して上位を絞り込む
# ============================================================
# 実務で本当に必要なのは「1対1」ではなく「1対大量」の比較。
# 埋め込み計算はコストが高いので、そのまま全件にハンガリアン法を
# かけると遅すぎる。そこで、検索エンジンと同じ2段階方式を取る：
#   ①まず全件を「文書全体の平均ベクトル」同士の単純なコサイン類似度
#     で高速に絞り込む（速いが粗い）
#   ②絞り込んだ上位だけ、精密な意味マッチング（②で作ったハンガリアン法）
#     で改めてスコアをつけ直す（遅いが正確）

def build_patent_database(records, show_progress=True):
    """
    records: [(id, 請求項テキスト), ...] のリスト
    （idは特許番号や管理番号など、何でもよい）

    各請求項をあらかじめSAO解析し、文書全体の平均埋め込みベクトルを
    計算してデータベース（リスト）として返す。
    このデータベースは一度作れば使い回せるので、検索のたびに
    全件を解析し直す必要がなくなる。
    """
    import numpy as np

    model = _get_embed_model()
    database = []
    total = len(records)
    for i, (rid, text) in enumerate(records):
        if show_progress:
            print(f"[{i+1}/{total}] {rid} を解析中...")
        try:
            _, relations = analyze_claim(text)
        except Exception as e:
            print(f"  → 解析エラー、スキップします: {e}")
            continue
        if not relations:
            continue
        triples = sorted(relations_to_triple_set(relations, normalize_numbers=True))
        texts = [_triple_to_text(t) for t in triples]
        embeddings = model.encode(texts, normalize_embeddings=True)
        doc_embedding = np.mean(embeddings, axis=0)
        doc_embedding = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-8)

        database.append({
            "id": rid,
            "text": text,
            "relations": relations,
            "doc_embedding": doc_embedding,
        })
    return database


def search_similar_claims(query_text, database, top_k=10, rerank_k=5):
    """
    query_text（新しく調べたい請求項）を、build_patent_database() で
    作ったデータベースの中から検索し、似ているものを上位から返す。

    ①文書全体の平均ベクトルによる高速な粗いスコアで、
      データベース全件からtop_k件に絞り込む
    ②その中の上位rerank_k件だけ、精密な意味マッチング
      （ハンガリアン法）でスコアを付け直す

    戻り値: [{"id":.., "fast_score":.., "precise_score":(あれば),
              "text":.., "matches":(rerankした場合のみ)}, ...]
             fast_scoreの高い順（rerank後はprecise_scoreの高い順）
    """
    import numpy as np

    _, query_relations = analyze_claim(query_text)
    if not query_relations:
        return []

    model = _get_embed_model()
    triples = sorted(relations_to_triple_set(query_relations, normalize_numbers=True))
    texts = [_triple_to_text(t) for t in triples]
    embeddings = model.encode(texts, normalize_embeddings=True)
    query_embedding = np.mean(embeddings, axis=0)
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

    scored = []
    for entry in database:
        fast_score = float(np.dot(query_embedding, entry["doc_embedding"]))
        scored.append({"id": entry["id"], "text": entry["text"],
                        "relations": entry["relations"], "fast_score": fast_score})

    scored.sort(key=lambda x: -x["fast_score"])
    top_candidates = scored[:top_k]

    for entry in top_candidates[:rerank_k]:
        precise_score, matches = semantic_similarity(query_relations, entry["relations"])
        entry["precise_score"] = precise_score
        entry["matches"] = matches

    reranked = [e for e in top_candidates if "precise_score" in e]
    not_reranked = [e for e in top_candidates if "precise_score" not in e]
    reranked.sort(key=lambda x: -x["precise_score"])

    return reranked + not_reranked


def print_search_results(results, query_name="検索クエリ"):
    """search_similar_claims() の結果を、人が読みやすい形で表示する"""
    print(f"=== 「{query_name}」に似ている請求項（上位{len(results)}件） ===")
    for i, r in enumerate(results):
        precise = f", 精密スコア={r['precise_score']:.3f}" if "precise_score" in r else ""
        print(f"{i+1}. [{r['id']}] 粗いスコア={r['fast_score']:.3f}{precise}")
    return results


# ============================================================
# ⑳ 従属請求項の展開
# ============================================================
# 「請求項１に記載の◯◯」という従属請求項は、親請求項の内容を
# 文章として繰り返さないため、そのままanalyze_claim()に渡しても
# 追加された限定文言しか抽出できず、親請求項が本来持っている
# 構成要素が全部抜け落ちてしまう。
# ここでは、親請求項の本文と、従属請求項の追加限定を自動でつなぎ
# 合わせ、単独で解析できる完全な文章に組み立て直す。

import re as _re_dep


def _split_claim_title(text):
    """
    請求項テキストの末尾にある「発明の名称」を、コンマの位置に
    頼らず、既存の構成要素抽出ロジックを再利用して正確に切り出す。
    戻り値: (発明の名称を除いた本文, 発明の名称)
    """
    text = text.strip()
    doc = nlp(text)
    components = extract_patent_components_general(doc)

    last_i = len(doc) - 1
    while last_i > 0 and doc[last_i].pos_ == "PUNCT":
        last_i -= 1

    title_comp = find_component_by_token(components, last_i)
    if title_comp is None:
        return "", text

    start_char = doc[title_comp["start"]].idx
    end_char = doc[last_i].idx + len(doc[last_i].text)
    body = text[:start_char]
    title = text[start_char:end_char]
    return body, title


_CLAIM_REF_PATTERN = _re_dep.compile(
    r"請求項(?P<nums>(?:請求項|[0-9０-９]+|[、,及びおよび又はまたはからー～\-乃至])+)"
    r"(?:のいずれか)?(?:[0-9０-９一二三四五六七八九十]+項?)?"
    r"(?:に)?(?:記載の|おいて)"
)


def _parse_claim_ref(text):
    """
    「請求項１に記載の」「請求項１又は２に記載の」「請求項１記載の」
    「請求項１から３のいずれか一項に記載の」
    「請求項１乃至請求項４のいずれか一において」等から、
    参照している請求項番号と、その表現の位置を取り出す。
    参照が見つからなければ None を返す（＝独立請求項）。
    """
    m = _CLAIM_REF_PATTERN.search(text)
    if not m:
        return None
    span = m.group("nums")
    span_half = span.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    nums = [int(n) for n in _re_dep.findall(r"\d+", span_half)]
    if not nums:
        return None
    if any(c in span for c in ("から", "-", "ー", "～", "乃至")) and len(nums) >= 2:
        nums = list(range(nums[0], nums[-1] + 1))
    is_preamble_style = m.group(0).endswith("おいて")
    match_end = m.end()
    if not is_preamble_style:
        # 「請求項１に記載の◯◯であって、〜」のように、参照表現の直後に
        # 発明の名称を挟んでから「であって」「において」が続く場合も、
        # 新しい限定文言がその後ろに続く「前置き型」とみなす
        # （「において」が参照表現に直接くっついていない、この変則パターン）。
        next_period = text.find("。", match_end)
        search_end = next_period if next_period != -1 else len(text)
        connector_m = _re_dep.search(r"(?:であって|において)", text[match_end:search_end])
        if connector_m is not None:
            is_preamble_style = True
            match_end = match_end + connector_m.end()
    return {
        "numbers": sorted(set(nums)),
        "match_start": m.start(),
        "match_end": match_end,
        "is_preamble_style": is_preamble_style,
    }


def _build_claim_chain(claim_number, claim_texts, prefer_parent=None, _seen=None):
    """
    claim_number を頂点として、祖先の請求項を根から順に並べた鎖にする。

    戻り値: [(請求項番号, その請求項固有の追加限定文（親の内容は含まない）), ...]
            リストの先頭が一番根本の独立請求項。
    """
    if _seen is None:
        _seen = set()
    if claim_number in _seen:
        raise ValueError(f"請求項の参照が循環しています: {claim_number}")
    _seen = _seen | {claim_number}

    text = claim_texts.get(claim_number)
    if text is None:
        raise KeyError(f"請求項{claim_number}の本文が見つかりません")

    ref = _parse_claim_ref(text)
    if ref is None:
        # 他の請求項を参照していない、独立請求項。全文がそのまま固有の内容になる。
        return [(claim_number, text.strip())]

    parent_num = prefer_parent if prefer_parent in ref["numbers"] else ref["numbers"][0]
    chain = _build_claim_chain(parent_num, claim_texts, _seen=_seen)

    # 「請求項◯に記載の」のように参照が末尾寄りにある場合は、それより
    # 前の部分が追加の限定文言。「請求項◯において、」のように参照が
    # 冒頭にある場合は、それより後ろの部分（発明の名称を除く）が
    # 追加の限定文言になる。
    if ref["is_preamble_style"]:
        after = text[ref["match_end"]:].strip().lstrip("、，,")
        additional, _title = _split_claim_title(after) if after else ("", "")
        additional = additional.strip()
        if additional.endswith(("、", "，")):
            additional = additional[:-1]
    else:
        additional = text[:ref["match_start"]].strip()
        if additional.endswith(("、", "，")):
            additional = additional[:-1]
    if additional and not additional.endswith("。"):
        additional += "。"

    return chain + [(claim_number, additional)]


def resolve_dependent_claim(claim_number, claim_texts, prefer_parent=None):
    """
    claim_texts: {請求項番号(int): 本文(str)} の辞書
    claim_number: 展開したい請求項の番号

    「請求項１又は２に記載の」のように複数の請求項を参照している
    場合、prefer_parentでどちらを親とみなすか指定できる
    （省略時は一番小さい番号を使う）。

    戻り値: 親請求項の内容も含めて、人間が読める形につなげた
            完全な請求項テキスト（表示用。解析には
            analyze_dependent_claim() を使う）。
    """
    chain = _build_claim_chain(claim_number, claim_texts, prefer_parent=prefer_parent)
    parts = [text for _, text in chain if text]
    return "\n".join(parts)


def analyze_dependent_claim(claim_number, claim_texts, prefer_parent=None):
    """
    従属請求項を、親請求項の内容も含めて解析する。

    請求項の数が増えるほど、全部を1つの巨大な文としてGiNZAに
    渡すと、誤読解や、行き場を失った構成要素が根っこに大量に
    直接ぶら下がる「孤立ノードの急増」を招きやすい。
    そこで、各請求項が持つ「固有の追加限定文」を1件ずつ別々に
    解析し、その関係リストだけを最後にまとめて合体させる方式を取る。
    「前記電極」のように共通する表現は、同じ文字列のノードとして
    後段で自動的につながるので、文を分けても情報は失われない。
    """
    chain = _build_claim_chain(claim_number, claim_texts, prefer_parent=prefer_parent)

    all_relations = []
    for i, (num, fragment_text) in enumerate(chain):
        if not fragment_text:
            continue
        if i == 0:
            # 一番根本の独立請求項は、通常通りフルに解析する
            _, relations = analyze_claim(fragment_text)
        else:
            # 追加の限定文はそれ単体では不完全な断片なので、
            # 孤立ノードを無理に根へ繋げる処理はまだかけない
            # （全部合体させたあとで、最後に1回だけ行う）
            _, relations, _ = _extract_raw_relations(fragment_text)
        for r in relations:
            r = dict(r)
            r["claim_number"] = num
            all_relations.append(r)

    seen_keys = set()
    unique_relations = []
    for r in all_relations:
        key = (r["source"], r["relation"], r["target"], r["type"])
        if key in seen_keys:
            # 同じ関係が複数の請求項で繰り返し登場する場合、
            # 一番最初（＝一番根本の請求項）に登場した方の
            # claim_numberを採用する
            continue
        seen_keys.add(key)
        unique_relations.append(r)

    # 全部合体させたあとで、最後にもう1回だけ階層整理をかける
    # （独立請求項由来の「有する」エッジが十分にあるので、
    #  正しい根はそこから見つかる）
    final_relations = _simplify_hierarchy(unique_relations)

    full_text = resolve_dependent_claim(claim_number, claim_texts, prefer_parent=prefer_parent)
    return (None, final_relations), full_text


def analyze_dependent_claim_with_components(claim_number, claim_texts, prefer_parent=None):
    """
    analyze_dependent_claim()と同じ要領で従属請求項を親請求項の内容も
    含めて解析するが、記載チェック（構成要素の未接続・表記ゆれ・用語の
    不一致チェック）で使うために、各請求項の追加限定文から抽出した生の
    構成要素リストも合わせて返す。

    「請求項１に記載の」等の引用表現そのものは _build_claim_chain() の
    段階で本文から取り除かれているため、この構成要素リストに
    「請求項」「記載」等の語が紛れ込むことはない
    （単に全文をanalyze_claim()に丸ごと渡した場合と違う点）。

    戻り値: (components, final_relations, full_text)
    """
    chain = _build_claim_chain(claim_number, claim_texts, prefer_parent=prefer_parent)

    all_components = []
    all_relations = []
    seen_component_texts = set()
    for i, (num, fragment_text) in enumerate(chain):
        if not fragment_text:
            continue
        if i == 0:
            # 一番根本の独立請求項は、通常通りフルに解析する
            fragment_components, raw_relations, doc = _extract_raw_relations(fragment_text)
            relations = _simplify_hierarchy(raw_relations, doc, fragment_components)
        else:
            # 追加の限定文はそれ単体では不完全な断片なので、
            # 孤立ノードを無理に根へ繋げる処理はまだかけない
            fragment_components, relations, _ = _extract_raw_relations(fragment_text)

        for c in fragment_components:
            term = _normalize_component_text(c["text"])
            if term and term not in seen_component_texts:
                seen_component_texts.add(term)
                all_components.append(c)
        for r in relations:
            r = dict(r)
            r["claim_number"] = num
            all_relations.append(r)

    seen_keys = set()
    unique_relations = []
    for r in all_relations:
        key = (r["source"], r["relation"], r["target"], r["type"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_relations.append(r)

    final_relations = _simplify_hierarchy(unique_relations)
    full_text = resolve_dependent_claim(claim_number, claim_texts, prefer_parent=prefer_parent)
    return all_components, final_relations, full_text


def parse_claims_block(text):
    """
    「【請求項１】（本文）【請求項２】（本文）…」という、
    実際の特許公報でそのまま使われている標準的な書式のテキストを、
    区切りの手作業なしで直接パースする。

    全角・半角どちらの数字にも対応する。
    【請求項N】の目印が1つも見つからない場合は、テキスト全体を
    請求項１本文とみなす（1件だけコピペした場合への対応）。

    戻り値: {請求項番号(int): 本文(str)} の辞書
    """
    pattern = _re_dep.compile(r"【\s*請求項\s*([0-9０-９]+)\s*】")
    matches = list(pattern.finditer(text))

    if not matches:
        stripped = text.strip()
        return {1: stripped} if stripped else {}

    result = {}
    for i, m in enumerate(matches):
        num_str = m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        num = int(num_str)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            result[num] = body
    return result


# ============================================================
# ㉑ Explorer：SAOキーワード探査（自社 vs 競合の比較）
# ============================================================
# 単なる単語頻度ではなく、SAO解析で抽出した「構成要素」「動詞」を
# キーワードとして使うことで、技術文書としての意味のある比較を行う。

# 「有する」「含む」「である」等は、ほぼ全ての請求項に出てくる
# 構造的な言い回しであり、技術内容とは無関係なので、キーワードとしては
# 拾わない（自社・競合比較で「共通語」に紛れ込んでも意味がないため）。
GENERIC_KEYWORD_VERBS = {
    "有する", "備える", "具備する", "含む", "含み", "である",
    "有し", "備え", "含める", "含める", "とは異なる",
}


def extract_keywords_from_relations(relations, kind="both"):
    """
    1件の請求項の関係リストから、キーワードの集合を取り出す。
    kind: "component"（構成要素のみ）, "verb"（動詞のみ）, "both"（両方）

    「有する」「備える」のような構造的な動詞（技術内容と無関係で、
    ほぼ全ての請求項に出てくる語）は、動詞キーワードから除外する。
    """
    keywords = set()
    for r in relations:
        if kind in ("component", "both"):
            keywords.add(r["source"])
            keywords.add(r["target"])
        if kind in ("verb", "both"):
            if r["relation"] not in GENERIC_KEYWORD_VERBS and r["type"] != "has":
                keywords.add(r["relation"])
    return keywords


def build_keyword_frequency(database, ids=None, kind="both", source="自動", exclude=None):
    """
    database（build_patent_database() 等の戻り値）から、指定したid群
    （省略時は全件）のキーワード出現頻度を集計する。
    「出現した件数」で数える（1件の中で同じ語が何度出ても1件とする）。

    source: "自動"（各エントリの持っている情報に応じて自動判定。
            従来の挙動）、"請求項"（請求項をSAO解析した構成要素を
            強制的に使う。"relations"を持たないエントリは無視される）、
            "発明の名称"（発明の名称から抽出したキーワードを強制的に
            使う。kind="verb"の場合は動詞を取れないため空になる）
    exclude: 除外したいキーワードのリスト（集計結果から取り除く）
    """
    from collections import Counter

    counter = Counter()
    target_ids = set(ids) if ids is not None else None
    exclude_set = set(exclude) if exclude else set()
    for entry in database:
        if target_ids is not None and entry["id"] not in target_ids:
            continue
        if source == "請求項":
            keywords = (
                extract_keywords_from_relations(entry["relations"], kind=kind)
                if "relations" in entry else set()
            )
        elif source == "発明の名称":
            if kind == "verb":
                keywords = set()
            else:
                title = entry.get("発明の名称", "") or ""
                keywords = extract_abstract_keywords(title) if title else set()
        else:
            keywords = _entry_keywords(entry, kind=kind)
        if exclude_set:
            keywords = keywords - exclude_set
        counter.update(keywords)
    return counter


def compare_keyword_groups(database, group_a_ids, group_b_ids, kind="both", top_n=30):
    """
    2つのグループ（例：自社群 vs 競合群）のキーワード頻度を比較する。

    戻り値: {
        "common": [(語, A頻度, B頻度), ...]（両方に出てくる語、頻度の合計順）,
        "only_a": [(語, 頻度), ...]（Aだけに出てくる語）,
        "only_b": [(語, 頻度), ...]（Bだけに出てくる語）,
        "freq_a": Counter, "freq_b": Counter,
    }
    """
    freq_a = build_keyword_frequency(database, group_a_ids, kind=kind)
    freq_b = build_keyword_frequency(database, group_b_ids, kind=kind)

    words_a = set(freq_a.keys())
    words_b = set(freq_b.keys())

    common = sorted(
        ((w, freq_a[w], freq_b[w]) for w in (words_a & words_b)),
        key=lambda x: -(x[1] + x[2])
    )[:top_n]
    only_a = sorted(((w, freq_a[w]) for w in (words_a - words_b)), key=lambda x: -x[1])[:top_n]
    only_b = sorted(((w, freq_b[w]) for w in (words_b - words_a)), key=lambda x: -x[1])[:top_n]

    return {"common": common, "only_a": only_a, "only_b": only_b, "freq_a": freq_a, "freq_b": freq_b}


def print_keyword_comparison(result, name_a="自社", name_b="競合"):
    """compare_keyword_groups() の結果を、人が読みやすい形で表示する"""
    print(f"=== 共通する語（上位{len(result['common'])}件） ===")
    for w, fa, fb in result["common"]:
        print(f"  {w:15s} {name_a}:{fa:3d}件 / {name_b}:{fb:3d}件")
    print()
    print(f"=== {name_a}だけに出てくる語（上位{len(result['only_a'])}件） ===")
    for w, f in result["only_a"]:
        print(f"  {w:15s} {f:3d}件")
    print()
    print(f"=== {name_b}だけに出てくる語（上位{len(result['only_b'])}件） ===")
    for w, f in result["only_b"]:
        print(f"  {w:15s} {f:3d}件")


def plot_wordcloud(freq_counter, title="キーワード頻度", font_path=None):
    """
    build_keyword_frequency() 等で作った頻度カウンタから、
    ワードクラウドの画像（matplotlib Figure）を作る。
    """
    from wordcloud import WordCloud

    if font_path is None:
        font_candidates = glob.glob("/tmp/NotoSansJP-Regular.ttf") + glob.glob(
            "/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True
        )
        font_path = font_candidates[0] if font_candidates else None

    wc = WordCloud(
        font_path=font_path,
        width=900, height=500,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(dict(freq_counter))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=16)
    plt.tight_layout()
    return fig


# ============================================================
# ㉒ Saturn V：意味的俯瞰マップ
# ============================================================
# build_patent_database() で計算済みの埋め込みベクトル（doc_embedding）を
# PCAで2次元に落とし込み、意味的に近い特許同士が近くに配置される
# 「地図」を作る。

def build_semantic_map(database, n_components=2, method="pca",
                        n_neighbors=15, min_dist=0.1, random_state=42):
    """
    database（build_patent_database()の戻り値、doc_embeddingを含む）から、
    2次元（または指定した次元数）の座標を計算する。

    method: "pca"（主成分分析。計算が速く、軸に「寄与率」という意味を
                    持たせられる。ただし複雑なクラスタ構造は潰れて見えがち）
            "umap"（UMAP。似た特許同士の近さ・クラスタ構造をより保った
                    非線形な配置になりやすい。軸自体には「寄与率」に相当
                    する意味はない）
    n_neighbors, min_dist: method="umap"のときのみ有効なパラメータ。
            n_neighborsは近傍点の数（小さいほど局所的、大きいほど大域的な
            構造を重視）、min_distは点同士に許容する最小距離（小さいほど
            密集したクラスタになる）。

    戻り値: (points, explained_variance)
        points: [{"id":.., "text":.., "x":.., "y":..}, ...]
        explained_variance: PCAの場合は寄与率の配列。UMAPの場合は
                             軸に寄与率の意味がないためNone。
    """
    import numpy as np

    if len(database) < 2:
        raise ValueError("2件以上のデータが必要です")

    embeddings = np.array([e["doc_embedding"] for e in database])

    if method == "umap":
        import umap

        n_comp = min(n_components, len(database) - 1)
        n_neighbors_eff = max(2, min(n_neighbors, len(database) - 1))
        reducer = umap.UMAP(
            n_components=n_comp,
            n_neighbors=n_neighbors_eff,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
        )
        coords = reducer.fit_transform(embeddings)
        explained_variance = None
    else:
        from sklearn.decomposition import PCA

        n_comp = min(n_components, len(database) - 1, embeddings.shape[1])
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_ratio_

    points = []
    for entry, xy in zip(database, coords):
        points.append({
            "id": entry["id"],
            "text": entry["text"],
            "x": float(xy[0]),
            "y": float(xy[1]) if n_comp > 1 else 0.0,
        })
    return points, explained_variance


def plot_semantic_map(points, explained_variance=None, groups=None, title="意味的俯瞰マップ",
                       theme="deepsea", max_labels=40, method="pca"):
    """
    build_semantic_map() の結果を、散布図として描画する。

    groups: {id: グループ名, ...} を渡すと、グループごとに色分けする
            （例：自社 vs 競合の比較地図にする場合）。
    max_labels: ラベル（文献番号等）を表示する点の最大数。
                点の数がこれを超える場合、ラベルはランダムに間引いて
                表示する（全部表示すると重なって読めなくなるため）。
                Noneを指定すると、件数に関わらず全部表示する。
    """
    if theme == "deepsea":
        bg, fg, grid = "#04121C", "#E8FBFF", "#1a3a4a"
        palette = [s["border"] for s in _DEEPSEA_PALETTE]
    else:
        bg, fg, grid = "#FFFFFF", "#233044", "#dddddd"
        palette = [s["edge"] for s in _BRANCH_PALETTE]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if groups:
        group_names = sorted(set(groups.values()))
        color_of = {g: palette[i % len(palette)] for i, g in enumerate(group_names)}
        for g in group_names:
            xs = [p["x"] for p in points if groups.get(p["id"]) == g]
            ys = [p["y"] for p in points if groups.get(p["id"]) == g]
            ax.scatter(xs, ys, s=90, alpha=0.85, color=color_of[g], edgecolors=fg,
                       linewidths=0.6, label=g)
        ax.legend(prop=FONT_PROP, facecolor=bg, labelcolor=fg, edgecolor=grid)
    else:
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        ax.scatter(xs, ys, s=90, alpha=0.85, color=palette[0], edgecolors=fg, linewidths=0.6)

    if max_labels is not None and len(points) > max_labels:
        import random
        rng = random.Random(0)
        labeled_points = rng.sample(points, max_labels)
    else:
        labeled_points = points

    for p in labeled_points:
        ax.annotate(str(p["id"]), (p["x"], p["y"]), fontsize=8, fontproperties=FONT_PROP,
                    color=fg, xytext=(4, 4), textcoords="offset points")

    ax.set_title(title, fontproperties=FONT_PROP, fontsize=16, color=fg)
    if method == "umap":
        ax.set_xlabel("UMAP次元1", fontproperties=FONT_PROP, color=fg)
        ax.set_ylabel("UMAP次元2", fontproperties=FONT_PROP, color=fg)
    elif explained_variance is not None and len(explained_variance) >= 2:
        ax.set_xlabel(f"第1主成分（寄与率 {explained_variance[0]*100:.1f}%）", fontproperties=FONT_PROP, color=fg)
        ax.set_ylabel(f"第2主成分（寄与率 {explained_variance[1]*100:.1f}%）", fontproperties=FONT_PROP, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(grid)
    ax.grid(True, color=grid, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_semantic_map_interactive(points, explained_variance=None, groups=None,
                                   title="意味的俯瞰マップ", theme="deepsea", method="pca"):
    """
    plot_semantic_map() のインタラクティブ版（Plotly）。
    普段は丸だけを表示し、カーソルを合わせた点だけ文献番号（id）を
    ツールチップで表示する。戻り値は plotly.graph_objects.Figure で、
    Streamlitでは st.plotly_chart(戻り値) で表示できる。
    """
    import plotly.graph_objects as go

    if theme == "deepsea":
        bg, fg, grid = "#04121C", "#E8FBFF", "#1a3a4a"
        palette = [s["border"] for s in _DEEPSEA_PALETTE]
    else:
        bg, fg, grid = "#FFFFFF", "#233044", "#dddddd"
        palette = [s["edge"] for s in _BRANCH_PALETTE]

    fig = go.Figure()

    if groups:
        group_names = sorted(set(groups.values()))
        color_of = {g: palette[i % len(palette)] for i, g in enumerate(group_names)}
        for g in group_names:
            pts = [p for p in points if groups.get(p["id"]) == g]
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in pts], y=[p["y"] for p in pts],
                mode="markers", name=g,
                marker=dict(size=11, color=color_of[g], line=dict(width=1, color=fg)),
                text=[p["id"] for p in pts],
                hovertemplate="%{text}<extra>" + g + "</extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in points], y=[p["y"] for p in points],
            mode="markers",
            marker=dict(size=11, color=palette[0], line=dict(width=1, color=fg)),
            text=[p["id"] for p in points],
            hovertemplate="%{text}<extra></extra>",
        ))

    if method == "umap":
        xlabel, ylabel = "UMAP次元1", "UMAP次元2"
    else:
        xlabel = "第1主成分"
        ylabel = "第2主成分"
        if explained_variance is not None and len(explained_variance) >= 2:
            xlabel = f"第1主成分（寄与率 {explained_variance[0]*100:.1f}%）"
            ylabel = f"第2主成分（寄与率 {explained_variance[1]*100:.1f}%）"

    fig.update_layout(
        title=title,
        xaxis_title=xlabel, yaxis_title=ylabel,
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, scaleanchor="x", scaleratio=1),
        legend=dict(bgcolor=bg, bordercolor=grid),
        width=700, height=700,
    )
    return fig


# ============================================================
# ㉒-2 類似度ネットワーク図（NetworkX）
# ============================================================
# doc_embedding同士のコサイン類似度が閾値を超えたペアをエッジで結び、
# 特許同士の「似ている／似ていない」の関係をネットワーク図として可視化する。
# Saturn V（PCA／UMAPマップ）が「全体としての配置・クラスタ傾向」を見るのに
# 向いているのに対し、こちらは「どの特許とどの特許が具体的に似ているか」を
# 直接確認するのに向いている。

def build_similarity_network(database, threshold=0.75, max_neighbors=5):
    """
    database（doc_embeddingを含む）から、コサイン類似度に基づく
    networkx.Graph を作る（doc_embeddingは正規化済みのため、
    内積がそのままコサイン類似度になる）。

    threshold: この類似度以上のペアだけをエッジにする
    max_neighbors: 1ノードあたり、類似度が高い順に最大何件まで
                   エッジを残すか（図が線だらけになるのを防ぐ）

    戻り値: networkx.Graph（各エッジは weight=類似度 属性を持つ。
             閾値を超えるペアが1件もないノードは含まれない）
    """
    import numpy as np

    if len(database) < 2:
        raise ValueError("2件以上のデータが必要です")

    ids = [e["id"] for e in database]
    embeddings = np.array([e["doc_embedding"] for e in database])
    sim_matrix = embeddings @ embeddings.T

    G = nx.Graph()
    G.add_nodes_from(ids)

    n = len(ids)
    for i in range(n):
        neighbors = [
            (j, sim_matrix[i, j]) for j in range(n)
            if j != i and sim_matrix[i, j] >= threshold
        ]
        neighbors.sort(key=lambda x: -x[1])
        for j, sim in neighbors[:max_neighbors]:
            if not G.has_edge(ids[i], ids[j]):
                G.add_edge(ids[i], ids[j], weight=float(sim))

    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def plot_similarity_network_interactive(G, groups=None, title="特許類似度ネットワーク図", theme="deepsea"):
    """
    build_similarity_network() の結果を、Plotlyのインタラクティブな
    ネットワーク図として描画する。ノードにカーソルを合わせると文献番号が
    表示される。ノードの大きさは、そのノードに繋がっているエッジの数
    （＝似ている特許の多さ）に応じて大きくなる。
    """
    import plotly.graph_objects as go

    if G.number_of_nodes() == 0:
        raise ValueError("エッジが1件もありません。類似度の閾値を下げてください。")

    if theme == "deepsea":
        bg, fg, grid = "#04121C", "#E8FBFF", "#1a3a4a"
        palette = [s["border"] for s in _DEEPSEA_PALETTE]
    else:
        bg, fg, grid = "#FFFFFF", "#233044", "#dddddd"
        palette = [s["edge"] for s in _BRANCH_PALETTE]

    pos = nx.spring_layout(G, seed=42, weight="weight")

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color=grid), hoverinfo="none", showlegend=False,
    ))

    node_ids = list(G.nodes())
    degrees = dict(G.degree())

    if groups:
        group_names = sorted(set(groups.get(i, "その他") for i in node_ids))
        color_of = {g: palette[k % len(palette)] for k, g in enumerate(group_names)}
        for g in group_names:
            ids_in_group = [i for i in node_ids if groups.get(i, "その他") == g]
            if not ids_in_group:
                continue
            fig.add_trace(go.Scatter(
                x=[pos[i][0] for i in ids_in_group],
                y=[pos[i][1] for i in ids_in_group],
                mode="markers", name=g,
                marker=dict(
                    size=[10 + degrees[i] * 3 for i in ids_in_group],
                    color=color_of[g], line=dict(width=1, color=fg),
                ),
                text=[i for i in ids_in_group],
                hovertemplate="%{text}<extra>" + g + "</extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in node_ids],
            y=[pos[i][1] for i in node_ids],
            mode="markers", showlegend=False,
            marker=dict(
                size=[10 + degrees[i] * 3 for i in node_ids],
                color=palette[0], line=dict(width=1, color=fg),
            ),
            text=[i for i in node_ids],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        showlegend=bool(groups),
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(bgcolor=bg, bordercolor=grid),
        width=700, height=700,
    )
    return fig


# ============================================================
# ㉓ CORE：論理式分類・ヒートマップ
# ============================================================
# 独自の論理式（キーワードの組み合わせ）を定義して、特許群を
# 「課題」軸と「解決手段」軸などで分類し、ヒートマップにする。
# 値が0のマスは、まだ誰も出願していない技術の組み合わせ
# （ホワイトスペース）の候補になる。

def evaluate_formula(keywords, formula):
    """
    keywords: 1件の特許から抽出したキーワードの集合
              （extract_keywords_from_relations()の戻り値）
    formula: {"any_of": [...], "all_of": [...], "none_of": [...]}
             のいずれかを指定した辞書（省略したキーの条件は無視する）
             ・any_of: このうち1つでも含まれていればよい（OR）
             ・all_of: これら全部が含まれている必要がある（AND）
             ・none_of: これらが1つも含まれていてはいけない（NOT）
    """
    if "any_of" in formula and not any(k in keywords for k in formula["any_of"]):
        return False
    if "all_of" in formula and not all(k in keywords for k in formula["all_of"]):
        return False
    if "none_of" in formula and any(k in keywords for k in formula["none_of"]):
        return False
    return True


def classify_patents(database, axis1_formulas, axis2_formulas, kind="both"):
    """
    axis1_formulas / axis2_formulas: {カテゴリ名: formula辞書, ...}

    各特許のキーワード集合を、両方の軸それぞれについて、
    マッチする全カテゴリに分類する（1件が複数のカテゴリに
    同時に該当してもよい。例：複数の課題を同時に解決している特許）。
    どちらの軸にもマッチするカテゴリがない場合は「(未分類)」に入れる。

    戻り値: {(axis1のカテゴリ名, axis2のカテゴリ名): [id, id, ...], ...}
    """
    from collections import defaultdict

    matrix = defaultdict(list)
    for entry in database:
        keywords = _entry_keywords(entry, kind=kind)
        matched1 = [name for name, f in axis1_formulas.items() if evaluate_formula(keywords, f)]
        matched2 = [name for name, f in axis2_formulas.items() if evaluate_formula(keywords, f)]
        if not matched1:
            matched1 = ["(未分類)"]
        if not matched2:
            matched2 = ["(未分類)"]
        for a1 in matched1:
            for a2 in matched2:
                matrix[(a1, a2)].append(entry["id"])
    return matrix


def classify_patents_by_sections(database, axis1_formulas, axis2_formulas,
                                  axis1_section="課題", axis2_section="解決手段"):
    """
    build_abstract_database() で作った、【課題】【解決手段】等の
    セクションに分かれた要約データベース専用の分類関数。

    classify_patents() は「1件の特許が持つ全キーワード」を両方の軸に
    使うが、この関数は縦軸を「課題」セクションのキーワードだけ、
    横軸を「解決手段」セクションのキーワードだけで判定するので、
    より精密に「どんな課題を、どんな手段で解決しているか」の
    マトリクスを作れる。

    axis1_section / axis2_section: 各entryの"sections"辞書から
    参照する見出し名（省略時は"課題"/"解決手段"）。
    """
    from collections import defaultdict

    matrix = defaultdict(list)
    for entry in database:
        sections = entry.get("sections", {})
        text1 = sections.get(axis1_section, "")
        text2 = sections.get(axis2_section, "")
        keywords1 = extract_abstract_keywords(text1) if text1 else set()
        keywords2 = extract_abstract_keywords(text2) if text2 else set()

        matched1 = [name for name, f in axis1_formulas.items() if evaluate_formula(keywords1, f)]
        matched2 = [name for name, f in axis2_formulas.items() if evaluate_formula(keywords2, f)]
        if not matched1:
            matched1 = ["(未分類)"]
        if not matched2:
            matched2 = ["(未分類)"]
        for a1 in matched1:
            for a2 in matched2:
                matrix[(a1, a2)].append(entry["id"])
    return matrix


def plot_classification_heatmap(matrix, axis1_names, axis2_names, title="論理式分類ヒートマップ"):
    """
    classify_patents() の結果をヒートマップとして描画する。
    値が0のマス（青枠で強調）が、ホワイトスペースの候補になる。
    """
    import numpy as np

    arr = np.zeros((len(axis1_names), len(axis2_names)), dtype=int)
    for i, a1 in enumerate(axis1_names):
        for j, a2 in enumerate(axis2_names):
            arr[i, j] = len(matrix.get((a1, a2), []))

    fig, ax = plt.subplots(figsize=(max(6, len(axis2_names) * 1.3), max(4, len(axis1_names) * 0.9)))
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(axis2_names)))
    ax.set_xticklabels(axis2_names, rotation=30, ha="right", fontproperties=FONT_PROP, fontsize=10)
    ax.set_yticks(range(len(axis1_names)))
    ax.set_yticklabels(axis1_names, fontproperties=FONT_PROP, fontsize=10)

    vmax = arr.max() if arr.max() > 0 else 1
    for i in range(len(axis1_names)):
        for j in range(len(axis2_names)):
            val = int(arr[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="black" if val < vmax / 2 else "white", fontsize=10)
            if val == 0:
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                 edgecolor="#3b7dd8", linewidth=1.8))

    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("件数", fontproperties=FONT_PROP)
    plt.tight_layout()
    return fig


def print_white_space_cells(matrix, axis1_names, axis2_names):
    """0件のマス（ホワイトスペース候補）だけを一覧表示する"""
    print("=== ホワイトスペース候補（0件のマス） ===")
    for a1 in axis1_names:
        for a2 in axis2_names:
            if len(matrix.get((a1, a2), [])) == 0:
                print(f"  「{a1}」×「{a2}」")


# ============================================================
# ㉔ 要約データの活用：J-PlatPatで一括取得できる「要約」を、
#    ポートフォリオ分析（Explorer / Saturn V / CORE）に使えるようにする
# ============================================================
# 要約は請求項と違って「〜を有する」という定型構文を持たない、
# より自然な文章であり、かつ多くの場合【課題】【解決手段】という
# 見出しが付いている。この見出しを頼りに、そのままCORE
# （縦軸＝課題、横軸＝解決手段）に使える形に整理する。

def parse_abstract(text):
    """
    「【課題】〜。【解決手段】〜。」のような、要約に含まれる
    見出しタグを頼りに、セクションごとの本文に分割する。

    戻り値: {見出し名: 本文, ...}（見出しが1つも見つからなければ
             {"全文": text} を返す）
    """
    pattern = _re_dep.compile(r"【([^】]+)】")
    matches = list(pattern.finditer(text))
    if not matches:
        return {"全文": text.strip()}

    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections[name] = body
    return sections


def extract_abstract_keywords(text):
    """
    請求項のような「を有する」構造を前提としない、一般的な日本語文
    （要約等）から、名詞句をキーワードとして抽出する。
    """
    doc = nlp(_clean_claim_text(text))
    components = extract_patent_components_general(doc)
    return {c["text"] for c in components}


def build_abstract_database(records, show_progress=True):
    """
    records: [(id, 要約テキスト), ...]

    請求項用の build_patent_database() とは違い、「有する」の
    階層構造は作らない（要約は請求項特有の構文を持たないため）。
    代わりに、名詞句のキーワード抽出と、文章全体の埋め込み
    ベクトル計算だけを行う。

    Explorer・Saturn V・CORE は、いずれもこのデータベースを
    build_patent_database() の代わりにそのまま使える。
    """
    model = _get_embed_model()
    database = []
    total = len(records)
    for i, (rid, text) in enumerate(records):
        if show_progress:
            print(f"[{i+1}/{total}] {rid} を解析中...")
        try:
            sections = parse_abstract(text)
            keywords = set()
            for sec_text in sections.values():
                keywords |= extract_abstract_keywords(sec_text)
            embedding = model.encode([text], normalize_embeddings=True)[0]
        except Exception as e:
            if show_progress:
                print(f"  → 解析エラー、スキップします: {e}")
            continue

        database.append({
            "id": rid,
            "text": text,
            "sections": sections,
            "keywords": keywords,
            "doc_embedding": embedding,
        })
    return database


def _entry_keywords(entry, kind="both"):
    """
    Explorer・COREの内部で使う共通ヘルパー。
    build_patent_database()（請求項、relationsを持つ）と
    build_abstract_database()（要約、keywordsを持つ）の
    どちらの形式のデータベースが来ても、同じようにキーワード
    集合を取り出せるようにする。
    """
    if "relations" in entry:
        return extract_keywords_from_relations(entry["relations"], kind=kind)
    return set(entry.get("keywords", set()))


# ============================================================
# ㉕ 構成部位ランキング・件数分布・レーダーチャート
# ============================================================

def rank_components(database, ids=None, kind="component", top_n=20):
    """
    ポートフォリオ全体（またはグループ）で、よく出てくる構成要素・動詞を
    頻度順にランキングする（「構成部位」分析に相当）。
    """
    freq = build_keyword_frequency(database, ids=ids, kind=kind)
    return freq.most_common(top_n)


def plot_component_ranking(ranking, title="構成部位ランキング", theme="deepsea"):
    """rank_components() の結果を横棒グラフにする"""
    if theme == "deepsea":
        bg, fg, bar = "#04121C", "#E8FBFF", "#5FD4E0"
    else:
        bg, fg, bar = "#FFFFFF", "#233044", "#4C87C6"

    labels = [w for w, _ in ranking][::-1]
    values = [c for _, c in ranking][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.35)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.barh(labels, values, color=bar)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=FONT_PROP, color=fg, fontsize=9)
    ax.set_xlabel("出現件数", fontproperties=FONT_PROP, color=fg)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    plt.tight_layout()
    return fig


def compute_scope_distribution(database, ids=None):
    """
    build_patent_database()（請求項データベース、relationsを持つ）から、
    各特許の「広さ・狭さスコア」を計算し、分布（件数分布）を作る。
    要約データベースには使えない（請求項の構造が必要なため）。
    """
    target_ids = set(ids) if ids is not None else None
    scores = []
    for entry in database:
        if target_ids is not None and entry["id"] not in target_ids:
            continue
        if "relations" not in entry:
            continue
        narrowness, breadth, detail = compute_claim_scope_score(entry["relations"])
        scores.append({"id": entry["id"], "narrowness": narrowness, "breadth": breadth})
    return scores


def plot_scope_distribution(scores, title="クレームの広さ・狭さの分布", theme="deepsea", bins=10):
    """compute_scope_distribution() の結果をヒストグラムにする"""
    if theme == "deepsea":
        bg, fg, bar = "#04121C", "#E8FBFF", "#5FD4E0"
    else:
        bg, fg, bar = "#FFFFFF", "#233044", "#4C87C6"

    values = [s["narrowness"] for s in scores]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.hist(values, bins=bins, color=bar, edgecolor=fg, alpha=0.85)
    ax.set_xlabel("狭さスコア（0=広い　1=狭い）", fontproperties=FONT_PROP, color=fg)
    ax.set_ylabel("件数", fontproperties=FONT_PROP, color=fg)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    plt.tight_layout()
    return fig


def compute_group_profile(database, ids):
    """
    グループ（自社／競合等）の「特徴プロファイル」を計算する。
    レーダーチャート用に、複数の指標を0〜1に正規化してまとめる。
    請求項データベース（relationsを持つ）が必要。
    """
    scope_list = compute_scope_distribution(database, ids)
    target_ids = set(ids)
    entries = [e for e in database if e["id"] in target_ids and "relations" in e]

    if not entries:
        return {}

    avg_narrowness = sum(s["narrowness"] for s in scope_list) / len(scope_list) if scope_list else 0.0
    avg_components = sum(
        len({r["source"] for r in e["relations"]} | {r["target"] for r in e["relations"]})
        for e in entries
    ) / len(entries)
    avg_relations = sum(len(e["relations"]) for e in entries) / len(entries)
    avg_attribute = sum(
        sum(1 for r in e["relations"] if r["type"] == "attribute") for e in entries
    ) / len(entries)
    unique_components = len(build_keyword_frequency(database, ids, kind="component"))

    return {
        "平均の狭さスコア": avg_narrowness,
        "平均構成要素数": avg_components,
        "平均関係数": avg_relations,
        "平均数値スペック数": avg_attribute,
        "構成要素の種類数": unique_components,
    }


def plot_radar_chart(profiles, title="グループ特徴比較", theme="deepsea"):
    """
    compute_group_profile() の結果を、複数グループ分まとめて
    レーダーチャートにする。
    profiles: {グループ名: compute_group_profile()の戻り値, ...}
    """
    import numpy as np

    if theme == "deepsea":
        bg, fg, grid = "#04121C", "#E8FBFF", "#1a3a4a"
        palette = [s["border"] for s in _DEEPSEA_PALETTE]
    else:
        bg, fg, grid = "#FFFFFF", "#233044", "#dddddd"
        palette = [s["edge"] for s in _BRANCH_PALETTE]

    labels = list(next(iter(profiles.values())).keys())
    n = len(labels)

    # 指標ごとに、グループ間の最大値で正規化する（0〜1にそろえる）
    max_per_label = {l: max(p[l] for p in profiles.values()) or 1 for l in labels}

    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for i, (name, profile) in enumerate(profiles.items()):
        values = [profile[l] / max_per_label[l] for l in labels]
        values += values[:1]
        color = palette[i % len(palette)]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=FONT_PROP, color=fg, fontsize=10)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(grid)
    ax.grid(color=grid)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=fg, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), prop=FONT_PROP, facecolor=bg, labelcolor=fg)
    plt.tight_layout()
    return fig


# ============================================================
# ㉖ Mission Control：メタデータ付きCSVの読み込み
# ============================================================
# 文献番号・出願番号・出願日・公知日・発明の名称・出願人/権利者・FI・
# 要約・公開番号・公告番号・登録番号・審判番号・その他・ステージ・
# イベント詳細・文献URL、という列を持つCSVを読み込み、
# 全モジュール（ATLAS・MEGA・Saturn V・Explorer・CORE）で
# 共通して使える形に正規化する。

PATENT_METADATA_COLUMNS = [
    "文献番号", "出願番号", "出願日", "公知日", "発明の名称", "出願人/権利者",
    "FI", "要約", "公開番号", "公告番号", "登録番号", "審判番号",
    "その他", "ステージ", "イベント詳細", "文献URL",
]


def load_patent_metadata_csv(csv_text):
    """
    上記の列を持つCSVのテキストを読み込み、
    [{"id":.., "出願日":datetime, "出願人":[...], "FI":[...], "要約":.., ...}, ...]
    のリストにして返す（Mission Controlの役割）。

    ・出願人/権利者は「／」「、」「,」等で複数人書かれていることがあるので、
      リストに分割しておく。
    ・FIも同様に複数書かれていることがあるので、空白や「;」等で分割する。
    ・出願日／公知日は日付型に変換する（変換できない場合はNoneのまま）。
    """
    import csv
    import io
    from datetime import datetime

    def _split_applicants(value, seps=("／", "、", ",", ";", "；")):
        if not value:
            return []
        text = value
        for s in seps[1:]:
            text = text.replace(s, seps[0])
        return [v.strip() for v in text.split(seps[0]) if v.strip()]

    def _split_fi(value, seps=("；", ";", "、", ",")):
        # FIコード自体に「/」（メイングループ/サブグループの区切り）が
        # 含まれるため、出願人の分割とは違い「/」では分割しない。
        # 「＠Ｚ」「＠Ａ」等は、FIコードの正式な一部（展開記号）であり、
        # ノイズではないので取り除かない。
        # 一方、「Ｈ１０Ｋ８５／６０，１５０」や「Ｂ２３Ｋ３５／３０，３１０＠Ｃ」
        # のように、FIコードの後ろにコンマ＋「数字（＋＠記号）」だけの
        # 「展開記号（細分）」が続くことがあり、これは新しいFIコードでは
        # なく、直前のコードの一部である。このパターンに一致するトークンは
        # 独立したコードとして分割せず、直前のコードに結合する。
        if not value:
            return []
        text = value
        for s in seps[1:]:
            text = text.replace(s, seps[0])
        raw_tokens = [v.strip() for v in text.split(seps[0]) if v.strip()]

        sub_position_pattern = _re_dep.compile(r"^[0-9]+(?:[＠@][A-Za-zＡ-Ｚａ-ｚ0-9０-９]+)?$")
        merged = []
        for t in raw_tokens:
            if sub_position_pattern.match(t) and merged:
                merged[-1] = merged[-1] + "," + t
            else:
                merged.append(t)
        return merged

    def _parse_date(value):
        if not value:
            return None
        value = value.strip().replace("/", "-").replace(".", "-")
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    reader = csv.DictReader(io.StringIO(csv_text))
    records = []
    for i, row in enumerate(reader):
        rid = row.get("文献番号") or row.get("出願番号") or row.get("公開番号") or f"行{i+1}"
        stage = row.get("ステージ", "").strip()
        registration_no = row.get("登録番号", "").strip()
        publication_no = row.get("公開番号", "").strip()
        if not stage:
            # ステージ欄が空の場合、登録番号・公開番号の有無から推測する
            if registration_no:
                stage = "登録"
            elif publication_no:
                stage = "公開"
        records.append({
            "id": rid,
            "出願番号": row.get("出願番号", ""),
            "出願日": _parse_date(row.get("出願日", "")),
            "公知日": _parse_date(row.get("公知日", "")),
            "発明の名称": row.get("発明の名称", ""),
            "出願人": _split_applicants(row.get("出願人/権利者", "")),
            "FI": _split_fi(row.get("FI", "")),
            "要約": row.get("要約", ""),
            "公開番号": publication_no,
            "公告番号": row.get("公告番号", "").strip(),
            "登録番号": registration_no,
            "審判番号": row.get("審判番号", "").strip(),
            "ステージ": stage,
            "文献URL": row.get("文献URL", ""),
        })
    return records


def build_full_database(metadata_records, show_progress=True):
    """
    load_patent_metadata_csv() の結果から、要約（あれば）と発明の名称の
    両方からキーワードを抽出し、メタデータと1つにまとめたデータベースを
    作る。ATLAS・MEGA・Saturn V・Explorer・COREのすべてに共通して
    使える、一番リッチな形式。

    要約が空の行でも、発明の名称からキーワード・埋め込みベクトルを
    計算するので、キーワードが空になって後段のワードクラウド等が
    エラーになることはない。
    """
    model = _get_embed_model()
    total = len(metadata_records)
    database = []
    for i, r in enumerate(metadata_records):
        if show_progress:
            print(f"[{i+1}/{total}] {r['id']} を解析中...")
        entry = dict(r)
        title = (r.get("発明の名称") or "").strip()
        abstract = (r.get("要約") or "").strip()
        combined_text = (title + "。" + abstract) if abstract else title

        try:
            sections = parse_abstract(abstract) if abstract else {}
            keywords = set()
            for sec_text in sections.values():
                keywords |= extract_abstract_keywords(sec_text)
            if title:
                keywords |= extract_abstract_keywords(title)
            embed_source = combined_text or r["id"]
            embedding = model.encode([embed_source], normalize_embeddings=True)[0]
        except Exception as e:
            if show_progress:
                print(f"  → 解析エラー、スキップします: {e}")
            keywords = set()
            sections = {}
            embedding = None

        entry["sections"] = sections
        entry["keywords"] = keywords
        entry["doc_embedding"] = embedding
        entry["text"] = combined_text
        database.append(entry)
    return database


# ============================================================
# ㉗ ATLAS：基礎特許マップ
# ============================================================
# 出願件数の時系列推移、出願人ランキング、FI（IPC）ランキングなど、
# 特許分析において最も基本的な統計グラフを描画する。

def _atlas_style(theme="deepsea"):
    if theme == "deepsea":
        return {"bg": "#04121C", "fg": "#E8FBFF", "bar": "#5FD4E0", "grid": "#1a3a4a"}
    return {"bg": "#FFFFFF", "fg": "#233044", "bar": "#4C87C6", "grid": "#dddddd"}


def plot_filing_trend(database, date_field="出願日", freq="Y", title="出願件数の推移", theme="deepsea"):
    """
    出願日（または公知日）を使って、件数の時系列推移を折れ線グラフにする。
    freq: "Y"（年単位）または "M"（月単位）
    """
    from collections import Counter

    counter = Counter()
    for entry in database:
        d = entry.get(date_field)
        if d is None:
            continue
        key = d.year if freq == "Y" else (d.year, d.month)
        counter[key] += 1

    keys_sorted = sorted(counter.keys())
    if freq == "Y":
        labels = [str(k) for k in keys_sorted]
    else:
        labels = [f"{k[0]}-{k[1]:02d}" for k in keys_sorted]
    values = [counter[k] for k in keys_sorted]

    s = _atlas_style(theme)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(s["bg"])
    ax.set_facecolor(s["bg"])
    ax.plot(labels, values, marker="o", color=s["bar"], linewidth=2)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=s["fg"])
    ax.set_ylabel("件数", fontproperties=FONT_PROP, color=s["fg"])
    ax.tick_params(colors=s["fg"], rotation=45)
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_PROP)
    for spine in ax.spines.values():
        spine.set_color(s["grid"])
    ax.grid(True, color=s["grid"], alpha=0.3)
    plt.tight_layout()
    return fig


def rank_by_field(database, field="出願人", top_n=15):
    """
    出願人やFIのような「リストを持つフィールド」で、
    出現件数のランキングを作る。
    """
    from collections import Counter

    counter = Counter()
    for entry in database:
        values = entry.get(field) or []
        counter.update(set(values))
    return counter.most_common(top_n)


def fi_to_subclass(fi_code):
    """
    FIコードからサブクラスを取り出す（先頭4文字。例：「H10K85/60,150」→「H10K」）。
    """
    return fi_code[:4]


def fi_to_maingroup(fi_code):
    """
    FIコードからメイングループを取り出す。
    先頭6文字（例：「H10K85/60,150」→「H10K85」）。
    6文字に満たない場合は先頭5文字だけを使う。
    """
    if len(fi_code) >= 6:
        return fi_code[:6]
    return fi_code[:5]


def rank_fi(database, level="サブクラス", top_n=15):
    """
    FIコードを、指定した粒度（"サブクラス"＝先頭4文字、
    "メイングループ"＝先頭6文字（無ければ5文字）、"そのまま"＝元のコード）
    に丸めてから集計する。粒度を粗くすることで、細分番号の違いに
    埋もれがちな技術分野ごとの傾向が見えやすくなる。
    """
    from collections import Counter

    if level == "サブクラス":
        convert = fi_to_subclass
    elif level == "メイングループ":
        convert = fi_to_maingroup
    else:
        convert = lambda x: x

    counter = Counter()
    for entry in database:
        values = entry.get("FI") or []
        counter.update({convert(v) for v in values})
    return counter.most_common(top_n)


def plot_ranking_bar(ranking, title="ランキング", xlabel="件数", theme="deepsea"):
    """rank_by_field() の結果を横棒グラフにする"""
    s = _atlas_style(theme)
    labels = [w for w, _ in ranking][::-1]
    values = [c for _, c in ranking][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.4)))
    fig.patch.set_facecolor(s["bg"])
    ax.set_facecolor(s["bg"])
    ax.barh(labels, values, color=s["bar"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=FONT_PROP, color=s["fg"], fontsize=9)
    ax.set_xlabel(xlabel, fontproperties=FONT_PROP, color=s["fg"])
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=s["fg"])
    ax.tick_params(colors=s["fg"])
    for spine in ax.spines.values():
        spine.set_color(s["fg"])
    plt.tight_layout()
    return fig


def plot_applicant_fi_bubble(database, top_applicants=10, top_fi=10, fi_level="サブクラス",
                              title="出願人×FI バブルチャート", theme="deepsea"):
    """
    出願人 × FI の組み合わせごとの件数を、対数スケールのバブルの
    大きさで表す散布図（バブルチャート）にする。

    fi_level: "サブクラス"（先頭4文字）、"メイングループ"（先頭6文字。
              無ければ5文字）、"そのまま"（元のコード）から選ぶ。
              細分番号まで含めた元のコードのままだと、同じ技術分野の
              コードが細かく分散してしまい傾向が見えにくいため、
              既定値は"サブクラス"にしている。
    """
    from collections import Counter

    if fi_level == "サブクラス":
        convert = fi_to_subclass
    elif fi_level == "メイングループ":
        convert = fi_to_maingroup
    else:
        convert = lambda x: x

    applicant_counter = Counter()
    fi_counter = Counter()
    for entry in database:
        applicant_counter.update(set(entry.get("出願人") or []))
        fi_counter.update({convert(v) for v in (entry.get("FI") or [])})

    top_applicant_names = [a for a, _ in applicant_counter.most_common(top_applicants)]
    top_fi_names = [f for f, _ in fi_counter.most_common(top_fi)]

    pair_counter = Counter()
    for entry in database:
        for a in set(entry.get("出願人") or []):
            if a not in top_applicant_names:
                continue
            for f in {convert(v) for v in (entry.get("FI") or [])}:
                if f not in top_fi_names:
                    continue
                pair_counter[(a, f)] += 1

    if not pair_counter:
        raise ValueError("出願人・FIの組み合わせデータが見つかりません。")

    import numpy as np

    s = _atlas_style(theme)
    fig, ax = plt.subplots(figsize=(max(8, len(top_fi_names) * 0.9), max(5, len(top_applicant_names) * 0.6)))
    fig.patch.set_facecolor(s["bg"])
    ax.set_facecolor(s["bg"])

    for (a, f), count in pair_counter.items():
        x = top_fi_names.index(f)
        y = top_applicant_names.index(a)
        size = 80 * np.log1p(count) ** 2 + 40
        ax.scatter(x, y, s=size, color=s["bar"], alpha=0.7, edgecolors=s["fg"], linewidths=0.5)
        ax.text(x, y, str(count), ha="center", va="center", fontsize=8, color=s["bg"])

    ax.set_xticks(range(len(top_fi_names)))
    ax.set_xticklabels(top_fi_names, rotation=45, ha="right", fontproperties=FONT_PROP, color=s["fg"], fontsize=9)
    ax.set_yticks(range(len(top_applicant_names)))
    ax.set_yticklabels(top_applicant_names, fontproperties=FONT_PROP, color=s["fg"], fontsize=9)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=s["fg"])
    ax.tick_params(colors=s["fg"])
    for spine in ax.spines.values():
        spine.set_color(s["grid"])
    ax.grid(True, color=s["grid"], alpha=0.2)
    plt.tight_layout()
    return fig


# ============================================================
# ㉘ 出願人×FI（IPCサブクラス等）のレーダーチャート
# ============================================================

def build_applicant_fi_radar_data(database, applicants=None, fi_level="サブクラス",
                                   top_applicants=5, top_fi=6, axis_selection="複数社共通"):
    """
    出願人ごとに、よく使うFI（サブクラス等）の件数をまとめ、
    plot_radar_chart() にそのまま渡せる形（{出願人名: {FI名: 件数, ...}, ...}）
    にする。

    applicants: 対象にする出願人名のリスト（省略時は出現件数が多い順に
                top_applicants件を自動選択する）
    fi_level: fi_to_subclass/fi_to_maingroupと同じ粒度指定
    top_fi: レーダーの軸として使うFIの数
    axis_selection: "複数社共通"（複数の出願人にまたがって出てくるFIを
                    優先して軸にする。1社だけが突出したFIが軸になり、
                    他社が軒並み0になって尖った形になるのを防ぐ）
                    "全体件数順"（単純に全体の件数が多い順）
    """
    from collections import Counter

    if fi_level == "サブクラス":
        convert = fi_to_subclass
    elif fi_level == "メイングループ":
        convert = fi_to_maingroup
    else:
        convert = lambda x: x

    if applicants is None:
        applicant_counter = Counter()
        for entry in database:
            applicant_counter.update(set(entry.get("出願人") or []))
        applicants = [a for a, _ in applicant_counter.most_common(top_applicants)]

    fi_counter = Counter()
    for entry in database:
        matched_applicants = set(entry.get("出願人") or []) & set(applicants)
        if not matched_applicants:
            continue
        fi_counter.update({convert(v) for v in (entry.get("FI") or [])})

    # FIごとに「何社が使っているか」を数える
    fi_applicant_sets = {}
    for entry in database:
        matched_applicants = set(entry.get("出願人") or []) & set(applicants)
        if not matched_applicants:
            continue
        for fi in {convert(v) for v in (entry.get("FI") or [])}:
            fi_applicant_sets.setdefault(fi, set()).update(matched_applicants)

    if axis_selection == "複数社共通":
        # 「使っている社数」を最優先、同数なら全体件数が多い順にする
        fi_axes = sorted(
            fi_counter.keys(),
            key=lambda f: (-len(fi_applicant_sets.get(f, set())), -fi_counter[f]),
        )[:top_fi]
    else:
        fi_axes = [f for f, _ in fi_counter.most_common(top_fi)]

    profiles = {}
    for applicant in applicants:
        counts = Counter()
        for entry in database:
            if applicant not in (entry.get("出願人") or []):
                continue
            counts.update({convert(v) for v in (entry.get("FI") or [])})
        profiles[applicant] = {fi: counts.get(fi, 0) for fi in fi_axes}

    return profiles


# ============================================================
# ㉙ キーワード地形図（等高線ヒートマップ）
# ============================================================
# よく出てくるキーワードを、意味の近さに基づいて地図上に配置し、
# 頻度を「山の高さ」として等高線で表現する。密集している場所ほど
# 赤く盛り上がった「山」になり、技術用語の集中地帯が一目で分かる。

def build_keyword_landscape(database, top_n=40, kind="component"):
    """
    ポートフォリオ全体でよく出てくるキーワードを取り出し、
    意味的な近さに基づいて2次元の座標を計算する。

    戻り値: [{"word":.., "freq":.., "x":.., "y":..}, ...]
    """
    import numpy as np
    from sklearn.decomposition import PCA

    freq = build_keyword_frequency(database, kind=kind)
    top_words = freq.most_common(top_n)
    if not top_words:
        return []

    words = [w for w, _ in top_words]
    freqs = [f for _, f in top_words]

    model = _get_embed_model()
    embeddings = model.encode(words, normalize_embeddings=True)

    n_comp = min(2, max(len(words) - 1, 1))
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(embeddings)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    points = []
    for w, f, xy in zip(words, freqs, coords):
        points.append({"word": w, "freq": f, "x": float(xy[0]), "y": float(xy[1])})
    return points


def plot_keyword_landscape(points, title="キーワード地形図", grid_size=200, bandwidth=None):
    """
    build_keyword_landscape() の結果を、等高線の地形図（ヒートマップ）
    として描画する。山（赤い部分）が、意味的に近いキーワードが
    密集している＝技術的に厚みのある領域を表す。
    """
    import numpy as np
    import matplotlib.patheffects as pe

    if not points:
        raise ValueError("キーワードが見つかりませんでした")

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    freqs = np.array([p["freq"] for p in points], dtype=float)

    x_pad = (xs.max() - xs.min()) * 0.25 + 1e-6
    y_pad = (ys.max() - ys.min()) * 0.25 + 1e-6
    x_lin = np.linspace(xs.min() - x_pad, xs.max() + x_pad, grid_size)
    y_lin = np.linspace(ys.min() - y_pad, ys.max() + y_pad, grid_size)
    X, Y = np.meshgrid(x_lin, y_lin)

    if bandwidth is None:
        span = max(xs.max() - xs.min(), ys.max() - ys.min())
        bandwidth = span / 7 + 1e-6

    Z = np.zeros_like(X)
    for x, y, f in zip(xs, ys, freqs):
        Z += f * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * bandwidth ** 2))

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.contourf(X, Y, Z, levels=30, cmap="turbo")
    ax.contour(X, Y, Z, levels=12, colors="white", linewidths=0.3, alpha=0.35)

    max_freq = freqs.max() if freqs.max() > 0 else 1
    for p in points:
        size = 9 + 9 * (p["freq"] / max_freq)
        ax.text(
            p["x"], p["y"], p["word"], fontsize=size, fontproperties=FONT_PROP,
            ha="center", va="center", color="white",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            zorder=5,
        )

    ax.set_title(title, fontproperties=FONT_PROP, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# ㉚ キーワード×FIサブクラスのホワイトスペースマップ（全自動）
# ============================================================
# COREとは違い、カテゴリを手入力する必要がない。発明の名称から
# 自動でキーワードを抽出し、縦軸＝キーワード、横軸＝FIサブクラス
# （データに含まれる全種類）の出願件数マトリクスを自動で作る。
# 色が濃いマスほど出願が多く、白いマスがホワイトスペース候補。

def build_keyword_fi_matrix(database, top_keywords=25, top_fi=25, fi_level="サブクラス",
                             title_field="発明の名称", source="発明の名称", applicant_filter=None):
    """
    database: build_full_database() や build_claims_metadata_database() 等の戻り値

    source: "発明の名称"（発明の名称からキーワードを抽出）、
            "請求項"（請求項本文をSAO解析した構成要素をキーワードとして使う。
            database の各エントリが "relations" を持っている必要がある）
    applicant_filter: 出願人名のリストを指定すると、その出願人が
                       関わる特許だけに絞り込んでからマトリクスを作る

    戻り値: (matrix, keyword_list, fi_list)
        matrix: {(キーワード, FI): [id, id, ...], ...}
        keyword_list: 縦軸に使うキーワード（出現件数が多い順）
        fi_list: 横軸に使うFI（出現件数が多い順）
    """
    from collections import Counter

    if fi_level == "サブクラス":
        convert = fi_to_subclass
    elif fi_level == "メイングループ":
        convert = fi_to_maingroup
    else:
        convert = lambda x: x

    if applicant_filter:
        applicant_set = set(applicant_filter)
        database = [e for e in database if set(e.get("出願人") or []) & applicant_set]

    entry_keywords = []
    keyword_counter = Counter()
    fi_counter = Counter()
    for entry in database:
        if source == "請求項":
            kws = _entry_keywords(entry, kind="component")
        else:
            title = entry.get(title_field, "") or ""
            kws = extract_abstract_keywords(title) if title else set()
        entry_keywords.append(kws)
        keyword_counter.update(kws)
        fi_counter.update({convert(v) for v in (entry.get("FI") or [])})

    keyword_list = [w for w, _ in keyword_counter.most_common(top_keywords)]
    fi_list = [f for f, _ in fi_counter.most_common(top_fi)]
    keyword_set = set(keyword_list)
    fi_set = set(fi_list)

    matrix = {}
    for entry, kws in zip(database, entry_keywords):
        fis = {convert(v) for v in (entry.get("FI") or [])} & fi_set
        for kw in kws & keyword_set:
            for fi in fis:
                matrix.setdefault((kw, fi), []).append(entry["id"])

    return matrix, keyword_list, fi_list


def plot_keyword_fi_heatmap(matrix, keyword_list, fi_list, title="キーワード×FI ホワイトスペースマップ"):
    """build_keyword_fi_matrix() の結果をヒートマップにする"""
    import numpy as np

    if not keyword_list or not fi_list:
        # 出願人フィルタ等の条件によって、キーワード・FIのどちらかが
        # 0件になることがある。この場合 arr が空配列（size=0）になり、
        # 後段の arr.max() が「zero-size array」のValueErrorで落ちてしまう
        # ため、先にわかりやすいエラーメッセージで弾く。
        raise ValueError("キーワードまたはFIが0件のため、マップを作成できません。絞り込み条件を緩めてください。")

    arr = np.zeros((len(keyword_list), len(fi_list)), dtype=int)
    for i, kw in enumerate(keyword_list):
        for j, fi in enumerate(fi_list):
            arr[i, j] = len(matrix.get((kw, fi), []))

    fig, ax = plt.subplots(figsize=(max(8, len(fi_list) * 0.5), max(6, len(keyword_list) * 0.35)))
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(fi_list)))
    ax.set_xticklabels(fi_list, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keyword_list)))
    ax.set_yticklabels(keyword_list, fontproperties=FONT_PROP, fontsize=9)

    vmax = arr.max() if arr.max() > 0 else 1
    for i in range(len(keyword_list)):
        for j in range(len(fi_list)):
            val = int(arr[i, j])
            if val == 0:
                continue
            ax.text(j, i, str(val), ha="center", va="center",
                    color="black" if val < vmax / 2 else "white", fontsize=7)

    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("出願件数", fontproperties=FONT_PROP)
    plt.tight_layout()
    return fig


# ============================================================
# ㉛ MEGA：動態分析（活動量×勢いのフェーズ診断）
# ============================================================

def compute_activity_momentum(database, group_by="出願人", recent_years=3, compare_years=3):
    """
    各グループ（出願人 or FI）ごとに、直近recent_years年間の
    出願件数（活動量）と、その直前compare_years年間からのCAGR
    （年平均成長率＝勢い）を計算する。

    group_by: "出願人" または "FI"（どちらもリストを持つフィールド）
    """
    from collections import defaultdict

    if group_by == "FI":
        def get_groups(entry):
            return {fi_to_subclass(v) for v in (entry.get("FI") or [])}
    else:
        def get_groups(entry):
            return set(entry.get(group_by) or [])

    year_counts = defaultdict(lambda: defaultdict(int))
    for entry in database:
        d = entry.get("出願日")
        if d is None:
            continue
        for g in get_groups(entry):
            year_counts[g][d.year] += 1

    all_years = sorted({y for counts in year_counts.values() for y in counts.keys()})
    if not all_years:
        return {}, None, recent_years, compare_years

    latest_year = all_years[-1]
    earliest_year = all_years[0]
    span = latest_year - earliest_year + 1

    # 指定された期間（recent_years + compare_years）がデータの実際の
    # 年範囲より広い場合、そのままでは「比較期間」に実績が存在せず、
    # 全グループが判定不能な「新興」扱いに落ちてしまう。
    # その場合は、実際の年範囲を半分ずつに自動で割り直す。
    if span < recent_years + compare_years:
        half = max(1, span // 2)
        recent_years = half
        compare_years = span - half if span - half > 0 else half

    recent_range = range(latest_year - recent_years + 1, latest_year + 1)
    compare_range = range(latest_year - recent_years - compare_years + 1, latest_year - recent_years + 1)

    result = {}
    for g, counts in year_counts.items():
        recent_total = sum(counts.get(y, 0) for y in recent_range)
        compare_total = sum(counts.get(y, 0) for y in compare_range)
        total_all = sum(counts.values())
        if compare_total > 0:
            cagr = (recent_total / compare_total) ** (1.0 / recent_years) - 1
        elif recent_total > 0:
            cagr = 1.0  # 直前期間に実績がなく、直近だけ出願がある＝新興とみなす
        else:
            cagr = 0.0
        result[g] = {
            "活動量": recent_total,
            "総出願件数": total_all,
            "勢い": cagr,
            "年別件数": dict(sorted(counts.items())),
        }
    return result, latest_year, recent_years, compare_years


def classify_phase(activity, momentum, activity_threshold):
    """活動量と勢いから、リーダー/新興/成熟/衰退の4象限に分類する"""
    if activity >= activity_threshold:
        return "リーダー" if momentum >= 0 else "成熟"
    else:
        return "新興" if momentum >= 0 else "衰退"


def plot_mega_chart(mega_data, title="MEGA：活動量×勢い", top_n=15, theme="deepsea"):
    """
    compute_activity_momentum() の結果を、活動量(x)×勢い(y)の
    散布図にする。4象限がそれぞれ「リーダー・新興・成熟・衰退」に
    対応し、点の位置からその技術・出願人が今どのフェーズにあるかが
    分かる。
    """
    import numpy as np

    if theme == "deepsea":
        bg, fg, grid = "#04121C", "#E8FBFF", "#1a3a4a"
        palette = [s["border"] for s in _DEEPSEA_PALETTE]
    else:
        bg, fg, grid = "#FFFFFF", "#233044", "#dddddd"
        palette = [s["edge"] for s in _BRANCH_PALETTE]

    items = sorted(mega_data.items(), key=lambda x: -x[1]["総出願件数"])[:top_n]
    if not items:
        raise ValueError("データが見つかりませんでした")

    activities = [v["総出願件数"] for _, v in items]
    momentums = [v["勢い"] for _, v in items]
    activity_threshold = sorted(activities)[len(activities) // 2] if activities else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    x_max = max(activities) * 1.3 + 1
    y_max = max(max(momentums), 0.1) * 1.3
    y_min = min(min(momentums), -0.1) * 1.3

    # 4象限の背景色（薄く）
    ax.axvspan(activity_threshold, x_max, 0.5, 1, color=palette[0], alpha=0.08)
    ax.axvspan(0, activity_threshold, 0.5, 1, color=palette[1], alpha=0.08)
    ax.axvspan(activity_threshold, x_max, 0, 0.5, color=palette[2], alpha=0.08)
    ax.axvspan(0, activity_threshold, 0, 0.5, color=palette[3], alpha=0.08)

    ax.axhline(0, color=grid, linewidth=1)
    ax.axvline(activity_threshold, color=grid, linewidth=1, linestyle="--")

    for i, (name, v) in enumerate(items):
        color = palette[i % len(palette)]
        ax.scatter(v["総出願件数"], v["勢い"], s=140, color=color, edgecolors=fg, linewidths=0.8, zorder=5)
        ax.annotate(name, (v["総出願件数"], v["勢い"]), fontsize=9, fontproperties=FONT_PROP,
                    color=fg, xytext=(6, 6), textcoords="offset points", zorder=6)

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("総出願量", fontproperties=FONT_PROP, color=fg)
    ax.set_ylabel("勢い（年平均成長率 CAGR）", fontproperties=FONT_PROP, color=fg)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=16, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(grid)

    label_style = dict(fontproperties=FONT_PROP, fontsize=11, color=fg, alpha=0.6)
    ax.text(x_max * 0.98, y_max * 0.92, "リーダー", ha="right", **label_style)
    ax.text(activity_threshold * 0.4, y_max * 0.92, "新興", ha="center", **label_style)
    ax.text(x_max * 0.98, y_min * 0.92, "成熟", ha="right", **label_style)
    ax.text(activity_threshold * 0.4, y_min * 0.92, "衰退", ha="center", **label_style)

    plt.tight_layout()
    return fig


def plot_mega_chart_interactive(mega_data, title="MEGA：活動量×勢い", top_n=15, theme="deepsea"):
    """
    plot_mega_chart() のインタラクティブ版（Plotly）。
    普段は丸だけを表示し、カーソルを合わせた点だけ出願人（またはFI）名を
    ツールチップで表示する。戻り値は plotly.graph_objects.Figure。
    """
    import plotly.graph_objects as go

    if theme == "deepsea":
        bg, fg, grid = "#04121C", "#E8FBFF", "#1a3a4a"
        palette = [s["border"] for s in _DEEPSEA_PALETTE]
    else:
        bg, fg, grid = "#FFFFFF", "#233044", "#dddddd"
        palette = [s["edge"] for s in _BRANCH_PALETTE]

    items = sorted(mega_data.items(), key=lambda x: -x[1]["総出願件数"])[:top_n]
    if not items:
        raise ValueError("データが見つかりませんでした")

    names = [name for name, _ in items]
    activities = [v["総出願件数"] for _, v in items]
    momentums = [v["勢い"] for _, v in items]
    activity_threshold = sorted(activities)[len(activities) // 2] if activities else 0

    x_max = max(activities) * 1.3 + 1
    y_max = max(max(momentums), 0.1) * 1.3
    y_min = min(min(momentums), -0.1) * 1.3

    fig = go.Figure()
    # 4象限の背景色
    fig.add_shape(type="rect", x0=activity_threshold, x1=x_max, y0=0, y1=y_max,
                  fillcolor=palette[0], opacity=0.08, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=activity_threshold, y0=0, y1=y_max,
                  fillcolor=palette[1], opacity=0.08, line_width=0)
    fig.add_shape(type="rect", x0=activity_threshold, x1=x_max, y0=y_min, y1=0,
                  fillcolor=palette[2], opacity=0.08, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=activity_threshold, y0=y_min, y1=0,
                  fillcolor=palette[3], opacity=0.08, line_width=0)
    fig.add_hline(y=0, line_color=grid)
    fig.add_vline(x=activity_threshold, line_color=grid, line_dash="dash")

    colors = [palette[i % len(palette)] for i in range(len(items))]
    fig.add_trace(go.Scatter(
        x=activities, y=momentums, mode="markers", text=names,
        marker=dict(size=16, color=colors, line=dict(width=1, color=fg)),
        hovertemplate="%{text}<br>総出願件数: %{x}<br>勢い: %{y:.2f}<extra></extra>",
    ))

    for label, x_pos, y_pos, anchor in [
        ("リーダー", x_max * 0.98, y_max * 0.95, "right"),
        ("新興", activity_threshold * 0.4, y_max * 0.95, "center"),
        ("成熟", x_max * 0.98, y_min * 0.95, "right"),
        ("衰退", activity_threshold * 0.4, y_min * 0.95, "center"),
    ]:
        fig.add_annotation(x=x_pos, y=y_pos, text=label, showarrow=False,
                           font=dict(color=fg, size=13), opacity=0.6, xanchor=anchor)

    fig.update_layout(
        title=title,
        xaxis_title="総出願量", yaxis_title="直近の成長率（CAGR）",
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, range=[0, x_max]),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, range=[y_min, y_max]),
        showlegend=False,
    )
    return fig


# ============================================================
# ㉜ 出願人ごとの自動グループ化（手作業のグループ分け不要）
# ============================================================

def get_applicant_groups(database, top_n=5):
    """
    データベースに「出願人」情報が含まれる場合、出願件数が多い順に
    上位top_n件の出願人ごとに、idのリストをまとめる。
    手作業でのグループ分けをせず、自動で比較グループを作るために使う。

    戻り値: {出願人名: [id, id, ...], ...}（出願件数が多い順）
            出願人情報が無いデータベースの場合は空の辞書を返す。
    """
    from collections import Counter, defaultdict

    if not any(e.get("出願人") for e in database):
        return {}

    applicant_counter = Counter()
    for entry in database:
        applicant_counter.update(set(entry.get("出願人") or []))

    top_applicants = [a for a, _ in applicant_counter.most_common(top_n)]
    groups = defaultdict(list)
    for entry in database:
        for a in set(entry.get("出願人") or []):
            if a in top_applicants:
                groups[a].append(entry["id"])
    # 出願件数の多い順を維持する
    return {a: groups[a] for a in top_applicants if a in groups}


# ============================================================
# ㉝ メタデータ（発明の名称・FI）だけで完結する分布・プロファイル
# ============================================================
# 請求項データベース（relationsを持つ）がなくても、
# 「発明の名称」から抽出したキーワード数や、FIコード数を使って、
# クレームの広さ・狭さの分布や、出願人ごとの特徴比較の代わりにする。

def compute_metadata_distribution(database, ids=None, metric="キーワード数"):
    """
    請求項データベースを使わず、発明の名称のキーワード数や
    FIコード数の分布を作る（compute_scope_distribution() の代わり）。

    metric: "キーワード数"（発明の名称から抽出した語の種類数）
            "FIコード数"（付与されているFIコードの種類数）
    """
    target_ids = set(ids) if ids is not None else None
    scores = []
    for entry in database:
        if target_ids is not None and entry["id"] not in target_ids:
            continue
        if metric == "FIコード数":
            value = len(set(entry.get("FI") or []))
        else:
            value = len(_entry_keywords(entry, kind="component"))
        scores.append({"id": entry["id"], "value": value})
    return scores


def plot_metadata_distribution(scores, metric="キーワード数", title=None, theme="deepsea", bins=10):
    """compute_metadata_distribution() の結果をヒストグラムにする"""
    if theme == "deepsea":
        bg, fg, bar = "#04121C", "#E8FBFF", "#5FD4E0"
    else:
        bg, fg, bar = "#FFFFFF", "#233044", "#4C87C6"

    values = [s["value"] for s in scores]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.hist(values, bins=bins, color=bar, edgecolor=fg, alpha=0.85)
    ax.set_xlabel(metric, fontproperties=FONT_PROP, color=fg)
    ax.set_ylabel("件数", fontproperties=FONT_PROP, color=fg)
    ax.set_title(title or f"{metric}の分布", fontproperties=FONT_PROP, fontsize=14, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    plt.tight_layout()
    return fig


def compute_metadata_group_profile(database, ids):
    """
    請求項データベースを使わず、発明の名称・FIのメタデータだけで
    グループ（出願人等）の特徴プロファイルを作る
    （compute_group_profile() の代わり）。レーダーチャート用。
    """
    target_ids = set(ids)
    entries = [e for e in database if e["id"] in target_ids]
    if not entries:
        return {}

    avg_keywords = sum(len(_entry_keywords(e, kind="component")) for e in entries) / len(entries)
    avg_fi = sum(len(set(e.get("FI") or [])) for e in entries) / len(entries)
    unique_keywords = len(build_keyword_frequency(database, ids, kind="component"))
    unique_fi = len({fi_to_subclass(v) for e in entries for v in (e.get("FI") or [])})

    return {
        "出願件数": len(entries),
        "平均キーワード数": avg_keywords,
        "平均FIコード数": avg_fi,
        "キーワードの種類数": unique_keywords,
        "FIサブクラスの種類数": unique_fi,
    }


# ============================================================
# ㉞ 出願人の名寄せ（グループ会社をまとめる）
# ============================================================
# 「株式会社東芝」「東芝デバイス＆ストレージ株式会社」「東芝マテリアル
# 株式会社」のような、同じグループの子会社・関連会社がバラバラの
# 出願人として扱われてしまう問題に対応する。

DEFAULT_APPLICANT_GROUP_KEYWORDS = ["富士電機", "三菱電機", "ローム", "東芝"]


def normalize_applicant_name(name, group_keywords=None):
    """
    出願人名を、グループ会社に共通する親会社名に正規化する。
    例：「東芝デバイス＆ストレージ株式会社」→「東芝」

    group_keywords で指定した語のいずれかが出願人名に含まれていれば、
    その語（＝親会社名）を正規化後の名前として返す。どれにも一致
    しなければ、元の名前をそのまま返す。
    複数の語に一致する場合は、一番長く一致した語を優先する
    （誤って短い語に丸められるのを防ぐため）。
    """
    keywords = group_keywords if group_keywords is not None else DEFAULT_APPLICANT_GROUP_KEYWORDS
    matched = [kw for kw in keywords if kw and kw in name]
    if not matched:
        return name
    return max(matched, key=len)


def apply_applicant_normalization(database, group_keywords=None, field="出願人"):
    """
    データベース全体の出願人名を正規化した、新しいデータベースを返す
    （元のデータベースは変更しない）。正規化前の名前は
    "{field}_元" というキーにそのまま保存しておく。
    """
    new_db = []
    for entry in database:
        new_entry = dict(entry)
        original = entry.get(field) or []
        normalized = sorted({normalize_applicant_name(n, group_keywords) for n in original})
        new_entry[field] = normalized
        new_entry[f"{field}_元"] = original
        new_db.append(new_entry)
    return new_db


# ============================================================
# ㉟ 登録率分析：どのグループが権利化に成功しやすいか
# ============================================================

def _is_registered(entry):
    """1件の特許が登録済みかどうかを判定する"""
    if entry.get("登録番号"):
        return True
    stage = entry.get("ステージ") or ""
    return "登録" in stage


def compute_registration_rate(database, group_by="出願人", fi_level="サブクラス", top_n=15):
    """
    グループ（出願人・FI・発明の名称のキーワード）ごとに、
    登録済みの割合（登録率）を計算する。

    group_by: "出願人"、"FI"、"キーワード"（発明の名称から抽出）
    戻り値: [{"グループ": .., "総数": .., "登録数": .., "登録率": ..}, ...]
            （総数が多い順、上位top_n件）
    """
    from collections import defaultdict

    if fi_level == "サブクラス":
        convert = fi_to_subclass
    elif fi_level == "メイングループ":
        convert = fi_to_maingroup
    else:
        convert = lambda x: x

    counts = defaultdict(lambda: {"総数": 0, "登録数": 0})
    for entry in database:
        registered = _is_registered(entry)
        if group_by == "出願人":
            groups = set(entry.get("出願人") or [])
        elif group_by == "FI":
            groups = {convert(v) for v in (entry.get("FI") or [])}
        else:
            groups = extract_abstract_keywords(entry.get("発明の名称") or "")
        for g in groups:
            counts[g]["総数"] += 1
            if registered:
                counts[g]["登録数"] += 1

    result = []
    for g, c in counts.items():
        result.append({
            "グループ": g,
            "総数": c["総数"],
            "登録数": c["登録数"],
            "登録率": c["登録数"] / c["総数"] if c["総数"] else 0.0,
        })
    result.sort(key=lambda x: -x["総数"])
    return result[:top_n]


def plot_registration_rate(rows, title="登録率", theme="deepsea"):
    """compute_registration_rate() の結果を横棒グラフにする（登録率でソート）"""
    if theme == "deepsea":
        bg, fg, bar = "#04121C", "#E8FBFF", "#5FD4E0"
    else:
        bg, fg, bar = "#FFFFFF", "#233044", "#4C87C6"

    rows_sorted = sorted(rows, key=lambda x: x["登録率"])
    labels = [f'{r["グループ"]}（{r["登録数"]}/{r["総数"]}）' for r in rows_sorted]
    values = [r["登録率"] * 100 for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.4)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.barh(labels, values, color=bar)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=FONT_PROP, color=fg, fontsize=9)
    ax.set_xlabel("登録率（%）", fontproperties=FONT_PROP, color=fg)
    ax.set_xlim(0, 100)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    plt.tight_layout()
    return fig


# ============================================================
# ㊱ 権利化期間分析：どのグループ・技術が早く／遅く公開されるか
# ============================================================

def compute_time_to_publication(database, group_by="出願人", fi_level="サブクラス", top_n=15):
    """
    グループ（出願人・FI）ごとに、出願日から公知日までの日数（権利化に
    かかった期間の目安）の平均・中央値を計算する。
    出願日・公知日の両方がある行だけを対象にする。

    戻り値: [{"グループ": .., "件数": .., "平均日数": .., "中央値日数": ..}, ...]
            （件数が多い順、上位top_n件）
    """
    from collections import defaultdict
    import statistics

    if fi_level == "サブクラス":
        convert = fi_to_subclass
    elif fi_level == "メイングループ":
        convert = fi_to_maingroup
    else:
        convert = lambda x: x

    days_by_group = defaultdict(list)
    for entry in database:
        d1 = entry.get("出願日")
        d2 = entry.get("公知日")
        if d1 is None or d2 is None:
            continue
        days = (d2 - d1).days
        if days < 0:
            continue
        if group_by == "出願人":
            groups = set(entry.get("出願人") or [])
        else:
            groups = {convert(v) for v in (entry.get("FI") or [])}
        for g in groups:
            days_by_group[g].append(days)

    result = []
    for g, days_list in days_by_group.items():
        result.append({
            "グループ": g,
            "件数": len(days_list),
            "平均日数": statistics.mean(days_list),
            "中央値日数": statistics.median(days_list),
        })
    result.sort(key=lambda x: -x["件数"])
    return result[:top_n]


def plot_time_to_publication(rows, title="出願から公知までの期間", theme="deepsea"):
    """compute_time_to_publication() の結果を横棒グラフにする（平均日数でソート）"""
    if theme == "deepsea":
        bg, fg, bar = "#04121C", "#E8FBFF", "#5FD4E0"
    else:
        bg, fg, bar = "#FFFFFF", "#233044", "#4C87C6"

    rows_sorted = sorted(rows, key=lambda x: x["平均日数"])
    labels = [f'{r["グループ"]}（{r["件数"]}件）' for r in rows_sorted]
    values = [r["平均日数"] / 365.25 for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.4)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.barh(labels, values, color=bar)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=FONT_PROP, color=fg, fontsize=9)
    ax.set_xlabel("平均期間（年）", fontproperties=FONT_PROP, color=fg)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    plt.tight_layout()
    return fig


# ============================================================
# ㊲ 技術の「先願者」年表：どのマスを誰が最初に押さえたか
# ============================================================

def find_first_filers(database, top_keywords=25, top_fi=25, fi_level="サブクラス",
                       title_field="発明の名称"):
    """
    build_keyword_fi_matrix() と同じ「キーワード×FI」のマス目について、
    各マスに最も早く出願したのが誰（どの出願人）で、いつだったかを求める。
    「この技術の組み合わせは、実はどの会社が先行して押さえていたか」を
    可視化するために使う。

    戻り値: [{"キーワード":.., "FI":.., "最初の出願人":.., "最初の出願日":..,
              "件数":..}, ...]（キーワード×FIのマスごとに1行、出願日が早い順）
    """
    matrix, keyword_list, fi_list = build_keyword_fi_matrix(
        database, top_keywords=top_keywords, top_fi=top_fi, fi_level=fi_level, title_field=title_field
    )
    db_by_id = {e["id"]: e for e in database}

    rows = []
    for (kw, fi), ids in matrix.items():
        entries = [db_by_id[i] for i in ids if i in db_by_id and db_by_id[i].get("出願日") is not None]
        if not entries:
            continue
        entries.sort(key=lambda e: e["出願日"])
        first = entries[0]
        rows.append({
            "キーワード": kw,
            "FI": fi,
            "最初の出願人": "、".join(first.get("出願人") or ["（不明）"]),
            "最初の出願日": first["出願日"],
            "件数": len(entries),
        })
    rows.sort(key=lambda r: r["最初の出願日"])
    return rows


# ============================================================
# ㊳ 請求項データベース（メタデータ付き）の読み込み・構築
# ============================================================
# 「id,特許番号,出願日,出願人,発明の名称,FI/IPC,請求項番号,請求項本文」
# 形式のCSV（Dataset A用テンプレート）を読み込み、請求項本文をSAO解析
# した上で、出願人・FIのメタデータと合わせたデータベースを作る。
# これにより「請求項から抽出した構成要素×FI」のホワイトスペースマップ
# や、出願人別の絞り込みができるようになる。

def load_claims_with_metadata_csv(csv_text):
    """
    請求項本文とメタデータ（出願人・FI等）を両方持つCSVを読み込む。
    列名: id（省略可）, 出願人（または出願人/権利者）, FI（またはFI/IPC）,
          請求項本文, 発明の名称（任意）, 出願日（任意）, 特許番号（任意）
    """
    import csv
    import io
    from datetime import datetime

    def _split_multi(value, seps=("／", "、", ",", ";", "；")):
        if not value:
            return []
        text = value
        for s in seps[1:]:
            text = text.replace(s, seps[0])
        return [v.strip() for v in text.split(seps[0]) if v.strip()]

    def _parse_date(value):
        if not value:
            return None
        value = value.strip().replace("/", "-").replace(".", "-")
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    reader = csv.DictReader(io.StringIO(csv_text))
    records = []
    for i, row in enumerate(reader):
        text = (row.get("請求項本文") or row.get("text") or "").strip()
        if not text:
            continue
        rid = row.get("id") or row.get("特許番号") or f"行{i+1}"
        applicant_raw = row.get("出願人") or row.get("出願人/権利者") or ""
        fi_raw = row.get("FI/IPC") or row.get("FI") or ""
        records.append({
            "id": rid,
            "出願人": _split_multi(applicant_raw),
            "FI": _split_multi(fi_raw),
            "請求項本文": text,
            "発明の名称": (row.get("発明の名称") or "").strip(),
            "出願日": _parse_date(row.get("出願日", "")),
            "特許番号": (row.get("特許番号") or "").strip(),
        })
    return records


def build_claims_metadata_database(records, show_progress=True):
    """
    load_claims_with_metadata_csv() の結果から、各行の請求項本文を
    SAO解析し、メタデータ（出願人・FI等）と合わせたデータベースを作る。

    build_patent_database()/build_abstract_database()等と同様に、
    文書全体の平均埋め込みベクトル（doc_embedding）も合わせて計算する
    （類似度ネットワーク図・意味的俯瞰マップ等、doc_embeddingを前提とする
    機能をこのデータベースでも使えるようにするため）。
    """
    import numpy as np

    model = _get_embed_model()
    database = []
    total = len(records)
    for i, r in enumerate(records):
        if show_progress:
            print(f"[{i+1}/{total}] {r['id']} を解析中...")
        try:
            _, relations = analyze_claim(r["請求項本文"])
        except Exception as e:
            if show_progress:
                print(f"  → 解析エラー、スキップします: {e}")
            continue
        entry = dict(r)
        entry["relations"] = relations
        entry["text"] = r["請求項本文"]

        triples = sorted(relations_to_triple_set(relations, normalize_numbers=True)) if relations else []
        if triples:
            texts = [_triple_to_text(t) for t in triples]
            embeddings = model.encode(texts, normalize_embeddings=True)
            doc_embedding = np.mean(embeddings, axis=0)
            doc_embedding = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-8)
        else:
            # SAOが1件も抽出できなかった請求項は、代わりに請求項本文
            # そのものを埋め込む（他の関数のように行ごと捨ててしまうと、
            # メタデータ一覧・出願人フィルタ等、他の機能で件数が
            # 合わなくなってしまうため、このデータベースでは行を残す）。
            embedding = model.encode([r["請求項本文"]], normalize_embeddings=True)[0]
            doc_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        entry["doc_embedding"] = doc_embedding

        database.append(entry)
    return database


# ============================================================
# ㊴ キーワード → FI推薦（J-PlatPat検索式の作成支援）
# ============================================================
# 単純なAI推測ではなく、「実際のデータの中で、そのキーワードを含む
# 特許にどのFIが多く付与されているか」を統計的に集計し、根拠（出現率・
# 実例）付きで推薦する。J-PlatPatで検索条件（FI）を決める際の
# 参考情報として使う。

def _entry_searchable_text(entry):
    """
    出願人検索用に、そのエントリが持っているテキスト情報
    （発明の名称・要約・請求項本文）を全部つなげたものを返す。
    """
    parts = []
    for field in ("発明の名称", "text", "請求項本文"):
        v = entry.get(field)
        if v:
            parts.append(v)
    return "".join(parts)


def recommend_fi_for_keywords(database, keywords, match_mode="いずれか", fi_level="サブクラス", top_n=10):
    """
    指定したキーワードを含む特許を検索し、その特許群でよく使われている
    FIを、出現率（スコア）付きで推薦する。

    keywords: 検索したいキーワードのリスト
    match_mode: "いずれか"（OR、キーワードのうち1つでも含まれていればよい）
                "すべて"（AND、全部のキーワードを含む特許だけを対象にする。
                キーワードの組み合わせによる推薦に使う）
    fi_level: fi_to_subclass/fi_to_maingroupと同じ粒度指定

    戻り値: {
        "matched_count": マッチした特許の件数,
        "recommendations": [
            {"FI": .., "件数": .., "スコア": .., "サンプルid": [id, ...]}, ...
        ]（スコアが高い順）
    }
    """
    from collections import Counter, defaultdict

    if fi_level == "サブクラス":
        convert = fi_to_subclass
    elif fi_level == "メイングループ":
        convert = fi_to_maingroup
    else:
        convert = lambda x: x

    keywords = [k for k in keywords if k]
    matched = []
    for entry in database:
        text = _entry_searchable_text(entry)
        if not text:
            continue
        if match_mode == "すべて":
            ok = all(kw in text for kw in keywords)
        else:
            ok = any(kw in text for kw in keywords)
        if ok:
            matched.append(entry)

    fi_counter = Counter()
    fi_examples = defaultdict(list)
    for entry in matched:
        fis = {convert(v) for v in (entry.get("FI") or [])}
        for fi in fis:
            fi_counter[fi] += 1
            if len(fi_examples[fi]) < 5:
                fi_examples[fi].append(entry["id"])

    total = len(matched)
    recommendations = []
    for fi, count in fi_counter.most_common(top_n):
        recommendations.append({
            "FI": fi,
            "件数": count,
            "スコア": count / total if total else 0.0,
            "サンプルid": fi_examples[fi],
        })

    return {"matched_count": total, "recommendations": recommendations}


def plot_fi_recommendations(result, title="キーワード→FI推薦", theme="deepsea"):
    """recommend_fi_for_keywords() の結果を横棒グラフ（出現率）にする"""
    if theme == "deepsea":
        bg, fg, bar = "#04121C", "#E8FBFF", "#5FD4E0"
    else:
        bg, fg, bar = "#FFFFFF", "#233044", "#4C87C6"

    recs = list(reversed(result["recommendations"]))
    labels = [f'{r["FI"]}（{r["件数"]}/{result["matched_count"]}件）' for r in recs]
    values = [r["スコア"] * 100 for r in recs]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.4)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.barh(labels, values, color=bar)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=FONT_PROP, fontsize=9, color=fg)
    ax.set_xlabel("出現率（%）", fontproperties=FONT_PROP, color=fg)
    ax.set_xlim(0, 100)
    ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    plt.tight_layout()
    return fig


# ============================================================
# ㊵ SAO抽出精度の定量評価（適合率・再現率・F1値）
# ============================================================
# 人手で作った正解SAOトリプルと、システムの抽出結果を比較して、
# 定量的に精度を評価する。卒論の「検証実験」章にそのまま使える。

def _normalize_relation_for_match(text):
    """
    動詞の活用形の違い（「有し」「有する」「有している」等）を吸収する
    ために、関係のテキストから漢字部分だけを取り出して正規化する。
    """
    return "".join(_re_dep.findall(r"[一-龥]+", text)) or text


# 請求項の中では「有する」「備える」「具備する」「含む」のように、
# 漢字表記が全く異なるのに実質的に同じ意味（全体が部分を持つ、という
# 関係）で使われる動詞群がある。_normalize_relation_for_match の
# 「漢字部分の一致」だけではこれらは別の関係として扱われてしまい、
# 正解データの語彙選択とシステムの語彙選択がたまたま違うだけで
# 不一致（誤り）とカウントされてしまう。評価の趣旨は「意味として
# 正しい関係を抽出できているか」なので、既知の同義語グループは
# 同じ関係とみなして比較する。
RELATION_SYNONYM_GROUPS = [
    # 「の」は正解データ側で「有する/含む」に相当する関係（全体が部分・
    # 属性を持つ、の意）をそのまま属格「の」で表記している箇所が
    # 532件中186クレーム・485件存在する（例:「スイッチ素子の高電位端子」
    # →source=スイッチ素子, relation=の, target=高電位端子）。全て
    # source=全体・target=部分/属性で方向は一貫しているため、同義語として
    # 扱う（本人に確認済み）。
    {"有する", "備える", "具備する", "含む", "含める", "の"},
    {"配置される", "配置", "設けられる", "設置される", "設置"},
    {"接続される", "接続", "連結される", "連結"},
    {"接触する", "接触", "当接する", "当接"},
    {"形成される", "形成"},
    {"固定される", "固定"},
    # 以下、532件の正解データで実際の用例を確認し、方向（source=全体・
    # target=部分/材料、または受動態でsource=被覆・搭載される側）が
    # 一貫していることを確認した上で追加。能動態（「構成する」「搭載する」）
    # は所有の向きが逆（source=部分側）になるため、あえて含めていない。
    {"からなる", "構成される"},
    {"搭載される", "実装される"},
    {"覆う", "被覆する"},
]


def _relation_synonym_match(rel_a, rel_b):
    """relation文字列が既知の同義語グループで一致するかを判定する"""
    for group in RELATION_SYNONYM_GROUPS:
        a_in = any(g in rel_a for g in group)
        b_in = any(g in rel_b for g in group)
        if a_in and b_in:
            return True
    return False


def evaluate_triples(predicted_relations, gold_triples, lenient_relation_match=True):
    """
    システムが抽出したSAOトリプル（predicted_relations）と、
    人手で作った正解トリプル（gold_triples）を比較し、
    適合率（Precision）・再現率（Recall）・F1値を計算する。

    predicted_relations / gold_triples: どちらも
        {"source":.., "relation":.., "target":..} の形の辞書のリスト
        （"type"キーは比較に使わない）

    lenient_relation_match: Trueの場合、relation（動詞部分）の比較を、
        漢字部分だけを取り出して行う（「有し」「有する」「有している」の
        ような活用の違いを吸収するため）。片方がもう片方の漢字部分を
        含んでいればよい（「決め」と「位置決めされる」のような、
        複合語の一部一致も許容する）。
        Falseの場合はrelationも完全一致でなければ正解としない。

    戻り値: {
        "precision":.., "recall":.., "f1":..,
        "正解数":.., "システム抽出数":.., "正解データ数":..,
        "matched_pred": [...]（正解した抽出結果）,
        "unmatched_pred": [...]（システムが誤って抽出した関係）,
        "unmatched_gold": [...]（システムが見逃した関係）,
    }
    """
    def _match(p, g):
        if p["source"] != g["source"] or p["target"] != g["target"]:
            return False
        if p["relation"] == g["relation"]:
            return True
        if lenient_relation_match:
            pn = _normalize_relation_for_match(p["relation"])
            gn = _normalize_relation_for_match(g["relation"])
            if pn == gn or pn in gn or gn in pn:
                return True
            return _relation_synonym_match(p["relation"], g["relation"])
        return False

    matched_gold_idx = set()
    matched_pred_idx = set()
    for pi, p in enumerate(predicted_relations):
        for gi, g in enumerate(gold_triples):
            if gi in matched_gold_idx:
                continue
            if _match(p, g):
                matched_pred_idx.add(pi)
                matched_gold_idx.add(gi)
                break

    matched_pred = [predicted_relations[i] for i in sorted(matched_pred_idx)]
    unmatched_pred = [p for i, p in enumerate(predicted_relations) if i not in matched_pred_idx]
    unmatched_gold = [g for i, g in enumerate(gold_triples) if i not in matched_gold_idx]

    tp = len(matched_pred_idx)
    precision = tp / len(predicted_relations) if predicted_relations else 0.0
    recall = tp / len(gold_triples) if gold_triples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "正解数": tp,
        "システム抽出数": len(predicted_relations),
        "正解データ数": len(gold_triples),
        "matched_pred": matched_pred,
        "unmatched_pred": unmatched_pred,
        "unmatched_gold": unmatched_gold,
    }


def print_evaluation_report(predicted_relations, gold_triples, name="請求項", lenient_relation_match=True):
    """evaluate_triples() の結果を、人が読みやすい形で表示する"""
    result = evaluate_triples(predicted_relations, gold_triples, lenient_relation_match=lenient_relation_match)
    print(f"=== {name} の評価 ===")
    print(f"適合率(Precision): {result['precision']*100:5.1f}%  （{result['正解数']}/{result['システム抽出数']}）")
    print(f"再現率(Recall):    {result['recall']*100:5.1f}%  （{result['正解数']}/{result['正解データ数']}）")
    print(f"F1値:              {result['f1']*100:5.1f}%")
    if result["unmatched_pred"]:
        print("--- システムが誤って抽出した関係（誤検出） ---")
        for r in result["unmatched_pred"]:
            print(f"  {r['source']} --{r['relation']}--> {r['target']}")
    if result["unmatched_gold"]:
        print("--- システムが見逃した関係（未検出） ---")
        for r in result["unmatched_gold"]:
            print(f"  {r['source']} --{r['relation']}--> {r['target']}")
    return result


# ============================================================
# 緩い評価（ノード表記の正規化＋意味的類似度）
# ============================================================
# 【方針転換】2026年時点でユーザーの明示的な判断により、「評価方法は
# 意味が伝わればいいくらいに緩くてよい」という方針に変更した。
# 従来のevaluate_triples/batch_evaluate（source/targetの完全一致必須）は
# GiNZA初期版0.277→改良版0.421という、これまでの厳格な比較の土台として
# そのまま残す（後方互換のため一切変更しない）。今後のLLM版の評価は、
# こちらの「緩い」評価を主指標として使う。
#
# 「緩さ」の内訳（ユーザー確認済み）：
#   ①表記・字体の揺れ（簡体字/日本字体の混同、全角/半角）を正規化
#   ②数量詞・修飾語（「複数の」「少なくとも１つの」等）の有無を無視
#   ③上記正規化後もまだ一致しない場合は、埋め込みモデルによる意味的
#     類似度（コサイン類似度が閾値以上）で一致とみなす
#
# ①②は無料（決定的・高速）なので必ず先に試し、③（埋め込み計算）は
# ①②で一致しなかったペアにだけ使う（全ペアを毎回embeddingにかけると
# 遅く、かつ①②で明らかに同じと分かるものまで確率的な閾値判定に
# 委ねてしまうため）。

import unicodedata as _unicodedata_lenient

# LLM（Ollama）が日本語のつもりで簡体字/類似字体を出力することがある
# （実運用ログで「収容する」→「收容する」を確認）。意味は同じなので、
# 緩い評価では同一視する。新しい混同パターンが見つかり次第、追加する。
_LENIENT_CHAR_VARIANTS = {
    "收": "収",
    "载": "載",  # 特開2025-174033で「载置する」「载置工程」を確認
}

# 埋め込みモデルが使えない場合の警告を、実行中に1回だけ表示するためのフラグ
# （batch_evaluate_lenientで532件回すと、警告なしだと毎回同じ理由で
# 静かにフォールバックし続けてしまい、精度低下の原因に気づきにくいため）。
_EMBED_MODEL_WARNING_SHOWN = False

# _QUANTIFIER_WORDS（構成要素の所有者プレフィックス除外用に既存定義済み）
# に加え、「１つの」「三つの」のような数詞＋助数詞パターンも接頭辞として
# 除去する。
_QUANTIFIER_PREFIX_RE = _re_symbolic_label.compile(
    "^(複数の|いくつかの|幾つかの|各|全ての|すべての|少なくとも|一部の|"
    "双方の|任意の|それぞれの|[0-9一二三四五六七八九十]+つの|[0-9]+個の)"
)


def _normalize_node_text_lenient(text):
    """
    緩い評価のための、source/targetテキストの正規化。
    「意味が伝わればいい」という基準を、①字体②数量詞・修飾語の観点で
    具体化したもの（③の意味的類似度は、この正規化を通した上で
    それでも不一致の場合にのみ埋め込みモデルにかける）。
    """
    if not text:
        return text
    t = text
    for a, b in _LENIENT_CHAR_VARIANTS.items():
        t = t.replace(a, b)
    t = _unicodedata_lenient.normalize("NFKC", t)
    for prefix in ("前記", "該"):
        if t.startswith(prefix) and t != prefix:
            t = t[len(prefix):]
    prev = None
    while prev != t:
        prev = t
        m = _QUANTIFIER_PREFIX_RE.match(t)
        if m and len(t) > len(m.group(0)):
            t = t[len(m.group(0)):]
    return t.strip()


def evaluate_triples_lenient(predicted_relations, gold_triples, lenient_relation_match=True,
                              semantic_threshold=0.75, use_semantic=True):
    """
    「意味が伝わっていれば正解」という基準でのF値（evaluate_triplesの
    緩和版）。source/targetは①字体・数量詞の正規化→②（それでも
    不一致なら）埋め込みモデルによる意味的類似度、の2段階でマッチを試みる。
    relationはevaluate_triplesと同じ緩和ロジック（漢字部分一致・同義語）。

    戻り値はevaluate_triples()と同じキー構成に加え、
    "match_method"（"exact_or_normalized" / "semantic"）別の内訳を
    "正解内訳" として返す。
    """
    def _rel_match(p_rel, g_rel):
        if p_rel == g_rel:
            return True
        if not lenient_relation_match:
            return False
        pn = _normalize_relation_for_match(p_rel)
        gn = _normalize_relation_for_match(g_rel)
        if pn == gn or pn in gn or gn in pn:
            return True
        return _relation_synonym_match(p_rel, g_rel)

    norm_pred = [
        (i, _normalize_node_text_lenient(p["source"]), _normalize_node_text_lenient(p["target"]), p["relation"])
        for i, p in enumerate(predicted_relations)
    ]
    norm_gold = [
        (i, _normalize_node_text_lenient(g["source"]), _normalize_node_text_lenient(g["target"]), g["relation"])
        for i, g in enumerate(gold_triples)
    ]

    matched_pred_idx = {}
    matched_gold_idx = set()

    # ①②: 正規化後の完全一致を先に確定させる
    for pi, p_src, p_tgt, p_rel in norm_pred:
        for gi, g_src, g_tgt, g_rel in norm_gold:
            if gi in matched_gold_idx:
                continue
            if p_src == g_src and p_tgt == g_tgt and _rel_match(p_rel, g_rel):
                matched_pred_idx[pi] = ("normalized", gi)
                matched_gold_idx.add(gi)
                break

    # ③: 残りは埋め込みモデルによる意味的類似度でマッチを試みる
    remaining_pred = [(pi, p) for pi, p in enumerate(predicted_relations) if pi not in matched_pred_idx]
    remaining_gold = [(gi, g) for gi, g in enumerate(gold_triples) if gi not in matched_gold_idx]
    if use_semantic and remaining_pred and remaining_gold:
        try:
            model = _get_embed_model()
            pred_texts = [
                _triple_to_text((
                    _normalize_node_text_lenient(p["source"]), p["relation"],
                    _normalize_node_text_lenient(p["target"]),
                ))
                for _, p in remaining_pred
            ]
            gold_texts = [
                _triple_to_text((
                    _normalize_node_text_lenient(g["source"]), g["relation"],
                    _normalize_node_text_lenient(g["target"]),
                ))
                for _, g in remaining_gold
            ]
            emb_pred = model.encode(pred_texts, normalize_embeddings=True)
            emb_gold = model.encode(gold_texts, normalize_embeddings=True)
            sim = emb_pred @ emb_gold.T

            candidates = []
            for a in range(sim.shape[0]):
                for b in range(sim.shape[1]):
                    candidates.append((sim[a, b], a, b))
            candidates.sort(key=lambda x: -x[0])

            used_gold_local = set()
            for s, a, b in candidates:
                if s < semantic_threshold:
                    break
                pi, _ = remaining_pred[a]
                gi, _ = remaining_gold[b]
                if pi in matched_pred_idx or b in used_gold_local or gi in matched_gold_idx:
                    continue
                matched_pred_idx[pi] = ("semantic", gi)
                matched_gold_idx.add(gi)
                used_gold_local.add(b)
        except Exception as e:
            # sentence-transformers未インストール、初回モデルダウンロードに
            # 必要なネットワークが使えない等、埋め込みモデルが使えない
            # 環境では③をスキップし、①②の正規化一致のみで評価する
            # （use_semantic=Falseと同じ挙動にフォールバックする）。
            # ImportErrorだけでなく、モデルダウンロード時のネットワーク
            # エラー（huggingface_hubの取得失敗等）もここで拾う必要がある。
            global _EMBED_MODEL_WARNING_SHOWN
            if not _EMBED_MODEL_WARNING_SHOWN:
                print(
                    f"[警告] 意味的類似度によるマッチングを利用できません"
                    f"（{type(e).__name__}: {e}）。表記正規化のみで評価します。"
                )
                _EMBED_MODEL_WARNING_SHOWN = True

    matched_pred = [predicted_relations[i] for i in sorted(matched_pred_idx)]
    unmatched_pred = [p for i, p in enumerate(predicted_relations) if i not in matched_pred_idx]
    unmatched_gold = [g for i, g in enumerate(gold_triples) if i not in matched_gold_idx]

    tp = len(matched_pred_idx)
    precision = tp / len(predicted_relations) if predicted_relations else 0.0
    recall = tp / len(gold_triples) if gold_triples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    n_normalized = sum(1 for method, _ in matched_pred_idx.values() if method == "normalized")
    n_semantic = sum(1 for method, _ in matched_pred_idx.values() if method == "semantic")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "正解数": tp,
        "システム抽出数": len(predicted_relations),
        "正解データ数": len(gold_triples),
        "matched_pred": matched_pred,
        "unmatched_pred": unmatched_pred,
        "unmatched_gold": unmatched_gold,
        "正解内訳": {"表記正規化一致": n_normalized, "意味的類似度一致": n_semantic},
    }


def batch_evaluate_lenient(test_cases, lenient_relation_match=True, semantic_threshold=0.75,
                            use_semantic=True, use_ollama=False):
    """
    evaluate_triples_lenient() を複数件まとめて実行する、batch_evaluate()の
    緩い評価版。返り値の形はbatch_evaluate()と同じ（micro/macro）。
    """
    analyze_fn = analyze_claim if use_ollama else analyze_claim_ginza_only

    results = []
    for case in test_cases:
        if len(case) == 4:
            name, text, gold, category = case
        else:
            name, text, gold = case
            category = None
        try:
            _, predicted = analyze_fn(text)
        except Exception:
            predicted = []
        result = evaluate_triples_lenient(
            predicted, gold, lenient_relation_match=lenient_relation_match,
            semantic_threshold=semantic_threshold, use_semantic=use_semantic,
        )
        result["name"] = name
        result["category"] = category
        results.append(result)

    total_tp = sum(r["正解数"] for r in results)
    total_pred = sum(r["システム抽出数"] for r in results)
    total_gold = sum(r["正解データ数"] for r in results)
    micro_precision = total_tp / total_pred if total_pred else 0.0
    micro_recall = total_tp / total_gold if total_gold else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )
    macro_precision = sum(r["precision"] for r in results) / len(results) if results else 0.0
    macro_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
    macro_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0

    return {
        "results": results,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "semantic_threshold": semantic_threshold,
    }


def batch_evaluate(test_cases, lenient_relation_match=True, use_ollama=False):
    """
    複数の請求項をまとめて評価する。

    test_cases: [(名前, 請求項テキスト, 正解トリプルのリスト), ...]
                または、難易度カテゴリ別集計もしたい場合は
                [(名前, 請求項テキスト, 正解トリプルのリスト, 難易度カテゴリ), ...]

    use_ollama=False（デフォルト）はGiNZAルールベースのみで解析する
    （再現性があり、Ollama未起動でも動く「厳格F1」用）。
    use_ollama=True にすると analyze_claim()（Ollama補完込み）を使う。

    マイクロ平均（全件の正解数・抽出数・正解データ数を合算してから
    Precision/Recall/F1を計算。件数の多い請求項の影響が大きくなる）と、
    マクロ平均（各件のF1を単純平均。すべての請求項を対等に扱う）の
    両方を返す。難易度カテゴリを指定していれば、カテゴリ別の集計も返す。
    """
    from collections import defaultdict

    analyze_fn = analyze_claim if use_ollama else analyze_claim_ginza_only

    results = []
    for case in test_cases:
        if len(case) == 4:
            name, text, gold, category = case
        else:
            name, text, gold = case
            category = None
        _, predicted = analyze_fn(text)
        result = evaluate_triples(predicted, gold, lenient_relation_match=lenient_relation_match)
        result["name"] = name
        result["category"] = category
        results.append(result)

    total_tp = sum(r["正解数"] for r in results)
    total_pred = sum(r["システム抽出数"] for r in results)
    total_gold = sum(r["正解データ数"] for r in results)
    micro_precision = total_tp / total_pred if total_pred else 0.0
    micro_recall = total_tp / total_gold if total_gold else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    macro_precision = sum(r["precision"] for r in results) / len(results) if results else 0.0
    macro_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
    macro_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0

    category_summary = {}
    if any(r["category"] for r in results):
        by_cat = defaultdict(list)
        for r in results:
            by_cat[r["category"] or "(未分類)"].append(r)
        for cat, rs in by_cat.items():
            tp = sum(r["正解数"] for r in rs)
            pred = sum(r["システム抽出数"] for r in rs)
            gold_n = sum(r["正解データ数"] for r in rs)
            p = tp / pred if pred else 0.0
            rcl = tp / gold_n if gold_n else 0.0
            f = 2 * p * rcl / (p + rcl) if (p + rcl) > 0 else 0.0
            category_summary[cat] = {"件数": len(rs), "precision": p, "recall": rcl, "f1": f}

    return {
        "results": results,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "category_summary": category_summary,
    }


def print_batch_evaluation_report(batch_result):
    """batch_evaluate() の結果を、人が読みやすい形で表示する"""
    print("=== 全体（マイクロ平均） ===")
    m = batch_result["micro"]
    print(f"適合率: {m['precision']*100:5.1f}%  再現率: {m['recall']*100:5.1f}%  F1: {m['f1']*100:5.1f}%")
    print()
    print("=== 全体（マクロ平均） ===")
    M = batch_result["macro"]
    print(f"適合率: {M['precision']*100:5.1f}%  再現率: {M['recall']*100:5.1f}%  F1: {M['f1']*100:5.1f}%")

    if batch_result["category_summary"]:
        print()
        print("=== 難易度カテゴリ別 ===")
        for cat, s in batch_result["category_summary"].items():
            print(f"{cat}（{s['件数']}件）: 適合率{s['precision']*100:5.1f}%  再現率{s['recall']*100:5.1f}%  F1{s['f1']*100:5.1f}%")

    print()
    print("=== 請求項ごとの詳細 ===")
    for r in batch_result["results"]:
        cat_label = f"[{r['category']}] " if r.get("category") else ""
        print(
            f"{cat_label}{r['name']}: 適合率{r['precision']*100:5.1f}%  再現率{r['recall']*100:5.1f}%  "
            f"F1{r['f1']*100:5.1f}%  （抽出{r['システム抽出数']}件中{r['正解数']}件正解、正解データ{r['正解データ数']}件）"
        )


# ============================================================
# ㊶ 信頼度フラグ：この請求項の抽出結果は信頼できるか
# ============================================================
# これまでの検証で分かった「SAO抽出が苦手なパターン」を請求項の
# テキストから事前に検出し、抽出結果をそのまま信じてよいか、
# それとも人間が読み直すべきかの目安を示す。知財担当者が、
# 「怪しい部分だけ確認する」形で読む速度を上げるために使う。

def assess_claim_confidence(text):
    """
    請求項のテキストを解析し、既知の弱点パターンに当てはまるかどうかから、
    抽出結果の信頼度を「高」「中」「低」で判定する。

    戻り値: {"level": "高"|"中"|"低", "score": 数値, "reasons": [検出理由, ...]}
    """
    reasons = []
    score = 0

    length = len(text)
    if length > 400:
        reasons.append(f"文字数が非常に多い長文です（{length}文字）")
        score += 2
    elif length > 250:
        reasons.append(f"文字数がやや多いです（{length}文字）")
        score += 1

    # 深い所有格の連鎖（「Ａの…Ｂの…Ｃの」のように「の」が何度も連なる）
    deep_no = len(_re_dep.findall(r"の(?:前記)?[^、。]{1,15}の(?:前記)?[^、。]{1,15}の", text))
    if deep_no >= 1:
        reasons.append("深い所有格の連鎖（AのBのCの…）が含まれています")
        score += 2

    # 3つ以上の並列列挙
    enum_count = len(_re_dep.findall(r"及び|および|又は|または", text))
    if enum_count >= 2:
        reasons.append(f"並列列挙（及び／又は等）が複数箇所あります（{enum_count}箇所）")
        score += 1

    # 「〜に対して〜される」という特殊な受身構文
    if _re_dep.search(r"に対し(?:て)?[^。]{0,20}(?:さ|られ)れ", text):
        reasons.append("「〜に対して〜される」という特殊な受身構文が含まれています")
        score += 1

    # 条件節・空間配置の表現
    if "場合" in text or "平面視" in text:
        reasons.append("条件節や空間配置の表現（「〜場合」「平面視において」等）が含まれています")
        score += 1

    # 「含む」「有する」の入れ子が深い
    nest_count = len(_re_dep.findall(r"含み|含む|有し|有する|備え|備える", text))
    if nest_count >= 5:
        reasons.append(f"「含む」「有する」「備える」の入れ子が多いです（{nest_count}箇所）")
        score += 1

    # 修飾語を伴わないと意味が定まりにくい一般的な語
    generic_terms = ["一方", "他方", "ゲート", "ソース", "ドレイン"]
    found_generic = [t for t in generic_terms if t in text]
    if found_generic:
        reasons.append(f"単体では意味が定まりにくい語（{'、'.join(found_generic)}）が含まれています")
        score += 1

    if score >= 5:
        level = "低"
    elif score >= 2:
        level = "中"
    else:
        level = "高"

    return {"level": level, "score": score, "reasons": reasons}


# ============================================================
# ㊷ 記載チェック：「前記」の整合性確認、明確性要件のリスク検出
# ============================================================
# SAO抽出の精度とは別に、知財実務で実際にチェックされている観点
# （「前記」参照の整合性、明確性要件違反になりやすい表現）を、
# 請求項のテキストから自動で検出する。読む速度を上げるための、
# 実務直結の機能。

def check_zenki_consistency(claim_number, claim_texts, prefer_parent=None):
    """
    請求項Ｎで使われている「前記Ｘ」「該Ｘ」が、その従属先
    （さらにその従属先を含む）より前に一度も登場していない場合、
    記載不備（明確性要件違反）の疑いがあるとして検出する。

    claim_texts: {番号: 本文} の辞書（parse_claims_block()の戻り値）

    戻り値: [{"claim_number": 検出元の請求項番号, "term": "Ｘ"}, ...]
    """
    chain = _build_claim_chain(claim_number, claim_texts, prefer_parent=prefer_parent)
    warnings = []
    accumulated_text = ""
    seen_terms = set()

    for num, fragment_text in chain:
        if not fragment_text:
            continue
        cleaned = _clean_claim_text(fragment_text)
        doc = nlp(cleaned)
        for i, tok in enumerate(doc):
            if tok.text not in ("前記", "該"):
                continue
            words = []
            j = i + 1
            while j < len(doc) and doc[j].pos_ in ("NOUN", "PROPN"):
                if doc[j].text in ("前記", "該") or not doc[j].text.strip():
                    break
                words.append(doc[j].text)
                j += 1
            if not words:
                continue
            term = _normalize_component_text("".join(words))
            if not term or term in seen_terms:
                continue
            # 「前記」「該」より前に登場していればOK。判定対象は
            # 「これまでの請求項（従属先）の蓄積テキスト」＋「同じ請求項内で
            # この語より前の部分」。後者を含めないと、同一請求項内で先に
            # 定義した構成要素を後段で「前記」で受けているだけの、ごく普通の
            # 記載まで誤検出してしまう（例：独立請求項である請求項1は
            # 従属先を持たないため、accumulated_textが常に空になる）。
            text_before_here = accumulated_text + cleaned[:tok.idx]
            if term not in text_before_here:
                warnings.append({"claim_number": num, "term": term})
                seen_terms.add(term)
        accumulated_text += cleaned

    return warnings


# 明確性要件（特許法36条6項2号）違反を指摘されやすい表現パターン。
# 判例（知財高裁 平17(行ケ)10015号、平21(行ケ)10395号 等）を根拠とする。
CLARITY_RISK_PATTERNS = [
    ("所定の", "明細書等で具体的に特定されていないと不明確と指摘される可能性があります（知財高裁 平21(行ケ)10395号）"),
    ("一定の", "具体的な基準が明細書で示されていないと不明確と指摘される可能性があります"),
    ("適切な", "主観的な表現であり、判断基準が明確でないと指摘される可能性があります"),
    ("必要に応じて", "条件が明確に特定されていないと指摘される可能性があります"),
    ("好ましくは", "任意的な限定であることが明確でも、権利範囲の外延が曖昧になりやすい表現です"),
]

# 抽象的で、具体的な裏付けがないと不明確と指摘されやすい語（単独使用時に注意）
CLARITY_RISK_ABSTRACT_NOUNS = ["構造", "手段", "機構"]


def check_clarity_risks(text):
    """
    明確性要件違反（特許法36条6項2号）を指摘されやすい表現パターンを
    テキストから検出する。

    戻り値: [{"phrase": 検出された表現, "reason": 根拠・説明}, ...]
    """
    findings = []
    for phrase, reason in CLARITY_RISK_PATTERNS:
        if phrase in text:
            findings.append({"phrase": phrase, "reason": reason})

    for noun in CLARITY_RISK_ABSTRACT_NOUNS:
        if noun in text:
            findings.append({
                "phrase": noun,
                "reason": f"「{noun}」は、具体的な構成（ハードウェア・処理内容等）と結びついた記載でないと、明確性要件違反（知財高裁 平17(行ケ)10015号）を指摘される可能性があります",
            })

    return findings


# ============================================================
# ㉔ 構成要素の未接続チェック
# ============================================================
# analyze_claim() が抽出した構成要素のうち、SAO関係（has・直接関係・
# 位置関係を問わず）を一度も持たない、つまりグラフ上で孤立している
# ものを検出する。既存のSAO抽出結果をそのまま利用できる。

def check_disconnected_components(components, relations):
    """
    components, relations: analyze_claim() / analyze_dependent_claim() の
    戻り値（構成要素リスト・関係リスト）。

    戻り値: [{"term": 構成要素名}, ...]（他の構成要素と一切関係を
             持たなかったものの一覧。抽出漏れの兆候である可能性がある）
    """
    connected = set()
    for r in relations:
        connected.add(_normalize_component_text(r["source"]))
        connected.add(_normalize_component_text(r["target"]))

    warnings = []
    seen = set()
    for c in components:
        term = _normalize_component_text(c["text"])
        if not term or term in seen:
            continue
        seen.add(term)
        if term not in connected:
            warnings.append({"term": term})
    return warnings


# ============================================================
# ㉕ 用語の表記ゆれ・不一致チェック
# ============================================================
# 「第１端子」と「第1端子」のように、全角/半角の違いなど見た目だけが
# 違う表記ゆれ（check_notation_variants）と、「第１端子」で定義したのに
# 後段で「前記第１電極」のように番号は同じでも名詞が食い違っている
# ケース（check_ordinal_term_consistency）、「制御部」と「制御装置」の
# ように、幹は同じで末尾の役割語（装置／部／手段等）だけが違う用語の
# ペア（check_similar_terms）を検出する。

_ZENKAKU_DIGITS_TO_HANKAKU = str.maketrans("０１２３４５６７８９", "0123456789")


def _canonicalize_for_variant_check(term):
    """
    表記ゆれ判定用に、全角数字→半角、スペース除去などの正規化を行う
    （意味は変えず、見た目の表記だけを揃える）。
    """
    t = term.translate(_ZENKAKU_DIGITS_TO_HANKAKU)
    t = t.replace(" ", "").replace("　", "")
    return t


def check_notation_variants(components):
    """
    構成要素名のうち、正規化（全角/半角数字の統一等）すると同じに
    なるのに、異なる表記のまま複数種類登場しているものを検出する。

    戻り値: [{"canonical": 正規化後の形, "variants": [表記1, 表記2, ...]}, ...]
    """
    groups = {}
    for c in components:
        term = _normalize_component_text(c["text"])
        if not term:
            continue
        canon = _canonicalize_for_variant_check(term)
        groups.setdefault(canon, set()).add(term)

    warnings = []
    for canon, variants in groups.items():
        if len(variants) > 1:
            warnings.append({"canonical": canon, "variants": sorted(variants)})
    return warnings


_ORDINAL_TERM_PATTERN = re.compile(r"^第(?P<num>[0-9]+)の?(?P<rest>.+)$")


def check_ordinal_term_consistency(components):
    """
    「第１端子」のように序数＋名詞で定義された構成要素について、
    同じ序数番号なのに末尾の名詞（端子／電極 等）が請求項内で
    食い違っていないかを検出する（例：「第１端子」と定義したのに、
    後段で「前記第１電極」と記載されている）。

    戻り値: [{"num": 序数（文字列）, "terms": [用語1, 用語2, ...]}, ...]
    """
    groups = {}
    seen = set()
    for c in components:
        term = _normalize_component_text(c["text"])
        if not term or term in seen:
            continue
        seen.add(term)
        m = _ORDINAL_TERM_PATTERN.match(term.translate(_ZENKAKU_DIGITS_TO_HANKAKU))
        if not m:
            continue
        groups.setdefault(m.group("num"), set()).add(term)

    warnings = []
    for num, terms in groups.items():
        if len(terms) > 1:
            warnings.append({"num": num, "terms": sorted(terms)})
    return warnings



# 「制御部」「制御装置」のように、末尾に付く「役割語」（同じ機能を指すのに
# 名詞の言い換えとしてよく使われる語）の一覧。ここに挙がっている語で終わる
# 構成要素名は、それより前の部分（＝幹）と役割語に分けて扱う。
# 長い候補から先にマッチさせるため長さ降順に並べる。
_ROLE_SUFFIXES = sorted(
    ["装置", "デバイス", "部品", "ユニット", "モジュール", "手段", "機構",
     "回路", "素子", "ブロック", "セクション", "システム", "部", "器", "機"],
    key=len, reverse=True,
)


def _split_stem_and_role_suffix(term):
    """
    「制御装置」→ ("制御", "装置") のように、末尾の役割語（装置・部・
    手段等）を切り出す。該当する役割語で終わっていなければ (term, None)。
    """
    for suf in _ROLE_SUFFIXES:
        if term.endswith(suf) and len(term) > len(suf):
            return term[: -len(suf)], suf
    return term, None


def check_similar_terms(components):
    """
    構成要素名のうち、中心となる語（幹）は同じなのに、末尾の役割語
    （装置／部／手段／ユニット等）だけが違う組み合わせを検出する
    （例：「制御部」と「制御装置」、「半導体素子」と「半導体デバイス」）。
    check_notation_variants()で検出できる、正規化すれば一致する表記ゆれは
    対象外（そちらの方が確度が高い別種の指摘のため）。

    以前は埋め込みベクトルによる意味的類似度で判定していたが、「表示部」と
    「通信部」のように役割語が同じだけで中身は無関係な語同士まで高い類似度
    が出てしまい、実用に耐えなかったため、幹の完全一致で判定する方式に
    変更した。その分、幹の表記まで違う言い換え（例：「制御部」と
    「コントローラ」）は検出できない。

    戻り値: [{"term_a":.., "term_b":.., "stem":..}, ...]
    """
    terms = sorted({_normalize_component_text(c["text"]) for c in components if c["text"].strip()})
    terms = [t for t in terms if t]

    by_stem = {}
    for t in terms:
        stem, suf = _split_stem_and_role_suffix(t)
        if suf is None or not stem:
            continue
        by_stem.setdefault(stem, set()).add(t)

    pairs = []
    for stem, terms_with_suffix in by_stem.items():
        if len(terms_with_suffix) < 2:
            continue
        terms_sorted = sorted(terms_with_suffix)
        for i in range(len(terms_sorted)):
            for j in range(i + 1, len(terms_sorted)):
                pairs.append({"term_a": terms_sorted[i], "term_b": terms_sorted[j], "stem": stem})

    return pairs


# ============================================================
# ㉖ 数値・単位チェック
# ============================================================
# 請求項に含まれる「数値＋単位」の組を抽出して一覧化し、同じ単位が
# 複数の表記（"mm"と"ｍｍ"、"um"と"μm"等）で混在していないかを検出する。

_UNIT_ALIASES = {
    "mm": "mm", "ｍｍ": "mm",
    "cm": "cm", "ｃｍ": "cm",
    "nm": "nm", "ｎｍ": "nm",
    "um": "μm", "μm": "μm",
    "℃": "℃", "度c": "℃", "度C": "℃",
    "mpa": "MPa", "MPa": "MPa", "Mpa": "MPa",
    "kpa": "kPa", "kPa": "kPa",
    "gpa": "GPa", "GPa": "GPa",
    "pa": "Pa", "Pa": "Pa",
    "%": "%", "％": "%", "パーセント": "%",
    "kg": "kg", "ｋｇ": "kg",
    "mg": "mg", "ｍｇ": "mg",
    "g": "g", "ｇ": "g",
    "kv": "kV", "kV": "kV",
    "mv": "mV", "mV": "mV",
    "v": "V", "V": "V", "ｖ": "V",
    "ma": "mA", "mA": "mA",
    "a": "A", "A": "A", "ａ": "A",
    "khz": "kHz", "kHz": "kHz",
    "mhz": "MHz", "MHz": "MHz",
    "ghz": "GHz", "GHz": "GHz",
    "hz": "Hz", "Hz": "Hz",
    "Ω": "Ω", "ω": "Ω", "オーム": "Ω",
    "ms": "ms", "秒": "s", "s": "s",
}

_NUMERIC_SPEC_PATTERN = re.compile(
    r"(?P<value>[0-9０-９]+(?:[.．][0-9０-９]+)?)"
    r"\s*"
    r"(?P<unit>mm|ｍｍ|cm|ｃｍ|nm|ｎｍ|μm|um|℃|度[Cc]|"
    r"MPa|Mpa|mpa|kPa|kpa|GPa|gpa|Pa|pa|%|％|パーセント|"
    r"kg|ｋｇ|mg|ｍｇ|g|ｇ|kV|kv|mV|mv|V|ｖ|mA|ma|A|ａ|"
    r"kHz|khz|MHz|mhz|GHz|ghz|Hz|hz|Ω|ω|オーム|ms|秒|s)"
)


def extract_numeric_specs(text):
    """
    請求項テキストから「数値＋単位」の組を抽出する
    （例：「10mm」「10 mm」「０．１ｍｍ」等）。

    戻り値: [{"raw": 元の表記, "value": 数値(float), "unit": 元の単位表記,
              "unit_normalized": 正規化後の単位, "position": 出現位置(文字index)}, ...]
    """
    results = []
    for m in _NUMERIC_SPEC_PATTERN.finditer(text):
        raw_value = m.group("value").translate(_ZENKAKU_DIGITS_TO_HANKAKU).replace("．", ".")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        unit_raw = m.group("unit")
        unit_normalized = _UNIT_ALIASES.get(unit_raw, unit_raw)
        results.append({
            "raw": m.group(0),
            "value": value,
            "unit": unit_raw,
            "unit_normalized": unit_normalized,
            "position": m.start(),
        })
    return results


def check_unit_notation_consistency(text):
    """
    extract_numeric_specs()の結果から、同じ単位（正規化後）が複数の
    異なる表記で書かれていないか（例："mm"と"ｍｍ"、"um"と"μm"）を検出する。

    戻り値: [{"unit_normalized": .., "raw_variants": [表記1, 表記2, ...]}, ...]
    """
    specs = extract_numeric_specs(text)
    groups = {}
    for s in specs:
        groups.setdefault(s["unit_normalized"], set()).add(s["unit"])

    warnings = []
    for unit_normalized, raws in groups.items():
        if len(raws) > 1:
            warnings.append({"unit_normalized": unit_normalized, "raw_variants": sorted(raws)})
    return warnings


# ============================================================
# ㉗ 請求項の引用関係
# ============================================================
# parse_claims_block() で分割した請求項群から、各請求項がどの請求項を
# 引用（従属）しているかを取り出し、引用関係図として可視化する。
# 「請求項◯に記載の」等の参照表現の解析には、_build_claim_chain()等が
# 既に使っている _parse_claim_ref() をそのまま利用する。

def build_claim_dependency_graph(claim_texts):
    """
    claim_texts: {番号: 本文} の辞書（parse_claims_block()の戻り値）

    戻り値: {請求項番号: [引用している親請求項番号, ...]}
            （独立請求項は空リスト。「請求項１又は２に記載の」のように
            複数を引用している場合は複数の番号が入る）
    """
    dependency = {}
    for num, text in claim_texts.items():
        ref = _parse_claim_ref(text)
        dependency[num] = ref["numbers"] if ref else []
    return dependency


def plot_claim_dependency_graphviz(dependency_graph, theme="deepsea"):
    """
    build_claim_dependency_graph()の結果を、請求項の引用関係図として
    Graphvizで描画する（例：請求項3 → 請求項1, 請求項2）。

    戻り値は graphviz.Digraph オブジェクト。Streamlitでは
    st.graphviz_chart(戻り値) でそのまま描画できる。
    """
    import graphviz

    g = graphviz.Digraph(engine="dot")
    g.attr(rankdir="LR", splines="spline", nodesep="0.3", ranksep="0.7", bgcolor="transparent")

    if theme == "deepsea":
        dep_fill, dep_border, dep_font = "#0E3A52", "#5FD4E0", "#E8FBFF"
        indep_fill, indep_border, indep_font = "#04121C", "#8FE0F0", "#FFFFFF"
        edge_color = "#5FD4E0"
    else:
        dep_fill, dep_border, dep_font = "#EAF2FF", "#5B8DEF", "#233044"
        indep_fill, indep_border, indep_font = "#3B4252", "#B0B7C6", "#FFFFFF"
        edge_color = "#5B8DEF"

    g.attr("node", shape="box", style="rounded,filled", fontname="IPAexGothic",
           fontsize="13", margin="0.2,0.12", penwidth="1.8")
    g.attr("edge", color=edge_color, penwidth="1.6", arrowsize="0.8")

    for num, parents in dependency_graph.items():
        is_independent = not parents
        g.node(
            str(num),
            label=f"請求項{num}" + ("\n（独立項）" if is_independent else ""),
            fillcolor=indep_fill if is_independent else dep_fill,
            color=indep_border if is_independent else dep_border,
            fontcolor=indep_font if is_independent else dep_font,
        )

    for num, parents in dependency_graph.items():
        for p in parents:
            if p in dependency_graph:
                g.edge(str(num), str(p))

    return g


# ============================================================
# ㊸ クレームの従属関係チェック（多項引用の多項引用・引用の不備）
# ============================================================
# ここから下は、SAO抽出の精度とは独立した「請求項ドラフティングの
# 品質チェック」機能。①は特許法施行規則上、機械的・客観的に判定できる
# 記載形式のルール違反であり、テキストの意味解釈が絡まないため、
# 他のcheck_*系（明確性リスク等）よりも高い精度が期待できる。

def check_multi_multi_dependent_claim(dependency_graph):
    """
    「多項引用の多項引用」（多数項引用形式請求項が、他の多数項引用形式
    請求項を引用すること）を検出する。

    日本の実務では、多項引用形式請求項（例：「請求項１又は２に記載の」
    のように2つ以上の請求項番号を1つの請求項が引用する形式）が、
    他の多項引用形式請求項を引用することは認められていない
    （特許法施行規則様式29の2 備考8）。

    dependency_graph: build_claim_dependency_graph() の戻り値
                       （{請求項番号: [引用している親請求項番号, ...]}）

    戻り値: [{"claim_number": 検出元（多項引用している請求項）,
             "cited_multi_claims": [引用先のうち、それ自体も多項引用に
             なっている請求項番号, ...]}, ...]
    """
    warnings = []
    for num, parents in dependency_graph.items():
        if len(parents) < 2:
            continue
        multi_parents = [p for p in parents if len(dependency_graph.get(p, [])) >= 2]
        if multi_parents:
            warnings.append({
                "claim_number": num,
                "cited_multi_claims": sorted(multi_parents),
            })
    return sorted(warnings, key=lambda w: w["claim_number"])


def check_claim_reference_defects(dependency_graph):
    """
    請求項の引用関係における、構造上の不備を検出する。
      ・存在しない請求項番号の引用
      ・自己引用（自分自身を引用）
      ・後方参照（自分より後ろ／同じ番号の請求項を引用。日本の実務では
        従属請求項は、それより前に記載された請求項のみ引用できる）

    後方参照が1件も無ければ引用関係は必ず非巡回（DAG）になるため、
    循環参照は後方参照の特殊ケースとして自動的に検出される
    （番号が単調に減少する経路だけでは、出発点に戻る閉路を作れないため）。

    dependency_graph: build_claim_dependency_graph() の戻り値

    戻り値: [{"claim_number": 検出元, "type": 種別, "detail": 説明}, ...]
    """
    warnings = []
    valid_numbers = set(dependency_graph.keys())

    for num in sorted(dependency_graph.keys()):
        for p in dependency_graph[num]:
            if p not in valid_numbers:
                warnings.append({
                    "claim_number": num,
                    "type": "存在しない請求項の引用",
                    "detail": f"請求項{num}が、存在しない請求項{p}を引用しています",
                })
            elif p == num:
                warnings.append({
                    "claim_number": num,
                    "type": "自己引用",
                    "detail": f"請求項{num}が、請求項{num}自身を引用しています",
                })
            elif p > num:
                warnings.append({
                    "claim_number": num,
                    "type": "後方参照",
                    "detail": (
                        f"請求項{num}が、より後ろの請求項{p}を引用しています"
                        "（従属請求項は、それより前に記載された請求項のみ引用できます）"
                    ),
                })
    return warnings


# ============================================================
# ㊹ 文章的特性チェック（長文・入れ子構造・主語の不明瞭さ）
# ============================================================
# 産業日本語研究会 特許文書分科会等で指摘されている、可読性に関する
# 代表的な観点をルールベースで検出する。①②と異なり、これらは
# 「読みにくさの目安」であり文法違反の断定ではないため、
# 他のcheck_clarity_risks等と同じく助言的な位置づけで扱う。

def check_long_claim(text, length_threshold=400, clause_threshold=12):
    """
    請求項本文が長すぎないかを判定する。

    length_threshold: これを超える文字数を「長文」の目安とする
    clause_threshold: 読点（、／，）で区切られた節の数がこれを超える場合も
                       「長文」の目安とする（文字数はそれほど多くなくても、
                       単純な列挙が延々と続くケースを拾うため）

    戻り値: {"is_long": bool, "length": 文字数, "clause_count": 節数,
             "reasons": [理由, ...],
             "split_suggestions": [分割候補となる読点付近の文脈, ...]}
             （split_suggestionsは、どこで独立請求項や別の従属請求項に
             切り出せそうかの「目安」であり、実際の分割可否は発明の
             内容次第で人間の判断が必要）
    """
    length = len(text)
    clauses = [c for c in re.split(r"[、，]", text) if c.strip()]
    clause_count = len(clauses)

    reasons = []
    if length > length_threshold:
        reasons.append(f"文字数が{length}文字あり、目安の{length_threshold}文字を超えています")
    if clause_count > clause_threshold:
        reasons.append(f"読点で区切られた節が{clause_count}個あり、目安の{clause_threshold}個を超えています")

    is_long = bool(reasons)

    split_suggestions = []
    if is_long and clause_count >= 4:
        mid_points = {clause_count // 3, (clause_count * 2) // 3}
        cursor = 0
        for i, clause in enumerate(clauses):
            cursor += len(clause) + 1
            if i in mid_points:
                context_start = max(0, cursor - 15)
                context_end = min(len(text), cursor + 15)
                split_suggestions.append(text[context_start:context_end])

    return {
        "is_long": is_long,
        "length": length,
        "clause_count": clause_count,
        "reasons": reasons,
        "split_suggestions": split_suggestions,
    }


def check_nested_clause_complexity(text, depth_threshold=3):
    """
    連体修飾節（acl：「〜する◯◯」のように、動詞が後ろの名詞を修飾する節）
    が何重にも入れ子になっている箇所を検出する。

    入れ子が深いほど「どの動詞がどの名詞にかかっているか」が
    人間にも構文解析器にも読み取りづらくなるため、可読性の観点で
    分割・言い換えを検討する目安として使う（複文構造の複雑さの指標）。

    depth_threshold: これ以上の深さを「入れ子が深い」とみなす目安
                      （2〜3階層程度までは一般的な特許請求項でもよく
                      見られるため、それより深いものだけを拾う）

    戻り値: {"max_depth": 最大の入れ子の深さ,
             "deep_phrases": [{"text": 該当箇所の先頭付近のテキスト,
             "depth": 深さ}, ...]}
    """
    cleaned = _clean_claim_text(text)
    doc = nlp(cleaned)

    depth_cache = {}

    def _acl_depth(token):
        if token.i in depth_cache:
            return depth_cache[token.i]
        depth_cache[token.i] = 0  # 循環防止の仮値
        max_child_depth = 0
        for child in token.children:
            if child.dep_ == "acl":
                max_child_depth = max(max_child_depth, _acl_depth(child) + 1)
        depth_cache[token.i] = max_child_depth
        return max_child_depth

    candidates = []
    max_depth = 0
    for token in doc:
        if token.dep_ != "acl":
            continue
        d = _acl_depth(token) + 1
        max_depth = max(max_depth, d)
        if d >= depth_threshold:
            indices = [c.i for c in token.subtree]
            start, end = min(indices), max(indices)
            phrase = "".join(doc[i].text for i in range(start, min(end + 1, start + 60)))
            candidates.append({"text": phrase, "depth": d})

    # 同じ入れ子構造の内側にある浅いacl由来の重複表示を避けるため、
    # 検出された中で最も深いものだけを残す。
    deep_phrases = []
    if candidates:
        best = max(c["depth"] for c in candidates)
        seen_text = set()
        for c in candidates:
            if c["depth"] == best and c["text"] not in seen_text:
                deep_phrases.append(c)
                seen_text.add(c["text"])

    return {"max_depth": max_depth, "deep_phrases": deep_phrases}


def check_missing_subject_clauses(text):
    """
    請求項本文の中から、その節（動詞）自身の主語（nsubj）がGiNZAの
    係り受け上見つからない、「主語が読み取りにくい節」を検出する。

    「Ａは、Ｂを検出し、Ｃを出力する。」のように、同じ主語Ａを複数の
    節にわたって省略するのは日本語として自然であり（連用中止形・テ形で
    前の動詞と主語を共有する場合）、SAO抽出でも正しく解決できるため
    対象外とする。また「〜する◯◯」のように名詞を修飾する連体修飾節
    （acl）は、その名詞自体が意味上の主語の役割を果たすため対象外とする。

    それ以外で、直前の動詞から主語を引き継ぐ構造でもなく、かつ節自身にも
    主語（nsubj）が見つからない述語だけを、読み直しの候補として拾う。

    戻り値: [{"clause": 節の先頭付近のテキスト, "verb": 述語の見出し語}, ...]
            （厳密な文法エラー検出ではなく、確認を促すための助言的な出力。
            他のcheck_*系と同様、断定ではなく「候補」として扱うこと）
    """
    cleaned = _clean_claim_text(text)
    doc = nlp(cleaned)

    # GiNZAは、「により」「に対して」「ており」のような機能表現の構成語や、
    # 「供給ため」「並列接続」のような複合名詞・修飾語の一部を、文脈によって
    # pos_="VERB"のまま dep_="compound"／"obl"／"fixed"等（＝独立した節の
    # 述語ではない関係）として解析してしまうことがある。実際に節の述語と
    # なっているのは dep_ が "ROOT"／"advcl"／"conj"／"ccomp" のいずれかの
    # 場合だけなので、それ以外は主語の有無を問う対象から除外する
    # （このフィルタを入れる前は532件中231件で誤検出しており、
    # 実用に耐えなかったため、判明した誤検出の傾向から追加した制約）。
    _PREDICATE_DEPS = {"ROOT", "advcl", "conj", "ccomp"}

    def _has_topic_before(verb):
        # 同じ文（句点をまたがない）の中で、この動詞より前に「は」で示す
        # 主題が一度でも登場していれば、その主題を引き継いでいる可能性が
        # 高いとみなす（日本語では、一度「Ａは」と示した主題は、
        # 後続の複数の節にわたって省略されるのがごく普通の書き方であり、
        # そのたびに毎回「Ａは」を繰り返す方がむしろ不自然なため）。
        for i in range(verb.i - 1, -1, -1):
            t = doc[i]
            if t.text == "。":
                break
            if t.text == "は" and t.dep_ == "case":
                return True
        return False

    # 「Ａと、Ｂと、Ｃとを備える、Ｄ。」のように、「有する」「備える」
    # 「含む」等の列挙構文は、文中に「Ｘは」が一度も無くても、末尾の
    # 発明の名称（Ｄ）自体が暗黙の主語になる、というごく普通の書き方が
    # 非常に多い。これはSAO抽出側で別途（root_componentへのフォール
    # バックとして）正しく解決されている、正常なパターンであって
    # 「主語が無い」欠陥ではないため、この動詞群は対象外にする。
    _ENUMERATION_HAS_LEMMAS = HAS_LEMMAS | {"含む"}

    findings = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        if token.text in ("前記", "該"):
            # 「少なくとも前記〜」等の文脈で、GiNZAが「前記」自体を
            # pos_="VERB"として誤解析することがある（「前記」は連体詞的な
            # 接頭語であり、そもそも述語になり得ない）。
            continue
        if token.dep_ not in _PREDICATE_DEPS:
            continue
        if token.lemma_ in _ENUMERATION_HAS_LEMMAS:
            continue
        if token.dep_ in ("advcl", "conj") and token.head.pos_ == "VERB":
            continue
        if any(c.dep_ == "nsubj" for c in token.children):
            continue
        if _has_topic_before(token):
            continue
        indices = [c.i for c in token.subtree]
        start, end = min(indices), max(indices)
        clause_text = "".join(doc[i].text for i in range(start, min(end + 1, start + 60)))
        findings.append({"clause": clause_text, "verb": token.lemma_})
    return findings


# ============================================================
# app.py タブ⑥（精度検証）用に移植：健全性チェック・gold比較
# ============================================================

_PARTITIVE_GENERIC_WORDS = {"一部", "部分", "全部", "全体"}

_CLAIM_ENDING_NOUN_RE = re.compile(
    r"([一-龥ァ-ヶー０-９0-9]{2,20})。\s*$"
)


def _extract_claim_ending_noun(text):
    """「…電力変換装置。」のように、請求項の末尾に置かれる発明の名称を直接取り出す。"""
    m = _CLAIM_ENDING_NOUN_RE.search(text.strip())
    return m.group(1) if m else None


def evaluate_claim_health(text, components, relations):
    """
    1件の請求項について、analyze_claim() が返した components/relations の
    構造的な健全性をチェックし、チェック項目ごとの合否と総合合否を返す。
    """
    checks = {}

    checks["has_components"] = len(components) > 0
    checks["has_relations"] = len(relations) > 0

    checks["no_self_loop"] = not any(
        r["source"] == r["target"] for r in relations
    )

    bad_words = RELATION_WORDS | GENERIC_NOUNS | _PARTITIVE_GENERIC_WORDS
    checks["no_bare_generic_node"] = not any(
        c["text"] in bad_words for c in components
    )

    ending_noun = _extract_claim_ending_noun(text)
    if ending_noun:
        checks["ending_noun_present"] = any(
            ending_noun in c["text"] or c["text"] in ending_noun
            for c in components
        )
    else:
        checks["ending_noun_present"] = True

    if components:
        used = set()
        for r in relations:
            used.add(r["source"])
            used.add(r["target"])
        orphan_ratio = 1 - (sum(1 for c in components if c["text"] in used) / len(components))
        checks["orphan_ratio_ok"] = orphan_ratio <= 0.5
    else:
        checks["orphan_ratio_ok"] = False

    incoming = {}
    for r in relations:
        incoming.setdefault(r["target"], set()).add(r["relation"])
    max_fan_in_labels = max((len(v) for v in incoming.values()), default=0)
    checks["no_fan_in_anomaly"] = max_fan_in_labels <= max(4, len(components) // 2)

    passed = all(checks.values())
    return passed, checks


def evaluate_corpus_health(records, analyze_fn=None, progress_callback=None):
    """
    (id, text) のリスト（またはtextだけのリスト）を受け取り、各請求項を解析して
    evaluate_claim_health() の結果を集計する。
    """
    if analyze_fn is None:
        analyze_fn = analyze_claim

    results = []
    for idx, item in enumerate(records):
        if isinstance(item, (tuple, list)):
            claim_id, text = item[0], item[1]
        else:
            claim_id, text = idx, item

        row = {"id": claim_id, "text": text}
        try:
            components, relations = analyze_fn(text)
            passed, checks = evaluate_claim_health(text, components, relations)
            row.update({
                "passed": passed,
                "n_components": len(components),
                "n_relations": len(relations),
                "error": None,
                **{f"check_{k}": v for k, v in checks.items()},
            })
        except Exception as e:
            row.update({
                "passed": False,
                "n_components": 0,
                "n_relations": 0,
                "error": str(e),
            })
        results.append(row)
        if progress_callback:
            progress_callback(idx + 1, len(records))

    total = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    n_errors = sum(1 for r in results if r.get("error"))
    summary = {
        "total": total,
        "passed": n_passed,
        "pass_rate": (n_passed / total) if total else 0.0,
        "errors": n_errors,
        "avg_components": (sum(r["n_components"] for r in results) / total) if total else 0.0,
        "avg_relations": (sum(r["n_relations"] for r in results) / total) if total else 0.0,
    }
    return results, summary


_VERB_TAIL_SUFFIXES = sorted(
    [
        "されている", "されており", "されていた", "される", "された", "され",
        "している", "しており", "していた", "する", "した", "して", "し",
        "られている", "られており", "られる", "られた", "られ",
        "を有する", "を備える", "を含む",
    ],
    key=len,
    reverse=True,
)


def _relation_core(text):
    t = text.strip()
    for suf in _VERB_TAIL_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf):
            return t[: -len(suf)]
    return t


def _relation_matches(gold_relation, extracted_relation):
    g = _relation_core(gold_relation)
    e = _relation_core(extracted_relation)
    if not g or not e:
        return gold_relation == extracted_relation
    return g in e or e in g


def _node_matches(gold_node, extracted_node):
    g = _normalize_component_text(gold_node.strip())
    e = _normalize_component_text(extracted_node.strip())
    if not g or not e:
        return g == e
    return g == e or g in e or e in g


def _triple_matches(gold_triple, extracted_triple):
    gs, gr, gt = gold_triple
    es, er, et = extracted_triple
    if not _relation_matches(gr, er):
        return False
    forward = _node_matches(gs, es) and _node_matches(gt, et)
    backward = _node_matches(gs, et) and _node_matches(gt, es)
    return forward or backward


def compare_with_gold(gold_by_claim, texts_by_claim, analyze_fn=None, progress_callback=None):
    """
    正解SAOと、実際にanalyze_fnで解析した結果を突き合わせて、
    請求項ごと・全体の適合率・再現率・F値を計算する（app.pyタブ6用、20件比較機能）。
    """
    if analyze_fn is None:
        analyze_fn = analyze_claim

    per_claim = []
    claim_ids = list(gold_by_claim.keys())
    for idx, claim_id in enumerate(claim_ids):
        gold_triples = gold_by_claim[claim_id]
        text = texts_by_claim.get(claim_id, "")
        row = {"claim_id": claim_id, "n_gold": len(gold_triples)}
        if not text:
            row.update({"error": "対応する請求項本文が見つかりません", "n_extracted": 0,
                        "n_matched": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
            per_claim.append(row)
            continue
        try:
            components, relations = analyze_fn(text)
            extracted_triples = [(r["source"], r["relation"], r["target"]) for r in relations]

            matched_gold = set()
            matched_extracted = set()
            for gi, g in enumerate(gold_triples):
                for ei, e in enumerate(extracted_triples):
                    if ei in matched_extracted:
                        continue
                    if _triple_matches(g, e):
                        matched_gold.add(gi)
                        matched_extracted.add(ei)
                        break

            n_gold = len(gold_triples)
            n_extracted = len(extracted_triples)
            n_matched = len(matched_gold)
            precision = (n_matched / n_extracted) if n_extracted else 0.0
            recall = (n_matched / n_gold) if n_gold else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            row.update({
                "n_extracted": n_extracted,
                "n_matched": n_matched,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "error": None,
                "missed_gold": [g for gi, g in enumerate(gold_triples) if gi not in matched_gold],
                "extra_extracted": [e for ei, e in enumerate(extracted_triples) if ei not in matched_extracted],
            })
        except Exception as e:
            row.update({"error": str(e), "n_extracted": 0, "n_matched": 0,
                        "precision": 0.0, "recall": 0.0, "f1": 0.0})
        per_claim.append(row)
        if progress_callback:
            progress_callback(idx + 1, len(claim_ids))

    total_gold = sum(r["n_gold"] for r in per_claim)
    total_extracted = sum(r["n_extracted"] for r in per_claim)
    total_matched = sum(r["n_matched"] for r in per_claim)
    overall_precision = (total_matched / total_extracted) if total_extracted else 0.0
    overall_recall = (total_matched / total_gold) if total_gold else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) else 0.0
    )
    summary = {
        "n_claims": len(per_claim),
        "total_gold": total_gold,
        "total_extracted": total_extracted,
        "total_matched": total_matched,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "avg_f1_per_claim": (sum(r["f1"] for r in per_claim) / len(per_claim)) if per_claim else 0.0,
    }
    return per_claim, summary


# ============================================================
# 実務寄りの「意味的F1」（厳格F1を置き換えるものではなく、追加の参考指標）
# ============================================================
# 厳格F1はsource/targetの完全一致を要求するため、実務上「人間が読めば
# 同じ構成要素・関係だと分かる」ケース（表記ゆれ、粒度の違い等）まで
# 不正解に数えてしまい、数値が実態より低く出すぎるという指摘があった。
# そこで、埋め込みモデルでトリプル同士の意味的な近さを測り、閾値以上を
# 正解とみなす評価も追加する。既存の厳格F1（evaluate_triples/batch_evaluate）
# はそのまま残し、この意味的F1は常に厳格F1と併記して報告する
# （数値を良く見せるために評価方法をすり替えるのではなく、
#   「完全一致」と「意味的に近い」の両方を透明に示すのが目的）。

def evaluate_triples_semantic(predicted_relations, gold_triples, threshold=0.75):
    """
    埋め込みモデルによる意味的な近さで正誤を判定するF値。

    predicted_relations: [{"source":..,"relation":..,"target":..}, ...]
    gold_triples: [(source, relation, target), ...] または同型の辞書リスト
    threshold: これ以上のコサイン類似度を「意味的に一致」とみなす
               （目安：0.85=ほぼ同義、0.75=人間が見て近いと感じる程度、
                0.6以下=関連はあるが別物と見た方がよい）
    """
    def _as_dict(t):
        if isinstance(t, dict):
            return t
        return {"source": t[0], "relation": t[1], "target": t[2]}

    pred = [_as_dict(p) for p in predicted_relations]
    gold = [_as_dict(g) for g in gold_triples]

    if not pred or not gold:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "正解数": 0, "システム抽出数": len(pred), "正解データ数": len(gold),
            "matched_pairs": [], "unmatched_pred": pred, "unmatched_gold": gold,
        }

    model = _get_embed_model()
    pred_texts = [_triple_to_text((p["source"], p["relation"], p["target"])) for p in pred]
    gold_texts = [_triple_to_text((g["source"], g["relation"], g["target"])) for g in gold]

    emb_pred = model.encode(pred_texts, normalize_embeddings=True)
    emb_gold = model.encode(gold_texts, normalize_embeddings=True)
    sim = emb_pred @ emb_gold.T

    # 類似度が高いペアから貪欲に確定させる（1つのpred・goldは1回だけマッチ）
    pairs = []
    for pi in range(sim.shape[0]):
        for gi in range(sim.shape[1]):
            pairs.append((sim[pi, gi], pi, gi))
    pairs.sort(key=lambda x: -x[0])

    matched_pred_idx = set()
    matched_gold_idx = set()
    matched_pairs = []
    for s, pi, gi in pairs:
        if s < threshold:
            break
        if pi in matched_pred_idx or gi in matched_gold_idx:
            continue
        matched_pred_idx.add(pi)
        matched_gold_idx.add(gi)
        matched_pairs.append((pred[pi], gold[gi], float(s)))

    tp = len(matched_pred_idx)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "正解数": tp,
        "システム抽出数": len(pred),
        "正解データ数": len(gold),
        "matched_pairs": matched_pairs,
        "unmatched_pred": [p for i, p in enumerate(pred) if i not in matched_pred_idx],
        "unmatched_gold": [g for i, g in enumerate(gold) if i not in matched_gold_idx],
    }


def batch_evaluate_semantic(test_cases, threshold=0.75, use_ollama=False, progress_callback=None):
    """
    evaluate_triples_semantic() を複数件まとめて実行し、マイクロ/マクロ平均を返す。
    test_cases: [(名前, 請求項テキスト, 正解トリプルのリスト), ...]
    """
    analyze_fn = analyze_claim if use_ollama else analyze_claim_ginza_only

    results = []
    for i, case in enumerate(test_cases):
        name, text, gold = case[0], case[1], case[2]
        try:
            _, predicted = analyze_fn(text)
        except Exception:
            predicted = []
        result = evaluate_triples_semantic(predicted, gold, threshold=threshold)
        result["name"] = name
        results.append(result)
        if progress_callback is not None:
            progress_callback(i + 1, len(test_cases))

    total_tp = sum(r["正解数"] for r in results)
    total_pred = sum(r["システム抽出数"] for r in results)
    total_gold = sum(r["正解データ数"] for r in results)
    micro_precision = total_tp / total_pred if total_pred else 0.0
    micro_recall = total_tp / total_gold if total_gold else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )
    macro_precision = sum(r["precision"] for r in results) / len(results) if results else 0.0
    macro_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
    macro_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0

    return {
        "results": results,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "threshold": threshold,
    }


def analyze_claim_translate(text, model=None, host=None, debug=False, debug_out=None):
    """
    「タグ化 → ローカルLLM(Ollama)で英訳 → 英語で依存構造解析 → タグを
    元の日本語に戻す」新方式（translate_sao.py）でSAOを抽出する。

    まだ実験的な方式であり、オラクル分割＋手動翻訳での検証（macro F1
    96〜98%程度）しか行っていない。自動タグ付け＋実際のOllama翻訳での
    end-to-end精度は eval_translate_sao.py で別途確認すること。

    analyze_claim() / analyze_claim_ginza_only() と同じ
    (components, relations) の形で返すので、evaluate_triples や
    build_graphviz にそのまま渡せる。
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    try:
        import translate_sao as _ts
    except ImportError as e:
        raise RuntimeError(
            "新方式（タグ化→英訳）の実行に必要なファイルが見つからないか、"
            "必要なパッケージが不足しています。translate_sao.py と "
            "en_relation_rules.py を patent_pipeline.py と同じフォルダに置き、"
            "`pip install spacy ollama` と "
            "`python -m spacy download en_core_web_sm` を実行してください。"
            f"（詳細: {e}）"
        ) from e

    kwargs = {"pp": sys.modules[__name__], "host": host, "debug": debug, "debug_out": debug_out}
    if model:
        kwargs["model"] = model
    else:
        kwargs["model"] = _ts.DEFAULT_MODEL

    return _ts.analyze_claim_translate(text, **kwargs)


# ###########################################################################
# ここから下は、卒業研究の途中で別ファイルに分けていたモジュールを統合したもの。
# 各節の先頭に元のファイル名を示す。同じ名前の関数が複数のファイルにあったものは、
# 末尾に番号などを付けて区別している（例：実験12の選別モデル → Selector12）。
# 元のファイル名で呼べるように、最後に名前空間（translate_sao・sao_selector12 など）
# も用意してある。
# ###########################################################################
import pathlib as _pathlib
import types as _types


# ===========================================================================
# 【統合】en_relation_rules.py
# 名前の付け替え: nlp → nlp_en
# ===========================================================================
"""
en_relation_rules.py
=====================
英訳された特許クレーム（構成要素は COMPONENT_<番号> というタグに
置き換えられている前提）から、spaCy(en_core_web_sm)の依存構造解析結果を
使って (source, relation, target) のSAOトリプルを抽出する。

設計の考え方:
  日本語は「1文に長い連体修飾節が何重にも入れ子になる」「格助詞（と/に等）が
  多義的」「主語省略・長距離係り受け誤り」のために、GiNZA単体でのSAO抽出には
  大量の特例ロジックが必要になっていた（patent_pipeline.py参照）。
  英訳すると、SVO構造・前置詞・並列(conj)がほぼ一意に決まるため、
  少数の一般的な依存構造パターン（HAS＋並列、能動他動詞、受身＋前置詞、
  比較級、コピュラ＋of等）だけでSAOの大部分をカバーできる、という
  仮説を検証するために書いた。

  ただし「動詞ごとに正しい日本語ラベルを知っている」必要はあるので、
  各パターンには小さな動詞辞書がある。日本語側のRELATION_WORDS/
  HAS_LEMMAS等に相当するもので、ここが今後の主なメンテナンス対象になる。

  10クレーム程度のオラクル検証（正解ノード分割を使った検証。
  /tmp/proto_run.py 相当）でmacro F1 98%程度を確認済み
  （GiNZA単体は同じ8クレームでmacro F1 43.8%）。
  ただしこれは「ノード分割は完璧」「翻訳はClaudeが手作業」という前提での
  ベストケースなので、実際にOllama翻訳と自動タグ付けを組み合わせた
  end-to-endの精度は別途 eval_translate_sao.py で検証すること。
"""
import re

import spacy

# ------------------------------------------------------------------
# nlp（en_core_web_sm）の遅延読み込み
# ------------------------------------------------------------------
# このモジュールはtranslate_sao.pyからモジュールレベルで
# `import en_relation_rules as err` されるが、実際にnlp（英語モデル）が
# 必要になるのは、本モジュールのextract_relations_from_english
# （英訳経由のSAO抽出。現在は analyze_claim_llm_direct 方式に置き換えられた
# 旧方式）が呼ばれたときだけである。以前はここで即座に
# `spacy.load("en_core_web_sm")` していたため、en_core_web_smが
# インストールされていない環境（Streamlit Community Cloud等、公開デモ用
# のクラウド環境にはインストールしていない）では、
# `import translate_sao` の時点で即座にOSError（モデル未検出）となり、
# analyze_claim_llm_direct（英語モデルを一切使わない方式）しか使わない
# アプリまで起動できなくなってしまっていた。
# nlpを遅延読み込みにすることで、実際にextract_relations_from_englishを
# 呼ぶまではen_core_web_smを読み込まないようにし、この問題を回避する。
# 呼び出し側（`err.nlp(text)`）から見た挙動は変わらない。
_nlp_instance = None


def _load_nlp():
    global _nlp_instance
    if _nlp_instance is None:
        _nlp_instance = spacy.load("en_core_web_sm")
    return _nlp_instance


class _LazyNLP:
    """spacy.load("en_core_web_sm")の遅延ラッパー。
    nlp(text)としての呼び出しも、nlp.pipe(...)等の属性アクセスも、
    実際に使われた時点で初めてモデルを読み込む。"""

    def __call__(self, *args, **kwargs):
        return _load_nlp()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_load_nlp(), name)


nlp_en = _LazyNLP()

TAG_RE = re.compile(r"^C\d+$")


def is_tag(token):
    return bool(TAG_RE.match(token.text))


def conj_chain(token):
    # 「A, B, C and D」のようなコンマ区切りの並列は、spaCyの小型モデルが
    # conjではなくapposとして解析することがある（Oxfordコンマなし特有の癖）
    # ため、conj・appos どちらの依存関係でも並列項として辿る。
    out = [token]
    for c in token.children:
        if c.dep_ in ("conj", "appos"):
            out.extend(conj_chain(c))
    return out


def _verb_key(token):
    # 見慣れないタグ（COMPONENT_1等）に挟まれると、spaCyの小型モデルが
    # 直後の動詞の原形（lemma）を誤判定することがあるため、lemma優先で、
    # ダメなら表層形（小文字化・末尾のs除去）にフォールバックする。
    if token.lemma_ and token.lemma_ != token.text:
        return token.lemma_
    return token.text.lower().rstrip("s")


# 「有する」「備える」に相当する、能動＋並列(conj)で目的語を複数取れる動詞
HAS_VERBS = {
    "have": "有する", "comprise": "備える", "include": "含む",
    "contain": "含む", "use": "用いる", "carry": "搭載する",
}
# 「AがBをVする」の単純な能動他動詞（対象を1つずつ取る）
ACTIVE_VERB_LABELS = {
    "accommodate": "収容する", "face": "対向する", "control": "制御する",
    "join": "接合する", "support": "支える", "inject": "射出する",
    "connect": "接続される", "cover": "覆う", "hold": "保持する",
    "press": "押圧する", "detect": "検出する", "generate": "生成する",
}
ACTIVE_PREP_VERBS = {"communicate": {"with": "連通される"}}
CONSIST_OF_VERBS = {"consist": "からなる"}
ACOMP_LABELS = {"opposite": "反対側の面である", "higher": "より高い", "lower": "より低い", "parallel": "平行である"}
CAPABLE_OF_GERUND_LABELS = {"inject": "射出可能である"}
CONFIGURE_XCOMP_LABELS = {"separate": "分離するように構成される"}

# 「Xに接続される/配置される/形成される」等の受身動詞。前置詞ごとに
# ラベルを変えたい場合は"_default"以外のキーを追加する。
PASSIVE_VERB_LABELS = {
    "connect": {"_default": "接続される"},
    "bond": {"_default": "接合される", "via": "を介して接合される"},
    "form": {"_default": "形成される"},
    "separate": {"_default": "離れている"},
    "fix": {"_default": "固定される"},
    "position": {"_default": "位置決めされる"},
    "dispose": {"_default": "配置される"},
    "arrange": {"_default": "配設される"},
    "provide": {"_default": "設けられる"},
    "insert": {"_default": "挿入される"},
    "mount": {"_default": "搭載される"},
    "attach": {"_default": "取り付けられる"},
    "cover": {"_default": "覆われる"},
    "hold": {"_default": "保持される"},
    "surround": {"_default": "囲まれる"},
    "embed": {"_default": "埋め込まれる"},
    "insulate": {"_default": "絶縁される"},
    "expose": {"_default": "露出される"},
    # 最初は"C2"止まりで「of C1」まで辿れず悪化したが（旧コメント参照）、
    # _scan_passive_targetsに「Xの主面」複合形の生成（ネストof連鎖解消）を
    # 実装した後に再検証し、188284でf1が悪化しないことを確認できたため採用。
    "configure": {"_default": "配置される"},
}
PASSIVE_ADVMOD_OVERRIDES = {
    ("separate", "electrically"): "電気的に離れて配置される",
}
# 日本語の原文では能動態（「ケースが半導体チップを収容し、」）で書かれている
# 動詞が、Ollama翻訳で受身（"C3 is accommodated in C6"）に変換されてしまう
# ことがある（実際に532件のうちの検証claimで確認済み）。この種の動詞は、
# 正解データでは一貫して「容器・全体側」を主語にした能動形（例:
# 「ケース 収容する 半導体チップ」）で記録されているため、通常のPASSIVE
# ruleとは逆に、source/targetを入れ替えて能動ラベルを使う。
REVERSED_PASSIVE_VERB_LABELS = {
    "accommodate": {"_default": "収容する"},
}
SURFACE_WORDS = {"surface": "表面に", "face": "面に"}
# 「in <noun> with」型の熟語的表現 → 日本語ラベル
PREP_NOUN_PATTERNS = {
    ("in", "communication", "with"): "連通される",
    ("in", "contact", "with"): "接触する",
}


def _scan_passive_targets(verb_node, vkey, override, labels_dict=None):
    """
    受身動詞（またはそれに相当する語）の子から、前置詞＋目的語のペアを
    すべて拾い、(対象テキスト, 関係ラベル) のリストを返す。

    「to X」がspaCyの小型モデルにより前置詞句でなくxcomp（不定詞句）と
    誤解析され、Xがそのまま裸のタグとして残ることがある
    （例: "C3 is bonded to C6." → bonded-xcomp->C6）。これも
    prep+pobjと同じ意味なので、あわせて対象にする。

    対象タグ自身が「C2 of C1」のようにネストした「of」属格を持つ場合
    （例:「C3 configured on C2 of C1」＝「C3がC1のC2に配置される」）、
    対象テキストを単なる「C2」ではなく「C1のC2」という複合形にする。
    こうしておくと、C2が実は「主面」等の面・部位を表す語だった場合、
    後段のpatent_pipeline._merge_surface_location_nodes（GiNZA単体版と
    共通のロジック）が、同じクレーム内に別途出現する「C1」ノードを
    手がかりに自動でC1へ統合してくれる（本人の指示「常に統合する」と
    同じ扱いになる）。C2が実際には別の独立した構成要素だった場合も、
    「C1のC2」という複合ノードのまま残るだけで、情報を失うわけではない。
    """
    labels = (labels_dict or PASSIVE_VERB_LABELS).get(vkey, {"_default": vkey})
    out = []
    for child in verb_node.children:
        pobj = None
        prep_text = None
        if child.dep_ == "prep":
            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
            prep_text = child.text
            if pobj is not None and pobj.lemma_ == "respect" and prep_text == "with":
                # "positioned with respect to X"のような多語前置詞。
                # spaCyの小型モデルは"with"のpobjを（意味の無い）"respect"に
                # してしまい、本当の対象Xは"respect"の子の"to"のpobjとして
                # さらに一段深くにぶら下がる（実際に検証claim（188284）の
                # 「C5 is positioned with respect to C3 by the C9」で確認）。
                # 「by C9」（手段）は"positioned"の直接の子ではなく"respect"の
                # 子になってしまうが、ここでは"to"だけを辿るので混入しない。
                to_prep = next((c for c in pobj.children if c.dep_ == "prep" and c.text == "to"), None)
                pobj = next((c for c in to_prep.children if c.dep_ == "pobj"), None) if to_prep else None
                prep_text = "with_respect_to"
        elif child.dep_ == "xcomp" and is_tag(child):
            # 誤解析: "to"がaux扱いになり、本来のpobjがxcompとして直接ぶら下がる
            pobj = child
            prep_text = "to"
        if pobj is None or pobj.pos_ == "VERB":
            continue
        target_label = override or labels.get(prep_text, labels["_default"])
        real_target = pobj
        if pobj.lemma_ in SURFACE_WORDS:
            of_prep = next((c for c in pobj.children if c.dep_ == "prep" and c.text == "of"), None)
            real_pobj = next((c for c in of_prep.children if c.dep_ == "pobj"), None) if of_prep else None
            if real_pobj is not None and is_tag(real_pobj):
                real_target = real_pobj
                target_label = SURFACE_WORDS[pobj.lemma_] + labels["_default"]
        for tgt in conj_chain(real_target):
            if not is_tag(tgt):
                continue
            tgt_text = tgt.text
            # ネストof連鎖の複合化は、"on"/"at"（場所・面を指す前置詞）の
            # ときだけ行う。"connected to C1 of C3"のような"to"は、場所
            # ではなく別の独立した構成要素への関係を表すことが多く
            # （実際に検証claim（187080）で、これも複合化すると
            # 「半導体チップの第１電極」のような不要な複合ノードが増え、
            # 元は正解していた単純な「第１電極」との一致が崩れて悪化した）、
            # ここでは対象外にする。
            if prep_text in ("on", "at"):
                of_prep2 = next((c for c in tgt.children if c.dep_ == "prep" and c.text == "of"), None)
                of_owner = next((c for c in of_prep2.children if c.dep_ == "pobj"), None) if of_prep2 else None
                if of_owner is not None and is_tag(of_owner) and of_owner.text != tgt.text:
                    tgt_text = f"{of_owner.text}の{tgt.text}"
            out.append((tgt_text, target_label))
    return out


def extract_relations(doc):
    rels = []
    for verb in doc:
        # spaCyの小型モデルは、直前に見慣れない大文字タグ（COMPONENT_1等）が
        # あると、直後の動詞をNOUN/PROPNに誤タグ付けすることがある
        # （品詞タグより依存関係ラベルの方が安定している）。そのため
        # 品詞（pos_）では絞らず、「文の述語になり得る位置」
        # （ROOT・従属節の述語等）かどうかで判定し、実際に動詞かどうかは
        # 後続の辞書照合（HAS_VERBS等に載っているか）に委ねる。
        # pcomp: "with X embedded in Y" のような with句の中の分詞構文
        # （withの補語として動詞が来るケース）も拾う。
        if verb.dep_ not in ("ROOT", "advcl", "xcomp", "ccomp", "relcl", "acl", "conj", "pcomp"):
            continue
        children = list(verb.children)
        has_auxpass = any(c.dep_ == "auxpass" for c in children)
        vkey = _verb_key(verb)

        # --- HAS rule（能動: have/comprise/include/contain/use + conjチェーン） ---
        if vkey in HAS_VERBS and not has_auxpass:
            subj = next((c for c in children if c.dep_ == "nsubj" and is_tag(c)), None)
            dobj = next((c for c in children if c.dep_ == "dobj"), None)
            if dobj is None:
                # "C4 includes C11 provided on C10..." のように、目的語が
                # dobjではなくccomp（従属節）の主語として誤解析される
                # ケースも拾う（"provided"側の関係は別途ccompとして
                # 独立に処理されるので、ここではC4-含む-C11のリンクだけ補う）。
                ccomp = next((c for c in children if c.dep_ == "ccomp"), None)
                if ccomp is not None:
                    dobj = next((c for c in ccomp.children if c.dep_ == "nsubj" and is_tag(c)), None)
            if subj is not None and dobj is not None:
                label = HAS_VERBS[vkey]
                if any(c.dep_ == "advmod" and c.lemma_ in ("mainly", "primarily") for c in children):
                    label = "主成分とする"
                seen_objs = set()
                for obj in conj_chain(dobj):
                    if is_tag(obj) and obj.text != subj.text:
                        rels.append({"source": subj.text, "relation": label, "target": obj.text})
                        seen_objs.add(obj.text)
                # "C9 comprises C7 electrically connected to C2... and C8
                # electrically insulated from C7" のように、2番目以降の
                # 列挙項目が名詞の並列(dobjのconj)ではなく、修飾する動詞同士の
                # 並列(HAS動詞自体のconj/advcl)として解析されることがある。
                # その場合、並列した動詞の主語を列挙項目として拾う。
                for sib in verb.children:
                    if sib.dep_ not in ("conj", "advcl"):
                        continue
                    sib_subj = next(
                        (c for c in sib.children if c.dep_ in ("nsubj", "nsubjpass") and is_tag(c)),
                        None,
                    )
                    if sib_subj is not None and sib_subj.text not in seen_objs and sib_subj.text != subj.text:
                        rels.append({"source": subj.text, "relation": label, "target": sib_subj.text})
                        seen_objs.add(sib_subj.text)
            continue

        # --- CONSIST-OF rule（consist of） ---
        if vkey in CONSIST_OF_VERBS:
            subj = next((c for c in children if c.dep_ == "nsubj" and is_tag(c)), None)
            prep = next((c for c in children if c.dep_ == "prep" and c.text == "of"), None)
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None) if prep else None
            if subj is not None and pobj is not None:
                for obj in conj_chain(pobj):
                    if is_tag(obj):
                        rels.append({"source": subj.text, "relation": CONSIST_OF_VERBS[vkey], "target": obj.text})
            continue

        # --- ACTIVE-PREP rule（communicate with等、dobjを取らない能動動詞） ---
        if vkey in ACTIVE_PREP_VERBS and not has_auxpass:
            subj = next((c for c in children if c.dep_ == "nsubj" and is_tag(c)), None)
            for prep in children:
                if prep.dep_ != "prep":
                    continue
                label = ACTIVE_PREP_VERBS[vkey].get(prep.text)
                if label is None:
                    continue
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if subj is not None and pobj is not None and is_tag(pobj):
                    rels.append({"source": subj.text, "relation": label, "target": pobj.text})
            continue

        # --- ACTIVE-TRANSITIVE rule（能動他動詞: nsubj + dobj） ---
        # "connect"のように能動(ACTIVE_VERB_LABELS)・受身(PASSIVE_VERB_LABELS)
        # 両方の辞書に載っている動詞がある。nsubj+dobjが揃った「能動」構文の
        # ときだけこのルールで処理してcontinueする。揃わない場合（例:
        # 主語のない分詞句 "C7 electrically connected to C2"）は、continueせずに
        # 下のPASSIVE ruleに処理を委ねる。
        if vkey in ACTIVE_VERB_LABELS and not has_auxpass:
            subj = next((c for c in children if c.dep_ == "nsubj" and is_tag(c)), None)
            dobj = next((c for c in children if c.dep_ == "dobj"), None)
            if subj is not None and dobj is not None:
                label = ACTIVE_VERB_LABELS[vkey]
                series_prep = next(
                    (c for c in children if c.dep_ == "prep" and any(
                        g.dep_ == "pobj" and g.lemma_ == "series" for g in c.children)),
                    None,
                )
                if series_prep is not None:
                    label = "直列に接続される"
                for obj in conj_chain(dobj):
                    if is_tag(obj) and obj.text != subj.text:
                        rels.append({"source": subj.text, "relation": label, "target": obj.text})
                continue

        # --- CONFIGURE + xcomp rule（is configured to separate等） ---
        if vkey == "configure" and has_auxpass:
            nsubjpass = next((c for c in children if c.dep_ == "nsubjpass" and is_tag(c)), None)
            xcomp = next((c for c in children if c.dep_ in ("xcomp", "advcl") and c.pos_ == "VERB"), None)
            if nsubjpass is not None and xcomp is not None and xcomp.lemma_ in CONFIGURE_XCOMP_LABELS:
                inner_dobj = next((c for c in xcomp.children if c.dep_ == "dobj"), None)
                if inner_dobj is not None and is_tag(inner_dobj):
                    rels.append({
                        "source": nsubjpass.text,
                        "relation": CONFIGURE_XCOMP_LABELS[xcomp.lemma_],
                        "target": inner_dobj.text,
                    })
            continue

        # --- PASSIVE rule（受身: nsubjpass + 前置詞連鎖。surface-of補正込み） ---
        nsubjpass = next((c for c in children if c.dep_ == "nsubjpass" and is_tag(c)), None)
        if nsubjpass is None and verb.dep_ == "acl" and is_tag(verb.head):
            # "C1 comprises at least one C3 configured on C2 of C1." のように、
            # be動詞も関係代名詞も無い縮約分詞構文（名詞を直接後置修飾する形）
            # では、この分詞(acl)の主語は常にそれが修飾している名詞そのもの
            # （UD文法上のacl.head）である。この関係は構造的に一意に決まるので、
            # verb.headがタグ自身であれば無条件に暗黙の主語として使ってよい。
            # （最初はこれだけだと"C2"止まりで悪化したが、_scan_passive_targets
            # 側にネストof連鎖の解消を実装した後は悪化しないことを確認済み。）
            nsubjpass = verb.head
        if nsubjpass is None and verb.tag_ in ("VBN", "VBD") and not any(c.dep_ == "dobj" for c in children):
            # "with C5 embedded in C6" のような、be動詞を伴わない分詞構文
            # （本来nsubjpassになるはずが、is/wasが省略されているため
            #  spaCyがnsubjとして解析してしまうケース）も受身として拾う。
            # tag_はVBN/VBDどちらに解析されるかが不安定なので両方許容する。
            nsubjpass = next((c for c in children if c.dep_ == "nsubj" and is_tag(c)), None)
            if nsubjpass is None and verb.dep_ in ("advcl", "acl"):
                # 主語まで省略された分詞句（"C9 comprises C7 electrically
                # connected to C2..."のように、"C7 [which is] connected..."の
                # 主語C7が丸ごと落ちるケース）。この場合、実際の主語は
                # 親動詞（comprises等）の目的語であることが多いので、それを
                # 暗黙の主語とみなす。
                nsubjpass = next(
                    (c for c in verb.head.children if c.dep_ == "dobj" and is_tag(c)), None
                )
        # --- REVERSED-PASSIVE rule（原文は能動だが翻訳で受身化された動詞。
        # 「容器・全体側」を主語にした能動ラベルでsource/targetを入れ替える） ---
        if nsubjpass is not None and vkey in REVERSED_PASSIVE_VERB_LABELS:
            for tgt, target_label in _scan_passive_targets(verb, vkey, None, labels_dict=REVERSED_PASSIVE_VERB_LABELS):
                if tgt != nsubjpass.text:
                    rels.append({"source": tgt, "relation": target_label, "target": nsubjpass.text})
            continue

        if nsubjpass is not None and vkey in PASSIVE_VERB_LABELS:
            advmod = next((c for c in children if c.dep_ == "advmod"), None)
            override = PASSIVE_ADVMOD_OVERRIDES.get((vkey, advmod.lemma_)) if advmod else None
            for tgt, target_label in _scan_passive_targets(verb, vkey, override):
                if tgt != nsubjpass.text:
                    rels.append({"source": nsubjpass.text, "relation": target_label, "target": tgt})
            continue

        # --- COPULA rule（be + acomp [+ prep] : opposite to / higher than 等） ---
        if vkey == "be":
            nsubj = next((c for c in children if c.dep_ == "nsubj" and is_tag(c)), None)
            acomp = next((c for c in children if c.dep_ == "acomp"), None)
            # spaCyの小型モデルが受身動詞をROOTでなくacomp（形容詞的補語）として
            # 誤解析することがある（"C4 is disposed on C8"のdisposed等）。
            # その場合もPASSIVE ruleと同じ要領で処理する。
            if nsubj is not None and acomp is not None and acomp.pos_ == "VERB":
                acomp_key = _verb_key(acomp)
                if acomp_key in PASSIVE_VERB_LABELS:
                    found = False
                    for tgt, target_label in _scan_passive_targets(acomp, acomp_key, None):
                        if tgt != nsubj.text:
                            rels.append({"source": nsubj.text, "relation": target_label, "target": tgt})
                            found = True
                    if found:
                        continue
            if nsubj is not None and acomp is not None:
                if acomp.text in ACOMP_LABELS:
                    prep = next((c for c in acomp.children if c.dep_ == "prep"), None)
                    pobj = next((c for c in prep.children if c.dep_ == "pobj"), None) if prep else None
                    if pobj is not None and is_tag(pobj):
                        rels.append({"source": nsubj.text, "relation": ACOMP_LABELS[acomp.text], "target": pobj.text})
                        continue
                if acomp.text == "capable":
                    prep = next((c for c in acomp.children if c.dep_ == "prep" and c.text == "of"), None)
                    pcomp = next((c for c in prep.children if c.dep_ == "pcomp"), None) if prep else None
                    if pcomp is not None and pcomp.lemma_ in CAPABLE_OF_GERUND_LABELS:
                        dobj = next((c for c in pcomp.children if c.dep_ == "dobj"), None)
                        if dobj is not None:
                            label = CAPABLE_OF_GERUND_LABELS[pcomp.lemma_]
                            for obj in conj_chain(dobj):
                                if is_tag(obj):
                                    rels.append({"source": nsubj.text, "relation": label, "target": obj.text})
                        continue
            # COPULA-GENITIVE rule（X is the <attr> of Y → 「の」関係、向きはY→X）
            attr = next((c for c in children if c.dep_ == "attr"), None)
            if nsubj is not None and attr is not None:
                prep = next((c for c in attr.children if c.dep_ == "prep" and c.text == "of"), None)
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None) if prep else None
                if pobj is not None and is_tag(pobj):
                    rels.append({"source": pobj.text, "relation": "の", "target": nsubj.text})
                    continue
            # COPULA-PREP-NOUN-PREP rule（X is in <noun> with Y 等の慣用表現）
            if nsubj is not None:
                prep1 = next((c for c in children if c.dep_ == "prep" and c.text in ("in", "into")), None)
                noun = next((c for c in prep1.children if c.dep_ == "pobj"), None) if prep1 else None
                if noun is not None and not is_tag(noun):
                    prep2 = next((c for c in noun.children if c.dep_ == "prep"), None)
                    pobj2 = next((c for c in prep2.children if c.dep_ == "pobj"), None) if prep2 else None
                    key = (prep1.text, noun.lemma_, prep2.text if prep2 else None)
                    label = PREP_NOUN_PATTERNS.get(key)
                    if label is not None and pobj2 is not None and is_tag(pobj2):
                        rels.append({"source": nsubj.text, "relation": label, "target": pobj2.text})
            continue
    return rels


def dedup(rels):
    seen = set()
    out = []
    for r in rels:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def extract_relations_from_text(english_text):
    """英訳済みテキスト（COMPONENT_N タグ入り）からSAOトリプルを抽出する。"""
    doc = nlp_en(english_text)
    return dedup(extract_relations(doc))


# ===========================================================================
# 【統合】translate_sao.py
# 名前の付け替え: analyze_claim_translate → analyze_claim_translate_llm
# ===========================================================================
"""
translate_sao.py
=================
「日本語クレーム → 構成要素をタグ化 → ローカルLLM(Ollama)で英訳
→ 英語の依存構造解析でSAO抽出 → タグを元の日本語に戻す」方式の
実装。

10クレームでのオラクル検証（正解ノード分割を使った検証）では
macro F1 96〜98%程度を確認済み（GiNZA単体では同じクレームで
macro F1 43.8%）。ただしそれは「ノード分割は完璧」という前提での
ベストケースなので、この実装（自動タグ付け＋実際のOllama翻訳）での
end-to-endの精度は必ず eval_translate_sao.py で確認すること。

使い方（自分のPC上、Ollamaが起動している状態で）:
    pip install spacy ollama
    python -m spacy download en_core_web_sm

    import sys
    sys.path.insert(0, "/path/to/real_app")  # patent_pipeline.pyのあるフォルダ
    ts = sys.modules[__name__]  # 統合後はこのファイル自身
    components, relations = ts.analyze_claim_translate(claim_text)

タグ形式について:
    構成要素は日本語原文中では "《C1》" のような記号付きタグに置き換える
    （LLM翻訳時に「これは翻訳しない特殊トークンだ」と認識されやすいように、
    通常の英数字だけのタグより機械的に目立つ記号《》を使っている）。
    英訳後は 《C1》→C1 のように記号を外してからspaCyで解析する
    （C1のような短い記号のほうが、英語の小型モデルの依存構造解析が
    安定することを確認済み。COMPONENT_1のような長いタグだと、
    直後の動詞の解析を誤ることがあった）。
"""
import hashlib
import os
import re
import sys

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

try:
    import deepl
except ImportError:  # pragma: no cover
    deepl = None

pass  # （統合済み）import en_relation_rules as err

DEFAULT_MODEL = "qwen3.5:9b"  # VRAM 16GB で余裕をもって動く（以前の実験は qwen2.5:7b）

_TAG_OPEN = "《"
_TAG_CLOSE = "》"
# 《C1》が本来の形だが、翻訳モデルが《》の代わりに〈〉や<>、{}を使って
# しまうことがあるため、フォールバックとして主要な括弧パターンも許容する。
_BRACKET_TAG_RE = re.compile(r"[《〈<{\[]\s*(C\d+)\s*[》〉>}\]]")


def _load_pipeline(pipeline_dir=None):
    if pipeline_dir:
        sys.path.insert(0, pipeline_dir)
    pp = sys.modules[__name__]  # 統合後はこのファイル自身
    return pp


_NON_COMPONENT_WORDS = {
    "複数", "多数", "一対", "互い", "それぞれ", "各々", "夫々", "いくつか", "幾つか", "全て", "すべて", "双方",
    "少なくとも一つ", "少なくとも１つ", "少なくとも1つ", "一つ", "１つ", "1つ",
}


_GA_NEXT_RE = re.compile(r"(、|，|及び|および|又は|または|並びに|ならびに)")


def tag_components(text, pp):
    """
    日本語クレームの構成要素をGiNZAで検出し、《C1》のようなタグに
    置き換えたテキストと、タグ→元のテキストの対応表を返す。

    同じ文字列の構成要素（複数回出てくる「前記半導体チップ」等）には
    同じタグを割り当てる。重複・入れ子の区間は、テキスト上で先に
    始まる方を優先し、後から重なる区間は捨てる（貪欲区間スケジューリング）。

    GiNZA単体版（analyze_claim_ginza_only）と同じく、まず
    _clean_claim_text で表記ゆれを正規化してから解析する
    （正規化前のテキストのままでは構成要素検出の精度が落ちるため）。
    戻り値のtagged_textは、この正規化後のテキストをベースにしている。
    """
    text = pp._clean_claim_text(text)
    doc = pp.nlp(text)
    comps = pp.extract_patent_components_general(doc)

    # 「パワーモジュールの製造方法」のように、クレームタイトル（依存構造上の
    # ROOT）が「Ｘの＜基本語＞」という所有格付き複合語になっている場合、
    # GiNZA単体版のextract_has_relations同様、直前の「Ｘの」チェーンを
    # 含めた最大限の複合語に拡張してから1つのタグにする（532件の検証で、
    # このタイトル拡張がGiNZA単体版のプレシジョン・リコールの両方を
    # 改善することを確認済み）。ここでタグ付けの段階で拡張しておかないと、
    # LLM直接抽出方式ではタイトルの「Ｘ」と「＜基本語＞」が別々の2つのタグ
    # になってしまい、LLMがどう頑張っても正解データの結合したタイトル名
    # （例："パワーモジュールの製造方法"）を1つのSubjectとして出力できない。
    root_token = next((t for t in doc if t.head == t), None)
    if root_token is not None:
        for i, c in enumerate(comps):
            if c["end"] == root_token.i:
                extended = pp.find_full_title_component(doc, comps, c)
                if extended is not c:
                    comps[i] = extended
                break

    spans = []
    for c in comps:
        if c["start"] < 0 or c["end"] < 0 or c["end"] >= len(doc):
            continue
        # 「電気的に」「機械的に」のような副詞的表現の語幹（「電気」等）が、
        # 単独の構成要素として誤検出されることがある。直後が「的」なら
        # 実体を指す名詞ではないので、タグ化の対象から外す
        # （タグ化してしまうと「電気的に」が壊れ、英訳が意味不明になる）。
        if c["end"] + 1 < len(doc) and doc[c["end"] + 1].text == "的":
            continue
        # GiNZA/SudachiPyの辞書に「延在する」が複合サ変動詞として登録されて
        # おらず、「延」(NOUN)＋「在す」(VERB)＋「る」(AUX) に誤分割される
        # ケースを確認した（「介在する」「点在する」「対向する」「混在する」
        # 「存在する」等、他の「〜在する」型動詞はすべて正しく1つの動詞
        # トークンとして解析される中、「延在する」だけがこの誤分割の対象に
        # なっていた）。この場合、直後のトークンが「在す」で始まるVERBに
        # なるので、そのパターンを検出したら「延」側は動詞の一部であって
        # 独立した構成要素ではないため、タグ化の対象から外す
        # （タグ化すると「延在する」が壊れ、LLM側のSAO抽出で
        # 「壁部 在する 延」のような意味不明な関係が生成されてしまう）。
        if (c["end"] + 1 < len(doc)
                and doc[c["end"] + 1].pos_ == "VERB"
                and doc[c["end"] + 1].text.startswith("在す")):
            continue
        # 「複数の多穴管」の「複数」、「互いに」の「互い」のような数量・指示の語は、構成要素ではない。
        # タグにすると「《C1》の《C2》」となり、LLM が「複数」を1つの部品として扱ってしまう
        # （「複数｜の｜多穴管」のような関係や、「複数」と「多穴管」が別々のノードになる）ため、タグにしない。
        if c["text"] in _NON_COMPONENT_WORDS:
            continue
        # 「（ａ）」「（ｂ）」の工程ラベル除外は、extract_patent_components_general
        # 側（patent_pipeline.py）で一元的に対応済み（_is_paren_step_label）。
        # ここでは重複対応しない。
        start_char = doc[c["start"]].idx
        end_tok = doc[c["end"]]
        end_char = end_tok.idx + len(end_tok.text)
        comp_text = c["text"]
        # 「しょうが及び落花生」の「しょうが」は「しょう＋が」と切られることがある。「が」の直後が「及び」や読点なら、
        # その「が」は助詞ではなく名前の一部なので、名前に含める
        if text[end_char:end_char + 1] == "が" and _GA_NEXT_RE.match(text, end_char + 1):
            end_char += 1
            comp_text += "が"
        spans.append((start_char, end_char, comp_text))

    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    text_to_tag = {}
    next_id = 1
    chosen = []
    last_end = -1
    for start_char, end_char, comp_text in spans:
        if start_char < last_end:
            continue
        if comp_text not in text_to_tag:
            text_to_tag[comp_text] = f"C{next_id}"
            next_id += 1
        chosen.append((start_char, end_char, text_to_tag[comp_text]))
        last_end = end_char

    chosen.sort(key=lambda s: s[0])
    pieces = []
    cursor = 0
    for start_char, end_char, tag in chosen:
        pieces.append(text[cursor:start_char])
        pieces.append(f"{_TAG_OPEN}{tag}{_TAG_CLOSE}")
        cursor = end_char
    pieces.append(text[cursor:])
    tagged_text = "".join(pieces)

    tag_to_text = {tag: comp_text for comp_text, tag in text_to_tag.items()}
    return tagged_text, tag_to_text, doc, comps


_SYSTEM_PROMPT = (
    "You are a technical patent translator.\n\n"
    "TASK: Translate the Japanese patent claim text into English. "
    "Break it into SEPARATE, SIMPLE, GRAMMATICALLY COMPLETE sentences — "
    "exactly ONE clear subject-verb-object relationship per sentence. "
    "Never output a bare noun phrase or a phrase missing its verb "
    "(e.g. never write \"an electrically connected component C4 to C1\" — "
    "instead write \"C4 is electrically connected to C1.\"). "
    "Every sentence must have an explicit subject and a real verb "
    "(is / comprises / has / includes / is connected to / is provided on / "
    "is inserted into / is embedded in / etc.).\n\n"
    f"The text contains placeholder tokens shaped exactly like {_TAG_OPEN}C1{_TAG_CLOSE}, "
    f"{_TAG_OPEN}C2{_TAG_CLOSE}, etc. — note the brackets are {_TAG_OPEN} and {_TAG_CLOSE} "
    "(NOT angle brackets < >, NOT 〈 〉, NOT parentheses). "
    f"Copy every such token EXACTLY as it appears, including the {_TAG_OPEN}{_TAG_CLOSE} "
    "brackets and the number — never translate, reorder, merge, or drop them, never change "
    "the bracket characters, and never invent new ones.\n\n"
    "IMPORTANT — Japanese patent claims typically name the overall device/apparatus "
    "as its OWN tagged token at the very END of the claim, right before the final "
    "\"。\" (e.g. \"...を備え、(details)、"
    f"{_TAG_OPEN}C9{_TAG_CLOSE}。\" — here {_TAG_OPEN}C9{_TAG_CLOSE} IS the device). "
    "You MUST NOT drop or ignore this final tag. Whenever earlier text lists items "
    "with \"を備え\"/\"を有し\"/\"を含み\" (comprising/having/including), output an "
    "explicit sentence naming that final device tag as the subject, e.g. "
    f"\"{_TAG_OPEN}C9{_TAG_CLOSE} comprises ...\" listing the top-level items that "
    "were introduced with \"を備え\" — never silently omit this sentence just because "
    "the tag happens to sit at the end of the Japanese sentence.\n\n"
    "EXAMPLE 1\n"
    f"Input: {_TAG_OPEN}C1{_TAG_CLOSE}と{_TAG_OPEN}C2{_TAG_CLOSE}とを有する"
    f"{_TAG_OPEN}C3{_TAG_CLOSE}と、前記{_TAG_OPEN}C3{_TAG_CLOSE}の前記{_TAG_OPEN}C1{_TAG_CLOSE}"
    f"に電気的に接続された{_TAG_OPEN}C4{_TAG_CLOSE}と、を備える装置。\n"
    "Output:\n"
    f"{_TAG_OPEN}C3{_TAG_CLOSE} comprises {_TAG_OPEN}C1{_TAG_CLOSE} and "
    f"{_TAG_OPEN}C2{_TAG_CLOSE}. "
    f"{_TAG_OPEN}C4{_TAG_CLOSE} is electrically connected to the "
    f"{_TAG_OPEN}C1{_TAG_CLOSE} of {_TAG_OPEN}C3{_TAG_CLOSE}.\n\n"
    "EXAMPLE 2 (the device itself is a tagged token at the end — the common case)\n"
    f"Input: {_TAG_OPEN}C1{_TAG_CLOSE}と、{_TAG_OPEN}C2{_TAG_CLOSE}とを備え、"
    f"前記{_TAG_OPEN}C1{_TAG_CLOSE}は{_TAG_OPEN}C3{_TAG_CLOSE}を含み、"
    f"{_TAG_OPEN}C4{_TAG_CLOSE}。\n"
    "Output:\n"
    f"{_TAG_OPEN}C4{_TAG_CLOSE} comprises {_TAG_OPEN}C1{_TAG_CLOSE} and "
    f"{_TAG_OPEN}C2{_TAG_CLOSE}. "
    f"{_TAG_OPEN}C1{_TAG_CLOSE} includes {_TAG_OPEN}C3{_TAG_CLOSE}.\n\n"
    "EXAMPLE 3 (two coordinated items, EACH with its own modifier, both "
    "belonging to the SAME containing 'includes' list — do not merge them "
    "into one sentence, and do not drop either item)\n"
    f"Input: 前記{_TAG_OPEN}C1{_TAG_CLOSE}に電気的に接続された{_TAG_OPEN}C2{_TAG_CLOSE}と"
    f"前記{_TAG_OPEN}C2{_TAG_CLOSE}から電気的に絶縁された{_TAG_OPEN}C3{_TAG_CLOSE}とを含む"
    f"{_TAG_OPEN}C4{_TAG_CLOSE}と、を備え、\n"
    "Output:\n"
    f"{_TAG_OPEN}C4{_TAG_CLOSE} includes {_TAG_OPEN}C2{_TAG_CLOSE} and "
    f"{_TAG_OPEN}C3{_TAG_CLOSE}. "
    f"{_TAG_OPEN}C2{_TAG_CLOSE} is electrically connected to {_TAG_OPEN}C1{_TAG_CLOSE}. "
    f"{_TAG_OPEN}C3{_TAG_CLOSE} is electrically insulated from {_TAG_OPEN}C2{_TAG_CLOSE}.\n"
    "(Note: the WRONG way to translate this — do not do this — would be to "
    f"write something like \"{_TAG_OPEN}C4{_TAG_CLOSE} comprises {_TAG_OPEN}C1{_TAG_CLOSE} "
    f"that is connected to {_TAG_OPEN}C2{_TAG_CLOSE}\", which drops "
    f"{_TAG_OPEN}C3{_TAG_CLOSE} entirely and puts the wrong tag as the object of "
    "'comprises'. Always give the containing item — here "
    f"{_TAG_OPEN}C4{_TAG_CLOSE} — BOTH coordinated tags as its direct objects in "
    "one sentence, then describe each one's own modifier in its own separate "
    "sentence.)\n\n"
    "Output ONLY the translated English text as plain sentences separated by "
    "periods, nothing else (no bullet points, no numbering, no explanations)."
)


_CHUNK_MAX_TAGS = 4  # 1回のOllama呼び出しに含めるタグ数の目安上限


def _split_into_chunks(tagged_text, max_tags=_CHUNK_MAX_TAGS):
    """
    タグ付き原文を、Ollamaに一度に渡すタグ数を抑えるため、節境界の「、」で
    分割する（タグ数が多いクレームほど、1回の翻訳呼び出しで主語の取り違えや
    列挙の脱落が起きやすいことが確認されたため）。

    分割候補として使えるのは、直前の文字が「》」ではない「、」だけにする。
    「《C1》と、《C2》とを備え、」の「と、」や「…を含み、」「…されている、」
    のような動詞の連用形＋「、」はここで切ってよいが、
    「《C6》、《C7》および《C8》」のようにタグの直後にそのまま「、」が続く
    並列列挙の区切りは分割点にしない（そこで切ると列挙が分断され、
    片方のチャンクだけでは意味が通らなくなるため）。

    もう一つ避けるべきパターンがある。「《C1》と、《C2》と、を備え、」の
    ように、列挙の最後の項目の直後の「と、」で切ってしまうと、「を備え」
    という動詞だけが主語（列挙全体）を失って次のチャンクの先頭に取り残され、
    翻訳がおかしくなる（実際に確認済み）。そこで、「、」の直後（空白・改行を
    除く）が「を＋動詞＋、」の形（「を備え、」「を有し、」「を含み、」等）で
    始まっている場合は分割点にしない。この「を〜、」自体はその内側にある
    独立した「、」で後から切れるので、結果的に「列挙＋を備え」の部分が
    ひとまとまりのチャンクになる（なお、この「Ｘを備える」というクレーム
    全体を貫く関係は、GiNZA単体版から借用する既存ロジックで別途正しく
    補完されるので、翻訳チャンク側で多少ぶつ切りになっても実害は無い）。

    実際に切るのは、直前の分割位置からのタグ数がmax_tags以上になった
    候補点だけ（タグ数が少ない間はまとめた方が文脈が保たれ、翻訳品質が
    上がるため）。候補が無い、またはタグ数がmax_tags未満の短いクレームは
    分割せず1チャンクのまま返す＝1回のOllama呼び出しのみで、従来と同じ
    挙動になる。
    """
    _VERB_CONTINUATION_RE = re.compile(r"^\s*を[^\s《》、。]{1,6}、")
    # 「Ａ、Ｂおよび《Ｃ》と、《Ｄ》との間に位置する」のように、列挙の
    # 締めの「と、」の直後が「を＋動詞」ではなく、さらに別のタグへの
    # 「タグとの」（比較・位置関係の格助詞）に続く場合も、動詞（位置する等）
    # から見た主語側の列挙が丸ごと切り離されてしまう（実際に特開2025-188284で
    # 確認済み：「間に位置する」の対象タグが翻訳から丸ごと消えた）。
    # この場合も分割点にしない。
    _AND_CONTINUATION_RE = re.compile(r"^\s*(前記)?《[^》]+》との")

    candidates = []
    for m in re.finditer("、", tagged_text):
        pos = m.start()
        if pos > 0 and tagged_text[pos - 1] == _TAG_CLOSE:
            continue  # タグ直後の「、」は並列列挙の区切りなのでスキップ
        if pos > 0 and tagged_text[pos - 1] == "は":
            continue  # 「Ｘは、」の直後は、まだ述語が来ていない主題化直後なのでスキップ
        if _VERB_CONTINUATION_RE.match(tagged_text[pos + 1:]):
            continue  # 「と、を備え、」のように、直後が列挙をまとめる動詞の続きならスキップ
        if _AND_CONTINUATION_RE.match(tagged_text[pos + 1:]):
            continue  # 「と、《Ｄ》との間に…」のように、直後が別タグへの「との」続きならスキップ
        candidates.append(pos + 1)

    if not candidates:
        return [tagged_text]

    chunks = []
    chunk_start = 0
    for cut in candidates:
        n_tags = len(_BRACKET_TAG_RE.findall(tagged_text[chunk_start:cut]))
        if n_tags >= max_tags:
            chunks.append(tagged_text[chunk_start:cut])
            chunk_start = cut
    if chunk_start < len(tagged_text):
        chunks.append(tagged_text[chunk_start:])

    # タグを一つも含まないチャンク（末尾の「。」だけ等）は単独でOllamaに
    # 渡す意味が無いので、前のチャンクに吸収する
    merged = []
    for c in chunks:
        if merged and len(_BRACKET_TAG_RE.findall(c)) == 0:
            merged[-1] += c
        else:
            merged.append(c)

    return merged if merged else [tagged_text]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# チャンク分割（_split_into_chunks）によって節の途中で切れた入力を、
# モデル（特にShisa V2.1等）が「入力が不完全に見える」と判断し、
# 翻訳本文の中に "(Note: The original Japanese text seems incomplete...)"
# のような注釈を混入させることがある（特開2025-174033/2023-126262の
# 実運用ログで確認済み）。この注釈は本来のシステムプロンプトの指示
# （「翻訳された英語テキストのみを出力する」）に反しており、これが
# そのまま英語の依存構造解析に渡ると、注釈内の単語（"incomplete"
# "specification"等）から無関係なSAOが誤抽出されてしまうため、
# 英語解析にかける前に除去する。
_NOTE_BLOCK_RE = re.compile(r"[\(\[]\s*note\s*:.*?[\)\]]", re.IGNORECASE | re.DOTALL)


# 特開2025-175399のように、正極側／負極側のような対称構造が何度も
# 繰り返される複雑なクレームでは、タグの種類・数が多くなり、ローカルの
# 小型モデル（qwen2.5:7b等）が同じような関係を延々と繰り返し出力する
# 「繰り返しループ」に陥ることがある（実運用ログで、この請求項でだけ
# eval_translate_sao.pyが応答なしのまま止まることを確認済み）。
# ollamaパッケージのデフォルトはtimeout=None（＝無制限に待つ）ため、
# こうなると1件のクレームで処理全体が無期限に固まってしまう。
# num_predictで生成トークン数の上限を設け、さらにHTTPタイムアウトも
# 設定することで、異常な1件のせいで全体が止まることを防ぐ
# （eval_translate_sao.py側は元々1件ごとにtry/exceptで囲んであるが、
# 例外が飛んでこない＝ハングする限り、そのtry/exceptは機能しない。
# タイムアウトを設定して初めて、ハングが「例外」に変換され、
# 既存のtry/exceptが機能するようになる）。
_OLLAMA_REQUEST_TIMEOUT_SECONDS = 400
_OLLAMA_MAX_OUTPUT_TOKENS = 1200
# 【追記】特開2025-175399は180秒でも間に合わずタイムアウトしていた
# （実運用ログで確認）。この請求項はスキップされず結果が欲しいとの
# 要望のため、180→400秒に延長した。


# 【重要】Ollama は、1回の呼び出しで読める長さ（コンテキスト長 num_ctx）を指定しないと、GPUのメモリが
# 24GB未満のPCでは 4096 トークン（古い版では 2048）で打ち切る。入りきらない部分は黙って捨てられるため、
# 長い指示文＋長い請求項＋多数の候補を渡すと、指示や請求項の一部が読まれないまま答えが返ってくる。
# ここで明示的に広げておく（環境変数 SAO_NUM_CTX で変更可。メモリが足りなければ 8192 などに下げる）。
OLLAMA_NUM_CTX = int(os.environ.get("SAO_NUM_CTX", "16384"))


def _ollama_chat(system_prompt, user_text, model=DEFAULT_MODEL, host=None):
    """
    Ollamaへの低レベル呼び出しを共通化したヘルパー。
    translate_tagged（タグ付き日本語→英訳）と、analyze_claim_llm_direct
    （タグ付き日本語→SAO直接抽出）の両方から使う。
    """
    if ollama is None:
        raise RuntimeError(
            "ollamaパッケージが見つかりません。`pip install ollama` を実行し、"
            "Ollamaサーバー（`ollama serve`）を起動してください。"
        )
    client = ollama.Client(host=host, timeout=_OLLAMA_REQUEST_TIMEOUT_SECONDS)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        options={"temperature": 0, "num_predict": _OLLAMA_MAX_OUTPUT_TOKENS, "num_ctx": OLLAMA_NUM_CTX},
    )
    # Qwen3系（Shisa V2.1等）はデフォルトで「思考モード」が有効で、実際の
    # 回答の前に長い<think>...</think>推論ブロックを出力する。これが
    # 長すぎると、回答本体を出す前にトークン上限に達してcontentが
    # 空文字列になることがある（実際に確認済み：188284等で英訳が空）。
    # think=Falseで思考モード自体を止めるのが第一の対策。
    # 古いollamaや、この設定に対応していないモデルではエラーになりうる
    # ので、その場合は思考モードありのままフォールバックする。
    try:
        response = client.chat(**kwargs, think=False)
    except TypeError:
        response = client.chat(**kwargs)
    content = response["message"]["content"]
    # think=Falseが効かない/未対応のモデルでも、<think>ブロックが
    # contentに残ったまま返ってくることがあるため、念のため除去する
    # （除去しないとその推論文がそのまま後段の解析に渡り、
    # 無関係なSAOが大量に誤抽出されてしまう）。
    content = _THINK_BLOCK_RE.sub("", content).strip()
    # 上記_NOTE_BLOCK_RE参照：チャンクが節の途中で切れたことに対する
    # モデル自身の注釈文を除去する。
    content = _NOTE_BLOCK_RE.sub("", content).strip()
    return content


def translate_tagged(tagged_text, model=DEFAULT_MODEL, host=None):
    return _ollama_chat(_SYSTEM_PROMPT, tagged_text, model=model, host=host)


_DEEPL_TAG_OPEN_RE = re.compile(r"《(C\d+)》")
_DEEPL_PLACEHOLDER_RE = re.compile(r"<(C\d+)\s*/>")


def translate_tagged_deepl(tagged_text, api_key=None, target_lang="EN-US"):
    """
    Ollamaの代わりにDeepL API（無料枠あり、1回だけ課金なしで合計100万文字まで
    利用可能なDeepL API Developerプランを想定）で翻訳する。
    DeepLは専用の機械翻訳エンジンなので、Ollamaの小型LLMのように「タグを
    翻訳するな」という指示を無視したり<think>ブロックで潰れたりする心配が
    ない一方、DeepL自身はプレーンテキスト中の《C1》のような記号を自然に
    保持する保証はないため、DeepLの「XMLタグ処理」機能
    （https://developers.deepl.com/docs/resources/examples-and-guides/placeholder-tags）
    を使い、《C1》を自己完結型のXMLタグ<C1/>に変換してから渡し、
    訳文中でDeepLが位置を保ったまま返してくる<C1/>を《C1》に戻す。
    """
    if deepl is None:
        raise RuntimeError(
            "deeplパッケージが見つかりません。`pip install deepl` を実行してください。"
        )
    api_key = api_key or os.environ.get("DEEPL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DeepL APIキーが必要です。https://www.deepl.com/ja/your-account/keys で取得した"
            "キーを環境変数 DEEPL_API_KEY に設定するか、api_key引数で渡してください。"
        )
    translator = deepl.Translator(api_key)
    xml_text = _DEEPL_TAG_OPEN_RE.sub(lambda m: f"<{m.group(1)}/>", tagged_text)
    result = translator.translate_text(
        xml_text, source_lang="JA", target_lang=target_lang, tag_handling="xml",
    )
    return _DEEPL_PLACEHOLDER_RE.sub(lambda m: f"{_TAG_OPEN}{m.group(1)}{_TAG_CLOSE}", result.text)


def strip_bracket_tags(english_text):
    """《C1》 → C1 （英語側の依存構造解析にかける前の整形）"""
    return _BRACKET_TAG_RE.sub(lambda m: m.group(1), english_text)


def extract_relations_from_english(english_text):
    """
    英訳済みテキスト（《C1》タグ入り、または既にC1形式）からSAOを抽出する。
    文単位で解析する（文をまたいだ誤った係り受けを避けるため）。
    """
    clean = strip_bracket_tags(english_text)
    doc = nlp_en(clean)
    rels = []
    for sent in doc.sents:
        rels.extend(extract_relations(sent.as_doc()))
    return dedup(rels)


# ============================================================
# 日本語タグ付きテキストからLLMに直接SAOを抽出させる方式
# ============================================================
# 「タグ付き日本語→英訳→英語で依存構造解析」という2段階では、英訳自体が
# 「Ａを備えたＢ」の主語・目的語を取り違えたり、「工程」を"device"に
# 変換したりするなど、SAO抽出の前段で不可逆な変形を起こしてしまうことが
# 判明した（特開2025-174033等の実運用ログで確認）。この誤差は翻訳段と
# 抽出段の2つが混ざったものであり、「LLM単独のSAO抽出精度」を測っている
# とは言えない。そこで、英訳を完全に外し、タグ付き日本語テキストを
# そのままLLMに渡してSAOを直接出力させる方式を用意する。
_SAO_EXTRACTION_PROMPT = (
    "あなたは特許請求項の構造解析器です。タグ付き日本語テキストから\n"
    "SAO（Subject-Action-Object）関係だけを抽出してください。\n"
    "\n"
    "【出力形式】\n"
    "1行に1関係、次の形式のみで出力する（他の文章・説明・見出し・番号付けは\n"
    "一切出力しない）：\n"
    "《タグ》 | 関係語 | 《タグ》\n"
    "\n"
    "【SAOのルール】\n"
    "1. SubjectとObjectは、必ずテキスト中の《C1》のようなタグそのものを使う\n"
    "   （タグの内容を書き換えたり、タグ以外の語をSubject/Objectにしたり\n"
    "   しない）。\n"
    "2. 「AをBに備える／有する／含む／具備する」は\n"
    "   B | 備える | A の向きにする（備える主体が常にSubject。AとBを\n"
    "   取り違えない）。\n"
    "3. 「Aに設けられたB」「Aに接続されたB」のような修飾関係は\n"
    "   B | 設けられる | A のように、修飾されている側（B）をSubjectにする。\n"
    "4. 「前記A」は同一請求項内の同じAを指すので、Aと同じタグを使う。\n"
    "5. 工程名（〜工程）も他の構成要素と全く同じ扱いにする（品詞や種類を\n"
    "   勝手に変換・言い換えない）。\n"
    "6. 原文に明示的に書かれている関係だけを抽出し、推測や補完で関係を\n"
    "   追加しない。\n"
    "7. SubjectとObjectの向きを変更しない。関係語は元の日本語の動詞・\n"
    "   表現をできるだけそのまま使う（言い換えない）。\n"
    "8. 「Ａに（おいて）Ｂを形成する／設ける／配置する／注入する」のように、\n"
    "   場所・対象Ａに対して新たにＢを作る・置くという能動文の場合は、\n"
    "   Ｂ | 形成される（またはその動詞の受動形） | Ａ の向きにする\n"
    "   （ルール3の能動文バージョン）。ただし、これは「〜する工程」\n"
    "   「〜すること」を置き換えるのではなく追加する処理である。\n"
    "   「装置は〜する工程を備える」のように工程・処理全体を備える／含む\n"
    "   の対象にしている記述があれば、そのタグ | 備える（または含む） | \n"
    "   工程のタグ の関係は必ず出力し、その上で、工程の内容にある\n"
    "   タグ同士の関係も原文にある範囲で追加で抽出する（両方を出す。\n"
    "   工程・処理を分解したからといって、備える／含むの関係を省略しない）。\n"
    "9. 「ＡはＢより／よりも大きい／小さい／高い／低い」のような比較文は、\n"
    "   Ａ | （原文の比較語をそのまま） | Ｂ の1関係として出力する\n"
    "   （比較語を「大きい」→「多い」等に言い換えない）。\n"
    "10. 同じSubject・Objectのタグの組に対して、意味が同じ関係を\n"
    "    言い換えて複数回出力しない（1つのタグの組につき、原文に基づく\n"
    "    関係は基本的に1つ）。例えば「Ｃ５は前記開口部により前記Ｃ３に\n"
    "    対して位置決めされており」という原文に対して、\n"
    "    Ｃ５|位置決めされる|Ｃ３ を既に出力したら、同じＣ５とＣ３の組に\n"
    "    対して Ｃ５|配置される|Ｃ３ のような言い換えを追加で出力しない。\n"
    "    比較文（ルール9）でも同様に、「ＡはＢより小さい」を\n"
    "    Ａ|より小さい|Ｂ として出力したら、逆方向の Ｂ|より大きい|Ａ を\n"
    "    重複して追加で出力しない（原文に書かれている向きの1関係だけでよい）。\n"
    "11. 「Ａと、Ｂと、Ｃとを備え（有し／含み）、…（Ａ・Ｂ・Ｃの説明が続く）…、\n"
    "    Ｄ。」のように、複数の要素をまず列挙してから「を備え」等で結び、\n"
    "    実際の主体（備える側）Ｄがクレームの最後（末尾のタグ、通常は句点\n"
    "    の直前）に1回だけ出てくる構造がある。この場合、Ｄ|備える|Ａ、\n"
    "    Ｄ|備える|Ｂ、Ｄ|備える|Ｃ のように、末尾のＤを備える主体にする\n"
    "    （列挙の途中に出てくるＡやＢを誤って備える主体にしない）。\n"
    "    重要：この「末尾のＤを主体にする」のは、冒頭の列挙＋備える／\n"
    "    有する／含むの関係だけに適用する。その後に続く別の文（例：\n"
    "    「前記Ｃの一部は…との間に位置する」）には、Ｄを主体として\n"
    "    使わない。それらの文の主体は、その文中に明示されている\n"
    "    「前記Ｘ」等のタグをそのまま使う。同じ関係（同じ意味の\n"
    "    Subject-Object）を、Ｄを主体にしたものと、本来の主体にした\n"
    "    ものの両方で重複して出力しない。\n"
    "12. 原文の文・読点区切りの記述を1つも読み飛ばさない。「前記Ｘは、\n"
    "    Ｙ、Ｚを含み」のような短い記述も、他の複雑な記述と同じように\n"
    "    必ず関係を抽出する（前後に複雑な記述があっても、この種の単純な\n"
    "    記述を省略しない）。\n"
    "\n"
    "EXAMPLE 1\n"
    f"Input: {_TAG_OPEN}C1{_TAG_CLOSE}と{_TAG_OPEN}C2{_TAG_CLOSE}とを有する"
    f"{_TAG_OPEN}C3{_TAG_CLOSE}と、前記{_TAG_OPEN}C3{_TAG_CLOSE}の前記{_TAG_OPEN}C1{_TAG_CLOSE}"
    f"に電気的に接続された{_TAG_OPEN}C4{_TAG_CLOSE}と、を備える装置。\n"
    "Output:\n"
    f"{_TAG_OPEN}C3{_TAG_CLOSE} | 有する | {_TAG_OPEN}C1{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C3{_TAG_CLOSE} | 有する | {_TAG_OPEN}C2{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C4{_TAG_CLOSE} | 電気的に接続される | {_TAG_OPEN}C1{_TAG_CLOSE}\n"
    "\n"
    "EXAMPLE 2（ルール8・9）\n"
    f"Input: 前記{_TAG_OPEN}C1{_TAG_CLOSE}の表面に{_TAG_OPEN}C2{_TAG_CLOSE}を形成することと、"
    f"前記{_TAG_OPEN}C3{_TAG_CLOSE}の{_TAG_OPEN}C4{_TAG_CLOSE}は、前記{_TAG_OPEN}C5{_TAG_CLOSE}"
    f"の{_TAG_OPEN}C6{_TAG_CLOSE}よりも小さい。\n"
    "Output:\n"
    f"{_TAG_OPEN}C2{_TAG_CLOSE} | 形成される | {_TAG_OPEN}C1{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C4{_TAG_CLOSE} | より小さい | {_TAG_OPEN}C6{_TAG_CLOSE}\n"
    "\n"
    "EXAMPLE 3（ルール11・12：列挙してから末尾で主体が示される構造、"
    "その後の文では末尾のタグを使い回さない、単純な記述も省略しない）\n"
    f"Input: {_TAG_OPEN}C1{_TAG_CLOSE}と、前記{_TAG_OPEN}C1{_TAG_CLOSE}に配置された"
    f"{_TAG_OPEN}C2{_TAG_CLOSE}と、{_TAG_OPEN}C3{_TAG_CLOSE}と、を備え、"
    f"前記{_TAG_OPEN}C3{_TAG_CLOSE}は、{_TAG_OPEN}C4{_TAG_CLOSE}および"
    f"{_TAG_OPEN}C5{_TAG_CLOSE}を含み、"
    f"前記{_TAG_OPEN}C2{_TAG_CLOSE}の一部は、前記{_TAG_OPEN}C4{_TAG_CLOSE}との間に位置する、"
    f"{_TAG_OPEN}C6{_TAG_CLOSE}。\n"
    "Output:\n"
    f"{_TAG_OPEN}C6{_TAG_CLOSE} | 備える | {_TAG_OPEN}C1{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C6{_TAG_CLOSE} | 備える | {_TAG_OPEN}C2{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C6{_TAG_CLOSE} | 備える | {_TAG_OPEN}C3{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C2{_TAG_CLOSE} | 配置される | {_TAG_OPEN}C1{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C3{_TAG_CLOSE} | 含む | {_TAG_OPEN}C4{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C3{_TAG_CLOSE} | 含む | {_TAG_OPEN}C5{_TAG_CLOSE}\n"
    f"{_TAG_OPEN}C2{_TAG_CLOSE} | 間に位置する | {_TAG_OPEN}C4{_TAG_CLOSE}\n"
    "（最後の文の主体は前記C2＝文中に明示されたタグであり、末尾のC6を\n"
    "ここで使い回して「C6|間に位置する|C4」のように出力してはいけない）\n"
)


_SAO_LINE_RE = re.compile(
    r"[《\[<{]\s*(C\d+)\s*[》\]>}]\s*[|｜]\s*(.+?)\s*[|｜]\s*[《\[<{]\s*(C\d+)\s*[》\]>}]"
)


def extract_relations_from_llm_output(llm_output, tag_to_text):
    """
    _SAO_EXTRACTION_PROMPTで指示した「《タグ》 | 関係語 | 《タグ》」形式の
    出力行をパースし、タグを元のテキストに戻したSAOリストを返す。

    形式に従わない行（説明文の混入等）は無視する（パースできる行だけを
    採用する、というfail-soft方針。英訳経由の方式と同じく、原文に無い
    説明文が混ざっても後段の評価に影響しないようにするため）。
    """
    relations = []
    seen = set()
    for line in llm_output.splitlines():
        m = _SAO_LINE_RE.search(line)
        if not m:
            continue
        source_tag, relation_text, target_tag = m.groups()
        source = tag_to_text.get(source_tag)
        target = tag_to_text.get(target_tag)
        if source is None or target is None:
            continue
        relation_text = relation_text.strip()
        if not relation_text or source == target:
            continue
        key = (source, relation_text, target)
        if key in seen:
            continue
        seen.add(key)
        relations.append({"source": source, "relation": relation_text, "target": target, "type": "llm_direct"})
    return relations


def analyze_claim_translate_llm(text, pipeline_dir=None, model=DEFAULT_MODEL, host=None, pp=None,
                             debug=False, debug_out=None, max_tags_per_chunk=_CHUNK_MAX_TAGS,
                             backend="ollama", deepl_api_key=None):
    """
    日本語クレームテキストを渡すと (構成要素リスト, 関係リスト) を返す。
    patent_pipeline.analyze_claim_ginza_only と同じ戻り値の形なので、
    既存の evaluate_triples / batch_evaluate にそのまま渡せる。

    タグ数が多いクレーム（目安でmax_tags_per_chunkを超える）は、
    _split_into_chunks で節境界ごとに分割し、それぞれ個別に翻訳させてから
    結果を連結する（1回の翻訳呼び出しに含まれるタグ数が多いと、主語の
    取り違えや列挙の脱落が起きやすいことが確認されたため）。
    タグ数が少ないクレームは分割されず、1回の呼び出しのみ。

    backend="ollama"（デフォルト）ならローカルのOllamaモデルを、
    backend="deepl"なら課金なしで使えるDeepL API（DEEPL_API_KEY環境変数、
    またはdeepl_api_key引数でキーを指定）を使って翻訳する。

    debug_out に辞書を渡すと、タグ付き原文・分割チャンク・英訳・タグ対応表を
    書き込む（Streamlit等、printではなく画面表示したい呼び出し元向け）。
    """
    pp = pp or _load_pipeline(pipeline_dir)
    tagged_text, tag_to_text, tag_doc, tag_comps = tag_components(text, pp)

    chunks = _split_into_chunks(tagged_text, max_tags=max_tags_per_chunk)
    if backend == "deepl":
        english_parts = [translate_tagged_deepl(c, api_key=deepl_api_key) for c in chunks]
    else:
        english_parts = [translate_tagged(c, model=model, host=host) for c in chunks]
    english = " ".join(p.strip() for p in english_parts if p and p.strip())

    if debug:
        print("---タグ付き原文---", tagged_text, sep="\n")
        if len(chunks) > 1:
            print(f"---{len(chunks)}個のチャンクに分割して翻訳---")
            for i, c in enumerate(chunks, 1):
                print(f"[chunk {i}] {c}")
        print("---英訳---", english, sep="\n")
    if debug_out is not None:
        debug_out["tagged_text"] = tagged_text
        debug_out["chunks"] = list(chunks)
        debug_out["english"] = english
        debug_out["tag_to_text"] = dict(tag_to_text)
    raw_rels = extract_relations_from_english(english)

    _TAG_TOKEN_RE = re.compile(r"C\d+")

    def _resolve_tag_text(s):
        # 通常は"C1"のような単一タグだが、en_relation_rules._scan_passive_targets
        # が「C2 of C1」のようなネストした属格を「C1のC2」という複合タグ文字列
        # として返すことがある。dict.get一発ではこの複合形を素通りさせてしまう
        # ため、文字列内のタグをすべて正規表現で個別に置換する
        # （単一タグの場合も同じ結果になるので後方互換）。
        return _TAG_TOKEN_RE.sub(lambda m: tag_to_text.get(m.group(0), m.group(0)), s)

    mapped = []
    seen = set()
    for r in raw_rels:
        source = _resolve_tag_text(r["source"])
        target = _resolve_tag_text(r["target"])
        if source == target:
            continue
        key = (source, r["relation"], target)
        if key in seen:
            continue
        seen.add(key)
        mapped.append({"source": source, "relation": r["relation"], "target": target, "type": "translate"})

    return _apply_ginza_fallback_and_normalize(text, mapped, pp, tag_doc, tag_comps, tag_to_text)


# ============================================================
# 【実験2】条件付きGiNZA（検証型）
# ============================================================
# 532件のbaseline結果を分析した結果、ginza_has_fallback|有する候補の
# Precisionを下げている主因は「構成要素数」ではなく「同一請求項内での
# 候補数（gh_n）」であることが判明した（gh_n 1〜8ではPrecision 68〜94%、
# gh_n 16以上の13件だけでPrecisionが25.3%まで落ち、しかもこの13件が
# 候補総数1208件のうち482件（約40%）を占めていた）。
# 単純に「gh_n>=16なら候補を全削除」するだけでもMICRO F1は改善する
# （+0.6pt、Recall低下は-1.2pt）が、その13件の中にも約36%は正しい候補が
# 含まれているため、削除ではなくLLMに個別確認させる「検証型」の方が
# Recallの犠牲をさらに抑えられる可能性がある、という南々香さんの判断で
# こちらを採用する。
_RISK_VERIFY_TYPE = "ginza_has_fallback"
_RISK_VERIFY_RELATION = "有する"
_DEFAULT_RISK_THRESHOLD = 16  # 上記分析に基づく既定値（gh_n>=16件で検証対象とする）

# ============================================================
# 【実験8】ginza_has_fallback|有するのfan-out型検証（1請求項の総数ではなく、
# 1つのsourceが何個の別々のtargetに繋がっているかで見る）
# ============================================================
# 実験2のgh_n（1請求項あたりのginza_has_fallback|有する候補の総数）は粗い
# 指標で、532件中わずか13件（gh_n>=16）しか捕まえられない。しかし実際の
# FP事例（「係合爪の各々」が「第１部材」「第２部材」「係合爪」「係合部」
# 「側壁」「半導体装置」の6つに、「外部端子」が「ベースプレート」
# 「絶縁基板」「絶縁板」「導体パターン」の4つに、それぞれ「有する」で
# 繋がってしまう、といったケース）を精査したところ、gh_nが16未満の請求項
# でも、1つのsourceが同じ請求項内で何個の別々のtargetと「有する」で
# 繋がっているか（fan-out数）で見ると、精度が急落することが判明した。
# 532件全体（GiNZAのraw候補×gold_sao_532_merged.jsonとの突き合わせによる
# シミュレーション）: fan-out=1で55.9%、2で66.5%、3で53.1%、4で49.2%、
# 5で27.3%、6以上で20.9%。gh_n（請求項全体の総数）とは独立した、より
# 細かい粒度のリスク指標であり、両方を併用できる（_verify_risky_candidates_
# in_listが同じ(type,relation)に対する複数ルールの発火をrisky_idxの
# 和集合として扱うため）。
# デフォルトはverify_fanout=False（無効）のため、実験1〜7のbaseline
# 再現性には影響しない。実際のF1への効果は、gh_n同様Ollamaでの532件本番
# 評価でのみ確定できる（ここでのシミュレーションは「raw候補×gold」の
# 突き合わせであり、LLM無言／一致／矛盾の分類を経た実際の採用候補数とは
# 厳密には異なるため、参考値として扱うこと）。
_DEFAULT_FANOUT_RISK_THRESHOLD = 3  # 上記分析に基づく既定値（同一source→3個以上の異なるtargetで検証対象とする）

# ============================================================
# 【実験3】検証型の対象を、ginza_has_fallback|有する以外にも拡張
# ============================================================
# 「削除するか残すかの0/1」ではなく、危険な(type, 関係語)を見つけたら
# 同じ「LLMに再確認させる」処理を適用する、という考え方に基づく。
# 532件baselineの精査（実験2の直前に実施）で見つかった、他に検証型が
# 有効そうな候補：
#   ・claim_title_ginza|有する：gh_nと同じ「1請求項あたりの候補数」で見ると、
#     全体は75.5%と健全だが、候補数14件以上の12クレームだけPrecisionが
#     54.0%まで落ちる（gh_nほど極端ではないが、同じ形の偏りがある）。
#     単純削除では常にF1が悪化することを確認済みなので、検証型で狙う。
#   ・attribute|の（Xの深さ、のような出自関係）：全体でPrecision53.5%だが、
#     候補数1件のクレームでも52.6%と、gh_nと違って「量が多いから危険」
#     ではなく最初から一様に怪しい。なので閾値は1（＝出現したら常に検証）。
#   ・llm_direct（LLM自身の直接抽出）側にも、文脈非依存でほぼ誤りな
#     関係語がある：「超える」（10.0%）「方向」（37.5%）「より小さい」
#     （55.6%）「位置する」（67.4%）。数値の閾値表現や方向・位置を表す
#     修飾句をSAO関係として誤抽出しているケースが多い（例：
#     「温度 超える 値」「通路 方向 全域」）。絶対件数は小さいが、
#     precisionが低いままにしておく理由もないので、同じ検証の仕組みに乗せる。
# ルールは {"type":..., "relation":..., "threshold":...} の辞書のリストで表す。
# typeによって検証対象のリストが変わる（_apply_ginza_fallback_and_normalize
# 側で振り分ける）：
#   "llm_direct" → LLM直接抽出の生候補（mapped、GiNZA補完より前）
#   "claim_title_ginza"/"ginza_has_fallback"とその*_conflict → GiNZA補完候補
#   "attribute" → 出自関係補完後のmapped
_LLM_DIRECT_RISK_RULES = [
    {"type": "llm_direct", "relation": "超える", "threshold": 1},
    {"type": "llm_direct", "relation": "方向", "threshold": 1},
    {"type": "llm_direct", "relation": "より小さい", "threshold": 1},
    {"type": "llm_direct", "relation": "位置する", "threshold": 1},
]
_DEFAULT_CLAIM_TITLE_RISK_THRESHOLD = 14  # 実験3時点の既定値（ct_n>=14でPrecisionが54.0%まで落ちる分析に基づく）
_GINZA_EXTRA_RISK_RULES = [
    {"type": "claim_title_ginza", "relation": "有する", "threshold": _DEFAULT_CLAIM_TITLE_RISK_THRESHOLD},
]
_ATTRIBUTE_RISK_RULES = [
    {"type": "attribute", "relation": "の", "threshold": 1},
]
# --verify-extra-risky 1つで全部まとめて有効化するための既定セット
# （ginza_has_fallback|有するは既存のverify_risky_ginza/risk_thresholdが担当するので含めない）
_ALL_EXTRA_RISK_RULES = _LLM_DIRECT_RISK_RULES + _GINZA_EXTRA_RISK_RULES + _ATTRIBUTE_RISK_RULES


def _build_extra_risk_rules(claim_title_risk_threshold=_DEFAULT_CLAIM_TITLE_RISK_THRESHOLD):
    """【実験5】claim_title_ginza|有するの検証閾値だけを差し替えられるようにした
    _ALL_EXTRA_RISK_RULES のビルダー版。

    実験4後の残存FP分析で、claim_title_ginza|有する（ct_n）・ginza_has_fallback|
    有する（gh_n）とも、既存の検証済み最上位バケットでまだ精度が低いままな
    上に、閾値のすぐ下（ct_n 10〜13でPrecision72.0%、gh_n 10〜13で55.6%）にも
    まだ検証対象外の危険な塊が残っていることが判明した。このうちgh_n側の
    閾値は既存の--risk-thresholdでそのまま変更できるが、ct_n側は
    _GINZA_EXTRA_RISK_RULESに埋め込まれた固定値だったため、ここだけ差し替え
    られるようにする（llm_direct側の4関係・attribute|のは実験3のまま変更しない）。
    引数を省略すればデフォルト（_DEFAULT_CLAIM_TITLE_RISK_THRESHOLD=14）となり、
    _ALL_EXTRA_RISK_RULESと完全に同じ内容を返すため、実験3までの再現性には
    影響しない。
    """
    ginza_rules = [
        {"type": "claim_title_ginza", "relation": "有する", "threshold": claim_title_risk_threshold},
    ]
    return _LLM_DIRECT_RISK_RULES + ginza_rules + _ATTRIBUTE_RISK_RULES

# ============================================================
# 【実験4】target無条件フィルタ（gh_n等の「危険候補数」ベースではなく、
# Goldデータそのものから「絶対に正解になり得ないtarget」を確定させる方式）
# ============================================================
# 実験3の532件結果（llm_direct|有するのFP事例）を精査したところ、
# 「基板 有する 複数」「配線基板 含む 互い」のように、targetが
# 「複数」「互い」のような抽象語・代名詞語**だけ**になっている誤抽出が
# 目立った（元の名詞句「複数の端子」等から、肝心の名詞部分が欠落した
# パターンとみられる）。
# gold_sao_532_merged.json（正解データ全10,497件、relation種別を問わず全件）
# を確認した結果、この18語がtargetになっている正解は**1件も存在しない**
# ことを確認済み。つまり「危険だから検証する」ではなく、「targetがこの
# 語なら、システム抽出は数学的に確定でFP」という決定的フィルタであり、
# LLMによる再確認は不要（Ollama呼び出しゼロで適用できる）。
# relationの種類やtypeに関わらず、target一致だけで判定する
# （llm_direct|有するに限定しない――根拠が「relationが危険」ではなく
# 「targetそのものがgold上で正解になり得ない」という性質のため）。
# 532件全体でのシミュレーション: 該当144件を除去してMICRO F1 80.14%→80.65%
# （+0.51pt）、MACRO F1 79.78%→80.27%（+0.49pt）。Recallコストは理論上ゼロ
# （goldに存在しない語なので、除去してもTPは一切減らない）。
# デフォルト（filter_invalid_targets=False）では一切変更を加えないため、
# 実験1〜3のbaseline再現性には影響しない。
#
# 追加検証（「一方」「他方」）: 「冷却器 有する 一方」「第１ヘッダ 備える 一方」
# のように、「○○の一方」「○○の他方」という名詞句から肝心の「○○」部分が
# 欠落し、「一方」「他方」単体がtargetになる誤抽出を確認。gold_sao_532_merged.json
# 全10,497件を確認したところ、「一方」「他方」がtargetまたはsourceとして
# 単体で（完全一致で）登場する正解は1件も存在しない（gold上では必ず
# 「一方の面」「絶縁基板の他方の面」のように、他の語と結合した複合語としてのみ
# 登場する）。実験4適用済みのExperiment4結果（532件）に対するシミュレーション:
# 該当13件（6請求項）を除去してMICRO F1 80.65%→80.70%（+0.05pt）、
# MACRO F1 80.27%→80.32%（+0.05pt）。Recallコストは理論上ゼロ（上記18語と同じ理由）。
#
# 追加検証（「それぞれの一方」「それぞれの他方」）: 「一方」「他方」を除去した後も、
# 「前記複数の多穴管のそれぞれの一方が接続される」のような原文から、LLMが
# タグを裸のまま使わず「それぞれの《C6》」のように周辺の語ごと出力してしまい、
# 結果としてtargetが「それぞれの一方」「それぞれの他方」という複合語のまま
# 残るケースを確認（実質「一方」「他方」と同じ問題の別表記）。同様に
# gold_sao_532_merged.json全10,497件を確認し、この2語が単体で（完全一致で）
# targetまたはsourceとして登場する正解が1件も存在しないことを確認済み。
_INVALID_TARGET_WORDS = {
    "複数", "互い", "こと", "もの", "場合", "状態", "様子",
    "全体", "一部", "両方", "それぞれ", "いずれか", "各々",
    "これ", "それ", "あれ", "ここ", "そこ",
    "一方", "他方", "それぞれの一方", "それぞれの他方",
}


def _filter_invalid_targets(relations):
    """targetが_INVALID_TARGET_WORDSに完全一致する関係を無条件で除外する。

    gold側にこれらの語がtargetとして一件も存在しないことを確認済みのため、
    typeやrelationの種類を問わず一律に除去してよい（詳細は上のコメント参照）。
    """
    return [r for r in relations if r.get("target", "").strip() not in _INVALID_TARGET_WORDS]

# ============================================================
# 【実験7】クレームタイトル（総称ノード）による二重所有の除去
# ============================================================
# 実験4後のFP分析（gold_sao_532_merged.jsonとの同一claim単位の突き合わせ）で、
# 「有する」系FPのうち、sourceがクレームタイトル相当の構成要素
# （_apply_ginza_fallback_and_normalize内でclaim_title_ginza判定に使っている
# ものと同じtitle_text）であるものだけを抜き出したところ、664件中371件
# （55.9%）は、targetがgold上で「titleとは別の、より具体的な下位構成要素」
# から正しく所有されていることが判明した（例：「パワーモジュール 有する
# 第１素子裏面」は誤りで、正しくは「第１パワー半導体素子 有する 第１素子
# 裏面」）。これは、階層構造の末端に近い構成要素を、本来の直接の親では
# なく、クレーム全体のタイトル（総称ノード）が二重に「有する」と主張して
# しまう過大包摂（over-generalization）パターンであり、GiNZA補完由来
# （claim_title_ginza）だけでなくLLM直接抽出（llm_direct）でも起きている
# ことを確認済み（type別内訳：claim_title_ginza 503件、llm_direct 146件、
# ginza_has_fallback 4件、claim_title_ginza_conflict 11件）。
# 532件全体でのシミュレーション: 該当371件を除去してMICRO F1 80.65%→80.99%
# 相当（+1.35pt、実験4単独の+0.51ptを上回る規模）。
# ただし実験4のtargetフィルタとは異なり、goldには稀に同一targetが複数の
# 妥当なsourceを持つケースが存在する（4,437件中61件、1.37%）ため、理論上は
# 正しいtitle起点の関係を誤って除去するリスクがゼロではない。そのため、
# この処理は「target無条件フィルタ」と異なり、Recallに与える実際の影響を
# 532件全体の再実行で確認することとし、デフォルト
# （filter_redundant_root_ownership=False）では一切変更を加えないため、
# 実験1〜6のbaseline再現性には影響しない。
#
# 【実験7実測後の追記（実験7b）】532件中268件分の実測結果で検証したところ、
# claim_title_ginza|有するのTPが997→699（-298件）、FPが301→97（-204件）と、
# 除去されたFPよりも除去された「正しい」関係（TP）の方が多く、Recallを
# マクロで3pt以上悪化させる、当初のシミュレーションに反する結果となった。
# 原因を実例（特開2023-128709等）で追跡したところ、_HAS_SYNONYMS_FOR_DEDUP
# に「の」を含めていたことが over-trigger の主因と判明した：「の」は
# 属性・部分参照（例：「主端子の一部」）を表す構文であり、「有する／備える」
# のような所有関係とは意味的に別物であるにもかかわらず、たまたま同じ
# targetを指す無関係な「の」関係が1件でも存在するだけで、
# targets_with_other_owner に登録され、タイトルの正当な所有関係が
# 「二重所有」と誤判定されて除去されてしまっていた（単体再現テスト済み、
# test_filter_redundant_root_ownership.py参照）。この設計は532件全件での
# 事前シミュレーション（word-list版・real-title版いずれも）では検出できて
# おらず、実際の関係集合でしか顕在化しない失敗モードだった。
# 対策として、「他のsourceが既に所有している」と判定するための関係語集合
# から「の」を除外し、真の所有関係を表す語（有する／備える／具備する／
# 含む／含める）のみに限定する。
def _filter_redundant_root_ownership(relations, title_text):
    """title_textを主体とする「有する」系関係のうち、同じtargetをtitle_text
    以外のsourceからの「有する」系関係で既に他の構成要素が所有している
    ものを、クレームタイトルによる二重所有（過大包摂）とみなして除去する。

    「他のsourceによる所有」の判定には、真の所有・包含関係を表す語
    （_OWNERSHIP_RELATIONS_FOR_DEDUP）のみを用いる。「の」による属性・
    部分参照は所有関係ではないため対象外とする（実験7の実測で、これを
    含めるとタイトルの正当な所有関係まで誤って除去されることが判明した
    ため、実験7bで除外した）。

    title_textがNone（クレームタイトルに相当する構成要素が検出できな
    かった場合）は何もしない。
    """
    if not title_text:
        return relations
    has_synonyms = _OWNERSHIP_RELATIONS_FOR_DEDUP
    targets_with_other_owner = {
        r["target"] for r in relations
        if r["relation"] in has_synonyms and r["source"] != title_text
    }
    return [
        r for r in relations
        if not (
            r["relation"] in has_synonyms
            and r["source"] == title_text
            and r["target"] in targets_with_other_owner
        )
    ]


# 【実験7b】真の所有・包含関係のみ（「の」による属性・部分参照は含めない）。
_OWNERSHIP_RELATIONS_FOR_DEDUP = {"有する", "備える", "具備する", "含む", "含める"}
# 後方互換のため旧名も残す（実験7の値と同一の意味では使わないこと）。
_HAS_SYNONYMS_FOR_DEDUP = _OWNERSHIP_RELATIONS_FOR_DEDUP

_VERIFY_PROMPT_SYSTEM = (
    "あなたは特許請求項の構造解析の検証者です。与えられた関係の候補一覧について、"
    "それぞれが請求項原文の記載から実際に読み取れる正しい関係かどうかを判定します。\n"
    "\n【出力形式】\n"
    "候補と同じ番号で、1行に1つ、次の形式のみで出力する（他の文章・説明・見出しは"
    "一切出力しない）：\n"
    "番号 | はい または いいえ\n"
    "\n【判定基準】\n"
    "・その関係が原文に明示的に書かれている、または一意に読み取れる場合のみ「はい」。\n"
    "・原文には別の構成要素との関係として書かれている、根拠が薄い、推測が必要な場合は"
    "「いいえ」。\n"
    "・候補は構文解析による自動抽出なので、誤って生成されたものも混ざっている前提で、"
    "厳密に判定してください。"
)

_VERIFY_VERDICT_RE = re.compile(r"^\s*(\d+)\s*[|｜:：]\s*(はい|いいえ|yes|no)", re.IGNORECASE)


def _build_verify_prompt_user(text, candidates):
    lines = [f"{i}. 「{r['source']}」は「{r['target']}」を「{r['relation']}」"
              for i, r in enumerate(candidates, 1)]
    return (
        "請求項の原文:\n" + text + "\n\n"
        "以下は、この請求項からGiNZA（構文解析）で自動抽出した関係の候補です。\n"
        "各候補が、請求項の記載から実際に読み取れる正しい関係かどうかを判定してください。\n\n"
        + "\n".join(lines)
    )


def _verify_ginza_candidates_llm(text, candidates, model=DEFAULT_MODEL, host=None,
                                  verify_cache=None, claim_id=None, debug=False):
    """
    risk_threshold件以上の「有する」候補が生成された請求項について、各候補を
    LLMに個別確認させ、確認できた候補だけを返す（判定できなかった候補は
    安全側＝従来通り採用に倒す）。

    verify_cache（dict）とclaim_idを渡すと、analyze_claim_llm_direct側の
    llm_cacheと同じ考え方で、候補一覧のハッシュが変わらない限りLLM呼び出しを
    再利用する（実験3以降でGiNZA側のロジックを変えずに再評価したい場合に、
    この検証呼び出し分もOllama不要にするため）。
    """
    if not candidates:
        return candidates, None

    cand_hash = _hash_text(_build_verify_prompt_user(text, candidates))
    cache_entry = verify_cache.get(claim_id) if (verify_cache is not None and claim_id is not None) else None

    if cache_entry is not None and cache_entry.get("candidates_hash") == cand_hash:
        raw = cache_entry["raw_verify_output"]
        from_cache = True
    else:
        raw = _ollama_chat(_VERIFY_PROMPT_SYSTEM, _build_verify_prompt_user(text, candidates),
                            model=model, host=host)
        from_cache = False
        if verify_cache is not None and claim_id is not None:
            verify_cache[claim_id] = {"candidates_hash": cand_hash, "raw_verify_output": raw}

    if debug:
        print(f"---GiNZA候補検証（{len(candidates)}件、{'キャッシュ利用' if from_cache else '新規呼び出し'}）---")
        print(raw)

    verdicts = {}
    for line in raw.splitlines():
        m = _VERIFY_VERDICT_RE.match(line)
        if m:
            verdicts[int(m.group(1))] = m.group(2).strip().lower() in ("はい", "yes")

    # 判定が得られなかった候補（LLM出力の解析失敗等）は、安全側＝従来通り
    # 採用に倒す（検証できなかったことを理由に正しい候補まで失わないため）。
    kept = [r for i, r in enumerate(candidates, 1) if verdicts.get(i, True)]
    return kept, from_cache


def _verify_risky_candidates_in_list(candidates, rules, text, model, host, verify_cache, claim_id,
                                      debug, verify_cache_key_suffix, stage_label):
    """
    【実験3】ginza_has_fallback|有する以外にも検証型を適用するための汎用ヘルパー。

    candidates: 関係リスト（source/relation/target/typeを持つ辞書のリスト）。
    rules: [{"type":..., "relation":..., "threshold":...}, ...]。
    同じcandidatesリストの中で複数のルールが同時に閾値を超えた場合も、
    LLM呼び出しは1回にまとめる（無駄なOllama呼び出しを増やさないため）。
    却下された候補だけをcandidatesから除いたリストを返す（1件もルールが
    発火しなければcandidatesをそのまま返す＝変更なし）。

    verify_cache_key_suffixは、同じclaim_idでも検証対象リスト（mapped側の
    llm_directなのか、extra_from_ginza側のGiNZA補完候補なのか、attribute側
    なのか）ごとにキャッシュエントリが衝突しないようにするための識別子。
    """
    risky_idx = set()
    triggered = []
    for rule in rules:
        idx = [i for i, r in enumerate(candidates)
               if r["type"] == rule["type"] and r["relation"] == rule["relation"]]
        if rule.get("group_by") == "source":
            # 【実験8】fan-out型：請求項全体での候補数ではなく、同じsourceを
            # 持つ候補どうしでグループ化し、グループごとに閾値判定する
            # （1つの語が何個の別々のtargetに繋がっているかを見る、より
            # 細かい粒度のリスク指標。既存のgh_n型ルールと同じ(type,relation)
            # を指定しても、risky_idxは和集合なので両方併用できる）。
            by_source = {}
            for i in idx:
                by_source.setdefault(candidates[i]["source"], []).append(i)
            for source_idx in by_source.values():
                if len(source_idx) >= rule["threshold"]:
                    risky_idx.update(source_idx)
                    triggered.append((rule, len(source_idx)))
        elif len(idx) >= rule["threshold"]:
            risky_idx.update(idx)
            triggered.append((rule, len(idx)))
    if not risky_idx:
        return candidates

    risky_candidates = [candidates[i] for i in sorted(risky_idx)]
    cache_key = f"{claim_id}{verify_cache_key_suffix}" if claim_id is not None else None
    kept, from_cache = _verify_ginza_candidates_llm(
        text, risky_candidates, model=model, host=host,
        verify_cache=verify_cache, claim_id=cache_key, debug=debug,
    )
    summary = ", ".join(f"{r['type']}|{r['relation']}={n}件（閾値{r['threshold']}）" for r, n in triggered)
    # --debugを付けていない通常実行でも、検証が実際に発火したことと結果が
    # 分かるようにする（532件のような長時間実行を眺めている間に「本当に
    # 発火しているか」を確認できるようにするため）。
    print(f"  [{claim_id or '?'}] 条件付き検証発火［{stage_label}］（{summary}） "
          f"→ 対象{len(risky_candidates)}件中{len(kept)}件採用"
          f"{'（verify-cache利用）' if from_cache else ''}")
    kept_keys = {(r["source"], r["relation"], r["target"]) for r in kept}
    return [
        r for i, r in enumerate(candidates)
        if i not in risky_idx or (r["source"], r["relation"], r["target"]) in kept_keys
    ]


def _apply_ginza_fallback_and_normalize(text, mapped, pp, tag_doc, tag_comps, tag_to_text,
                                         enable_has_fallback=True, verify_risky_ginza=False,
                                         risk_threshold=_DEFAULT_RISK_THRESHOLD,
                                         verify_fanout=False,
                                         fanout_risk_threshold=_DEFAULT_FANOUT_RISK_THRESHOLD,
                                         extra_risk_rules=None,
                                         model=DEFAULT_MODEL, host=None,
                                         verify_cache=None, claim_id=None, debug=False,
                                         filter_invalid_targets=False,
                                         filter_redundant_root_ownership=False):
    """
    英訳経由（analyze_claim_translate）・LLM直接抽出（analyze_claim_llm_direct）
    の両方で共通の後処理。どちらも「タグ付き日本語→タグ付き関係リスト」までは
    別々のロジックで作るが、そこから先（GiNZA単体版からの補完＋ノード正規化）
    は全く同じ処理なので、1箇所にまとめて重複・食い違いを防ぐ。

    「Ａと、Ｂと、…とを備え、（詳細説明）、Ｃ。」のように、装置名Ｃがクレームの
    一番最後にしか出てこない構造や、「ＸはＹとＺとを含む」のような並列列挙は、
    翻訳／LLM直接抽出のどちらでも取り違えや訳し漏らしが起きることがある一方、
    GiNZA単体版（analyze_claim_ginza_only）はこの「含む/有する/備える」系の
    直接関係を比較的安定して抽出できるので、そちらから補完する（既存の結果は
    上書きしない＝取り違えた誤った関係が残る可能性はあるが、正しい関係が
    足される分は必ず改善になる）。

    上記は、GiNZA単体版やanalyze_claim_translate（英訳経由）の抽出力が
    弱かった頃を前提にした設計だったが、532件実測（244/532件時点）で
    (type, 関係語)ごとのTP/FP/Precisionを集計・シミュレーションした結果、
    claim_title_ginza／ginza_has_fallbackを丸ごとON/OFFしたり、
    Precisionが低い(type, 関係語)を単純に不採用にしたりすると、削れる
    FPよりも失うTPの方が多く、F1は必ず下がることを確認済み（詳細は
    eval_translate_sao.pyのコメント参照）。つまり今の無条件統合が、
    単純な絞り込みでは超えられない実質的な最良構成になっている。

    【LLM無言／LLM矛盾の分離】そこで、GiNZAの各候補を、同じ(source, target)
    についてLLM（mapped）が何を言っているかで3つに分けるようにした。
    （1）LLM無言：LLMが何も出していない → 従来通りclaim_title_ginza／
    ginza_has_fallbackとして追加。（2）LLMと一致（同義語含む、
    _relation_synonym_match）：表記が違うだけの重複関係なので追加しない
    （【バグ修正】以前はclaim_title_ginza側にこのチェックが無く、LLMが
    既に「有する」を出していてもGiNZAが「備える」のような表記違いの
    候補を無条件に追加してしまっていた＝純粋なFPを生んでいた）。
    （3）LLM矛盾：LLMが同じ(source,target)について同義語ではない別の
    関係を出している → claim_title_ginza_conflict／
    ginza_has_fallback_conflictとして追加する（採用するかどうかはまだ
    決めない。まずeval_translate_sao.pyのtp_fp_by_type_relationで
    この新しいtypeのTP/FP/Precisionを実測してから、無言タイプとは別に
    採用/不採用を判断する）。

    【実験3：検証型の対象拡張】
    extra_risk_rules（_ALL_EXTRA_RISK_RULES等、[{"type":..., "relation":...,
    "threshold":...}, ...]の形）を渡すと、実験2で作ったLLM再確認の仕組みを
    ginza_has_fallback|有する以外の(type, 関係語)にも適用できる。ルールの
    typeによって検証対象のリストが変わる：
      - "llm_direct" → LLM直接抽出の生候補（mapped、GiNZA補完より前）
      - "claim_title_ginza"/"ginza_has_fallback"とその*_conflict →
        GiNZA補完候補（extra_from_ginza、mapped統合より前）
      - "attribute" → 出自関係補完後のmapped（_add_genitive_provenance_relations後）
    extra_risk_rules=None（デフォルト）では一切変更しない。

    【実験8：fan-out型検証】
    verify_fanout=Trueにすると、ginza_has_fallback|有するの候補群を、請求項
    全体での総数（gh_n、実験2）ではなく、同じsourceが何個の別々のtargetと
    繋がっているか（fan-out数）で見て、fanout_risk_threshold件以上の
    グループだけをLLMに個別確認させる。verify_risky_ginzaと独立に指定でき、
    両方Trueなら両方の基準（総数・fan-out）のいずれかに該当する候補が
    まとめて1回のLLM呼び出しで検証される。デフォルト（verify_fanout=False）
    では一切変更を加えないため、実験1〜7のbaseline再現性には影響しない。
    """
    # 【実験3】LLM直接抽出自体の危険候補（超える・方向 等）を検証する。
    # GiNZA補完より前の、LLMの生の出力（type="llm_direct"）だけが対象。
    llm_direct_rules = [r for r in (extra_risk_rules or []) if r["type"] == "llm_direct"]
    if llm_direct_rules:
        mapped = _verify_risky_candidates_in_list(
            mapped, llm_direct_rules, text, model, host, verify_cache, claim_id, debug,
            verify_cache_key_suffix="::llm_direct", stage_label="LLM直接抽出",
        )

    try:
        _, ginza_relations = pp.analyze_claim_ginza_only(text)
    except Exception:
        ginza_relations = []

    # 表現形式（順次列挙形式／構成要素列挙形式／ジェプソン的形式）によって、
    # GiNZA単体版からの補完（下記extra_from_ginza）をどこまで信用するかを
    # 切り替える。順次列挙形式（方法クレーム）では、GiNZA単体版の
    # extract_has_relationsにある「同じ語の繰り返し列挙→root_componentへ」
    # ヒューリスティックが、「〜すること」という動詞の名詞化節による列挙
    # （「基板とエピタキシャル成長層とを準備することと、…を有する、
    # 半導体基板構造体の製造方法」等）で誤爆し、方法名（製造方法等）から
    # 全く無関係な語への誤った関係を作ってしまうことを532件回帰の分析で
    # 確認済み（例：「製造方法 有する エピタキシャル成長層」「製造方法
    # 有する 真空中」）。一方、「〜工程」という名詞そのものの繰り返し列挙
    # （「製造方法 備える 供給工程」等）は同じヒューリスティックが正しく
    # 機能している。両者を区別する安全で簡単な基準として、対象
    # （target）のテキストが「工程」を含むものだけに絞って採用する
    # （タグ付け＝構成要素境界の認識ルール自体を形式ごとに作り直すのでは
    # なく、既存のGiNZA補完をどこまで信用するかを形式で絞る、という
    # 形での「表現形式ごとのルール適用」）。
    claim_format = pp.classify_claim_format(text)
    if claim_format == "順次列挙形式":
        def _ginza_fallback_target_ok(target_text):
            return "工程" in target_text
    else:
        def _ginza_fallback_target_ok(target_text):
            return True

    # 【LLM無言／LLM一致／LLM矛盾の分類】GiNZAの候補ごとに、同じ(source, target)
    # についてLLM（mapped）が何を言っているかで3つに分ける。
    #   ・LLM無言　　：その(source,target)についてLLMは何も出していない
    #                  → GiNZAの候補をそのまま採用候補として追加する
    #                    （type=claim_title_ginza / ginza_has_fallback、従来通り）。
    #   ・LLMと一致　：LLMが同じ意味（同義語グループで一致、_relation_synonym_match）
    #                  の関係を既に出している → 追加しない（表記だけ違う重複関係を
    #                  増やさない）。
    #                  【バグ修正】以前はclaim_title_ginza側にこのチェックが
    #                  一切無く、LLMが既に「有する」を出していてもGiNZAが
    #                  「備える」のような表記違いの候補を追加してしまうことが
    #                  あった（同じ関係の言い換え重複＝純粋なFPになる）。
    #   ・LLM矛盾　　：LLMが同じ(source,target)について同義語ではない別の関係を
    #                  出している → type=claim_title_ginza_conflict /
    #                  ginza_has_fallback_conflict として追加する（採用するか
    #                  どうかはまだ決めない。まずTP/FP/Precisionを実測してから
    #                  判断するための、無言タイプとは別カテゴリの候補として
    #                  記録する）。
    mapped_relations_by_pair = {}
    for r in mapped:
        mapped_relations_by_pair.setdefault((r["source"], r["target"]), []).append(r["relation"])

    def _relation_matches_any(candidate_relation, existing_relations):
        for rel in existing_relations:
            if rel == candidate_relation or pp._relation_synonym_match(candidate_relation, rel):
                return True
        return False

    # 【実験7】クレームタイトル相当の構成要素名。enable_has_fallback=Falseでも
    # _filter_redundant_root_ownershipで使えるよう、ブロックの外で初期化しておく。
    _claim_title_text_for_dedup = None

    extra_from_ginza = []
    if enable_has_fallback:
        added_pairs = set()  # GiNZAフォールバック候補として既に判定済みの(source,target)

        def _classify_and_maybe_add(r, type_no_conflict, type_conflict):
            pair = (r["source"], r["target"])
            if pair in added_pairs:
                return
            existing_rels = mapped_relations_by_pair.get(pair, [])
            if not existing_rels:
                extra_from_ginza.append({
                    "source": r["source"], "relation": r["relation"],
                    "target": r["target"], "type": type_no_conflict,
                })
            elif not _relation_matches_any(r["relation"], existing_rels):
                extra_from_ginza.append({
                    "source": r["source"], "relation": r["relation"],
                    "target": r["target"], "type": type_conflict,
                })
            # else: LLMと同義語一致＝重複なので追加しない
            added_pairs.add(pair)

        last_i = len(tag_doc) - 1
        while last_i > 0 and tag_doc[last_i].pos_ == "PUNCT":
            last_i -= 1
        claim_title_comp = pp.find_component_by_token(tag_comps, last_i)
        _claim_title_text_for_dedup = (
            claim_title_comp["text"] if claim_title_comp is not None else None
        )
        if claim_title_comp is not None:
            title_text = claim_title_comp["text"]
            for r in ginza_relations:
                if r.get("type") == "has" and r["source"] == title_text and _ginza_fallback_target_ok(r["target"]):
                    _classify_and_maybe_add(r, "claim_title_ginza", "claim_title_ginza_conflict")

        _HAS_SYNONYMS = next(
            (g for g in pp.RELATION_SYNONYM_GROUPS if "含む" in g),
            {"有する", "備える", "具備する", "含む", "含める"},
        )
        # 「工程」を含むかどうかの絞り込みは、実際に誤爆を確認した
        # root_component（＝クレーム題名）が所有者になっているケースだけに
        # 限定する。それ以外の所有者（工程自身が持つ内部の実体等、例えば
        # 「原子 有する 結合手」）は、今回診断した誤爆パターンとは無関係な
        # ので、絞り込まずに従来通り採用する（未診断のまま広く絞ると、
        # 正しい関係まで失うリスクがある）。
        _title_text_for_filter = claim_title_comp["text"] if claim_title_comp is not None else None

        for r in ginza_relations:
            if r["relation"] not in _HAS_SYNONYMS:
                continue
            if r["source"] == _title_text_for_filter and not _ginza_fallback_target_ok(r["target"]):
                continue
            _classify_and_maybe_add(r, "ginza_has_fallback", "ginza_has_fallback_conflict")

    # 【実験2＋3】GiNZA補完候補側の検証：ginza_has_fallback|有する（実験2、
    # verify_risky_ginza/risk_threshold）と、それ以外の追加ルール
    # （claim_title_ginza|有する等、extra_risk_rules）をまとめて1回の
    # LLM呼び出しで検証する。どちらも指定しなければ一切変更しない
    # （既定では実験1のbaselineと完全に同じ挙動になる）。
    ginza_rules = []
    if verify_risky_ginza:
        ginza_rules.append({"type": _RISK_VERIFY_TYPE, "relation": _RISK_VERIFY_RELATION,
                             "threshold": risk_threshold})
    # 【実験8】fan-out型：同じ(type,relation)でも、請求項全体での総数ではなく
    # 「同じsourceが何個の別々のtargetと繋がっているか」で判定する別ルール。
    # verify_risky_ginzaのgh_n型ルールと共存でき、risky_idxの和集合として
    # 扱われる（_verify_risky_candidates_in_list参照）。
    if verify_fanout:
        ginza_rules.append({"type": _RISK_VERIFY_TYPE, "relation": _RISK_VERIFY_RELATION,
                             "threshold": fanout_risk_threshold, "group_by": "source"})
    ginza_rules += [
        r for r in (extra_risk_rules or [])
        if r["type"] in ("claim_title_ginza", "ginza_has_fallback",
                          "claim_title_ginza_conflict", "ginza_has_fallback_conflict")
        and not (r["type"] == _RISK_VERIFY_TYPE and r["relation"] == _RISK_VERIFY_RELATION)
    ]
    if ginza_rules:
        extra_from_ginza = _verify_risky_candidates_in_list(
            extra_from_ginza, ginza_rules, text, model, host, verify_cache, claim_id, debug,
            verify_cache_key_suffix="::ginza", stage_label="GiNZA補完",
        )

    existing_keys = {(r["source"], r["relation"], r["target"]) for r in mapped}
    mapped = mapped + [
        r for r in extra_from_ginza
        if (r["source"], r["relation"], r["target"]) not in existing_keys
    ]

    # GiNZA単体版（analyze_claim_ginza_only）と同じく、「Ｘの主面」「Ｘの裏面」
    # のような面・部位を表す複合語は、常にＸそのものとして統合する
    # （本人の指示：「構成要素の一部を表す言葉は、その構成要素として扱う」）。
    # 出自関係（_add_genitive_provenance_relations）より先に実行する必要がある
    # 順序もGiNZA単体版と揃えている（先に統合しておかないと「Ｘの主面」が
    # 単独ノードのまま出自関係を持ってしまうため）。
    mapped = pp._merge_surface_location_nodes(mapped)

    # 「Ｘの深さ」のような複合語ノードに対する出自関係の補完も、
    # GiNZA単体版と同じロジックを流用する。
    mapped = pp._add_genitive_provenance_relations(mapped)

    # 【実験3】attribute|の候補の検証。attribute型は_add_genitive_provenance_
    # relationsでしか作られないので、このタイミングでしか対象を拾えない。
    attribute_rules = [r for r in (extra_risk_rules or []) if r["type"] == "attribute"]
    if attribute_rules:
        mapped = _verify_risky_candidates_in_list(
            mapped, attribute_rules, text, model, host, verify_cache, claim_id, debug,
            verify_cache_key_suffix="::attribute", stage_label="出自関係(attribute)",
        )

    # GiNZA単体版（analyze_claim_ginza_only）と同じく、「一部」「部分」のような
    # 裸の部分名詞は、係り元（の格）の実体名にマージする
    # （例: "第１端子の一部" が単独ノード「一部」のままにならないようにする）。
    mapped = pp._merge_partitive_nodes(mapped, tag_doc, tag_comps)

    # 【実験4】target無条件フィルタ。type/relationを問わず、_INVALID_TARGET_WORDS
    # に完全一致するtargetを持つ関係を最後に一律除去する（詳細は定数定義部の
    # コメント参照）。LLM呼び出し不要・デフォルトFalseでは実験1〜3のbaseline
    # 再現性に影響しない。
    if filter_invalid_targets:
        before_n = len(mapped)
        mapped = _filter_invalid_targets(mapped)
        removed_n = before_n - len(mapped)
        if removed_n:
            print(f"  [{claim_id or '?'}] target無条件フィルタ発火 → {removed_n}件除去")

    # 【実験7】クレームタイトルによる二重所有の除去。target無条件フィルタ
    # （実験4）と同じくLLM呼び出し不要の決定的な後処理だが、根拠が
    # 「targetの語そのもの」ではなく「同じtargetを既に別の構成要素が
    # 所有しているか」という関係同士の突き合わせである点が異なる（詳細は
    # 関数定義部のコメント参照）。
    if filter_redundant_root_ownership:
        before_n = len(mapped)
        mapped = _filter_redundant_root_ownership(mapped, _claim_title_text_for_dedup)
        removed_n = before_n - len(mapped)
        if removed_n:
            print(f"  [{claim_id or '?'}] クレームタイトル二重所有フィルタ発火 → {removed_n}件除去")

    components = [{"text": v, "start": -1, "end": -1} for v in tag_to_text.values()]
    return components, mapped


def _hash_text(s):
    """キャッシュ整合性チェック用のハッシュ（タグ付き原文が変わっていないか確認する）。"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def analyze_claim_llm_direct(text, pipeline_dir=None, model=DEFAULT_MODEL, host=None, pp=None,
                              debug=False, debug_out=None, llm_cache=None, claim_id=None,
                              verify_risky_ginza=False, risk_threshold=_DEFAULT_RISK_THRESHOLD,
                              verify_fanout=False,
                              fanout_risk_threshold=_DEFAULT_FANOUT_RISK_THRESHOLD,
                              extra_risk_rules=None, verify_cache=None,
                              filter_invalid_targets=False,
                              filter_redundant_root_ownership=False):
    """
    日本語クレームテキストを渡すと (構成要素リスト, 関係リスト) を返す。
    英訳を挟まず、タグ付き日本語テキストをそのままLLMに渡してSAOを
    直接出力させる方式（analyze_claim_translateとの違いは、
    「タグ付き日本語→タグ付き関係リスト」の生成方法だけで、それ以降の
    GiNZA補完・ノード正規化は_apply_ginza_fallback_and_normalizeを共有する）。

    まずはチャンク分割せず、クレーム全体を1回のLLM呼び出しで処理する
    （英訳のときのように主語取り違えが起きやすいかどうかは未検証のため、
    問題が出た場合はチャンク分割を検討する）。

    【LLM生出力キャッシュ（実験2以降のため）】
    llm_cache（dict）とclaim_id（str）を渡すと、次のように動作する:
      - llm_cache[claim_id]が存在し、そのtagged_text_hashが今回のタグ付き
        原文のハッシュと一致する場合 → Ollamaを呼ばず、キャッシュ済みの
        llm_outputを使う（GiNZAフォールバック側のロジック変更だけを
        再評価したい実験3以降で、Ollama呼び出しを完全に省くためのもの）。
      - それ以外（未キャッシュ or ハッシュ不一致）→ 通常通りOllamaを呼び、
        結果をllm_cache[claim_id]に書き込む（呼び出し元がファイルへ保存する）。
      - llm_cache=None（デフォルト）の場合は今まで通り毎回Ollamaを呼ぶ。
        既存の呼び出し元（引数を渡さない場合）の挙動は完全に変わらない。

    【実験2：条件付きGiNZA（検証型）】
    verify_risky_ginza=Trueにすると、_apply_ginza_fallback_and_normalizeの
    条件付き検証（risk_threshold件以上のginza_has_fallback|有する候補が
    生成された請求項だけ、LLMに個別確認させる）が有効になる。
    verify_cache（dict）を渡すと、この検証呼び出し分もllm_cacheと同じ考え方で
    キャッシュされる。デフォルト（verify_risky_ginza=False）では
    _apply_ginza_fallback_and_normalizeに一切変更を加えないため、実験1の
    baselineの再現性には影響しない。

    【実験3：検証型の対象拡張】
    extra_risk_rules（例：_ALL_EXTRA_RISK_RULES）を渡すと、同じ検証の仕組みを
    claim_title_ginza|有する・attribute|の・llm_direct側の「超える」「方向」
    等にも適用する。verify_risky_ginzaと独立に指定でき、両方Noneのままなら
    実験1のbaselineと完全に同じ挙動。

    【実験4：target無条件フィルタ】
    filter_invalid_targets=Trueにすると、type・relationを問わず、targetが
    _INVALID_TARGET_WORDS（「複数」「互い」等、gold中に一件も存在しないと
    確認済みの語）に完全一致する関係を最後に一律除去する。LLM呼び出し不要
    （Ollama非依存）で、検証系のverify_risky_ginza/extra_risk_rulesとは独立に
    指定できる。デフォルトFalseでは実験1〜3のbaseline再現性に影響しない。

    【実験8：ginza_has_fallback|有するのfan-out型検証】
    verify_fanout=Trueにすると、ginza_has_fallback|有するの候補を、請求項
    全体の総数（gh_n、実験2）ではなく、同じsourceが何個の別々のtargetと
    繋がっているか（fan-out数）で見て、fanout_risk_threshold件以上の
    グループだけをLLMに個別確認させる。verify_risky_ginzaと独立に指定でき、
    両方Trueなら1回のLLM呼び出しでまとめて検証する。デフォルトFalseでは
    実験1〜7のbaseline再現性に影響しない。
    """
    pp = pp or _load_pipeline(pipeline_dir)
    tagged_text, tag_to_text, tag_doc, tag_comps = tag_components(text, pp)

    tagged_hash = _hash_text(tagged_text)
    cache_entry = llm_cache.get(claim_id) if (llm_cache is not None and claim_id is not None) else None

    if cache_entry is not None and cache_entry.get("tagged_text_hash") == tagged_hash:
        llm_output = cache_entry["llm_output"]
        from_cache = True
    else:
        llm_output = _ollama_chat(_SAO_EXTRACTION_PROMPT, tagged_text, model=model, host=host)
        from_cache = False
        if llm_cache is not None and claim_id is not None:
            llm_cache[claim_id] = {"tagged_text_hash": tagged_hash, "llm_output": llm_output}

    if debug:
        print("---タグ付き原文---", tagged_text, sep="\n")
        print(f"---LLM出力（SAO）{'［キャッシュ利用］' if from_cache else ''}---", llm_output, sep="\n")
    if debug_out is not None:
        debug_out["tagged_text"] = tagged_text
        debug_out["llm_output"] = llm_output
        debug_out["tag_to_text"] = dict(tag_to_text)
        debug_out["llm_cache_hit"] = from_cache

    mapped = extract_relations_from_llm_output(llm_output, tag_to_text)
    # 【検討中・いったん保留】enable_has_fallback=Falseで「有する」系GiNZA
    # フォールバックを無効化する変更を試したが、Precision改善のためだけに
    # Recallを犠牲にする判断を実測データなしで行うべきではない、という
    # 指摘を受けて元のTrue（フォールバック有効）に戻した。理由・今後の
    # 判断手順は_apply_ginza_fallback_and_normalizeのdocstring
    # 【検討中・いったん保留】を参照。
    return _apply_ginza_fallback_and_normalize(
        text, mapped, pp, tag_doc, tag_comps, tag_to_text,
        verify_risky_ginza=verify_risky_ginza, risk_threshold=risk_threshold,
        verify_fanout=verify_fanout, fanout_risk_threshold=fanout_risk_threshold,
        extra_risk_rules=extra_risk_rules,
        model=model, host=host, verify_cache=verify_cache, claim_id=claim_id, debug=debug,
        filter_invalid_targets=filter_invalid_targets,
        filter_redundant_root_ownership=filter_redundant_root_ownership,
    )


# ===========================================================================
# 【統合】node_match_eval.py
# ===========================================================================
"""
node_match_eval.py
===================
主語どうし・目的語どうしを比べる評価（--eval-mode node）。

従来の緩い評価（evaluate_triples_lenient）は「ＡがＢをＲする」という文全体の
埋め込み類似度で一致を判定していたが、同じ請求項の構成要素をでたらめに
組み合わせた偽の関係でも97.8%がどれかの正解と類似度0.75以上になり、本文を
読まないでたらめな出力でもF1が67.1%になってしまうことが分かった。
そこで、意味的類似度は使いつつ、主語と目的語を別々に比べる。

一致の条件（1対1の対応付け）:
  ① 表記正規化で主語・目的語が一致し、関係も一致 → 一致（従来と同じ）
  ② 残りは、主語どうし・目的語どうしがそれぞれ次のどれかで対応し、かつ関係が一致:
       ・表記正規化で一致
       ・一方がもう一方を含む（例：「第１端」と「第１トランジスタの第１端」）。
         ただし短い方の番号（第１・Ａ等）が長い方にも含まれる場合のみ
       ・番号が同じで、埋め込みの類似度がθ（既定0.9）以上
     主語・目的語の対応の強さの小さい方が高い組から順に対応付ける。
  θ=0.9 の根拠：同じ請求項の別々の正解ノード同士（番号が同じ・包含関係なし）で
  類似度が0.9以上になるのは4.2%（0.75では27.2%）。
  関係の一致は従来の規則（表記正規化・漢字部分一致・同義語グループ）を使うが、
  同義語グループの「の」は関係全体が「の」の場合だけ認める（従来は「の」を含む
  あらゆる関係語が「有する」系と同義扱いになっていた）。
"""
import re

_KANJI_NUM = str.maketrans("一二三四五六七八九", "123456789")
_NUM_RE = re.compile(r"\d+|[a-zA-Z]")


def numbers(t):
    t = re.sub(r"第([一二三四五六七八九])", lambda m: "第" + m.group(1).translate(_KANJI_NUM), t)
    return tuple(_NUM_RE.findall(t))


def node_score(a, b, E, theta):
    if a == b:
        return 1.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 2 and short in long_ and set(numbers(short)) <= set(numbers(long_)):
        return 0.9
    if numbers(a) != numbers(b):
        return 0.0
    s = float(E[a] @ E[b])
    return s if s >= theta else 0.0


def rel_match(pp, p_rel, g_rel):
    if p_rel == g_rel:
        return True
    pn = pp._normalize_relation_for_match(p_rel)
    gn = pp._normalize_relation_for_match(g_rel)
    if pn == gn or pn in gn or gn in pn:
        return True
    for group in pp.RELATION_SYNONYM_GROUPS:
        a_in = any((g in p_rel) if len(g) > 1 else (g == p_rel) for g in group)
        b_in = any((g in g_rel) if len(g) > 1 else (g == g_rel) for g in group)
        if a_in and b_in:
            return True
    return False


def evaluate_triples_node(pp, predicted, gold, theta=0.9):
    n = pp._normalize_node_text_lenient
    mp, mg = {}, set()
    for i, p in enumerate(predicted):
        for j, g in enumerate(gold):
            if j in mg:
                continue
            if (n(p["source"]) == n(g["source"]) and n(p["target"]) == n(g["target"])
                    and rel_match(pp, p["relation"], g["relation"])):
                mp[i] = "normalized"
                mg.add(j)
                break
    rp = [i for i in range(len(predicted)) if i not in mp]
    rg = [j for j in range(len(gold)) if j not in mg]
    if rp and rg:
        nodes = list({n(predicted[i][k]) for i in rp for k in ("source", "target")}
                     | {n(gold[j][k]) for j in rg for k in ("source", "target")})
        model = pp._get_embed_model()
        E = dict(zip(nodes, model.encode(nodes, normalize_embeddings=True)))
        cands = []
        for i in rp:
            p = predicted[i]
            for j in rg:
                g = gold[j]
                if not rel_match(pp, p["relation"], g["relation"]):
                    continue
                sc = min(node_score(n(p["source"]), n(g["source"]), E, theta),
                         node_score(n(p["target"]), n(g["target"]), E, theta))
                if sc > 0:
                    cands.append((sc, i, j))
        cands.sort(key=lambda x: -x[0])
        for sc, i, j in cands:
            if i in mp or j in mg:
                continue
            mp[i] = "semantic"
            mg.add(j)
    tp = len(mp)
    P = tp / len(predicted) if predicted else 0.0
    R = tp / len(gold) if gold else 0.0
    return {
        "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0,
        "正解数": tp, "システム抽出数": len(predicted), "正解データ数": len(gold),
        "matched_pred": [predicted[i] for i in sorted(mp)],
        "unmatched_pred": [p for i, p in enumerate(predicted) if i not in mp],
        "unmatched_gold": [g for j, g in enumerate(gold) if j not in mg],
        "正解内訳": {"表記正規化一致": sum(1 for v in mp.values() if v == "normalized"),
                  "意味的類似度一致": sum(1 for v in mp.values() if v == "semantic")},
    }


def evaluate_triples_exact(pp, predicted, gold):
    """【主指標】トリプル完全一致（--eval-mode exact）。
    主語・目的語は表記正規化（_normalize_node_text_lenient：全角半角・「前記」・
    数量詞などの揺れを吸収）した上で完全一致、関係は rel_match（表記正規化・
    漢字部分一致・同義語グループ）で一致したものだけを正解とする。
    意味的類似度は使わない。"""
    n = pp._normalize_node_text_lenient
    mp, mg = set(), set()
    for i, p in enumerate(predicted):
        for j, g in enumerate(gold):
            if j in mg:
                continue
            if (n(p["source"]) == n(g["source"]) and n(p["target"]) == n(g["target"])
                    and rel_match(pp, p["relation"], g["relation"])):
                mp.add(i)
                mg.add(j)
                break
    tp = len(mp)
    P = tp / len(predicted) if predicted else 0.0
    R = tp / len(gold) if gold else 0.0
    return {
        "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0,
        "正解数": tp, "システム抽出数": len(predicted), "正解データ数": len(gold),
        "matched_pred": [predicted[i] for i in sorted(mp)],
        "unmatched_pred": [p for i, p in enumerate(predicted) if i not in mp],
        "unmatched_gold": [g for j, g in enumerate(gold) if j not in mg],
    }


# ---------------------------------------------------------------- 緩い一致（補助指標）
# 主指標はトリプル完全一致のまま。以下は規則だけで判定する補助指標で、埋め込みは使わない。
# 1つの正解には1つの抽出だけを対応させる（1対1）。
LOOSE_LEVELS = {1: "L1 ノードの部分一致", 2: "L2 関係名を問わない", 3: "L3 関係名・向きを問わない",
                4: "L4 部分一致＋関係名・向きを問わない"}


def _loose_node(a, b, partial):
    if a == b:
        return True
    if not partial:
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 2 and short in long_ and set(numbers(short)) <= set(numbers(long_))


def loose_match_count(pp, predicted, gold, level):
    """L1: 主語・目的語の片方がもう片方を含み（2文字以上）、番号が食い違わなければ一致（関係は完全一致と同じ）
    L2: 主語・目的語は完全一致、関係名は問わない　L3: 主語と目的語の組が向きを問わず完全一致
    L4: L1 と L3 を合わせたもの"""
    n = pp._normalize_node_text_lenient
    P = [(n(r["source"]), r["relation"], n(r["target"])) for r in predicted]
    G = [(n(r["source"]), r["relation"], n(r["target"])) for r in gold]
    partial = level in (1, 4)

    def ok(p, g):
        if level in (3, 4):
            return ((_loose_node(p[0], g[0], partial) and _loose_node(p[2], g[2], partial))
                    or (_loose_node(p[0], g[2], partial) and _loose_node(p[2], g[0], partial)))
        if not (_loose_node(p[0], g[0], partial) and _loose_node(p[2], g[2], partial)):
            return False
        return level == 2 or rel_match(pp, p[1], g[1])

    cands = [((p[0] == g[0]) + (p[2] == g[2]) + (p[1] == g[1]), i, j)
             for i, p in enumerate(P) for j, g in enumerate(G) if ok(p, g)]
    cands.sort(key=lambda x: -x[0])
    up, ug = set(), set()
    for _, i, j in cands:
        if i not in up and j not in ug:
            up.add(i)
            ug.add(j)
    return len(up)


# ---------------------------------------------------------------- 構造評価（完全一致と併用する2つ目の評価）
# 完全一致は「主語・関係・目的語がすべて正解と一致したSAO」を数える厳密な評価。構造評価は、
# 本研究で定めたノードの同一視の規則を正解と抽出の両方に同じようにかけてから、
#   構造F1        … 主語・目的語はノードの部分一致（「主面」と「放熱装置の主面」）、関係は完全一致と同じ規則
#   骨組みF1      … 主語・目的語はノードの部分一致、関係名と向きは問わない（どの部品とどの部品がつながるか）
#   要素別の一致率 … 正解の各SAOに最も近い抽出を1つずつ対応させ、主語・関係・目的語のそれぞれが合っている割合
# を出す。いずれも規則だけで判定し、埋め込み（意味の近さ）は使わない。
#
# ノードの同一視の規則（抽出の _merge_partitive_nodes・_merge_surface_location_nodes と同じ考え方）：
#   同じ請求項に「X」もあるとき、「Xの一部」「Xの部分」「Xの外側」「Xの内側」「Xの主面」「Xの第１主面」
#   「X上面」のような、X の部分・面・側を表す名前は X と同じノードとみなす。
IDENTITY_TAILS = ("一部", "部分", "外側", "内側", "主面", "上面", "下面", "表面", "裏面", "側面", "底面", "天面",
                  "端面", "内面", "外面", "接触面", "対向面", "外周面", "内周面")
_IDENTITY_REST_RE = re.compile(r"^の?(第?[0-9０-９一二三四五六七八九]+の?)?(" + "|".join(IDENTITY_TAILS) + r")$")


def identity_map(nodes):
    """{ノード: 同一視する相手のノード}。相手は同じ集合の中で一番長い前方一致のノード。"""
    nodes = set(nodes)
    out = {}
    for x in nodes:
        owners = [o for o in nodes if o != x and len(o) >= 2 and x.startswith(o)
                  and _IDENTITY_REST_RE.match(x[len(o):])]
        if owners:
            out[x] = max(owners, key=len)
    for x in list(out):  # 「Xの一部の表面」のように2段になっていても、最後の相手まで辿る
        seen = {x}
        while out[x] in out and out[x] not in seen:
            seen.add(out[x])
            out[x] = out[out[x]]
    return out


def canon_pair(pp, predicted, gold):
    """正解と抽出に、表記の正規化とノードの同一視を同じようにかける（自己ループは除き、重複はまとめる）。"""
    n = pp._normalize_node_text_lenient
    P = [(n(r["source"]), r["relation"], n(r["target"])) for r in predicted]
    G = [(n(r["source"]), r["relation"], n(r["target"])) for r in gold]
    m = identity_map({x for t in P + G for x in (t[0], t[2])})

    def canon(ts):
        out, seen = [], set()
        for s, r, t in ts:
            s2, t2 = m.get(s, s), m.get(t, t)
            if s2 != t2:  # 同一視すると自分自身への関係になる（「X｜有する｜Xの主面」）ときは、元の名前のまま比べる
                s, t = s2, t2
            if s == t or (s, r, t) in seen:
                continue
            seen.add((s, r, t))
            out.append((s, r, t))
        return out
    return canon(P), canon(G)


def _align(P, G, ok, score):
    cands = [(score(p, g), i, j) for i, p in enumerate(P) for j, g in enumerate(G) if ok(p, g)]
    cands.sort(key=lambda x: x[0], reverse=True)
    up, ug, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in up and j not in ug:
            up.add(i)
            ug.add(j)
            pairs.append((i, j))
    return pairs


def structure_counts(pp, predicted, gold):
    """構造評価の数を返す：{"n_pred", "n_gold", "struct", "skeleton", "comp_s", "comp_r", "comp_o"}
    （n_pred・n_gold は同一視の後の件数。struct・skeleton は1対1で対応した数。comp_* は正解のうち、対応した
    抽出と主語・関係・目的語がそれぞれ合っている数）"""
    P, G = canon_pair(pp, predicted, gold)

    def node(a, b):
        return _loose_node(a, b, True)

    def strict_score(p, g):
        return (p[0] == g[0]) + (p[2] == g[2]) + (p[1] == g[1])

    struct = _align(P, G, lambda p, g: node(p[0], g[0]) and node(p[2], g[2]) and rel_match(pp, p[1], g[1]),
                    strict_score)
    skeleton = _align(P, G, lambda p, g: (node(p[0], g[0]) and node(p[2], g[2]))
                      or (node(p[0], g[2]) and node(p[2], g[0])), strict_score)

    def parts(p, g):
        return node(p[0], g[0]), rel_match(pp, p[1], g[1]), node(p[2], g[2])

    comp = _align(P, G, lambda p, g: sum(parts(p, g)) >= 2,
                  lambda p, g: sum(parts(p, g)) * 10 + strict_score(p, g))
    cs = [parts(P[i], G[j]) for i, j in comp]
    return {"n_pred": len(P), "n_gold": len(G), "struct": len(struct), "skeleton": len(skeleton),
            "comp_s": sum(c[0] for c in cs), "comp_r": sum(c[1] for c in cs), "comp_o": sum(c[2] for c in cs)}


def prf_counts(tp, n_pred, n_gold):
    P = tp / n_pred if n_pred else 0.0
    R = tp / n_gold if n_gold else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


# ===========================================================================
# 【統合】nested_graph.py
# ===========================================================================
"""SAO関係を「構成（入れ子の箱）＋関係（矢印）」の1枚の図にするDOT生成。

・備える／有する等の所有関係 → 親の箱の中に子を入れる（入れ子）
・それ以外の関係（接続される・固定される等）→ 箱と箱をつなぐ矢印
・どの部品にも属さない語（方向・設置対象など）→ 装置の箱の外に点線の楕円
・同じ部品を複数の兄弟が持つ場合（第１ヘッダと第２ヘッダがそれぞれ壁部を持つ等）
  → それぞれの箱の中に複製して描く
・上位と下位の両方が同じ部品を持つ場合（題名と具体的な部品の二重所有）
  → より具体的な方（下位）の中だけに描く
"""

import html

HAS_RELATIONS = {"有する", "備える", "具備する", "含む", "含める", "の"}
FONT = "Noto Sans CJK JP,Yu Gothic,Meiryo,sans-serif"


RELATION_GROUP_NAMES = ["接続", "配置・位置", "覆う・収める", "動作・機能"]
RELATION_COLORS = [
    ("#2563eb", ("接続", "導通", "連結", "結合")),                                   # 接続
    ("#16a34a", ("配置", "設け", "位置", "形成", "積層", "搭載", "実装", "載置", "固定", "取り付",
                 "接合", "接着", "延在", "延び", "対向", "隣接", "間に", "上に", "下に")),      # 配置・位置
    ("#9333ea", ("覆", "封止", "収容", "収納", "囲", "挟", "埋め込", "貫通", "挿入")),       # 覆う・収める
    ("#ea580c", ("制御", "駆動", "供給", "出力", "入力", "検出", "流れ", "流す", "生成", "変換",
                 "スイッチング", "冷却", "放熱")),                                        # 動作・機能
]


def relation_color(rel, default="#475569"):
    """関係語の種類ごとの色（接続＝青、配置・位置＝緑、覆う・収める＝紫、動作・機能＝橙、その他＝灰）。"""
    for color, keys in RELATION_COLORS:
        if any(k in rel for k in keys):
            return color
    return "#475569"


def _esc(s):
    return html.escape(s).replace('"', '\\"')


def relation_group(rel):
    """関係語の種類（凡例・絞り込み用）。"""
    for (color, _), name in zip(RELATION_COLORS, RELATION_GROUP_NAMES):
        if color == relation_color(rel):
            return name
    return "その他"


def relations_to_nested_dot(relations, rel_color="#2563eb", direction="LR", focus=None, groups=None,
                            show_labels=True, merge=True):
    """focus: この部品に関わる矢印だけを描く（None ならすべて）。
    groups: 描く関係の種類（relation_group の名前の集合。None ならすべて）。
    merge: 同じ2つの箱をつなぐ矢印を1本にまとめる（関係名はまとめて表示）。"""
    rels = [r for r in relations if r["source"] != r["target"]]

    names = []
    for r in rels:
        for n in (r["source"], r["target"]):
            if n not in names:
                names.append(n)

    owners = {}
    for r in rels:
        if r["relation"] in HAS_RELATIONS:
            owners.setdefault(r["target"], [])
            if r["source"] not in owners[r["target"]]:
                owners[r["target"]].append(r["source"])

    def ancestors(n, seen=None):
        seen = set() if seen is None else seen
        out = set()
        for o in owners.get(n, []):
            if o in seen:
                continue
            seen.add(o)
            out.add(o)
            out |= ancestors(o, seen)
        return out

    # 他の所有者の祖先になっている所有者（題名などの上位）は外す
    direct_owners = {}
    for child, os_ in owners.items():
        keep = [o for o in os_
                if o != child and child not in ancestors(o)
                and not any(o in ancestors(p) for p in os_ if p != o)]
        if keep:
            direct_owners[child] = keep

    # インスタンス（描画上の箱）を作る。path = 根から自分までの名前の並び
    instances = {}  # name -> [path tuple]

    def build(name, stack=()):
        if name in stack:
            return []
        if name in instances:
            return instances[name]
        if name not in direct_owners:
            paths = [(name,)]
        else:
            paths = []
            for o in direct_owners[name]:
                for op in build(o, stack + (name,)):
                    paths.append(op + (name,))
            if not paths:
                paths = [(name,)]
        instances[name] = paths
        return paths

    for n in names:
        build(n)

    all_paths = [p for ps in instances.values() for p in ps]
    children = {}
    for p in all_paths:
        if len(p) > 1:
            children.setdefault(p[:-1], []).append(p)
    containers = set(children)
    ids = {p: f"n{i}" for i, p in enumerate(all_paths)}

    def anchor(p):
        return f"{ids[p]}_a" if p in containers else ids[p]

    lines = [
        "digraph SAO {",
        f'graph [compound=true, rankdir={direction}, newrank=true, nodesep=0.45, ranksep=0.9, splines=true, '
        f'fontname="{FONT}", fontsize=15, bgcolor="white", pad=0.3];',
        f'node [fontname="{FONT}", fontsize=14, shape=box, style="rounded,filled", margin="0.18,0.08", '
        'fillcolor="#ffffff", color="#94a3b8", penwidth=1.2];',
        f'edge [fontname="{FONT}", fontsize=12, arrowsize=0.8];',
    ]

    def emit(p, indent):
        pad = "  " * indent
        if p in containers:
            fill = ["#eef2ff", "#f0fdf4", "#fff7ed", "#ffffff"][min(len(p) - 1, 3)]
            lines.append(f"{pad}subgraph cluster_{ids[p]} {{")
            hot = focus and p[-1] == focus
            lines.append(f'{pad}  label=<<b>{html.escape(p[-1])}</b>>; labeljust=l; style="rounded,filled"; '
                         f'fillcolor="{"#fef9c3" if hot else fill}"; color="{"#ca8a04" if hot else "#64748b"}"; '
                         f'penwidth={2.4 if hot else 1.4}; margin=12;')
            lines.append(f'{pad}  {anchor(p)} [shape=point, width=0.01, style=invis, label=""];')
            for c in children[p]:
                emit(c, indent + 1)
            lines.append(f"{pad}}}")
        elif len(p) == 1:
            lines.append(f'{pad}{ids[p]} [label="{_esc(p[-1])}", shape=ellipse, style="dashed", '
                         'color="#94a3b8", fontcolor="#475569"];')
        elif focus and p[-1] == focus:
            lines.append(f'{pad}{ids[p]} [label="{_esc(p[-1])}", fillcolor="#fef9c3", color="#ca8a04", penwidth=2.4];')
        else:
            lines.append(f'{pad}{ids[p]} [label="{_esc(p[-1])}"];')

    for p in all_paths:
        if len(p) == 1:
            emit(p, 1)

    def closest(src_path, tgt_name):
        # 複製された相手のうち、根からの道筋を一番長く共有するものを選ぶ
        def shared(a, b):
            k = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                k += 1
            return k
        return max(instances[tgt_name], key=lambda q: shared(src_path, q))

    # 1つの関係は1本の矢印だけにし、同じ2つの箱をつなぐ矢印は1本にまとめる
    # （向きが両方あれば両矢印）。関係名は線の色（種類）と、まとめたラベルで示す。
    edges = {}
    order = []
    drawn = set()
    for r in rels:
        if r["relation"] in HAS_RELATIONS:
            continue
        key0 = (r["source"], r["relation"], r["target"])
        if key0 in drawn:
            continue
        drawn.add(key0)
        best = None
        for sp in instances[r["source"]]:
            tp = closest(sp, r["target"])
            if sp == tp:
                continue
            score = sum(1 for x, y in zip(sp, tp) if x == y)
            if best is None or score > best[0]:
                best = (score, sp, tp)
        if best is None:
            continue
        _, sp, tp = best
        # 自分を含む箱への矢印（箱の中の部品→外側の箱）は描くと箱の枠に刺さるだけなので省く
        if sp[:len(tp)] == tp or tp[:len(sp)] == sp:
            continue
        # 図の配置が変わらないように、絞り込みで外れた関係も「見えない線」「薄い線」として残す
        if groups is not None and relation_group(r["relation"]) not in groups:
            state = 0
        elif focus and focus not in (r["source"], r["target"]):
            state = 1
        else:
            state = 2
        k = (frozenset((sp, tp)), state) if merge else (sp, tp, r["relation"], state)
        if k not in edges:
            edges[k] = {"sp": sp, "tp": tp, "rels": [], "dirs": set(), "state": state}
            order.append(k)
        e = edges[k]
        if r["relation"] not in e["rels"]:
            e["rels"].append(r["relation"])
        e["dirs"].add((sp, tp))

    for k in order:
        e = edges[k]
        sp, tp = e["sp"], e["tp"]
        color = relation_color(e["rels"][0], rel_color)
        both = len(e["dirs"]) > 1
        shown = e["rels"][:2]
        text = "／".join(shown) + ("／他%d" % (len(e["rels"]) - 2) if len(e["rels"]) > 2 else "")
        many = sum(1 for x in edges.values() if x["state"] == 2) > 25
        if e["state"] == 0:
            a = ["style=invis"]
        elif e["state"] == 1:
            a = ['color="#cbd5e1"', "penwidth=0.8", "arrowsize=0.5"]
        else:
            # 矢印が多いときは少し透かして、重なった線の見分けをつきやすくする
            a = [f'color="{color}{"b3" if many and not focus else ""}"', "penwidth=1.4", "arrowsize=0.7"]
            if show_labels:
                a.append('label=<<table border="0" cellborder="0" cellpadding="1" cellspacing="0" bgcolor="white">'
                         f'<tr><td><font color="{color}" point-size="11">{html.escape(text)}</font></td></tr></table>>')
        a.append(f'tooltip="{_esc(sp[-1])} → {_esc(tp[-1])}: {_esc(" / ".join(e["rels"]))}"')
        if both:
            a.append("dir=both")
        if sp in containers:
            a.append(f"ltail=cluster_{ids[sp]}")
        if tp in containers:
            a.append(f"lhead=cluster_{ids[tp]}")
        lines.append(f"  {anchor(sp)} -> {anchor(tp)} [{', '.join(a)}];")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 元の形の関係図（深海テーマ）
# 箱の入れ子を使わず、すべての関係を「→ 関係」付きの矢印で描く、以前の見た目の図。
DEEPSEA_BG = "#071b2b"
DEEPSEA_NODE = {"fill": "#0E3A52", "border": "#5FD4E0", "font": "#E8FBFF"}
DEEPSEA_ROOT = {"fill": "#04121C", "border": "#8FE0F0", "font": "#FFFFFF"}
DEEPSEA_DIM = "#1f4a5a"


def relations_to_flat_dot(relations, title="構成要素間関係", direction="LR", focus=None, groups=None,
                          show_labels=True):
    """以前の「構成要素間関係図」と同じ見た目（濃紺の背景・発光する水色の角丸の箱・「→ 関係」の矢印）。
    focus を指定すると、その部品に関わる矢印以外を暗くする。groups で関係の種類を絞る（外れた矢印は描かない）。"""
    rels = []
    seen = set()
    for r in relations:
        k = (r["source"], r["relation"], r["target"])
        if r["source"] == r["target"] or k in seen:
            continue
        seen.add(k)
        rels.append(r)
    names = []
    indeg, outdeg = {}, {}
    for r in rels:
        for x in (r["source"], r["target"]):
            if x not in names:
                names.append(x)
        outdeg[r["source"]] = outdeg.get(r["source"], 0) + 1
        indeg[r["target"]] = indeg.get(r["target"], 0) + 1
    roots = [x for x in names if not indeg.get(x)]
    root = max(roots or names, key=lambda x: outdeg.get(x, 0)) if names else None
    ids = {x: f"n{i}" for i, x in enumerate(names)}
    lines = [
        "digraph SAO {",
        f'graph [rankdir={direction}, splines=spline, nodesep=0.3, ranksep=0.9, bgcolor="{DEEPSEA_BG}", pad=0.35, '
        f'fontname="{FONT}", fontsize=16, fontcolor="#E8FBFF", labelloc=t'
        + (f', label="{_esc(title)}"' if title else "") + "];",
        f'node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=13, margin="0.2,0.1", penwidth=2.2];',
        f'edge [fontname="{FONT}", fontsize=11, penwidth=1.8, arrowsize=0.9];',
    ]
    for x in names:
        s = DEEPSEA_ROOT if x == root else DEEPSEA_NODE
        extra = ', penwidth=3.4, color="#fde68a"' if focus and x == focus else f', color="{s["border"]}"'
        lines.append(f'  {ids[x]} [label="{_esc(x)}", fillcolor="{s["fill"]}", fontcolor="{s["font"]}"{extra}];')
    for r in rels:
        if groups is not None and relation_group(r["relation"]) not in groups:
            continue
        dim = focus and focus not in (r["source"], r["target"])
        color = DEEPSEA_DIM if dim else DEEPSEA_NODE["border"]
        a = [f'color="{color}"', f'tooltip="{_esc(r["source"])} → {_esc(r["relation"])} → {_esc(r["target"])}"']
        if show_labels and not dim:
            a.append(f'label="→ {_esc(r["relation"])}", fontcolor="{color}"')
        lines.append(f"  {ids[r['source']]} -> {ids[r['target']]} [{', '.join(a)}];")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 階層図（上位概念 → 下位概念の木）
TREE_STYLES = [
    {"fill": "#fde2ea", "border": "#e0527a", "font": "#6b0f2e", "role": "主語・上位概念"},
    {"fill": "#d9f5e8", "border": "#22a06b", "font": "#0f3d2c", "role": "下位概念"},
    {"fill": "#efe4fb", "border": "#9b6bd6", "font": "#3b1a66", "role": "さらに下位の概念"},
]
TREE_LINE = "#1e3a8a"
WRAP = 5  # 階層図で1段に並べる子の数（これを超えると次の段へ折り返す）


COMPONENT_PALETTE = [
    ("#dbeafe", "#2563eb", "#1e3a8a"), ("#dcfce7", "#16a34a", "#14532d"), ("#fee2e2", "#dc2626", "#7f1d1d"),
    ("#fef3c7", "#d97706", "#78350f"), ("#ede9fe", "#7c3aed", "#4c1d95"), ("#cffafe", "#0891b2", "#164e63"),
    ("#fce7f3", "#db2777", "#831843"), ("#ecfccb", "#65a30d", "#365314"), ("#ffedd5", "#ea580c", "#7c2d12"),
    ("#e0e7ff", "#4f46e5", "#312e81"), ("#ccfbf1", "#0d9488", "#134e4a"), ("#f5f5f4", "#57534e", "#292524"),
    ("#fae8ff", "#c026d3", "#701a75"), ("#e0f2fe", "#0284c7", "#0c4a6e"), ("#fef9c3", "#a16207", "#713f12"),
    ("#ffe4e6", "#e11d48", "#881337"), ("#d1fae5", "#059669", "#064e3b"), ("#f3e8ff", "#9333ea", "#581c87"),
    ("#e7e5e4", "#78716c", "#1c1917"), ("#fff7ed", "#c2410c", "#7c2d12"),
]


def component_colors(relations):
    """同じ構成要素には同じ色（塗り・枠・文字）を割り当てる。出てくる順に20色を使い回す。"""
    out = {}
    for r in relations:
        for x in (r["source"], r["target"]):
            if x not in out:
                f, b, t = COMPONENT_PALETTE[len(out) % len(COMPONENT_PALETTE)]
                out[x] = {"fill": f, "border": b, "font": t}
    return out


def _tree_levels(rels):
    """各ノードの階層（0＝一番上の上位概念）。どこからも矢印を受けないノードを上に置き、
    そこから矢印の向きにたどった段数を階層とする（輪になっている部分は、残りのうち矢印の多いノードから）。"""
    names, out, indeg = [], {}, {}
    for r in rels:
        for x in (r["source"], r["target"]):
            if x not in names:
                names.append(x)
        out.setdefault(r["source"], []).append(r["target"])
        indeg[r["target"]] = indeg.get(r["target"], 0) + 1
    level = {}
    roots = sorted([x for x in names if not indeg.get(x)], key=lambda x: -len(out.get(x, [])))
    while len(level) < len(names):
        if not roots:
            rest = [x for x in names if x not in level]
            roots = [max(rest, key=lambda x: len(out.get(x, [])))]
        frontier = [x for x in roots if x not in level]
        for x in frontier:
            level[x] = 0
        while frontier:
            nxt = []
            for u in frontier:
                for v in out.get(u, []):
                    if v not in level:
                        level[v] = level[u] + 1
                        nxt.append(v)
            frontier = nxt
        roots = []
    return names, level


def relations_to_tree_dot(relations, direction="TB", focus=None, groups=None, show_labels=True, show_roles=False,
                          show_cross=False, colors=None):
    """階層図：一番上に主語（上位概念）、その下に「関係」を挟んで目的語（下位概念）を並べる木の形の図。
    同じ主語・同じ関係の目的語はまとめて枝分かれさせる。階層を上に戻る関係や、同じ階層どうしの関係は、
    show_cross=True のときだけ点線で描く（既定では木の形を見やすくするため描かない）。"""
    rels, seen = [], set()
    for r in relations:
        k = (r["source"], r["relation"], r["target"])
        if r["source"] != r["target"] and k not in seen:
            seen.add(k)
            rels.append(r)
    if groups is not None:
        rels = [r for r in rels if relation_group(r["relation"]) in groups]
    names, level = _tree_levels(rels)
    ids = {x: f"n{i}" for i, x in enumerate(names)}
    lines = [
        "digraph SAO {",
        f'graph [rankdir={direction}, splines=ortho, nodesep=0.45, ranksep=0.42, bgcolor="white", pad=0.3, '
        f'fontname="{FONT}", newrank=true];',
        f'node [shape=box, style="rounded,filled", fontname="{FONT}", penwidth=2.0, margin="0.22,0.08"];',
        f'edge [color="{TREE_LINE}", penwidth=1.6, arrowsize=0.8, fontname="{FONT}"];',
    ]
    for x in names:
        s = dict(TREE_STYLES[min(level[x], 2)])
        if colors and x in colors:
            s.update(colors[x])  # 構成要素ごとの色（同じ部品は同じ色）
        hot = focus and x == focus
        role = (f'<br/><font point-size="10" color="{s["font"]}">（{s["role"]}）</font>' if show_roles else "")
        lines.append(f'  {ids[x]} [label=<<b><font point-size="15" color="{s["font"]}">{html.escape(x)}</font></b>{role}>, '
                     f'fillcolor="{s["fill"]}", color="{"#ca8a04" if hot else s["border"]}"'
                     f'{", penwidth=3.6" if hot else ""}];')
    # 下向きの関係は、主語と関係ごとに1つの「関係」ラベルを挟んで枝分かれさせる
    groups_down, others = {}, []
    for r in rels:
        if level[r["target"]] > level[r["source"]]:
            groups_down.setdefault((r["source"], r["relation"]), []).append(r["target"])
        else:
            others.append(r)
    for k, ((src, rel), tgts) in enumerate(groups_down.items()):
        dim = focus and focus != src and focus not in tgts
        color = "#cbd5e1" if dim else relation_color(rel, "#1d4ed8")
        line = "#cbd5e1" if dim else TREE_LINE
        rid = f"r{k}"
        lab = html.escape(rel) if show_labels else ""
        lines.append(f'  {rid} [shape=plaintext, style="", margin=0, label=<<b><font point-size="13" color="{color}">'
                     f'{lab}</font></b>>, width=0.1, height=0.1];')
        lines.append(f'  {ids[src]} -> {rid} [arrowhead=none, color="{line}"];')
        for t in tgts:
            lines.append(f'  {rid} -> {ids[t]} [color="{line}"];')
        # 子が多いときは、横に長くなりすぎないよう WRAP 個ずつ段を下げて折り返す（見えない線で下へ押す）
        for a in range(WRAP, len(tgts)):
            lines.append(f'  {ids[tgts[a - WRAP]]} -> {ids[tgts[a]]} [style=invis, weight=5];')
    for r in (others if show_cross else []):
        dim = focus and focus not in (r["source"], r["target"])
        color = "#cbd5e1" if dim else relation_color(r["relation"], "#475569")
        lab = f', xlabel=<<font point-size="11" color="{color}">{html.escape(r["relation"])}</font>>' if show_labels and not dim else ""
        lines.append(f'  {ids[r["source"]]} -> {ids[r["target"]]} [style=dashed, constraint=false, color="{color}"{lab}];')
    lines.append("}")
    return "\n".join(lines)


def relations_to_tree_html(relations, groups=None, max_depth=6, colors=None, font_size=".9rem"):
    """階層図と同じ構造を、字下げした文字の木で表す（「この図の構造」）。"""
    rels, seen = [], set()
    for r in relations:
        k = (r["source"], r["relation"], r["target"])
        if r["source"] != r["target"] and k not in seen:
            seen.add(k)
            rels.append(r)
    if groups is not None:
        rels = [r for r in rels if relation_group(r["relation"]) in groups]
    names, level = _tree_levels(rels)
    down = {}
    for r in rels:
        if level[r["target"]] > level[r["source"]]:
            down.setdefault(r["source"], {}).setdefault(r["relation"], []).append(r["target"])
    chip = ('<span style="background:{f};border:1px solid {b};color:{c};border-radius:5px;padding:1px 6px;'
            'font-weight:700">{t}</span>')
    out = []
    shown = set()

    def walk(x, depth):
        s = dict(TREE_STYLES[min(level[x], 2)])
        if colors and x in colors:
            s.update(colors[x])
        pad = depth * 1.4
        out.append(f'<div style="margin-left:{pad}em;margin-top:3px">{"└ " if depth else ""}'
                   + chip.format(f=s["fill"], b=s["border"], c=s["font"], t=html.escape(x))
                   + ('<span style="opacity:.6;font-size:.8em">（上に同じ）</span>' if x in shown else "") + "</div>")
        if x in shown or depth >= max_depth:
            return
        shown.add(x)
        for rel, tgts in down.get(x, {}).items():
            out.append(f'<div style="margin-left:{pad + 1.0}em;color:{relation_color(rel, "#1d4ed8")};font-weight:700">'
                       f'└ {html.escape(rel)}</div>')
            for t in tgts:
                walk(t, depth + 2)

    for x in names:
        if level[x] == 0:
            walk(x, 0)
    return f'<div style="font-size:{font_size};line-height:1.6">' + "".join(out) + "</div>"


def tree_cross_relations(relations, groups=None):
    """階層図で木の枝にならない関係（同じ階層どうし・上に戻る関係）の一覧。"""
    rels = [r for r in relations if r["source"] != r["target"]]
    if groups is not None:
        rels = [r for r in rels if relation_group(r["relation"]) in groups]
    _, level = _tree_levels(rels)
    return [r for r in rels if level[r["target"]] <= level[r["source"]]]


# ===========================================================================
# 【統合】claim_segmenter.py
# ===========================================================================
"""
claim_segmenter.py
===================
【実験10】手がかり句による請求項の分割と、区間ごとのGiNZA解析。

新森ら（2004）「手がかり句を用いた特許請求項の構造解析」の考え方に基づき、
長い請求項を記述断片に分割してから、区間ごとに GiNZA 単体版
（patent_pipeline.analyze_claim_ginza_only）で解析する。

分割の手がかり（新森ら 表2・表3 を元にした）:
  ・出願人が入れた改行（新森らの調査では、改行位置の87%が記述断片の境界）
  ・「において、」「に於いて、」「であって、」（ジェプソン的形式の区切り）
  ・「名詞＋と、」（構成要素列挙形式の区切り）
  ・「動詞・助動詞の連用形＋、」（順次列挙形式・「前記Xは、…を含み、」の区切り）

各区間は、末尾の「と、」「、」を「。」に置き換えて独立した文として解析する。
「を備え、」のような構成要素の列挙を締めくくるだけの区間は解析しない
（題名と構成要素の関係は、請求項全体のGiNZA解析で既に取れているため）。

オプション（実験10b）:
  distribute=True : 「前記A及び前記Bのそれぞれは、…」を「前記Aは、…」「前記Bは、…」
                    の2つの区間に展開してから解析する。
  merge_enzai=True: GiNZA（SudachiPy）が「延在する」を「延」＋「在す」に誤って分割する
                    問題を、解析の前に1語へ結合して防ぐ。
"""
import re
from contextlib import contextmanager

_JEPSON_RE = re.compile(r"(において|に於いて|に於て|であって|にあたり|に当たり)[、，]")
_COMPOSE_ONLY_RE = re.compile(r"^(と)?[、，]?(を|とを)?[、，]?(備え|具備し|有し|含み|設け)(る|た|ている|てなる)?[、，]?$")
_DISTRIB_RE = re.compile(
    r"^(?P<coord>.+?)の(?P<each>それぞれ|各々|各)(?P<part>は|が|に|を|には|にも|も)(?P<rest>.*)$")
_NEW_TOPIC_RE = re.compile(r"^\s*(前記|該|当該|上記)?[^、。，]{1,40}?(は|が)[、，]")
_COORD_SPLIT_RE = re.compile(r"、|，|及び|および|並びに|ならびに|と")


def _split_line(pp, line):
    """1行を、GiNZAの品詞情報を使って手がかり句で区間に分ける。"""
    line = line.strip()
    if not line:
        return []
    doc = pp.nlp(line)
    cuts = []
    toks = list(doc)
    for i, t in enumerate(toks):
        if t.text not in ("、", "，") or i == 0 or i == len(toks) - 1:
            continue
        prev = toks[i - 1]
        infl = "".join(prev.morph.get("Inflection"))
        if prev.pos_ in ("VERB", "AUX") and "連用形" in infl:
            # 「〜設けられ、〜流路を有する多穴管と」のような修飾部の途中では切らない。
            # 新しい主題・主語（「前記Yは、」等）が続く場合だけ区切りとみなす。
            if _NEW_TOPIC_RE.match(line[t.idx + len(t.text):]):
                cuts.append(t.idx + len(t.text))
        elif prev.text == "と" and prev.pos_ in ("ADP", "CCONJ") and i >= 2 and toks[i - 2].pos_ in ("NOUN", "PROPN", "NUM", "SYM"):
            # 「AとBとの間に位置する」のような並列は構成要素の区切りではないので切らない
            if "との" not in line[t.idx + len(t.text):]:
                cuts.append(t.idx + len(t.text))
    for m in _JEPSON_RE.finditer(line):
        cuts.append(m.end())
    cuts = sorted(set(c for c in cuts if 0 < c < len(line)))
    parts, start = [], 0
    for c in cuts:
        parts.append(line[start:c])
        start = c
    parts.append(line[start:])
    return [p.strip() for p in parts if p.strip()]


def split_claim(pp, text):
    segs = []
    for line in re.split(r"[\r\n]+", text):
        segs.extend(_split_line(pp, line))
    return segs


def to_sentence(seg):
    """区間を独立した文にする。構成要素の列挙を締めくくるだけの区間はNone。"""
    s = seg.strip()
    if _COMPOSE_ONLY_RE.match(s):
        return None
    s = re.sub(r"[。．]$", "", s)
    s = _JEPSON_RE.sub(lambda m: "", s) if _JEPSON_RE.search(s[-6:] if len(s) > 6 else s) else s
    s = re.sub(r"とを?[、，]?$", "", s)
    s = re.sub(r"[、，]$", "", s)
    s = s.strip()
    if len(s) < 2:
        return None
    return s + "。"


def distribute(sent):
    """「前記A及び前記Bのそれぞれは、…」→「前記Aは、…」「前記Bは、…」"""
    m = _DISTRIB_RE.match(sent)
    if not m:
        return [sent]
    items = [x.strip() for x in _COORD_SPLIT_RE.split(m.group("coord")) if x.strip()]
    if len(items) < 2 or any(len(x) > 40 for x in items):
        return [sent]
    return [f"{x}{m.group('part')}{m.group('rest')}" for x in items]


_ENZAI_NAME = "exp10_merge_enzai"


def _ensure_enzai_component(pp):
    from spacy.language import Language

    if _ENZAI_NAME not in Language.factories:
        @Language.component(_ENZAI_NAME)
        def _merge(doc):
            with doc.retokenize() as rt:
                for i in range(len(doc) - 1):
                    if doc[i].text == "延" and doc[i + 1].text.startswith("在す"):
                        rt.merge(doc[i:i + 2], attrs={"POS": "VERB", "LEMMA": "延在する"})
            return doc
    if _ENZAI_NAME not in pp.nlp.pipe_names and _ENZAI_NAME not in pp.nlp.disabled:
        pp.nlp.add_pipe(_ENZAI_NAME, first=True)
        pp.nlp.disable_pipe(_ENZAI_NAME)


@contextmanager
def _enzai(pp, on):
    if not on:
        yield
        return
    _ensure_enzai_component(pp)
    pp.nlp.enable_pipe(_ENZAI_NAME)
    try:
        yield
    finally:
        pp.nlp.disable_pipe(_ENZAI_NAME)


def segment_relations(pp, text, distribute_each=False, merge_enzai=False):
    """区間ごとにGiNZA単体版で解析した関係を返す（type に区間解析の種類を付ける）。"""
    out = []
    with _enzai(pp, merge_enzai):
        for seg in split_claim(pp, text):
            sent = to_sentence(seg)
            if sent is None:
                continue
            sents = distribute(sent) if distribute_each else [sent]
            for s in sents:
                try:
                    _, rels = pp.analyze_claim_ginza_only(s)
                except Exception:  # noqa: BLE001
                    continue
                for r in rels:
                    r = dict(r)
                    r["seg_type"] = r.get("type", "?")
                    r["distributed"] = len(sents) > 1
                    out.append(r)
    return out


# ===========================================================================
# 【統合】dep_pairs.py
# ===========================================================================
"""
dep_pairs.py
=============
【実験12】係り受けに基づく汎用の候補生成（Open IE 型）。

特定の書き方ごとの規則を足すのではなく、「述語（動詞・サ変名詞＋する・
形容詞）にかかる構成要素どうしは関係を持ちうる」という一般原則だけで候補を作る。
どの組を・どの向きで採るかは、格助詞（が・は・を・に・で・と・から・より…）、
受身かどうか、連体修飾かどうか、といった言語的な手がかりを特徴量として
選別モデルに学習させる（ClausIE 等の節ベース Open IE と同じ考え方）。

対象: 請求項全体と、手がかり句で分割した各区間（claim_segmenter）。
述語の項:
  ・述語に直接かかる名詞（nsubj / obj / obl / iobj / nmod）
  ・連体修飾（acl）の場合は、修飾される名詞
  ・連用形・て形で続く述語（conj / advcl）は、主題（「前記Xは、」）を引き継ぐ
"""
import re

pass  # （統合済み）import claim_segmenter as CS

ARG_DEPS = {"nsubj", "obj", "obl", "iobj", "nmod", "nsubj:pass", "csubj"}
CASES = ["が", "は", "を", "に", "で", "と", "から", "より", "へ", "の", "まで", "によって", "により", "に対して", "として", "間"]


def _case_of(tok):
    ps = [c.text for c in tok.children if c.dep_ in ("case", "mark") and c.i > tok.i]
    s = "".join(ps)
    for c in CASES:
        if s.startswith(c) or s == c:
            return c
    if any(c.text == "間" for c in tok.children) or any(c.text in ("間", "との間") for c in tok.head.children if c.i > tok.i and c.i < tok.head.i):
        return "間"
    return s or "-"


def _is_pred(t):
    if t.pos_ in ("VERB", "ADJ"):
        return True
    if t.pos_ == "NOUN" and any(c.lemma_ == "する" or c.text in ("さ", "し", "する", "され", "される") for c in t.children if c.dep_ in ("aux", "cop")):
        return True
    return False


def _label(pp, t):
    lab = pp._relation_label(t)
    passive = bool(re.search(r"(れ|られ|され)(る|た|て|ており|ている)?$", lab)) or any(
        c.lemma_ in ("れる", "られる") for c in t.children if c.dep_ == "aux")
    lab = re.sub(r"(ており|ている|ていた|て|た|、)$", "", lab)
    if t.pos_ == "NOUN" and not lab.endswith(("する", "される", "され", "し")):
        lab += "される" if passive else "する"
    return lab, passive


def _comp_map(pp, doc):
    comps = pp.extract_patent_components_general(doc)
    m = {}
    for c in comps:
        for i in range(c["start"], c["end"] + 1):
            m[i] = c
    return m


def _component_of(tok, cmap):
    c = cmap.get(tok.i)
    return c["text"] if c else None


def _coordinated(tok, depth=0):
    """「A及びB」「AとBと」のような並列を展開する（GiNZAでは nmod/conj＋cc で表される）。"""
    out = [tok]
    if depth > 2:
        return out
    has_cc = any(c.dep_ == "cc" for c in tok.children)
    for x in tok.children:
        if x.dep_ in ("nmod", "conj") and (has_cc or any(c.dep_ == "cc" for c in x.children)
                                            or _case_of(x) == "と" or x.dep_ == "conj"):
            out += _coordinated(x, depth + 1)
    return out


def pairs_from_doc(pp, doc, scope):
    cmap = _comp_map(pp, doc)
    out = []
    topic = None
    for t in doc:
        if not _is_pred(t):
            continue
        args = []
        for c in t.children:
            if c.dep_ in ARG_DEPS:
                case = _case_of(c)
                for k, x in enumerate(_coordinated(c)):
                    name = _component_of(x, cmap)
                    if name:
                        args.append((name, case, c.dep_ + ("" if k == 0 else "+並列"), x.i))
        if t.dep_ == "acl":
            for k, x in enumerate(_coordinated(t.head)):
                name = _component_of(x, cmap)
                if name:
                    args.append((name, "連体", "acl_head" + ("" if k == 0 else "+並列"), x.i))
        tops = [a for a in args if a[1] == "は"]
        if tops:
            topic = tops[0]
        elif topic is not None and t.dep_ in ("conj", "advcl", "ROOT") and not any(a[1] in ("が",) for a in args):
            if topic[0] not in [a[0] for a in args]:
                args.append((topic[0], "は(継承)", "topic", topic[3]))
        seen, uniq = set(), []
        for a in args:
            if a[0] not in seen:
                seen.add(a[0])
                uniq.append(a)
        if len(uniq) < 2 or len(uniq) > 6:
            continue
        lab, passive = _label(pp, t)
        for a in uniq:
            for b in uniq:
                if a[0] == b[0]:
                    continue
                out.append({
                    "source": a[0], "relation": lab, "target": b[0], "scope": scope,
                    "src_case": a[1], "tgt_case": b[1], "src_dep": a[2], "tgt_dep": b[2],
                    "passive": passive, "dist": abs(a[3] - b[3]), "src_before": a[3] < b[3],
                    "n_args": len(uniq), "pred_pos": t.pos_,
                })
    return out


def dependency_pairs(pp, text):
    with _enzai(pp, True):
        return _dependency_pairs(pp, text)


def _dependency_pairs(pp, text):
    res = []
    try:
        res += pairs_from_doc(pp, pp.nlp(pp._clean_claim_text(text)), "全体")
    except Exception:  # noqa: BLE001
        pass
    for seg in split_claim(pp, text):
        sent = to_sentence(seg)
        if sent is None:
            continue
        for s in distribute(sent):
            try:
                res += pairs_from_doc(pp, pp.nlp(pp._clean_claim_text(s)), "区間")
            except Exception:  # noqa: BLE001
                continue
    return res


# ===========================================================================
# 【統合】seg_pairs.py
# 名前の付け替え: HAS_VERBS → HAS_VERBS_sp
# ===========================================================================
"""
seg_pairs.py
=============
【実験13】区間の中の「主役」と他の構成要素を組にする候補生成。

実験12の誤り分析では、正解のSAOの37%がどの候補にも入っていなかった。そのうち最も多いのは
「両方のノードは候補にあるのに、その組がどの方法からも出ていない」もの（正解の13.5%）で、
その約7割は、同じ構成要素の説明区間（「…を有する第１ヘッダと、」のような、手がかり句で
区切った1つの区間）の中に両方のノードが出てくる。係り受けの経路が長くなる（6以上）ため、
述語の直接の項だけを組にする実験12の方法では拾えない。

そこで、係り受けの経路には頼らず、次の一般的な規則で候補を作る。
  ・区間ごとに「主役」を決める：区間の最後に出てくる構成要素（「…Ｘと、」のＸ）と、
    区間の先頭の主題（「前記Ｘは、…」のＸ）。
  ・主役と、同じ区間に出てくる他の構成要素とを、区間の中の各述語で結んだ候補を、両方の
    向きで作る。
  ・どの組・どの述語・どの向きを採るかは、選別モデルが特徴量（述語との距離、係り受けの
    経路の長さ、述語が2つの間にあるか後ろにあるか等）から判断する。
"""
import re

pass  # （統合済み）import claim_segmenter as CS
pass  # （統合済み）import dep_pairs as D


def _tree_dist(a, b):
    def anc(t):
        out = [t]
        while t.head.i != t.i:
            t = t.head
            out.append(t)
        return out
    A, B = anc(a), anc(b)
    pos = {t.i: k for k, t in enumerate(B)}
    for k, t in enumerate(A):
        if t.i in pos:
            return k + pos[t.i]
    return 99


def pairs_from_segment(pp, doc, scope="区間"):
    cmap = _comp_map(pp, doc)
    comps = {}
    for c in cmap.values():
        comps[(c["start"], c["end"])] = c
    comps = [comps[k] for k in sorted(comps)]
    if len(comps) < 2:
        return []
    preds = [t for t in doc if _is_pred(t)]
    if not preds:
        return []
    # 主役：区間の最後の構成要素、と、先頭の主題（「Ｘは、」「Ｘが、」）
    heads = {comps[-1]["text"]: comps[-1]}
    first = comps[0]
    after = [t.text for t in doc[first["end"] + 1:first["end"] + 3]]
    if after and after[0] in ("は", "が"):
        heads[first["text"]] = first
    out = []
    for hname, h in heads.items():
        ht = doc[h["end"]]
        for c in comps:
            if c["text"] == hname:
                continue
            ct = doc[c["end"]]
            for p in preds:
                lab, passive = _label(pp, p)
                if not lab or len(lab) > 12:
                    continue
                lo, hi = min(h["end"], c["end"]), max(h["end"], c["end"])
                info = {"scope": scope, "passive": passive, "pred_pos": p.pos_,
                        "tree_h": _tree_dist(ht, p), "tree_c": _tree_dist(ct, p),
                        "between": lo < p.i < hi, "after": p.i > hi, "gap": abs(p.i - ct.i),
                        "n_comps": len(comps), "n_preds": len(preds), "head_last": h is comps[-1]}
                out.append(dict(info, source=hname, relation=lab, target=c["text"], head_is="source"))
                out.append(dict(info, source=c["text"], relation=lab, target=hname, head_is="target"))
    return out


HAS_VERBS_sp = [("備え", "備える"), ("有し", "有する"), ("有する", "有する"), ("含み", "含む"), ("含む", "含む"),
             ("具備", "具備する"), ("から構成", "構成される"), ("からなる", "からなる")]


def segment_pairs(pp, text, title=None):
    """各区間の主役どうしの候補に加え、題名（請求項の最後の構成要素）が、列挙された各区間の
    主役を「備える／有する／含む…」で持つ候補も作る（構成要素列挙形式の骨格）。"""
    res, heads = [], []
    with _enzai(pp, True):
        for seg in split_claim(pp, text):
            sent = to_sentence(seg)
            if sent is None:
                continue
            for s in distribute(sent):
                try:
                    doc = pp.nlp(pp._clean_claim_text(s))
                    res += pairs_from_segment(pp, doc)
                    cmap = _comp_map(pp, doc)
                    if cmap:
                        last = max(cmap.values(), key=lambda c: c["end"])
                        heads.append(last["text"])
                except Exception:  # noqa: BLE001
                    continue
    if title:
        clean = pp._clean_claim_text(text)
        verbs = [v for stem, v in HAS_VERBS_sp if stem in clean]
        for k, h in enumerate(dict.fromkeys(heads)):
            if h == title:
                continue
            for v in dict.fromkeys(verbs):
                res.append({"scope": "題名", "passive": False, "pred_pos": "VERB", "tree_h": -1, "tree_c": -1,
                            "between": False, "after": True, "gap": k, "n_comps": len(heads), "n_preds": len(verbs),
                            "head_last": True, "source": title, "relation": v, "target": h, "head_is": "source"})
    return res


# ===========================================================================
# 【統合】sao_selector.py
# 名前の付け替え: build_candidates → build_candidates9, claim_features → claim_features9
# ===========================================================================
"""
sao_selector.py
================
候補選別モデル（実験9）。

LLM直接抽出・GiNZA補完・GiNZA単体版の出力をすべて「候補」として集め、
候補の関係ごとに「正解である確率」を学習済みモデルで予測して、
確率がしきい値以上のものだけを採用する。同じ2つの構成要素の組（向きを
問わない）からは、確率が最も高い1件だけを採る。

実験2〜8では「このtypeのこの関係語はLLMに再確認させる」「この語が目的語
なら消す」といったルールを1つずつ足してきたが、ここではそれらの判断材料
（抽出元、関係語の種類、fan-out、題名かどうか、本文中の位置など）を
特徴量としてまとめて与え、どう組み合わせるかは学習に任せる。
LLMの呼び出しは抽出の1回だけ（実験2〜4の検証LLMは使わない）。

学習データ（sao_selector_train.npz）は、532件の正解データから作った
候補ごとの特徴量とラベル。評価時は交差検証の分割ごとに、その分割を除いた
請求項だけで学習したモデルを使う（学習に使った請求項で評価しない）。
"""
import re
from collections import Counter
import pathlib as _pathlib

import numpy as np

HERE = _pathlib.Path(__file__).resolve().parent

INVALID = {"複数", "互い", "こと", "もの", "場合", "状態", "様子", "全体", "一部", "両方", "それぞれ",
           "いずれか", "各々", "これ", "それ", "あれ", "ここ", "そこ", "一方", "他方",
           "それぞれの一方", "それぞれの他方", "少なくとも", "前記", "所定"}
SIMPLIFIED = set("收离构设为们这个从与际图电气压热导线发开关实现应该总数据还样进过对时")
SRC_KEYS = ["E1:llm_direct", "E1:claim_title_ginza", "E1:claim_title_ginza_conflict",
            "E1:ginza_has_fallback", "E1:ginza_has_fallback_conflict", "E1:attribute",
            "G:direct", "G:positional", "G:has", "LLMraw", "REV", "VERB"]
FORMATS = ["順次列挙形式", "構成要素列挙形式", "ジェプソン的形式"]
# LLMが関係語に中国語の簡体字を混ぜることがある（例：收容する）ので日本の字体に直す
SIMP = str.maketrans({"收": "収", "离": "離", "构": "構", "并": "並", "设": "設", "为": "為", "电": "電",
                      "气": "気", "压": "圧", "热": "熱", "导": "導", "线": "線", "发": "発", "开": "開",
                      "关": "関", "实": "実", "现": "現", "应": "応", "总": "総", "进": "進", "过": "過",
                      "对": "対", "时": "時", "连": "連", "结": "結", "层": "層", "极": "極", "块": "塊",
                      "从": "従", "个": "個", "际": "際", "图": "図", "样": "様", "动": "動", "驱": "駆",
                      "边": "辺", "侧": "側", "间": "間", "备": "備", "装": "装", "据": "拠", "变": "変",
                      "绝": "絶", "缘": "縁", "传": "伝", "递": "逓", "输": "輸", "给": "給", "调": "調",
                      "节": "節", "测": "測", "检": "検", "术": "術", "处": "処", "理": "理", "盖": "蓋",
                      "载": "載", "阳": "陽", "阴": "陰", "储": "貯", "续": "続", "转": "転", "换": "換"})
KEPT_INDEX = len(SRC_KEYS)  # 実験4の検証LLMを通ったかどうか（本モデルでは常に0として使う）


def rel_group(pp, rel):
    for i, group in enumerate(pp.RELATION_SYNONYM_GROUPS):
        if any((g in rel) if len(g) > 1 else (g == rel) for g in group):
            return i
    return -1


def claim_features9(pp, info):
    """info = {"cands": [...], "tags": [...], "title": str|None, "cleaned": str, "format": str}"""
    n = pp._normalize_node_text_lenient
    surface = set(getattr(pp, "_SURFACE_LOCATION_WORDS", set()))
    n_groups = len(pp.RELATION_SYNONYM_GROUPS)
    cands = info["cands"]
    title = info["title"]
    tags = set(info["tags"])
    text = info["cleaned"]
    out_deg = Counter(c["source"] for c in cands)
    in_deg = Counter(c["target"] for c in cands)
    owners = Counter(c["target"] for c in cands if rel_group(pp, c["relation"]) == 0)
    pairs = Counter((c["source"], c["target"]) for c in cands)
    llm_pairs = {(c["source"], c["target"]) for c in cands
                 if "E1:llm_direct" in c["srcs"] or "LLMraw" in c["srcs"]}
    g_pairs = {(c["source"], c["target"]) for c in cands if any(s.startswith("G:") for s in c["srcs"])}
    rows = []
    for c in cands:
        s, r, t = c["source"], c["relation"], c["target"]
        f = [float(k in c["srcs"]) for k in SRC_KEYS]
        f += [float(c.get("kept", False)), float(len(c["srcs"]))]
        f += [float((s, t) in llm_pairs), float((s, t) in g_pairs), float((t, s) in llm_pairs),
              float((t, s) in g_pairs), float(pairs[(s, t)] - 1), float((t, s) in pairs)]
        g = rel_group(pp, r)
        f += [float(g == i) for i in range(-1, n_groups)]
        f += [float(len(r)), float("《" in r), float(r == "の"),
              float(bool(re.search(r"(される|られる|され|られ)$", r))),
              float(any(ch in SIMPLIFIED for ch in r))]
        for x in (s, t):
            f += [float(x == title), float(x in tags), float(len(x)), float("の" in x),
                  float(bool(re.search(r"\d|[０-９]", x))), float(x in INVALID or n(x) in INVALID),
                  float(x in surface), float(x in text)]
        f += [float(out_deg[s]), float(in_deg[t]), float(owners[t]), float(out_deg[t]), float(in_deg[s])]
        ps, pt = text.find(s), text.find(t)
        f += [float(ps), float(pt), float(abs(ps - pt) if ps >= 0 and pt >= 0 else -1),
              float(ps < pt if ps >= 0 and pt >= 0 else -1), float(len(text))]
        f += [float(len(tags)), float(len(cands))] + [float(info["format"] == x) for x in FORMATS]
        rows.append(f)
    X = np.array(rows, dtype=float) if rows else np.zeros((0, 1))
    if len(X):
        X[:, KEPT_INDEX] = 0.0
    return X


def build_candidates9(ts, pp, text, llm_output=None, llm_cache=None, claim_id=None,
                     model=None, host=None):
    """候補を集める。LLMの呼び出しは抽出の1回だけ（llm_cacheがあればそれを使う）。"""
    tagged_text, t2t, tag_doc, tag_comps = tag_components(text, pp)
    debug_out = {}
    kw = {} if model is None else {"model": model}
    if llm_output is not None:
        llm_cache = {"_": {"tagged_text_hash": _hash_text(tagged_text), "llm_output": llm_output}}
        claim_id = "_"
    _, e1 = analyze_claim_llm_direct(text, pp=pp, host=host, llm_cache=llm_cache,
                                        claim_id=claim_id, debug_out=debug_out, **kw)
    raw = extract_relations_from_llm_output(debug_out.get("llm_output", ""), t2t)
    try:
        _, g = pp.analyze_claim_ginza_only(text)
    except Exception:  # noqa: BLE001
        g = []
    cands = {}

    def add(r, src):
        rel = r["relation"].translate(SIMP)
        key = (r["source"], rel, r["target"])
        c = cands.setdefault(key, {"source": key[0], "relation": rel, "target": key[2],
                                   "srcs": [], "type": r.get("type", src)})
        if src not in c["srcs"]:
            c["srcs"].append(src)

    for r in e1:
        add(r, "E1:" + r["type"])
    for r in g:
        add(r, "G:" + r.get("type", "?"))
    for r in raw:
        add(r, "LLMraw")
    last_i = len(tag_doc) - 1
    while last_i > 0 and tag_doc[last_i].pos_ == "PUNCT":
        last_i -= 1
    tc = pp.find_component_by_token(tag_comps, last_i)
    return {"cands": list(cands.values()), "tags": sorted(set(t2t.values())),
            "title": tc["text"] if tc is not None else None,
            "cleaned": pp._clean_claim_text(text), "format": pp.classify_claim_format(text)}








def cv_folds(texts_by_id, n_folds=5, seed=42):
    """本文が同じ請求項を同じ分割に入れた、交差検証の分割（学習時と同じ分け方）。"""
    import random
    groups = {}
    for cid, text in texts_by_id.items():
        groups.setdefault("".join(text.split()), []).append(cid)
    glist = list(groups.values())
    random.Random(seed).shuffle(glist)
    return [[c for g in glist[i::n_folds] for c in g] for i in range(n_folds)]


# ===========================================================================
# 【統合】sao_selector10.py
# 名前の付け替え: build_candidates → build_candidates10, claim_features → claim_features10
# ===========================================================================
"""
sao_selector10.py
==================
【実験10】実験9（sao_selector.py）の候補に、手がかり句で分割した区間ごとの
GiNZA解析（claim_segmenter.py）の結果を追加した候補選別モデル。
実験9のコード・学習データには一切手を加えず、候補と特徴量を足すだけにしている。
"""
import pathlib as _pathlib

import numpy as np

pass  # （統合済み）import claim_segmenter as CS
pass  # （統合済み）import sao_selector as S

HERE = _pathlib.Path(__file__).resolve().parent
SEG_KEYS = ["GS:direct", "GS:positional", "GS:has", "GS:distrib"]


def build_candidates10(ts, pp, text, variant="a", **kw):
    """variant="a"（採用）: 分割のみ。"b": 分割＋「それぞれ」の展開＋「延在」の結合（効果なし）。"""
    info = build_candidates9(ts, pp, text, **kw)
    rels = segment_relations(pp, text, distribute_each=(variant == "b"), merge_enzai=(variant == "b"))
    index = {(c["source"], c["relation"], c["target"]): c for c in info["cands"]}
    for r in rels:
        rel = r["relation"].translate(SIMP)
        key = (r["source"], rel, r["target"])
        c = index.get(key)
        if c is None:
            c = {"source": key[0], "relation": rel, "target": key[2], "srcs": [], "type": "GS:" + r["seg_type"]}
            info["cands"].append(c)
            index[key] = c
        for src in ["GS:" + r["seg_type"]] + (["GS:distrib"] if r.get("distributed") else []):
            if src not in c["srcs"]:
                c["srcs"].append(src)
    return info


def claim_features10(pp, info):
    X = claim_features9(pp, info)
    if not len(info["cands"]):
        return X
    cands = info["cands"]
    gs_pairs = {(c["source"], c["target"]) for c in cands if any(s.startswith("GS:") for s in c["srcs"])}
    extra = []
    for c in cands:
        f = [float(k in c["srcs"]) for k in SEG_KEYS]
        f += [float((c["source"], c["target"]) in gs_pairs), float((c["target"], c["source"]) in gs_pairs),
              float(sum(s.startswith("GS:") for s in c["srcs"]))]
        extra.append(f)
    return np.hstack([X, np.array(extra, dtype=float)])


# ===========================================================================
# 【統合】sao_selector11.py
# 名前の付け替え: build_candidates → build_candidates11, claim_features → claim_features11
# ===========================================================================
"""
sao_selector11.py
==================
【実験11】ノードの粒度（「XのY」の結合）を、候補として選べるようにしたもの。

532件の正解データの分析（「XのY」でX・Yともにタグになっている1,341箇所）:
  ・正解で「XのY」が1つのノードになっている：526箇所
  ・正解ではYが単独のノードになっている　　：約310箇所
  → 無条件に結合すると約4割で誤るため、結合は規則で決めず「結合した候補」を
    追加して、実験9と同じ選別モデルにどちらを採るか判断させる。
  ・Yが本文で常に「〜のY」の形でしか出てこない場合は結合が約9割、単独でも
    出てくる場合は約8割（この情報を特徴量として与える）。
  ・正解に結合ノードがある場合、「X の XのY」という関係も約4分の1で付いている
    ので、その候補も追加する。

同じ箇所の「Y」と「XのY」は、どちらか一方しか採らない（選別時に同じ組とみなす）。

学習ラベルは主指標と同じ「トリプル完全一致」（node_match_eval.evaluate_triples_exact
と同じ基準）で付けている。主語・目的語ごとの意味的類似度（包含を一致とみなす）で
ラベルを付けると「Y」と「XのY」が同じ扱いになり、結合を学習できない（その条件では
F1が1.4pt下がった）。
実験9・10のコードと学習データには手を加えない。
"""
import re
import pathlib as _pathlib

import numpy as np

pass  # （統合済み）import sao_selector as S
pass  # （統合済み）import sao_selector10 as S10

HERE = _pathlib.Path(__file__).resolve().parent


def merge_occurrences(info):
    """本文中の「XのY」（X・Yともにタグ）を探し、Y→[X, ...] を返す。"""
    text = info["cleaned"]
    tags = sorted(set(info["tags"]), key=len, reverse=True)
    owners = {}
    for X in tags:
        for Y in tags:
            if X != Y and (X + "の" + Y) in text:
                owners.setdefault(Y, [])
                if X not in owners[Y]:
                    owners[Y].append(X)
    return owners


def build_candidates11(ts, pp, text, **kw):
    info = build_candidates10(ts, pp, text, variant="a", **kw)
    owners = merge_occurrences(info)
    text_c = info["cleaned"]
    index = {(c["source"], c["relation"], c["target"]): c for c in info["cands"]}
    base = {}

    def add(src, rel, tgt, srcs, bsrc, btgt):
        key = (src, rel, tgt)
        if key in index:
            c = index[key]
            for s in srcs:
                if s not in c["srcs"]:
                    c["srcs"].append(s)
            return
        c = {"source": src, "relation": rel, "target": tgt, "srcs": list(srcs), "type": "MRG"}
        info["cands"].append(c)
        index[key] = c
        base[key] = (bsrc, btgt)

    for c in list(info["cands"]):
        for side in ("source", "target"):
            Y = c[side]
            for X in owners.get(Y, []):
                M = X + "の" + Y
                other = c["target"] if side == "source" else c["source"]
                if other in (X, M):
                    continue
                new = dict(source=c["source"], target=c["target"])
                new[side] = M
                add(new["source"], c["relation"], new["target"],
                    [s for s in c["srcs"]] + ["MRG"], c["source"], c["target"])
    for Y, xs in owners.items():
        for X in xs:
            add(X, "の", X + "の" + Y, ["MRG:link"], X, Y)
    info["merge_owners"] = owners
    info["merge_base"] = {"|".join(k): list(v) for k, v in base.items()}
    info["y_always_no"] = {Y: (text_c.count(Y) == text_c.count("の" + Y)) for Y in owners}
    return info


def claim_features11(pp, info):
    X = claim_features10(pp, info)
    if not len(info["cands"]):
        return X
    owners = info.get("merge_owners", {})
    merged_nodes = {X_ + "の" + Y: Y for Y, xs in owners.items() for X_ in xs}
    extra = []
    for c in info["cands"]:
        ms = [merged_nodes[x] for x in (c["source"], c["target"]) if x in merged_nodes]
        ys = [x for x in (c["source"], c["target"]) if x in owners]
        y = ms[0] if ms else (ys[0] if ys else None)
        extra.append([
            float("MRG" in c["srcs"]), float("MRG:link" in c["srcs"]), float(bool(ms)), float(bool(ys)),
            float(info.get("y_always_no", {}).get(y, False)) if y else -1.0,
            float(len(owners.get(y, []))) if y else 0.0,
            float(bool(re.search(r"\d|[０-９一二三四五六七八九]", y))) if y else -1.0,
            float(len(y)) if y else 0.0,
        ])
    return np.hstack([X, np.array(extra, dtype=float)])


# ===========================================================================
# 【統合】sao_selector12.py
# 名前の付け替え: Selector → Selector12, TRAIN_FILE → TRAIN_FILE12, build_candidates → build_candidates12, claim_features → claim_features12, select → select12, structural_features → structural_features12
# ===========================================================================
"""
sao_selector12.py
==================
【実験12】抽出の欠点（余計な抽出・取り逃し・苦手な関係）を、個別の規則ではなく汎用的な
仕組みで改善する。試した仕組みと、交差検証（主指標：トリプル完全一致）での採否：

(1) 係り受けに基づく汎用の候補生成（dep_pairs.py）…… 採用（+1.28pt）
    述語（動詞・サ変名詞・形容詞）にかかる構成要素どうしをすべて候補の組にし、
    格助詞・受身・連体修飾・並列などを特徴量として選別モデルに与える（節ベースの
    Open IE と同じ考え方）。「接続される」「封止する」など有する系以外の関係に効く。
(2) 2段階の選別 …… 採用（+0.35pt、信頼区間は0をわずかにまたぐ）
    1段目の確率から、同じ組・逆向きの組の中での順位、同じ部品を持つ別の持ち主の確率、
    A→B→C の経路（近道）、各ノードにとって最も確からしい関係か、を作り、2段目で選び直す。
(3) 1組から2件目の関係も採る …… 不採用
    同じ組から採る関係の最大数（1 か 2）を内側の交差検証で選ばせたところ、全分割で1が選ばれた。
(4) 持ち主つきノード「XのY」の候補（OWN）…… 不採用
    75,464 件の候補のうち正解は 340 件（0.4%）、選ばれた 6 件はすべて誤りだった。
    （own=True で作れるが、配布した学習データは使っていない）
(5) 持ち主の情報で「XのY」と「Y」を同じ組とみなす（owner_canon）…… 不採用（−0.13pt）

しきい値・1組の最大件数を選ぶ内側の評価は、主指標と同じく「1つの正解には1つの抽出だけを
対応させる」数え方にした（同義の関係が2件とも正解に数えられる水増しを防ぐ）。
学習ラベル・モデル選択・評価はすべてトリプル完全一致（実験11と同じ基準）。
"""
import re
from collections import defaultdict
import pathlib as _pathlib

import numpy as np

pass  # （統合済み）import dep_pairs as D
pass  # （統合済み）import sao_selector as S
pass  # （統合済み）import sao_selector11 as S11

HERE = _pathlib.Path(__file__).resolve().parent
TRAIN_FILE12 = HERE / "sao_selector12_train.npz"
CASE_KEYS = CASES + ["連体", "は(継承)", "-"]
HAS = {"有する", "備える", "具備する", "含む", "含める", "の"}


def _is_has(rel):
    return rel in HAS or any(h in rel for h in ("有する", "備える", "具備する", "含む"))


def build_candidates12(ts, pp, text, own=False, owner_canon=False, **kw):
    """実験11の候補に、係り受けに基づく候補（DEP）を加える。
    owner_canon=True: 候補の「X 有する Y」から持ち主Xを求め、「XのY」と「Y」を同じ組とみなして
      重複を除く（選別時）。採用した実験12（dep）はこれを使わない設定で学習したので、既定は False。
    own=True: 持ち主つきノード「XのY」の候補（OWN）も作る。532件の検証で、OWN候補は75,464件中
      正解が340件（0.4%）しかなく、選別モデルが選んだものはすべて誤りだったため、既定では作らない。"""
    info = build_candidates11(ts, pp, text, **kw)
    index = {(c["source"], c["relation"], c["target"]): c for c in info["cands"]}
    for c in info["cands"]:
        c.setdefault("dep", None)

    def add(src, rel, tgt, tag, dep=None, base=None):
        key = (src, rel, tgt)
        c = index.get(key)
        if c is None:
            c = {"source": src, "relation": rel, "target": tgt, "srcs": [], "type": tag, "dep": None}
            info["cands"].append(c)
            index[key] = c
        if tag not in c["srcs"]:
            c["srcs"].append(tag)
        if dep is not None:
            agg = c["dep"] or {"n": 0, "scope": set(), "sc": set(), "tc": set(), "passive": False,
                               "before": False, "dist": 99, "n_args": 0, "noun_pred": False, "coord": False}
            agg["n"] += 1
            agg["scope"].add(dep["scope"])
            agg["sc"].add(dep["src_case"])
            agg["tc"].add(dep["tgt_case"])
            agg["passive"] |= dep["passive"]
            agg["before"] |= dep["src_before"]
            agg["dist"] = min(agg["dist"], dep["dist"])
            agg["n_args"] = max(agg["n_args"], dep["n_args"])
            agg["noun_pred"] |= dep["pred_pos"] == "NOUN"
            agg["coord"] |= "並列" in dep["src_dep"] or "並列" in dep["tgt_dep"]
            c["dep"] = agg
        if base is not None:
            c["own_base"] = base
        return c

    for d in dependency_pairs(pp, text):
        add(d["source"], d["relation"].translate(SIMP), d["target"], "DEP", dep=d)

    title = info["title"]
    owners = defaultdict(list)
    for c in info["cands"]:
        if _is_has(c["relation"]) and c["source"] != title and "の" not in c["target"] \
                and c["source"] not in owners[c["target"]] and len(owners[c["target"]]) < 3:
            owners[c["target"]].append(c["source"])
    if own:
        for c in list(info["cands"]):
            for side in ("source", "target"):
                Y = c[side]
                for X in owners.get(Y, []):
                    other = c["target"] if side == "source" else c["source"]
                    if other == Y or (other == X and side == "source"):
                        continue
                    new = {"source": c["source"], "target": c["target"]}
                    new[side] = X + "の" + Y
                    if new["source"] == new["target"]:
                        continue
                    add(new["source"], c["relation"], new["target"], "OWN", base=Y)
    info["own_owners"] = {k: v for k, v in owners.items()} if (owner_canon or own) else {}
    for c in info["cands"]:
        if c.get("dep") is not None:
            c["dep"] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in c["dep"].items()}
    return info


def canon(info):
    m = {}
    for Y, xs in info.get("merge_owners", {}).items():
        for X in xs:
            m[X + "の" + Y] = Y
    for Y, xs in info.get("own_owners", {}).items():
        for X in xs:
            m.setdefault(X + "の" + Y, Y)
    return m


def claim_features12(pp, info):
    X = claim_features11(pp, info)
    if not len(info["cands"]):
        return X
    rows = []
    for c in info["cands"]:
        d = c.get("dep")
        f = [float("DEP" in c["srcs"]), float("OWN" in c["srcs"])]
        if d:
            f += [float(d["n"]), float("全体" in d["scope"]), float("区間" in d["scope"]),
                  float(d["passive"]), float(d["before"]), float(d["dist"]), float(d["n_args"]),
                  float(d["noun_pred"]), float(d["coord"])]
            f += [float(k in d["sc"]) for k in CASE_KEYS] + [float(k in d["tc"]) for k in CASE_KEYS]
        else:
            f += [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0] + [0.0] * (2 * len(CASE_KEYS))
        f += [float(len(info.get("own_owners", {}).get(c.get("own_base", ""), [])))]
        rows.append(f)
    return np.hstack([X, np.array(rows, dtype=float)])


def structural_features12(pp, info, p):
    """1段目の確率 p から、組・持ち主・経路の構造的な特徴を作る（2段目用）。"""
    n = pp._normalize_node_text_lenient
    cm = canon(info)
    cands = info["cands"]
    key = [(n(cm.get(c["source"], c["source"])), n(cm.get(c["target"], c["target"]))) for c in cands]
    pair_ps = defaultdict(list)
    for i, (s, t) in enumerate(key):
        pair_ps[frozenset((s, t))].append(p[i])
    best_dir = defaultdict(float)
    for i, k in enumerate(key):
        best_dir[k] = max(best_dir[k], p[i])
    has_edge = defaultdict(float)
    owner_ps = defaultdict(list)
    out_best, in_best, node_best = defaultdict(float), defaultdict(float), defaultdict(float)
    for i, (c, (s, t)) in enumerate(zip(cands, key)):
        if _is_has(c["relation"]):
            has_edge[(s, t)] = max(has_edge[(s, t)], p[i])
            owner_ps[t].append((p[i], s))
        out_best[s] = max(out_best[s], p[i])
        in_best[t] = max(in_best[t], p[i])
        node_best[s] = max(node_best[s], p[i])
        node_best[t] = max(node_best[t], p[i])
    children = defaultdict(list)
    for (s, t), v in has_edge.items():
        children[s].append((t, v))
    rows = []
    for i, (c, (s, t)) in enumerate(zip(cands, key)):
        ps = sorted(pair_ps[frozenset((s, t))], reverse=True)
        rank = ps.index(p[i])
        others = [q for q in ps if q != p[i]] or [0.0]
        rev = best_dir.get((t, s), 0.0)
        own_other = max([q for q, o in owner_ps[t] if o != s] or [0.0])
        n_own = len({o for q, o in owner_ps[t] if q >= 0.3})
        shortcut = max([min(v, has_edge.get((m, t), 0.0)) for m, v in children[s] if m != t] or [0.0])
        via = max([min(has_edge.get((s2, s), 0.0), p[i]) for s2 in [] ] or [0.0])
        rows.append([
            p[i], float(rank), max(others), rev, p[i] - rev, own_other, p[i] - own_other, float(n_own),
            shortcut, p[i] - shortcut, node_best[s], node_best[t], float(p[i] >= node_best[s] - 1e-9),
            float(p[i] >= node_best[t] - 1e-9), out_best[s], in_best[t], float(len(ps)), via,
            float(_is_has(c["relation"])),
        ])
    return np.array(rows, dtype=float) if rows else np.zeros((0, 19))


def select12(pp, info, prob, threshold, max_per_pair=2):
    """確率の高い順に採用。同じ組（「XのY」と「Y」は同じノードとみなす）では最大 max_per_pair 件まで。
    2件目は、既に採った関係と同義でないものだけ。max_per_pair は交差検証の内側で選ぶ。"""
    pass  # （統合済み）import node_match_eval as NM

    n = pp._normalize_node_text_lenient
    cm = canon(info)
    out, chosen = [], defaultdict(list)
    for i in np.argsort(-prob):
        if prob[i] < threshold:
            break
        c = info["cands"][i]
        k = frozenset((n(cm.get(c["source"], c["source"])), n(cm.get(c["target"], c["target"]))))
        if any(rel_match(pp, c["relation"], r) or rel_match(pp, r, c["relation"]) for r in chosen[k]):
            continue
        if len(chosen[k]) >= max_per_pair:
            continue
        chosen[k].append(c["relation"])
        out.append({"source": c["source"], "relation": c["relation"], "target": c["target"],
                    "type": c["srcs"][0].split(":")[-1] if c["srcs"] else "selected",
                    "prob": float(prob[i])})
    return out


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


class Selector12:
    """2段階の選別モデル。学習データ（npz）には1段目の特徴量X1・ラベルY・請求項ID・
    学習時の1段目確率（交差検証の外側で求めたもの）P1・2段目の構造特徴X2を保存してある。"""

    def __init__(self, exclude_ids=None, train_file=TRAIN_FILE12, fold=None):
        """exclude_ids を除いて学習する（交差検証の評価用）。fold を渡すと、2段目の構造特徴に
        「その分割の学習用請求項だけで作った1段目の確率」から求めたもの（X2_fold{k}）を使い、
        交差検証（make_train12.py）と同じモデルを再現する。"""
        d = np.load(train_file, allow_pickle=False)
        keep = np.ones(len(d["Y"]), bool)
        if exclude_ids:
            keep = ~np.isin(d["ids"], np.array(sorted(exclude_ids)))
        X2_all = d["X2"]
        if fold is not None:
            key = "X2_fold%d" % fold
            # 分割ごとの構造特徴は大きいので、別ファイル（…_cv_folds.npz）に分けて置いてもよい
            extra = _pathlib.Path(train_file).with_name(_pathlib.Path(train_file).name.replace("_train.npz", "_cv_folds.npz"))
            if key in d.files:
                X2_all = d[key]
            elif extra.exists():
                X2_all = np.load(extra, allow_pickle=False)[key]
        X1, Y, X2 = d["X1"][keep], d["Y"][keep], X2_all[keep]
        self.fold_thresholds = [float(t) for t in d["fold_thresholds"]] if "fold_thresholds" in d.files else None
        X1 = X1.copy()
        X1[:, KEPT_INDEX] = 0.0
        self.m1 = _model().fit(X1, Y)
        self.m2 = _model().fit(np.hstack([X1, X2]), Y)
        self.threshold = float(d["threshold"])
        self.max_per_pair = int(d["max_per_pair"]) if "max_per_pair" in d.files else 2
        self.fold_max_per_pair = [int(x) for x in d["fold_max_per_pair"]] if "fold_max_per_pair" in d.files else None

    def predict(self, pp, info):
        if not len(info["cands"]):
            return np.zeros(0)
        X1 = claim_features12(pp, info)
        p1 = self.m1.predict_proba(X1)[:, 1]
        X2 = structural_features12(pp, info, p1)
        return self.m2.predict_proba(np.hstack([X1, X2]))[:, 1]


# ===========================================================================
# 【統合】sao_selector13.py
# 名前の付け替え: Selector → Selector13, TRAIN_FILE → TRAIN_FILE13, analyze_claim_selected → analyze_claim_selected13, build_candidates → build_candidates13, canon → canon13, claim_features → claim_features13, select → select13, structural_features → structural_features13
# ===========================================================================
"""
sao_selector13.py
==================
【実験13】候補の網羅性の改善。実験12（係り受け候補＋2段階選別）の候補に、区間の主役と他の
構成要素を組にする候補（seg_pairs.py）と、題名が列挙された各構成要素を持つ候補を加える。
選別の仕組み（2段階・1組1件・内側の交差検証でしきい値を選ぶ）は実験12と同じ。
"""
import pathlib as _pathlib

import numpy as np

pass  # （統合済み）import sao_selector as S
pass  # （統合済み）import sao_selector12 as S12
pass  # （統合済み）import seg_pairs as SP

HERE = _pathlib.Path(__file__).resolve().parent
TRAIN_FILE13 = HERE / "sao_selector13_train.npz"
canon13 = canon
select13 = select12
structural_features13 = structural_features12


def add_segment_candidates(info, seg):
    """seg_pairs の出力を候補に加える（同じ候補は1つにまとめ、区間の手がかりを集計する）。"""
    index = {(c["source"], c["relation"], c["target"]): c for c in info["cands"]}
    for d in seg:
        rel = d["relation"].translate(SIMP)
        key = (d["source"], rel, d["target"])
        c = index.get(key)
        if c is None:
            c = {"source": key[0], "relation": rel, "target": key[2], "srcs": [], "type": "SEG", "dep": None}
            info["cands"].append(c)
            index[key] = c
        tag = "TITLE" if d["scope"] == "題名" else "SEG"
        if tag not in c["srcs"]:
            c["srcs"].append(tag)
        a = c.get("seg") or {"n": 0, "tree_h": 99, "tree_c": 99, "between": False, "after": False, "gap": 99,
                             "n_comps": 0, "n_preds": 0, "head_src": False, "head_tgt": False, "passive": False,
                             "title": False}
        a["n"] += 1
        if d["tree_h"] >= 0:
            a["tree_h"] = min(a["tree_h"], d["tree_h"])
            a["tree_c"] = min(a["tree_c"], d["tree_c"])
        a["between"] |= bool(d["between"])
        a["after"] |= bool(d["after"])
        a["gap"] = min(a["gap"], d["gap"])
        a["n_comps"] = max(a["n_comps"], d["n_comps"])
        a["n_preds"] = max(a["n_preds"], d["n_preds"])
        a["head_src"] |= d["head_is"] == "source"
        a["head_tgt"] |= d["head_is"] == "target"
        a["passive"] |= bool(d["passive"])
        a["title"] |= d["scope"] == "題名"
        c["seg"] = a
    return info


def build_candidates13(ts, pp, text, **kw):
    info = build_candidates12(ts, pp, text, own=False, owner_canon=False, **kw)
    return add_segment_candidates(info, segment_pairs(pp, text, title=info["title"]))


def claim_features13(pp, info):
    X = claim_features12(pp, info)
    if not len(info["cands"]):
        return X
    rows = []
    for c in info["cands"]:
        a = c.get("seg")
        f = [float("SEG" in c["srcs"]), float("TITLE" in c["srcs"]),
             float(len([s for s in c["srcs"] if s not in ("SEG", "TITLE")]))]
        if a:
            f += [float(a["n"]), float(a["tree_h"]), float(a["tree_c"]), float(a["between"]), float(a["after"]),
                  float(a["gap"]), float(a["n_comps"]), float(a["n_preds"]), float(a["head_src"]),
                  float(a["head_tgt"]), float(a["passive"]), float(a["title"])]
        else:
            f += [0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rows.append(f)
    return np.hstack([X, np.array(rows, dtype=float)])


class Selector13(Selector12):
    """実験12と同じ2段階の選別モデル（特徴量に区間の手がかりを加えたもの）。"""

    def __init__(self, exclude_ids=None, train_file=TRAIN_FILE13, fold=None):
        super().__init__(exclude_ids=exclude_ids, train_file=train_file, fold=fold)

    def predict(self, pp, info):
        if not len(info["cands"]):
            return np.zeros(0)
        X1 = claim_features13(pp, info)
        p1 = self.m1.predict_proba(X1)[:, 1]
        X2 = structural_features13(pp, info, p1)
        return self.m2.predict_proba(np.hstack([X1, X2]))[:, 1]


def analyze_claim_selected13(ts, pp, selector, text, threshold=None, max_per_pair=None, **kw):
    info = build_candidates13(ts, pp, text, **kw)
    prob = selector.predict(pp, info)
    return info, select13(pp, info, prob, selector.threshold if threshold is None else threshold,
                        selector.max_per_pair if max_per_pair is None else max_per_pair)


# ===========================================================================
# 【統合】node_pairs.py
# ===========================================================================
"""
node_pairs.py
==============
【実験14】候補生成の網羅性を最大化するための、ノードの拡張と区間内の総当たりの組。

実験13でも正解のSAOの29.2%が候補に入っていなかった。その内訳（正解に占める割合）は
  ・一方のノードが本文にあるのに構成要素として切り出されていない（「ケースの外側」「外部」
    「基材の長手方向」など、位置・部分・属性を表す名詞句）          約7%
  ・一方のノードが「XのY」の形で本文にそのまま無い（「第1トランジスタの第1端」「接着剤の膜厚」
    のように、正解データが部分・属性に持ち主を付けて呼ぶもの）        約5%
  ・両方のノードは候補にあるが、同じ区間にあるのに組になっていない    約6%
  ・組はあるが関係名が合わない                                        約4%
だった。そこで、特定の書き方に頼らない次の一般的な規則で、ノードと組を増やす。

ノード（区間ごと）
  N1 構成要素（従来の抽出器）
  N2 名詞のまとまり（名詞・数詞・記号の連続。「前記」「複数の」等の前置きは除く）
  N3 「の」でつながった名詞句と、その途中からの部分（「Aの一方の面」「一方の面」）
  N4 持ち主つきの名前：区間の主役（先頭の主題「Xは、」と、最後の名詞「…Yと、」）＋「の」＋同じ区間の名詞
     （「…第1端を有する第1トランジスタ」→「第1トランジスタの第1端」）、および並列の展開
     （「A及びBの膜厚」→「Aの膜厚」「Bの膜厚」）
組（区間ごと）
  同じ区間の2つのノードを、その区間の述語（2つのうち前のノードより後ろにあるもの）で結ぶ。両方の向き。
  N3・N4 の「XのY」とその持ち主 X には「の」「有する」の組を作る。
どの組・どの述語・どの向きを採るかは、選別モデルが特徴量から判断する（実験12・13と同じ）。
"""
import re

pass  # （統合済み）import claim_segmenter as CS
pass  # （統合済み）import dep_pairs as D

NOUNISH = {"NOUN", "PROPN", "NUM", "SYM"}
LEAD = {"前記", "該", "上記", "当該", "各", "複数", "所定"}
FORMAL = {"とき", "場合", "こと", "もの", "ため", "それぞれ", "各々", "以上", "以下", "未満", "程度", "状態", "際",
          "少なくとも", "一部", "全部", "うち", "間", "後", "前", "上", "下", "中", "内", "外", "式", "数", "値"}
QTY_RE = re.compile(r"^[0-9０-９一二三四五六七八九十]+(個|つ|本|枚|箇所|か所|層)?(以上|以下)?$")
HAS_LABELS = ("有する",)
OWNER_LABELS = ("の", "有する")
MAX_NODES = 14


def _units(doc):
    """名詞のまとまり（開始・終了のトークン番号とテキスト）。"""
    out, cur = [], []
    for t in doc:
        if t.pos_ in NOUNISH and t.text not in ("、", "，", "。"):
            cur.append(t)
            continue
        if cur:
            out.append(cur)
        cur = []
    if cur:
        out.append(cur)
    units = []
    for toks in out:
        while toks and (toks[0].text in LEAD or toks[0].text in ("の",)):
            toks = toks[1:]
        if not toks:
            continue
        text = "".join(t.text for t in toks)
        if text in FORMAL or QTY_RE.match(text) or not re.search(r"[一-龥ァ-ヶA-Za-zＡ-Ｚａ-ｚ]", text):
            continue
        units.append({"start": toks[0].i, "end": toks[-1].i, "text": text})
    return units


def _chains(doc, units):
    """「の」でつながった名詞句。[[unit, unit, ...], ...]"""
    by_start = {u["start"]: u for u in units}
    chains, used = [], set()
    for u in units:
        if u["start"] in used:
            continue
        ch = [u]
        while True:
            e = ch[-1]["end"]
            if e + 2 < len(doc) and doc[e + 1].text == "の" and (e + 2) in by_start:
                nxt = by_start[e + 2]
                ch.append(nxt)
                used.add(nxt["start"])
            elif (e + 3 < len(doc) and doc[e + 1].text == "の" and doc[e + 2].text in LEAD
                  and (e + 3) in by_start):
                nxt = by_start[e + 3]
                ch.append(nxt)
                used.add(nxt["start"])
            else:
                break
        if len(ch) >= 2:
            chains.append(ch[:4])
    return chains


def _coord_partner(doc, unit, units):
    """「A及びBの…」「AとBとの…」「A、Bの…」の A（Bと並列の名詞）。"""
    s = unit["start"]
    j = s - 1
    while j >= 0 and doc[j].text in LEAD:
        j -= 1
    if j < 1 or doc[j].text not in ("及び", "および", "並びに", "ならびに", "と", "、", "又は", "または"):
        return None
    for u in units:
        if u["end"] == j - 1 or (u["end"] == j - 2 and doc[j - 1].text in LEAD):
            return u
    return None


def segment_nodes(pp, doc):
    """区間のノード一覧: {text: {"pos": 代表位置, "kind": 種類, "len": 連結数, "owner": 持ち主}}"""
    nodes = {}

    def add(text, pos, kind, n=1, owner=None, start=None):
        if not text or len(text) < 2 or text in FORMAL:
            return
        cur = nodes.get(text)
        if cur is None or ["N4", "N3", "N2", "N1"].index(kind) > ["N4", "N3", "N2", "N1"].index(cur["kind"]):
            nodes[text] = {"pos": pos, "kind": kind, "len": n, "owner": owner, "start": pos if start is None else start}

    cmap = _comp_map(pp, doc)
    comps = {}
    for c in cmap.values():
        comps[(c["start"], c["end"])] = c
    for c in comps.values():
        add(c["text"], c["end"], "N1", start=c["start"])
    units = _units(doc)
    for u in units:
        add(u["text"], u["end"], "N2", start=u["start"])
    for ch in _chains(doc, units):
        k = len(ch)
        for a in range(k):
            for b in range(a + 1, k):
                sub = ch[a:b + 1]
                add("の".join(x["text"] for x in sub), sub[-1]["end"], "N3", n=len(sub),
                    owner="の".join(x["text"] for x in sub[:-1]), start=sub[0]["start"])
        partner = _coord_partner(doc, ch[0], units)
        if partner is not None:
            add(partner["text"] + "の" + "の".join(x["text"] for x in ch[1:]), ch[-1]["end"], "N4", n=k,
                owner=partner["text"], start=partner["start"])
    # 区間の主役（先頭の主題と、最後の名詞）
    heads = []
    names = sorted(((v["pos"], t) for t, v in nodes.items() if v["kind"] in ("N1", "N2")))
    if names:
        heads.append(names[-1][1])
        first = min(names)
        after = [t.text for t in doc[first[0] + 1:first[0] + 3]]
        if after and after[0] in ("は", "が"):
            heads.append(first[1])
    for h in dict.fromkeys(heads):
        for t, v in list(nodes.items()):
            if t == h or v["kind"] not in ("N1", "N2") or h in t or t in h:
                continue
            add(h + "の" + t, v["pos"], "N4", n=2, owner=h, start=v["start"])
    if len(nodes) > MAX_NODES * 3:
        # 長すぎる区間は、構成要素と名詞のまとまりを優先して上限まで
        keep = sorted(nodes.items(), key=lambda kv: (["N1", "N2", "N3", "N4"].index(kv[1]["kind"]), kv[1]["pos"]))
        nodes = dict(keep[:MAX_NODES * 3])
    return nodes, heads


def pairs_in_segment(pp, doc, max_preds=3):
    nodes, heads = segment_nodes(pp, doc)
    if len(nodes) < 2:
        return []
    preds = []
    for t in doc:
        if _is_pred(t):
            lab, passive = _label(pp, t)
            if lab and len(lab) <= 12:
                preds.append((t.i, lab, passive, t.pos_))
    out = []
    items = list(nodes.items())
    for s, sv in items:
        for t, tv in items:
            if s == t:
                continue
            base = {"n_nodes": len(nodes), "n_preds": len(preds), "kind_s": sv["kind"], "kind_t": tv["kind"],
                    "len_s": sv["len"], "len_t": tv["len"], "head_s": s in heads, "head_t": t in heads,
                    "gap": abs(sv["pos"] - tv["pos"]), "s_before": sv["pos"] < tv["pos"]}
            # 持ち主と「XのY」
            if tv["owner"] == s:
                for lab in OWNER_LABELS:
                    out.append(dict(base, source=s, relation=lab, target=t, owner=True, pred_after=0,
                                    between=False, passive=False, rank=0, after_rank=0, pred_gap=0))
                continue
            if sv["owner"] == t:
                continue
            # 片方がもう片方の名前の一部（「基板」と「基板の面」）は、持ち主の組以外は作らない
            if s in t or t in s:
                continue
            # 範囲が重なるノード（同じ語の別の切り出し）は組にしない
            if not (sv["pos"] < tv["start"] or tv["pos"] < sv["start"]):
                continue
            lo, hi = min(sv["pos"], tv["pos"]), max(sv["pos"], tv["pos"])
            cand = [p for p in preds if p[0] > lo]
            between = [p for p in cand if p[0] < hi]
            after = [p for p in cand if p[0] > hi][:max_preds]
            for rank, p in enumerate(between + after):
                out.append(dict(base, source=s, relation=p[1], target=t, owner=False,
                                pred_after=int(p[0] > hi), between=p[0] < hi, passive=p[2], rank=rank,
                                after_rank=rank - len(between), pred_gap=p[0] - hi))
    return out


def keep_pair(d):
    """生成した組のうち、残すもの（120件で「新たに候補に入る正解の数／候補の数」を調べて決めた一般的な条件）。
    ・持ち主つきの名前（N4）どうし・名詞のまとまり（N2）どうしの組は、正解がほぼ無いので作らない
    ・述語は、2つのノードの間にあるものと、後ろ側のノードの直後の1つ（「…を有する」の締め）まで
    ・持ち主つきの名前（N4）は、持ち主との組と、近く（8語以内）で間に述語がある組だけ"""
    ks = {d["kind_s"], d["kind_t"]}
    if d["owner"]:
        return ks != {"N2", "N4"}
    if ks == {"N2"} or ks == {"N4"} or ks == {"N2", "N4"} or ks == {"N3", "N4"}:
        return False
    if "N4" in ks:
        return d["between"] and d["gap"] <= 8
    # 間に述語が無いときは、後ろ側のノードの直後の述語1つだけ
    return d["between"] or d["after_rank"] == 0


def node_pair_candidates(pp, text, max_preds=3, filtered=True):
    res = []
    with _enzai(pp, True):
        for seg in split_claim(pp, text):
            sent = to_sentence(seg)
            if sent is None:
                continue
            for s in distribute(sent):
                try:
                    doc = pp.nlp(pp._clean_claim_text(s))
                    res += [d for d in pairs_in_segment(pp, doc, max_preds=max_preds) if not filtered or keep_pair(d)]
                except Exception:  # noqa: BLE001
                    continue
    return res


# ===========================================================================
# 【統合】sao_selector14.py
# 名前の付け替え: Selector → Selector14, TRAIN_FILE → TRAIN_FILE14, analyze_claim_selected → analyze_claim_selected14, build_candidates → build_candidates14, canon → canon14, claim_features → claim_features14, select → select14
# ===========================================================================
"""
sao_selector14.py
==================
【実験14】候補生成の網羅性の最大化。実験13の候補に、区間内のノードを拡張（名詞のまとまり・
「の」でつながった名詞句・持ち主つきの名前）し、区間の述語で組にした候補（NP、node_pairs.py）を加える。
選別の仕組み（2段階・1組1件・内側の交差検証でしきい値を選ぶ）は実験12・13と同じ。

「有する」の付け足し（HASV：既にある組に関係「有する」を両方の向きで付ける候補）も試したが、
候補の上限は上がったものの主指標（完全一致F1）は上がらなかったため採用していない
（add_has_variants は研究の再現用に残してあるが、build_candidates では使わない）。
"""
import pathlib as _pathlib

import numpy as np

pass  # （統合済み）import node_pairs as NP
pass  # （統合済み）import sao_selector as S
pass  # （統合済み）import sao_selector12 as S12
pass  # （統合済み）import sao_selector13 as S13

HERE = _pathlib.Path(__file__).resolve().parent
TRAIN_FILE14 = HERE / "sao_selector14_train.npz"
canon14 = canon
select14 = select12
structural_features = structural_features12
HAS_REL = "有する"
HAS_LIKE = {"有する", "備える", "含む", "具備する", "の"}
KINDS = ("N1", "N2", "N3", "N4")


def add_node_pair_candidates(info, pairs):
    index = {(c["source"], c["relation"], c["target"]): c for c in info["cands"]}
    for d in pairs:
        rel = d["relation"].translate(SIMP)
        key = (d["source"], rel, d["target"])
        c = index.get(key)
        if c is None:
            c = {"source": key[0], "relation": rel, "target": key[2], "srcs": [], "type": "NP", "dep": None}
            info["cands"].append(c)
            index[key] = c
        if "NP" not in c["srcs"]:
            c["srcs"].append("NP")
        a = c.get("np") or {"n": 0, "kind_s": 9, "kind_t": 9, "owner": False, "between": False, "after_rank": 9,
                            "rank": 9, "gap": 99, "pred_gap": 99, "n_nodes": 0, "n_preds": 0, "head_s": False,
                            "head_t": False, "len_s": 0, "len_t": 0, "passive": False, "s_before": False}
        a["n"] += 1
        a["kind_s"] = min(a["kind_s"], KINDS.index(d["kind_s"]))
        a["kind_t"] = min(a["kind_t"], KINDS.index(d["kind_t"]))
        a["owner"] |= bool(d["owner"])
        a["between"] |= bool(d["between"])
        a["after_rank"] = min(a["after_rank"], max(d.get("after_rank", 0), -3))
        a["rank"] = min(a["rank"], d["rank"])
        a["gap"] = min(a["gap"], d["gap"])
        a["pred_gap"] = min(a["pred_gap"], abs(d.get("pred_gap", 0)))
        a["n_nodes"] = max(a["n_nodes"], d["n_nodes"])
        a["n_preds"] = max(a["n_preds"], d["n_preds"])
        a["head_s"] |= bool(d["head_s"])
        a["head_t"] |= bool(d["head_t"])
        a["len_s"] = max(a["len_s"], d["len_s"])
        a["len_t"] = max(a["len_t"], d["len_t"])
        a["passive"] |= bool(d["passive"])
        a["s_before"] |= bool(d["s_before"])
        c["np"] = a
    return info


def _pair_origin(c):
    srcs = set(c["srcs"])
    if srcs - {"SEG", "TITLE", "DEP", "NP", "HASV"}:
        return 3  # LLM・GiNZA などの従来の候補
    if "DEP" in srcs:
        return 2
    if srcs & {"SEG", "TITLE"}:
        return 1
    a = c.get("np")
    if a and not a["between"] and a["gap"] <= 8:
        return 0  # 区間内の組のうち、近くて後ろの述語で結んだもの
    return -1


def add_has_variants(info):
    """既にある組（従来の候補・区間の主役の組・係り受けの組・近い区間内の組）に、「有する」を両方の向きで付ける。"""
    index = {(c["source"], c["relation"], c["target"]): c for c in info["cands"]}
    pairs = {}
    for c in info["cands"]:
        if c["relation"] in HAS_LIKE:
            continue
        o = _pair_origin(c)
        if o < 0:
            continue
        for d, (s, t) in ((0, (c["source"], c["target"])), (1, (c["target"], c["source"]))):
            k = (s, t)
            pairs[k] = max(pairs.get(k, (-1, 0))[0], o), d
    for (s, t), (o, d) in pairs.items():
        key = (s, HAS_REL, t)
        c = index.get(key)
        if c is None:
            c = {"source": s, "relation": HAS_REL, "target": t, "srcs": [], "type": "HASV", "dep": None}
            info["cands"].append(c)
            index[key] = c
        if "HASV" not in c["srcs"]:
            c["srcs"].append("HASV")
        c["hasv"] = {"origin": o, "rev": d}
    return info


def build_candidates14(ts, pp, text, **kw):
    info = build_candidates13(ts, pp, text, **kw)
    return add_node_pair_candidates(info, node_pair_candidates(pp, text))


def claim_features14(pp, info):
    X = claim_features13(pp, info)
    if not len(info["cands"]):
        return X
    rows = []
    for c in info["cands"]:
        f = [float("NP" in c["srcs"]), float("HASV" in c["srcs"]), float(c["srcs"] in (["NP"], ["HASV"], ["NP", "HASV"], ["HASV", "NP"]))]
        a = c.get("np")
        if a:
            f += [float(a["n"]), float(a["kind_s"]), float(a["kind_t"]), float(a["owner"]), float(a["between"]),
                  float(a["after_rank"]), float(a["rank"]), float(a["gap"]), float(a["pred_gap"]), float(a["n_nodes"]),
                  float(a["n_preds"]), float(a["head_s"]), float(a["head_t"]), float(a["len_s"]), float(a["len_t"]),
                  float(a["passive"]), float(a["s_before"])]
        else:
            f += [0.0, -1.0, -1.0, 0.0, 0.0, -9.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        h = c.get("hasv")
        f += [float(h["origin"]), float(h["rev"])] if h else [-1.0, -1.0]
        rows.append(f)
    return np.hstack([X, np.array(rows, dtype=float)])


class Selector14(Selector13):
    """実験12・13と同じ2段階の選別モデル（特徴量に実験14の候補の手がかりを加えたもの）。"""

    def __init__(self, exclude_ids=None, train_file=TRAIN_FILE14, fold=None):
        super().__init__(exclude_ids=exclude_ids, train_file=train_file, fold=fold)

    def predict(self, pp, info):
        if not len(info["cands"]):
            return np.zeros(0)
        X1 = claim_features14(pp, info)
        p1 = self.m1.predict_proba(X1)[:, 1]
        X2 = structural_features(pp, info, p1)
        return self.m2.predict_proba(np.hstack([X1, X2]))[:, 1]


def analyze_claim_selected14(ts, pp, selector, text, threshold=None, max_per_pair=None, **kw):
    info = build_candidates14(ts, pp, text, **kw)
    prob = selector.predict(pp, info)
    return info, select14(pp, info, prob, selector.threshold if threshold is None else threshold,
                        selector.max_per_pair if max_per_pair is None else max_per_pair)


# ===========================================================================
# 【統合】struct_extra.py
# ===========================================================================
"""
struct_extra.py
================
【実験15】分野によらない2つの文の構造の規則で、候補を足す。

(1) 並列の展開（COORD）
    「A、B、C及びDから構成されるX」「A、B及びCを含み」のような列挙では、係り受け解析が並列を
    取り違えやすく（「しょうが」を「しょう＋が」と切る等）、最後の要素Dとの組しか候補に出ないことが多い。
    本文の「、」と「及び・および・並びに・又は・または」から列挙を見つけ、既にある候補 (X, r, D)・(D, r, X)
    の D を、列挙の他の要素に置き換えた候補を足す。正解データの作成基準でも、並列は要素ごとに
    別の三つ組にしている。

(2) 方法の請求項の工程（STEP）
    「〜を〜する工程と、」「〜段階と、」「〜ステップと、」で終わる区間を1つの工程とみなし、
    ・工程の名前：「混合工程」のような名詞ならそのまま、「原材料を容器に入れる段階」のような動詞句なら、
      目的語（なければ斜格の名詞）から「工程」までを名前にする（前置きの修飾と「前記」は除く）
    ・（題名, 含む／備える／有する, 工程）…題名は「Xの製造方法」のような長い名前も候補にする
    ・（工程, 動詞, 動詞の項の名詞）…「原材料を容器に入れる段階」→（…段階, 入れる, 原材料）（…段階, 入れる, 容器）
    を候補にする。どれを採るかは選別モデルが判断する。
"""
import re

pass  # （統合済み）import claim_segmenter as CS
pass  # （統合済み）import dep_pairs as D

CONJ = r"(?:及び|および|並びに|ならびに|又は|または)"
LEAD_RE = re.compile(r"^(前記|該|上記|当該|複数の|少なくとも一つの|少なくとも１つの)+")
NOT_ITEM_RE = re.compile(r"(を|に|で|は|から|より|する|した|され|れる|れた|て|であって|において|備え|有し|含み)$|[をにで]")
STEP_END_RE = re.compile(r"(工程|段階|ステップ)。?$")
HAS_STEMS = [("含み", "含む"), ("含む", "含む"), ("備え", "備える"), ("有し", "有する"), ("有する", "有する"),
             ("具備", "具備する")]


def _clean(x):
    x = LEAD_RE.sub("", x.strip())
    return x.replace("前記", "").replace("当該", "")


def find_lists(text, known):
    """本文中の列挙を見つけて [[要素, ...], ...] を返す。known: 既知のノード名（最後の要素の切り出しに使う）。"""
    out = []
    known_sorted = sorted({k for k in known if k}, key=len, reverse=True)
    for m in re.finditer(CONJ, text):
        left, right = text[:m.start()], text[m.end():]
        # 最後の要素：既知のノード名で一番長く一致するもの。無ければ助詞の手前まで
        r = LEAD_RE.sub("", right)
        last = next((k for k in known_sorted if r.startswith(k)), None)
        if last is None:
            mm = re.match(r"^([^、。，\s]{1,25}?)(?=から|を|に|が|は|で|と|の|、|。|より)", r)
            last = mm.group(1) if mm else None
        if not last:
            continue
        pieces = left.split("、")
        items = []
        for k in range(len(pieces) - 1, -1, -1):
            pc = _clean(pieces[k].split("\n")[-1])
            if k < len(pieces) - 1 and (not pc or len(pc) > 25 or NOT_ITEM_RE.search(pc)):
                break
            if k == len(pieces) - 1:
                # 接続詞の直前の要素（「しょうが」のように末尾が助詞に見える語もそのまま使う）
                if not pc or len(pc) > 25 or re.search(r"[をにで]$|する$|れる$|した$", pc):
                    items = []
                    break
            items.append(pc)
            if len(items) >= 30:
                break
        items = list(reversed(items))
        if items:
            out.append(items + [_clean(last)])
    return out


def coord_expansions(text, cands, n):
    """既にある候補の一方の端が列挙の要素なら、他の要素に置き換えた候補を作る。"""
    known = {c["source"] for c in cands} | {c["target"] for c in cands}
    lists = find_lists(text, known)
    out = []
    for items in lists:
        norm_items = [n(x) for x in items]
        pos = {x: i for i, x in enumerate(norm_items)}
        last = len(items) - 1
        for c in cands:
            other = {"source": n(c["target"]), "target": n(c["source"])}
            for side in ("source", "target"):
                v = n(c[side])
                # 係り受け解析は列挙の最後（か最初）の要素だけを相手と結ぶことが多いので、その候補だけを広げる
                if v not in pos or pos[v] not in (0, last) or other[side] in pos:
                    continue
                for j, it in enumerate(items):
                    if n(it) == v:
                        continue
                    d = {"source": c["source"], "relation": c["relation"], "target": c["target"]}
                    d[side] = it
                    out.append(dict(d, side=side, n_items=len(items), pos=j, from_pos=pos[v]))
    return out


def _span_text(doc, i0, i1):
    return "".join(t.text for t in doc[i0:i1 + 1])


def _np_left(tok, stop):
    """名詞句の左端（連体修飾の節・「前記」は含めない）。"""
    left = tok.i
    for c in tok.children:
        if c.i < tok.i and c.dep_ in ("compound", "nmod", "nummod", "case") and c.i > stop:
            if c.dep_ == "nmod":
                left = min(left, _np_left(c, stop))
            else:
                left = min(left, c.i)
    while left < tok.i and doc_text(tok.doc[left]) in ("前記", "該", "上記", "当該"):
        left += 1
    return left


def doc_text(t):
    return t.text


def step_candidates(pp, text, title, nodes):
    """方法の請求項の工程の候補（題名→工程、工程→動詞の項）。"""
    clean = pp._clean_claim_text(text)
    verbs = [v for stem, v in HAS_STEMS if stem in clean] or ["含む"]
    titles = [title] if title else []
    titles += [x for x in nodes if title and x != title and x.endswith(title) and x in clean]
    out = []
    steps = []
    with _enzai(pp, True):
        for seg in split_claim(pp, text):
            sent = to_sentence(seg)
            if sent is None or not STEP_END_RE.search(sent):
                continue
            try:
                doc = pp.nlp(pp._clean_claim_text(sent))
            except Exception:  # noqa: BLE001
                continue
            toks = [t for t in doc if t.text not in ("。",)]
            if not toks:
                continue
            root = toks[-1]
            if root.text not in ("工程", "段階", "ステップ"):
                continue
            names = []
            prev = doc[root.i - 1] if root.i > 0 else None
            if prev is not None and prev.pos_ in ("NOUN", "PROPN") and prev.dep_ == "compound":
                i0 = prev.i
                while i0 > 0 and doc[i0 - 1].dep_ == "compound" and doc[i0 - 1].head.i >= prev.i:
                    i0 -= 1
                names.append(_clean(_span_text(doc, i0, root.i)))
            acl = [c for c in root.children if c.dep_ in ("acl", "advcl") and c.pos_ in ("VERB", "AUX", "NOUN", "ADJ")]
            verb = max(acl, key=lambda c: c.i) if acl else None
            args = []
            if verb is not None:
                vs = [verb] + [c for c in verb.children if c.dep_ in ("advcl", "conj") and c.pos_ in ("VERB", "NOUN")]
                for v in vs:
                    for c in v.children:
                        if c.dep_ in ("obj", "obl", "nsubj", "iobj"):
                            args.append((v, c))
                objs = [c for v, c in args if v is verb and c.dep_ == "obj"] or [c for v, c in args if c.dep_ == "obj"]
                first = min(objs, key=lambda c: c.i) if objs else (min((c for _, c in args), key=lambda c: c.i)
                                                                   if args else None)
                if first is not None:
                    l1 = _np_left(first, -1)
                    names.append(_clean(_span_text(doc, l1, root.i)))
                    names.append(_clean(_span_text(doc, first.i, root.i)))
            names.append(_clean(_span_text(doc, 0, root.i)))
            names = [x for x in dict.fromkeys(names) if 2 <= len(x) <= 60]
            if not names:
                continue
            steps.append(names)
            for rank, nm in enumerate(names):
                for tt in titles:
                    for v in verbs:
                        out.append({"source": tt, "relation": v, "target": nm, "kind": 0, "rank": rank,
                                    "n_names": len(names)})
                for v, a in args:
                    lab, _ = _label(pp, v)
                    if not lab or len(lab) > 12:
                        continue
                    l = _np_left(a, -1)
                    for an in dict.fromkeys([_clean(_span_text(doc, l, a.i)), _clean(a.text)]):
                        if an and an != nm:
                            out.append({"source": nm, "relation": lab, "target": an, "kind": 1, "rank": rank,
                                        "n_names": len(names)})
    for d in out:
        d["n_steps"] = len(steps)
    return out


# ===========================================================================
# 【統合】comp_first.py
# 名前の付け替え: NOUNISH → NOUNISH_cf
# ===========================================================================
"""
comp_first.py
==============
構成要素を先に確定してから、GiNZA の規則で関係を取り出す（構成要素は分割しない）。

  1. LLM に、請求項の構成要素の名前を本文の表記のまま書き出させる
  2. 新森ら（2004）の「名詞まとまり」の考え方で、名前の境界を確かめる
     （本文にそのまま出てきて、末尾が名詞まとまりの末尾に一致するものだけを使う。LLM が文の途中で切った名前や、
       本文に無い言い換えを除く）
  3. 確かめた名前の範囲を、GiNZA の解析結果の中で1語に結合し（spaCy の retokenize）、その語を構成要素として、
     いつもの GiNZA の規則（係り受けの規則・「有する」木の整理など）で関係を取り出す

532件で、正解データの構成要素の名前をそのまま与えた場合（LLM が完璧に取り出せた場合の上限）、GiNZA の規則の F1 は
42.3% → 48.0% になった。一方、名詞まとまりの規則だけで名前を決めると 41.1% に下がったため、規則は名前を決める役ではなく、
LLM の名前の境界を確かめる役にしている。
"""
import hashlib
import re

COMPONENT_SYSTEM = """あなたは日本の特許請求項の構造を分析する専門家です。
請求項の本文から、構成要素（装置の部品・部材、材料、手段、方法の工程など、関係の主語や目的語になるもの）の名前を
すべて書き出してください。

きまり：
- 名前は本文に書かれている表記のまま書く（言い換えない・要約しない）
- 名前の前の「前記」「該」「複数の」「一対の」「少なくとも一つの」などは付けない
- 「第１の電極」「第１電極」のような番号付きの名前は、番号も含めて1つの名前にする
- 「Xの上面」「Xの第１面」「Xの一端」のように、ある部品の部位を表す名前は、そのまま1つの名前にする
- 「〜する工程」「〜する段階」は、「原料を加熱する工程」のように、工程の内容を含めた名前にする
- 請求項の題名（最後の名詞句。例：「半導体装置」「飲料の製造方法」）も含める
- 同じものは1回だけ書く

出力は、名前を1行に1つずつ書くだけにする。番号や記号、説明は書かない。

【例】
【請求項】
基台と、前記基台に固定された支柱と、前記支柱に回転可能に取り付けられたアームと、前記アームの先端に設けられた複数の光源とを備え、前記光源は、前記基台の上面に向けて光を照射する照明装置。
【構成要素】
基台
支柱
アーム
アームの先端
光源
基台の上面
光
照明装置"""

NOUNISH_cf = {"NOUN", "PROPN", "NUM", "SYM"}
BARE_STEPS = {"段階", "工程", "ステップ"}


def _nounish(t):
    return t.pos_ in NOUNISH_cf or t.tag_.startswith("接頭辞") or t.tag_.startswith("接尾辞-名詞")


def parse_components(out, clean_node=lambda x: x):
    names = []
    for line in str(out or "").splitlines():
        x = re.sub(r"^\s*(?:[-・*●○]|\d+[.)．）])\s*", "", line).strip().strip("「」『』\"'")
        if not x or x.startswith("【") or len(x) > 60 or "、" in x:
            continue
        x = clean_node(x)
        if x and x not in names:
            names.append(x)
    return names


def extract_components_llm(ts, text, model=None, host=None, cache=None, clean_node=lambda x: x):
    """1. LLM に構成要素の名前を書き出させる（結果は cache に保存）。"""
    user = f"【請求項】\n{text.strip()}\n【構成要素】"
    key = "CMP:" + hashlib.sha1((COMPONENT_SYSTEM + "\n" + user + "\n" + str(model)).encode("utf-8")).hexdigest()
    if cache is not None and key in cache:
        out = cache[key]
    else:
        kw = {} if model is None else {"model": model}
        out = ts._ollama_chat(COMPONENT_SYSTEM, user, host=host, **kw)
        if cache is not None:
            cache[key] = out
    return parse_components(out, clean_node), out


def _pattern(name):
    # 本文では「前記Xの前記Y」のように「の」の後に「前記」が入ることがあるので、それを許す
    return re.escape(name).replace("の", "の(?:前記|該|上記|当該)?")


def guard_names(doc, text, names):
    """2. 名前の境界の確認（新森らの「名詞まとまり」）。本文にそのまま出てきて、末尾が名詞まとまりの末尾に
    一致する（次の語が名詞・記号でない）名前だけを残す。"""
    keep = []
    for nm in dict.fromkeys(names):
        if not nm or nm in BARE_STEPS:
            continue
        for m in re.finditer(_pattern(nm), text):
            sp = doc.char_span(m.start(), m.end(), alignment_mode="strict")
            if sp is None:
                continue
            e = sp.end - 1
            nxt = doc[e + 1] if e + 1 < len(doc) else None
            if nxt is not None and _nounish(nxt):
                continue
            if not (_nounish(doc[e]) or doc[e].text == "が" or sp.text.endswith(("工程", "段階", "ステップ"))):
                continue
            keep.append(nm)
            break
    return keep


def forced_relations(pp, text, names):
    """3. 名前の範囲を1語に結合してから、GiNZA の規則で関係を取り出す（analyze_claim_ginza_only と同じ規則）。"""
    text = pp._clean_claim_text(text)
    doc = pp.nlp(text)
    spans = []
    for nm in sorted(set(names), key=len, reverse=True):
        for m in re.finditer(_pattern(nm), text):
            spans.append((m.start(), m.end(), nm))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen, last = [], -1
    for a, b, nm in spans:
        if a < last:
            continue
        sp = doc.char_span(a, b, alignment_mode="expand")
        if sp is None or (chosen and sp.start < chosen[-1][0].end):
            continue
        chosen.append((sp, nm))
        last = b
    offsets = [(sp.start_char, sp.end_char, nm) for sp, nm in chosen]
    with doc.retokenize() as r:
        for sp, _ in chosen:
            if len(sp) > 1:
                r.merge(sp)
    names_at = {}
    for a, b, nm in offsets:
        t = doc.char_span(a, b, alignment_mode="expand")
        if t is not None and len(t) == 1:
            names_at[t.start] = nm
    comps = [{"text": nm, "start": i, "end": i} for i, nm in names_at.items()]
    for c in pp.extract_patent_components_general(doc):
        if not any(c["start"] <= i <= c["end"] for i in names_at):
            comps.append(c)
    comps.sort(key=lambda c: c["start"])
    rw = pp.extract_relation_words_general(doc)
    rel = pp.combine_all_relations(
        pp.extract_positional_relations(doc, comps, rw) + pp.extract_has_location_relations(doc, comps)
        + pp.extract_installation_relations(doc, comps) + pp.extract_boundary_relations(doc, comps),
        pp.extract_direct_relations(doc, comps) + pp.extract_contact_relations(doc, comps)
        + pp.extract_capability_relations(doc, comps) + pp.extract_composition_relations(doc, comps)
        + pp.extract_attribute_relations(doc, comps) + pp.extract_numeric_threshold_relations(doc, comps)
        + pp.extract_copula_relations(doc, comps) + pp.extract_comparison_relations(doc, comps),
        pp.extract_has_relations(doc, comps))
    rel = pp._simplify_hierarchy(rel, doc, comps)
    rel = pp._merge_surface_location_nodes(rel)
    rel = pp._add_genitive_provenance_relations(rel)
    rel = pp._merge_partitive_nodes(rel, doc, comps)
    return rel


def llm_component_relations(ts, pp, text, model=None, host=None, cache=None, clean_node=lambda x: x):
    """1〜3 をまとめて実行する。戻り値：(関係のリスト, 使った名前, LLM が出した名前, LLM の生の出力)"""
    names, raw = extract_components_llm(ts, text, model=model, host=host, cache=cache, clean_node=clean_node)
    clean = pp._clean_claim_text(text)
    kept = guard_names(pp.nlp(clean), clean, names)
    return forced_relations(pp, text, kept), kept, names, raw


# ===========================================================================
# 【統合】llm_select.py
# 名前の付け替え: BARE_STEPS → BARE_STEPS_ls, analyze_claim → analyze_claim_ls
# ===========================================================================
"""
llm_select.py
==============
学習済みモデルを使わないSAO抽出（GiNZAの規則＋LLMの抽出 → LLMによる選別 → ルールで整理）。Recall（取りこぼしの
少なさ）を重視する。

  ① GiNZAの規則による抽出：請求項全体の係り受け解析の規則（G）、手がかり句で分けた区間ごとの解析（GS）、
     文の構造の規則（ST：列挙「A、B及びC」を要素ごとに広げる／方法の請求項の「〜段階と、」を工程として取り出す）
     pool="wide"（広め）では、さらに係り受けの組（DEP）・区間の主役どうしの組（SEG）・題名が持つ構成要素（TITLE）・
     同一ノードの結合（MRG）の規則の候補も加える（532件で、候補に正解が含まれる割合が 55.5% → 約71%）
  ② LLMの直接抽出：タグ付きの請求項を LLM に入れて、SAOをそのまま出させる（方式B/D）
     with_translate=True なら、タグ付きの請求項を LLM で英訳し、英語の文の規則で関係を取り出して日本語に戻した
     候補も加える（方式C：英訳経由）
  ③ LLMによる追加：GiNZAの規則（G）の結果はそのまま採用し（【抽出済み】として LLM にも見せる）、LLM には
     請求項の本文と例題を見せて、残りの候補を「2つの構成要素の組」ごとにまとめた【候補】から、本文に書かれている
     関係を追加させる（例題は532件に含まれない作例3件）。規則の結果の削除は LLM にさせない
     （qwen3.5:9b で試したところ、正しい関係までほぼすべて削除したため）。votes=2 なら2回聞いて一致したものだけ追加
  ④ ルールによる整理：同じ組に採用が複数あるときは1件だけ残し、残りは「要確認」に回す

どこにも学習データ（正解データで学習したモデル）を使わないので、分野によらず同じ仕組みで動く。
③の LLM の呼び出しに失敗したときは、①と②の両方が出した候補を採用にする。
②も失敗したとき（Ollama が起動していない等）は、①のうち GiNZA の規則（G）の結果を採用にする（規則だけの方法）。

判定：
  採用   … GiNZAの規則の結果＋LLM が【候補】から追加したもの（④で整理した後）
  要確認 … 2つ以上の系統（規則・LLM）が出したが追加されなかった候補／④で外した別表現
           （取りこぼしを防ぐため、人が確認できるように表に残す）
  除外   … それ以外
「確からしさ（目安）」は学習した確率ではなく、判定の根拠を数値にしたもの：
  採用＋2系統以上 1.0 ／ 採用＋1系統 0.8 ／ 採用されなかった2系統以上 0.5 ／ それ以外 0.1
"""
import hashlib
import re

pass  # （統合済み）import claim_segmenter as CS
pass  # （統合済み）import sao_selector as S
pass  # （統合済み）import comp_first as CF
pass  # （統合済み）import sao_selector13 as S13
pass  # （統合済み）import struct_extra as SE

SELECT_SYSTEM = """あなたは日本の特許請求項の構造を分析する専門家です。請求項に書かれている構成要素どうしの関係を、
「主語｜関係｜目的語」の形で、漏れなく、正しく整理するのが仕事です。

与えられるもの：
【請求項】本文
【抽出済み】規則ですでに抽出した関係（参考。これには答えない）
【候補】まだ抽出されていない関係の候補。「2つの構成要素の組」ごとに番号が付き、その下に言い方の候補が a, b, c… で並ぶ

やること：
【候補】のうち、請求項の本文に書かれている組だけを選び、本文に最も合う言い方を書く（「3a」のように番号と記号）。
- 【抽出済み】に同じ内容がすでにある組は選ばない
- 本文に書かれていない組、主語と目的語が逆の言い方、相手を取り違えた組は選ばない
- 本文にはっきり書かれていれば選ぶ。書かれていなければ選ばない

判断の基準：
- 「AとBとを備えるX」は X｜備える｜A と X｜備える｜B（全体が主語、部品が目的語。関係は本文の動詞を使う）
- 「Bに固定されたA」「AはBに固定される」は、どちらも A｜固定される｜B（修飾される側・述語の主題が主語）
- 「A、B及びCを含むX」のような列挙は、要素ごとに X｜含む｜A、X｜含む｜B、X｜含む｜C
- 「〜する混合工程と、…を含む方法」は 方法｜含む｜混合工程、工程の中身は 混合工程｜混合する｜材料
- 「Xの上面」「Xの第１面」などの部位は、X｜有する｜Xの上面 の関係も書く

出力は次の1行だけ。説明は書かない。選ぶものが無ければ「追加: なし」と書く。
追加: 番号と記号をカンマ区切り

【例1】
【請求項】
基台と、前記基台に固定された支柱と、前記支柱に回転可能に取り付けられたアームと、前記アームの先端に設けられた光源とを備える照明装置。
【抽出済み】
・照明装置｜備える｜基台
・照明装置｜備える｜支柱
・照明装置｜備える｜光源
【候補】
1. 照明装置 と アーム
   a) 照明装置｜備える｜アーム
   b) アーム｜備える｜照明装置
2. 支柱 と 基台
   a) 基台｜固定される｜支柱
   b) 支柱｜固定される｜基台
3. アーム と 支柱
   a) 支柱｜取り付けられる｜アーム
   b) アーム｜取り付けられる｜支柱
4. 光源 と アームの先端
   a) 光源｜設けられる｜アームの先端
5. アーム と アームの先端
   a) アーム｜有する｜アームの先端
6. 光源 と 支柱
   a) 光源｜設けられる｜支柱
7. 照明装置 と 先端
   a) 照明装置｜備える｜先端
答え：
追加: 1a,2b,3b,4a,5a

【例2】
【請求項】
原料を水に浸漬する浸漬工程と、浸漬された前記原料を加熱する加熱工程と、加熱された前記原料を粉砕する粉砕工程と、を含む飲料の製造方法。
【抽出済み】
・飲料の製造方法｜含む｜浸漬工程
・飲料の製造方法｜含む｜加熱工程
・浸漬工程｜浸漬する｜原料
【候補】
1. 飲料の製造方法 と 粉砕工程
   a) 飲料の製造方法｜含む｜粉砕工程
   b) 粉砕工程｜含む｜飲料の製造方法
2. 加熱工程 と 原料
   a) 加熱工程｜加熱する｜原料
3. 粉砕工程 と 原料
   a) 粉砕工程｜粉砕する｜原料
4. 加熱工程 と 粉砕工程
   a) 加熱工程｜含む｜粉砕工程
5. 原料 と 水
   a) 原料｜含む｜水
答え：
追加: 1a,2a,3a

【例3】
【請求項】
導体層と、前記導体層を覆う絶縁膜と、を有し、前記導体層は、銅、アルミニウム及びニッケルを含み、前記絶縁膜の一部は、前記導体層の上面に接する配線基板。
【抽出済み】
・配線基板｜有する｜導体層
・配線基板｜有する｜絶縁膜
・絶縁膜｜覆う｜導体層
・導体層｜含む｜ニッケル
【候補】
1. 導体層 と 銅
   a) 導体層｜含む｜銅
2. 導体層 と アルミニウム
   a) 導体層｜含む｜アルミニウム
3. 絶縁膜 と 導体層の上面
   a) 導体層の上面｜接する｜絶縁膜
   b) 絶縁膜｜接する｜導体層の上面
4. 導体層 と 導体層の上面
   a) 導体層｜有する｜導体層の上面
5. 銅 と ニッケル
   a) 銅｜含む｜ニッケル
6. 配線基板 と ニッケル
   a) 配線基板｜有する｜ニッケル
答え：
追加: 1a,2a,3b,4a"""

LLM_SRCS = ("E1:llm_direct", "LLMraw")
MAX_VARIANTS = 4      # 1つの組に並べる言い方の数の上限
MAX_PAIRS_PER_CALL = 35  # 1回の LLM の呼び出しで見せる組の数の上限（多いときは分けて呼ぶ）
POOLS = {"standard": "標準（GiNZAの規則＋LLM）",
         "wide": "広め（Recall重視：係り受け・区間・題名・同一ノードの結合の規則も加える）"}


def families(srcs):
    """候補の出どころの系統：LLM（タグ付き日本語から直接抽出）／英訳（英訳経由）／規則（GiNZAの規則）。
    MRG（同一ノードの結合で作った候補）は元の候補から派生したものなので、系統に数えない。"""
    fams = set()
    for x in srcs:
        if x in LLM_SRCS:
            fams.add("LLM")
        elif x.startswith("EN:"):
            fams.add("英訳")
        elif not x.startswith("MRG"):
            fams.add("規則")
    return fams or {"規則"}


def _family(src):  # 互換のため残す
    return next(iter(families([src])))


def _add(info, index, items, tag_of):
    for d in items:
        rel = d["relation"].translate(SIMP)
        key = (d["source"], rel, d["target"])
        c = index.get(key)
        if c is None:
            c = {"source": key[0], "relation": rel, "target": key[2], "srcs": [], "type": d.get("type", "?")}
            info["cands"].append(c)
            index[key] = c
        tag = tag_of(d)
        if tag not in c["srcs"]:
            c["srcs"].append(tag)


def translate_relations(ts, pp, text, model=None, host=None, cache=None):
    """方式C：タグ付きの請求項を LLM で英訳し、英語の文の規則で関係を取り出して日本語に戻す（英訳経由の抽出）。"""
    key = "EN:" + hashlib.sha1((text + "\n" + str(model)).encode("utf-8")).hexdigest()
    if cache is not None and key in cache:
        return cache[key]
    kw = {} if model is None else {"model": model}
    # 統合版の patent_pipeline.py では、translate_sao.py の関数は analyze_claim_translate_llm という名前になる
    fn = getattr(ts, "analyze_claim_translate_llm", None) or ts.analyze_claim_translate
    _, rels = fn(text, pp=pp, host=host, **kw)
    out = [{"source": r["source"], "relation": r["relation"], "target": r["target"], "type": r.get("type", "?")}
           for r in rels if r.get("type") == "translate"]
    if cache is not None:
        cache[key] = out
    return out


# 構成要素の名前の前に付く数量の言葉（「複数の多穴管」→「多穴管」）。正解データでも名前に含めない
_QUANT_PREFIX_RE = re.compile(r"^(前記|該|上記|当該|複数の|多数の|一対の|いくつかの|幾つかの|全ての|すべての|双方の|"
                              r"任意の|それぞれの|各々の|少なくとも[0-9０-９一二三四五六七八九十]+つの|"
                              r"[0-9０-９一二三四五六七八九十]+つの|[0-9０-９]+個の)")
# 構成要素になり得ない語（数量・指示の語、動詞の切れ端）。これが主語・目的語になっている候補は捨てる
NON_NODES = {"複数", "多数", "一対", "互い", "それぞれ", "各々", "夫々", "いくつか", "幾つか", "全て", "すべて", "双方",
             "こと", "もの", "場合", "状態", "備え", "有し", "含み", "具備し", "設け", "一つ", "１つ", "1つ"}


def clean_node(x):
    """「前記複数の多穴管」→「多穴管」のように、名前の前の数量・指示の言葉を外す（残りが空になるなら外さない）。"""
    prev = None
    while prev != x:
        prev = x
        m = _QUANT_PREFIX_RE.match(x)
        if m and len(x) > len(m.group(0)):
            x = x[len(m.group(0)):]
    return x


def normalize_candidates(cands):
    """全候補の主語・目的語に clean_node をかけ、同じになった候補は出どころをまとめて1つにする。
    数量・指示の語や動詞の切れ端が主語・目的語になった候補と、自分自身への関係は捨てる。"""
    out, index = [], {}
    for c in cands:
        s, t = clean_node(c["source"]), clean_node(c["target"])
        if s == t or s in NON_NODES or t in NON_NODES:
            continue
        key = (s, c["relation"], t)
        if key in index:
            for x in c["srcs"]:
                if x not in index[key]["srcs"]:
                    index[key]["srcs"].append(x)
            continue
        d = dict(c, source=s, target=t, srcs=list(c["srcs"]))
        index[key] = d
        out.append(d)
    return out


_HAS_RELS = {"有する", "備える", "含む", "の", "具備する", "構成される", "からなる", "有し", "備え", "含み"}


def grounded(pp, c, text_n):
    """候補の主語・目的語が本文に出てくる名前で、関係の漢字の語幹も本文にある（「有する」系は除く）か。
    532件では、正解に当たる候補はすべてこれを満たし、満たさない候補はすべて誤りだった（誤りの約4%）。"""
    n = pp._normalize_node_text_lenient
    if n(c["source"]) not in text_n or n(c["target"]) not in text_n:
        return False
    stem = re.match(r"[一-龥々]+", c["relation"])
    return c["relation"] in _HAS_RELS or not stem or stem.group(0)[:2] in text_n


_GA_FIX_RE = r"が(?=、|，|及び|および|又は|または|並びに|ならびに)"
BARE_STEPS_ls = {"段階", "工程", "ステップ"}


def structure_fixes(pp, text, info):
    """分野によらない構造の整理（学習なしの規則）。532件（半導体）ではF1がほぼ変わらない（±0.3pt）ことを確認済みで、
    主に半導体以外の請求項（方法の請求項・長い列挙）での崩れを防ぐためのもの。
    1. 「しょう」→「しょうが」：本文で「〜が及び」「〜が、」と続く名前は「が」まで含める
    2. 読点「、」を含む名前（列挙が1つにくっついたもの）の候補は捨てる（正解データでは約2万個中1個だけ）
    3. 方法の請求項で工程の名前（「原材料を容器に入れる段階」）が取り出せたときは、
       ・名前がただの「段階」「工程」「ステップ」の候補を捨てる（どの工程か分からないため。正解データでは0個）
       ・工程の候補（題名｜含む｜工程、工程｜動詞｜対象）を、規則の結果（土台）に加える
       ・題名が工程以外の部品を「有する」とする候補は捨てる（方法は工程を含むもので、部品は工程の中で使われる）
    4. 題名の言い方の揺れ（「製造方法」「乳化剤の製造方法」「冷凍食品の乳化剤の製造方法」）を、本文にある一番長い
       言い方にそろえる
    5. 題名の名前の中だけで閉じた候補（冷凍食品｜の｜冷凍食品の乳化剤、冷凍食品｜の｜冷凍食品の乳化剤の製造方法）は
       捨てる（題名は請求項全体の主語で、その名前の一部は別の構成要素ではない）"""
    cands = info["cands"]
    clean = pp._clean_claim_text(text)
    # 1
    fixes = {}
    for c in cands:
        for x in (c["source"], c["target"]):
            if x not in fixes and re.search(re.escape(x) + _GA_FIX_RE, clean) and (x + "が") in clean:
                fixes[x] = x + "が"
    # 2
    cands = [c for c in cands if not any(p in c["source"] + c["target"] for p in ("、", "，"))]
    # 3
    steps = [c for c in cands if "ST:工程" in c["srcs"]]
    if steps:
        step_names = {c["target"] for c in steps if c["relation"] in ("含む", "備える", "有する")} | \
                     {c["source"] for c in steps if c["relation"] not in ("含む", "備える", "有する")}
        cands = [c for c in cands if c["source"] not in BARE_STEPS_ls and c["target"] not in BARE_STEPS_ls]
        cands = [c for c in cands if not (any(x in ("G:has", "GF:has") for x in c["srcs"]) and c["source"].endswith("方法")
                                          and c["target"] not in step_names)]
    # 4
    title = info.get("title")
    if title:
        nodes = {c["source"] for c in cands} | {c["target"] for c in cands}
        # 「冷凍食品の乳化剤の製造方法であって、…製造方法。」のように、冒頭で題名を長く書いている場合も拾う
        m = re.match(r"^\s*(?:前記)?([^、。，\s]{0,40}?" + re.escape(title) + r")(?:であって|において|に於いて)", clean)
        extra = {m.group(1)} if m else set()
        extra |= {x for x in (info.get("components_used") or []) if x.endswith(title)}
        variants = [x for x in nodes | {title} | extra if (x.endswith(title) or (len(x) >= 2 and title.endswith(x)))
                    and x in clean]
        if variants:
            full = max(variants, key=len)
            for x in variants:
                if x != full:
                    fixes[x] = full
            info["title"] = full
    if fixes:
        for c in cands:
            c["source"], c["target"] = fixes.get(c["source"], c["source"]), fixes.get(c["target"], c["target"])
        cands = normalize_candidates(cands)
    # 5. 題名の中の「Xの」（「冷凍食品の乳化剤の製造方法」の「冷凍食品」）が、題名の親になる候補は捨てる
    #    （題名は請求項全体の主語で、その名前の一部は別の構成要素ではない）
    title = info.get("title")
    if title:
        inner = {x for x in {c["source"] for c in cands} | {c["target"] for c in cands} if (x + "の") in title}
        inner.add(title)
        cands = [c for c in cands if not (c["source"] in inner and c["target"] in inner)]
    return cands


def build_rule_and_llm_candidates(ts, pp, text, llm_cache=None, claim_id=None, model=None, host=None,
                                  llm_output=None, pool="standard", with_translate=False, translate_cache=None,
                                  components="ginza", component_cache=None):
    """① GiNZAの規則と ② LLMの直接抽出（と、指定すれば英訳経由の抽出）の候補を集める（出どころを srcs に残す）。
    pool="wide" では、係り受け（DEP）・区間の主役（SEG）・題名（TITLE）・同一ノードの結合（MRG）の規則の候補も加える。
    llm_output="" を渡すと LLM を呼ばない（①だけ）。"""
    kw = dict(llm_output=llm_output, llm_cache=llm_cache, claim_id=claim_id, model=model, host=host)
    info = build_candidates13(ts, pp, text, **kw) if pool == "wide" else build_candidates9(ts, pp, text, **kw)
    index = {(c["source"], c["relation"], c["target"]): c for c in info["cands"]}
    try:
        seg = segment_relations(pp, text, distribute_each=True, merge_enzai=True)
    except Exception:  # noqa: BLE001
        seg = []
    _add(info, index, seg, lambda r: "GS:" + r.get("seg_type", r.get("type", "?")))
    # 文の構造の規則（列挙の展開と、方法の請求項の工程）
    try:
        n = pp._normalize_node_text_lenient
        # 列挙の展開は、GiNZAの規則（全体・区間ごと）が出した候補からだけ広げる（区間の主役の組などから広げると、
        # 「段階｜ホタテ｜エビ」のような意味の無い候補が大量にできるため）
        seeds = [c for c in info["cands"] if any(x.startswith(("G:", "GS:")) for x in c["srcs"])]
        _add(info, index, coord_expansions(pp._clean_claim_text(text), seeds, n), lambda d: "ST:列挙")
        nodes = {c["source"] for c in info["cands"]} | {c["target"] for c in info["cands"]}
        steps = step_candidates(pp, text, info.get("title"), nodes)
        _add(info, index, [d for d in steps if d["rank"] == 0], lambda d: "ST:工程")
    except Exception:  # noqa: BLE001
        pass
    # 構成要素を LLM で先に取り出し、GiNZA はそれを分割せずに規則で関係を取る（comp_first.py）。その結果（GF）を土台にする
    info["component_error"], info["components_used"] = None, []
    if components == "llm" and llm_output != "":
        try:
            rels, used, _, _ = llm_component_relations(ts, pp, text, model=model, host=host, cache=component_cache,
                                                           clean_node=clean_node)
            _add(info, index, rels, lambda r: "GF:" + r.get("type", "?"))
            info["components_used"] = used
        except Exception as exc:  # noqa: BLE001
            info["component_error"] = str(exc)
    info["translate_error"] = None
    if with_translate and llm_output != "":
        try:
            _add(info, index, translate_relations(ts, pp, text, model=model, host=host, cache=translate_cache),
                 lambda d: "EN:translate")
        except Exception as exc:  # noqa: BLE001
            info["translate_error"] = str(exc)
    info["cands"] = normalize_candidates(info["cands"])
    info["cands"] = structure_fixes(pp, text, info)
    # 土台（規則で抽出済みとして採用する関係）：構成要素を LLM で固定した GiNZA の規則（GF）が取れていればそれ、
    # 無ければ通常の GiNZA の規則（G）。どちらも、方法の請求項の工程の規則（ST:工程）を加える
    use_gf = components == "llm" and any(x.startswith("GF:") for c in info["cands"] for x in c["srcs"])
    info["base_kind"] = "GF" if use_gf else "G"
    for c in info["cands"]:
        c["base"] = any(x.startswith("GF:" if use_gf else "G:") or x == "ST:工程" for x in c["srcs"])
    # 本文に根拠の無い候補（名前や動詞が本文に出てこない）は、LLM に見せる前に外す（GiNZAの規則Gの結果は残す）
    text_n = pp._normalize_node_text_lenient(pp._clean_claim_text(text).replace("前記", ""))
    info["cands"] = [c for c in info["cands"] if is_base(c) or grounded(pp, c, text_n)]
    info["pool"] = pool
    return info


def is_base(c):
    """土台にする「規則で抽出済みの関係」：GiNZAの係り受けの規則（G。構成要素をLLMで固定したときは GF）と、
    方法の請求項の工程の規則（ST:工程）が出した候補。"""
    if "base" in c:
        return c["base"]
    return any(x.startswith("G:") or x == "ST:工程" for x in c["srcs"])


def group_pairs(pp, cands, idx=None):
    """候補を「2つの構成要素の組（向きを問わない）」ごとにまとめる。組の中の言い方は、出どころの系統が多い順
    （同じなら出どころの数が多い順）に並べ、MAX_VARIANTS 件までにする。戻り値：[[候補の番号, ...], ...]"""
    n = pp._normalize_node_text_lenient
    groups = {}
    for i in (range(len(cands)) if idx is None else idx):
        c = cands[i]
        groups.setdefault(frozenset((n(c["source"]), n(c["target"]))), []).append(i)
    out = []
    for g in groups.values():
        g.sort(key=lambda i: (-len(families(cands[i]["srcs"])), -len(cands[i]["srcs"]), i))
        out.append(g[:MAX_VARIANTS])
    out.sort(key=lambda g: (-max(len(families(cands[i]["srcs"])) for i in g), min(g)))
    return out


def select_prompt(text, cands, groups=None, base=None):
    if groups is None:
        groups = [[i] for i in range(len(cands)) if not base or i not in base]
    lines = ["【抽出済み】"]
    for i in base or []:
        c = cands[i]
        lines.append(f"・{c['source']}｜{c['relation']}｜{c['target']}")
    if not base:
        lines.append("なし")
    lines.append("【候補】")
    for k, g in enumerate(groups, 1):
        a, b = cands[g[0]]["source"], cands[g[0]]["target"]
        lines.append(f"{k}. {a} と {b}")
        for v, i in enumerate(g):
            c = cands[i]
            lines.append(f"   {chr(97 + v)}) {c['source']}｜{c['relation']}｜{c['target']}")
    if not groups:
        lines.append("なし")
    return f"【請求項】\n{text.strip()}\n" + "\n".join(lines) + "\n答え："


_ITEM_RE = re.compile(r"(\d+)\s*[)）.]?\s*([a-zA-Z](?![a-zA-Z]))?")


def parse_selection(out, n_groups, groups=None):
    """「3a,5,7c」→ [(2, 0), (4, 0), (6, 2)]（組の番号・言い方の番号、どちらも0始まり）"""
    picked = []
    for m in _ITEM_RE.finditer(str(out or "")):
        k = int(m.group(1))
        v = ord(m.group(2).lower()) - 97 if m.group(2) else 0
        if not 1 <= k <= n_groups:
            continue
        if groups is not None and not 0 <= v < len(groups[k - 1]):
            v = 0
        if (k - 1, v) not in picked:
            picked.append((k - 1, v))
    return picked


def parse_answer(out, n_base, groups):
    """「削除: 3,5 ／ 追加: 1a,2b」を読む。戻り値：(削除する【A】の番号（0始まり）, 追加する (組, 言い方))。
    書式が崩れて「削除」「追加」の見出しが無いときは、全部を追加とみなす（取りこぼしを増やさない側に倒す）。"""
    text = str(out or "")
    m_del = re.search(r"削除\s*[:：]?(.*?)(?=追加|$)", text, re.S)
    m_add = re.search(r"追加\s*[:：]?(.*)$", text, re.S)
    if not m_del and not m_add:
        return set(), parse_selection(text, len(groups), groups)
    dels = set()
    if m_del:
        for m in re.finditer(r"\d+", m_del.group(1)):
            k = int(m.group(0))
            if 1 <= k <= n_base:
                dels.add(k - 1)
    adds = parse_selection(m_add.group(1), len(groups), groups) if m_add else []
    return dels, adds


def _chat_cached(ts, system, user, model, host, cache):
    key = hashlib.sha1((system + "\n" + user + "\n" + str(model)).encode("utf-8")).hexdigest()
    if cache is not None and key in cache:
        return cache[key]
    kw = {} if model is None else {"model": model}
    out = ts._ollama_chat(system, user, host=host, **kw)
    if cache is not None:
        cache[key] = out
    return out


def select_with_llm(ts, text, cands, model=None, host=None, cache=None, pp=None, votes=1):
    """③ 規則の結果（【抽出済み】）はそのまま採用し、LLM には【候補】から本文に書かれている関係を追加させる。
    （規則の結果の削除も LLM に任せていたが、qwen3.5:9b では正しい関係までほぼすべて削除してしまったため、
    削除はさせない。規則の誤りは、人が確認画面で外す）
    【候補】の組が多いときは MAX_PAIRS_PER_CALL ずつに分けて呼ぶ。
    votes=2 なら、組の並びを逆にしてもう一度聞き、2回とも選ばれたものだけを追加する（精度重視・時間は約2倍）。
    戻り値：(採用する候補の番号（0始まり）の集合, LLMの出力をつないだ文字列, {"deleted": [], "added": [...]})"""
    base = [i for i, c in enumerate(cands) if is_base(c)]
    rest = [i for i in range(len(cands)) if i not in set(base)]
    groups = group_pairs(pp, cands, rest) if pp is not None else [[i] for i in rest]
    outs, per_pass = [], []
    for p in range(max(1, votes)):
        order = groups if p == 0 else list(reversed(groups))
        chunks = [order[s0:s0 + MAX_PAIRS_PER_CALL] for s0 in range(0, len(order), MAX_PAIRS_PER_CALL)]
        added = set()
        for part in chunks:
            out = _chat_cached(ts, SELECT_SYSTEM, select_prompt(text, cands, part, base), model, host, cache)
            outs.append(out)
            _, adds = parse_answer(out, 0, part)
            for k, v in adds:
                added.add(part[k][v])
        per_pass.append(added)
    added = set.intersection(*per_pass) if per_pass else set()
    return set(base) | added, " ／ ".join(str(o) for o in outs), {"deleted": [], "added": sorted(added)}


def judge(cands, selected, llm_ok=True, mode=None, deleted=()):
    """候補ごとの判定（採用・要確認・除外）と確からしさ（目安）。
    mode: "llm"（③を使う）／"both"・"rules"（③または②③が失敗：GiNZAの規則Gの結果をそのまま採用）"""
    mode = mode or ("llm" if llm_ok else "both")
    deleted = set(deleted)
    out = []
    for i, c in enumerate(cands):
        fams = families(c["srcs"])
        multi = len(fams) >= 2
        g = is_base(c)
        if mode != "llm":
            sel = g
            out.append({"selected": sel, "score": 0.6 if sel else (0.5 if multi else 0.1),
                        "status": "採用" if sel else ("要確認" if multi else "除外"),
                        "basis": "GiNZAの規則の結果（LLMなし）" if sel else "規則の候補（LLMなし）"})
            continue
        sel = i in selected
        if sel:
            score = 1.0 if multi else 0.8
            basis = "規則の結果（LLMが確認）" if g else "LLMが追加"
            status = "採用"
        elif i in deleted:
            score, basis, status = 0.3, "規則の結果（LLMが削除）", "要確認"
        else:
            score = 0.5 if multi else 0.1
            basis = "2系統以上が出した（追加されず）" if multi else (
                "規則のみ" if fams == {"規則"} else "LLMのみ" if fams == {"LLM"} else "英訳のみ")
            status = "要確認" if multi else "除外"
        out.append({"selected": sel, "score": score, "status": status, "basis": basis})
    return out


def tidy(pp, cands, judged):
    """④ ルールによる整理：同じ2つの構成要素の組（向きを問わない）に採用が複数あれば、
    確からしさが高く、関係名が本文の動詞らしいもの（「の」以外）を1件だけ残し、残りを要確認にする。"""
    n = getattr(pp, "_normalize_node_text_lenient", lambda x: x)
    best = {}
    for i, (c, j) in enumerate(zip(cands, judged)):
        if not j["selected"]:
            continue
        key = frozenset((n(c["source"]), n(c["target"])))
        rank = (j["score"], c["relation"] != "の", -i)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, i)
    keep = {i for _, i in best.values()}
    for i, j in enumerate(judged):
        if j["selected"] and i not in keep:
            j.update(selected=False, status="要確認", score=min(j["score"], 0.5), basis="同じ組の別表現（④で整理）")
    return judged


def analyze_claim_ls(ts, pp, text, llm_cache=None, select_cache=None, claim_id=None, model=None, host=None,
                  pool="standard", with_translate=False, translate_cache=None, votes=1, components="ginza"):
    """①②③④をまとめて実行する。戻り値：(候補, 判定, LLMの選別の生の出力 または エラー)
    info["mode"] に、実際に使えた段階（"llm"／"both"／"rules"）を入れる。"""
    try:
        info = build_rule_and_llm_candidates(ts, pp, text, llm_cache=llm_cache, claim_id=claim_id,
                                             model=model, host=host, pool=pool, with_translate=with_translate,
                                             translate_cache=translate_cache, components=components,
                                             component_cache=select_cache)
    except Exception as exc:  # noqa: BLE001  LLM を呼べない → ①の規則だけで決める
        info = build_rule_and_llm_candidates(ts, pp, text, model=model, host=host, llm_output="", pool=pool)
        info["mode"] = "rules"
        return info, judge(info["cands"], set(), mode="rules"), f"LLMを呼べませんでした（GiNZAの規則だけで判定）: {exc}"
    cands = info["cands"]
    try:
        selected, raw, ops = select_with_llm(ts, text, cands, model=model, host=host, cache=select_cache, pp=pp,
                                             votes=votes)
        info["mode"] = "llm"
    except Exception as exc:  # noqa: BLE001
        selected, raw, info["mode"] = set(), f"選別のLLMを呼べませんでした: {exc}", "both"
        ops = {"deleted": [], "added": []}
    info["n_base"] = sum(is_base(c) for c in cands)
    info["n_deleted"], info["n_added"] = len(ops["deleted"]), len(ops["added"])
    return info, tidy(pp, cands, judge(cands, selected, mode=info["mode"], deleted=ops["deleted"])), raw


# =====================================================================================================
# 【最終方式】A2：構成要素を LLM で先に取り出して固定し、GiNZA の規則で関係を取り出す（学習なし）
# =====================================================================================================
def analyze_claim_a2(ts, pp, text, cache=None, model=None, host=None):
    """最終方式（A2）。LLM の呼び出しは構成要素の書き出しの1回だけ。
      1. LLM が構成要素の名前を本文の表記のまま書き出す（comp_first.COMPONENT_SYSTEM）
      2. 新森ら（2004）の「名詞まとまり」の考え方で、名前の境界を確かめる
      3. 確かめた名前を GiNZA の解析結果の中で1語に固定し、GiNZA の規則で関係を取り出す
      4. 分野によらない構造の整理（数量の言葉を外す・題名の揺れをそろえる・方法の工程など。structure_fixes）
    判定：採用＝3と4の結果（方法の請求項の工程の規則を含む）／要確認＝通常の GiNZA の規則（構成要素を固定しない）だけが
    出した関係（取りこぼしを人が拾えるように表に残す）／除外＝それ以外。
    LLM を呼べないときは、通常の GiNZA の規則の結果を採用にする（mode="rules"）。
    戻り値：(info, 判定, LLM の出力 または エラー)"""
    info = build_rule_and_llm_candidates(ts, pp, text, model=model, host=host, llm_output="", pool="standard")
    raw = ""
    info["mode"], info["components_used"], info["component_error"] = "rules", [], None
    try:
        rels, used, _, raw = llm_component_relations(ts, pp, text, model=model, host=host, cache=cache,
                                                        clean_node=clean_node)
        index = {(c["source"], c["relation"], c["target"]): c for c in info["cands"]}
        _add(info, index, rels, lambda r: "GF:" + r.get("type", "?"))
        # 列挙「A、B及びCから構成されるX」の展開：GiNZA は最後の要素 C としか結ばないことが多いので、LLM が構成要素として
        # 書き出した名前どうしに限って、他の要素にも同じ関係を広げる（ST:列挙A2。土台に加える）
        usedset = set(used)
        seeds = [c for c in info["cands"] if any(x.startswith("GF:") for x in c["srcs"])]
        exp = [d for d in coord_expansions(pp._clean_claim_text(text), seeds, pp._normalize_node_text_lenient)
               if d["source"] in usedset and d["target"] in usedset]
        _add(info, index, exp, lambda d: "ST:列挙A2")
        info["cands"] = structure_fixes(pp, text, dict(info, cands=normalize_candidates(info["cands"])))
        info["components_used"], info["mode"] = used, "llm"
    except Exception as exc:  # noqa: BLE001
        info["component_error"] = raw = f"LLMを呼べませんでした（通常のGiNZAの規則で判定）: {exc}"
    use_gf = info["mode"] == "llm" and any(x.startswith("GF:") for c in info["cands"] for x in c["srcs"])
    info["base_kind"] = "GF" if use_gf else "G"
    judged = []
    for c in info["cands"]:
        g = any(x.startswith("G:") for x in c["srcs"])
        gf = any(x.startswith("GF:") for x in c["srcs"])
        st_ = "ST:工程" in c["srcs"] or (use_gf and "ST:列挙A2" in c["srcs"])
        base = (gf if use_gf else g) or st_
        c["base"] = base
        if base:
            both = gf and g
            judged.append({"selected": True, "score": 1.0 if both or not use_gf else 0.8, "status": "採用",
                           "basis": ("構成要素を固定したGiNZAの規則" + ("（通常の規則とも一致）" if both else "")) if use_gf
                           else ("方法の工程の規則" if st_ and not g else "GiNZAの規則")})
            if "ST:列挙A2" in c["srcs"] and not gf:
                judged[-1]["basis"] = "列挙の展開（LLMの構成要素どうし）"
        elif g and use_gf:
            judged.append({"selected": False, "score": 0.5, "status": "要確認",
                           "basis": "通常のGiNZAの規則だけが出した関係"})
        else:
            judged.append({"selected": False, "score": 0.1, "status": "除外", "basis": "その他の規則の候補"})
    return info, tidy(pp, info["cands"], judged), raw


# ===========================================================================
# 【統合】platform_core.py
# 名前の付け替え: structural_features → claim_structure_features
# ===========================================================================
"""
platform_core.py
=================
特許分析プラットフォーム（app.py）のデータ処理部分。
画面（Streamlit）に依存しない処理だけをまとめる。

  ・コーパス（532件）の読み込み：corpus_sao_532.json
  ・採用／要確認／除外の判定（選別モデルの確率を、交差検証で較正した帯で区切る）
  ・人手の確認・修正結果の反映
  ・構造的特徴量（レーダーチャート用。特許の「強さ」ではなく請求項の書き方の構造）
  ・SAOネットワーク（全特許の構成要素を基本語にまとめたグラフ）
  ・類似性（SAOトリプルのTF-IDFとコサイン類似度）
  ・Excel／CSVへの書き出し
"""
import io
import json
import math
import re
from collections import Counter, defaultdict
import pathlib as _pathlib

import numpy as np
import pandas as pd

HERE = _pathlib.Path(__file__).resolve().parent
CORPUS_NAME = "corpus_sao_532.json"


def find_corpus_file():
    """corpus_sao_532.json を、このファイルと同じフォルダ・実行フォルダ・その下の
    SAO_platform_実験12 フォルダの順に探す（見つからなければ None）。"""
    for d in (HERE, _pathlib.Path.cwd(), HERE / "SAO_platform_実験12", _pathlib.Path.cwd() / "SAO_platform_実験12"):
        f = d / CORPUS_NAME
        if f.exists():
            return f
    return None


CORPUS_FILE = find_corpus_file() or (HERE / CORPUS_NAME)

STATUS_ACCEPT = "採用"
STATUS_REVIEW = "要確認"
STATUS_REJECT = "除外"
STATUS_ORDER = [STATUS_ACCEPT, STATUS_REVIEW, STATUS_REJECT]

HAS_WORDS = ("有する", "備える", "具備する", "含む", "含める")

# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------


def load_corpus(path=None):
    """corpus_sao_532.json を読み込む。
    戻り値: dict(meta=..., bands=..., patents=[{id, title, applicant, company, group,
    fi, fi_sub, year, url, text, x, y, z, map_x, map_y, relations=[...]}, ...])
    relations の各要素: source, relation, target, prob, status, origin"""
    path = path or find_corpus_file() or CORPUS_FILE
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify(prob, selected, bands):
    """選別モデルの確率を、採用／要確認／除外に振り分ける。
    bands = {"accept": 採用の下限, "threshold": 選別のしきい値, "review_low": 要確認の下限}
      ・選ばれた関係で確率が accept 以上 → 採用
      ・選ばれた関係で accept 未満、または選ばれなかったが review_low 以上 → 要確認
      ・それ以外 → 除外"""
    if selected and prob >= bands["accept"]:
        return STATUS_ACCEPT
    if selected or prob >= bands["review_low"]:
        return STATUS_REVIEW
    return STATUS_REJECT


# ---------------------------------------------------------------------------
# 人手の確認結果の反映
# ---------------------------------------------------------------------------


def effective_relations(patent, reviews=None):
    """人手の確認結果があればそれを、無ければ「採用」＋「要確認のうち選別モデルが
    選んだもの」を、その特許の現在のSAOとして返す（分析ページ共通の入力）。"""
    if reviews and patent["id"] in reviews:
        return [r for r in reviews[patent["id"]] if r.get("keep", True)]
    return [r for r in patent["relations"] if r.get("selected")]


def review_table(patent, reviews=None):
    """人手確認用の表（DataFrame）。確認済みならその内容を、未確認なら
    採用・要確認の候補を、採用=True/要確認は選別モデルの判断を初期値にして返す。"""
    if reviews and patent["id"] in reviews:
        rows = reviews[patent["id"]]
        return pd.DataFrame([{
            "採用する": bool(r.get("keep", True)), "主語(S)": r["source"], "関係(A)": r["relation"],
            "目的語(O)": r["target"], "確率": r.get("prob"), "判定": r.get("status", "人手追加"),
            "抽出元": r.get("origin", "人手"),
        } for r in rows])
    rows = [r for r in patent["relations"] if r["status"] != STATUS_REJECT]
    return pd.DataFrame([{
        "採用する": bool(r.get("selected")), "主語(S)": r["source"], "関係(A)": r["relation"],
        "目的語(O)": r["target"], "確率": round(float(r["prob"]), 3), "判定": r["status"],
        "抽出元": origin_label(r.get("origin", "")),
    } for r in sorted(rows, key=lambda r: -r["prob"])],
        columns=["採用する", "主語(S)", "関係(A)", "目的語(O)", "確率", "判定", "抽出元"])


def table_to_review(df):
    """人手確認の表（data_editor の結果）を、保存用のリストに戻す。"""
    out = []
    for _, row in df.iterrows():
        s, a, o = (str(row.get(k) or "").strip() for k in ("主語(S)", "関係(A)", "目的語(O)"))
        if not (s and a and o):
            continue
        prob = row.get("確率")
        out.append({
            "source": s, "relation": a, "target": o, "keep": bool(row.get("採用する", True)),
            "prob": None if prob is None or (isinstance(prob, float) and math.isnan(prob)) else float(prob),
            "status": row.get("判定") if isinstance(row.get("判定"), str) and row.get("判定") else "人手追加",
            "origin": row.get("抽出元") if isinstance(row.get("抽出元"), str) and row.get("抽出元") else "人手",
        })
    return out


def reviews_to_csv(reviews):
    rows = [{"特許番号": pid, **r} for pid, rs in reviews.items() for r in rs]
    return pd.DataFrame(rows, columns=["特許番号", "source", "relation", "target", "keep", "prob",
                                       "status", "origin"]).to_csv(index=False).encode("utf-8-sig")


def reviews_from_csv(data):
    df = pd.read_csv(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    out = defaultdict(list)
    for _, r in df.iterrows():
        out[str(r["特許番号"])].append({
            "source": str(r["source"]), "relation": str(r["relation"]), "target": str(r["target"]),
            "keep": str(r.get("keep", True)).lower() in ("true", "1", "yes"),
            "prob": None if pd.isna(r.get("prob")) else float(r["prob"]),
            "status": r.get("status") if isinstance(r.get("status"), str) else "人手追加",
            "origin": r.get("origin") if isinstance(r.get("origin"), str) else "人手",
        })
    return dict(out)


# ---------------------------------------------------------------------------
# 基本語（ネットワーク・類似度用のノード名の正規化）
# ---------------------------------------------------------------------------

_NUM = r"[0-9０-９一二三四五六七八九十]+"
_PREFIX_RE = re.compile(r"^(前記|当該|該|上記|各|複数の|少なくとも(一|１|1)つの|一対の|１対の|1対の)+")
_ORD_RE = re.compile(r"第" + _NUM + r"の?")
_TAIL_RE = re.compile(r"(" + _NUM + r"|[A-Za-zＡ-Ｚａ-ｚ]+)$")
_SUFFIX_RE = re.compile(r"(の各々|のそれぞれ|の各|それぞれ|各々)$")


def base_term(node):
    """「前記第１トランジスタの第２端」→「トランジスタの端」のように、
    番号・指示語を除いて、特許をまたいで比べられる基本語にする。"""
    t = str(node).strip()
    t = _PREFIX_RE.sub("", t)
    t = _SUFFIX_RE.sub("", t)
    t = _ORD_RE.sub("", t)
    t = _TAIL_RE.sub("", t)
    t = re.sub(r"\s+", "", t)
    return t or str(node)


_ORIGIN = [("LLM", ("LLMraw", "E1:llm_direct")),
           ("GiNZA補完", ("E1:claim_title_ginza", "E1:ginza_has_fallback", "E1:attribute",
                        "E1:ginza_has_fallback_conflict", "E1:claim_title_ginza_conflict")),
           ("GiNZA", ("G",)), ("分割GiNZA", ("GS",)), ("ノード結合", ("MRG",)), ("係り受け", ("DEP",)),
           ("持ち主つき", ("OWN",)), ("区間の主役", ("SEG",)), ("題名", ("TITLE",)), ("区間内の名詞句", ("NP",)),
           ("文の構造の規則", ("ST",)), ("英訳経由", ("EN",)),
           ("構成要素を固定したGiNZA", ("GF",))]


def origin_label(srcs):
    """候補の出どころ（内部の記号）を、画面用の短い名前にする。
    srcs: 記号のリスト（"E1:llm_direct" など）か、"+" でつないだ文字列。"""
    items = srcs.split("+") if isinstance(srcs, str) else list(srcs)
    out = []
    for name, keys in _ORIGIN:
        for x in items:
            if x in keys or (x.split(":")[0] in keys and ":" not in keys[0]):
                out.append(name)
    return "・".join(dict.fromkeys(out)) or (srcs if isinstance(srcs, str) else "")


def is_has(rel):
    return rel in ("の",) + HAS_WORDS or any(h in rel for h in HAS_WORDS)


# ---------------------------------------------------------------------------
# 構造的特徴量（レーダーチャート用）
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(
    r"[0-9０-９]+(\.[0-9０-９]+)?\s*(%|％|μm|um|ｍｍ|mm|nm|ｎｍ|℃|°C|V|Ｖ|A|Ａ|Ω|Hz|倍|度|原子%|atoms|cm)"
    r"|以上|以下|未満|より大きい|より小さい|超え")

RADAR_AXES = ["構成要素数", "SAO関係数", "階層の深さ", "分岐の多さ", "関係の多様性", "機能・配置の記述", "数値限定"]


def _longest_path(edges):
    children = defaultdict(set)
    nodes = set()
    for s, t in edges:
        if s != t:
            children[s].add(t)
            nodes |= {s, t}
    memo = {}

    def depth(n, stack):
        if n in memo:
            return memo[n]
        best = 0
        for c in children[n]:
            if c in stack:
                continue
            best = max(best, 1 + depth(c, stack | {c}))
        memo[n] = best
        return best

    return max((depth(n, {n}) for n in nodes), default=0)


def claim_structure_features(relations, text=""):
    """1件の請求項のSAOから構造的特徴量を求める（値の大小は特許の優劣ではない）。"""
    nodes = {x for r in relations for x in (r["source"], r["target"])}
    has_edges = [(r["source"], r["target"]) for r in relations if is_has(r["relation"])]
    out_deg = Counter(r["source"] for r in relations)
    rel_kinds = {re.sub(r"(され|させ|して|する|した|される)+$", "", r["relation"]) for r in relations}
    non_has = sum(1 for r in relations if not is_has(r["relation"]))
    return {
        "構成要素数": len(nodes),
        "SAO関係数": len(relations),
        "階層の深さ": _longest_path(has_edges),
        "分岐の多さ": round(float(np.mean(list(out_deg.values()))), 2) if out_deg else 0.0,
        "関係の多様性": len(rel_kinds),
        "機能・配置の記述": round(non_has / len(relations), 3) if relations else 0.0,
        "数値限定": len(_NUMERIC_RE.findall(text or "")),
        "請求項の文字数": len(text or ""),
    }


def feature_table(corpus, reviews=None):
    rows = []
    for p in corpus["patents"]:
        f = claim_structure_features(effective_relations(p, reviews), p["text"])
        rows.append({"特許番号": p["id"], "発明の名称": p["title"], "企業": p["company"], "出願年": p["year"], **f})
    return pd.DataFrame(rows)


def percentile_scores(df, axes=RADAR_AXES):
    """各軸をコーパス内の順位（0〜100のパーセンタイル）に変換する。"""
    out = df.copy()
    for a in axes:
        out[a] = (df[a].rank(pct=True, method="average") * 100).round(1)
    return out


# ---------------------------------------------------------------------------
# SAOネットワーク
# ---------------------------------------------------------------------------


def build_network(corpus, patent_ids=None, reviews=None, min_patents=2, top_n=60):
    """全特許（または指定特許）のSAOを基本語でまとめたネットワーク。
    ノード: 基本語。重み = その語が現れる特許の件数。
    エッジ: 基本語どうしのSAO関係。重み = その組が現れる特許の件数、代表の関係名。"""
    node_patents = defaultdict(set)
    edge_patents = defaultdict(set)
    edge_rel = defaultdict(Counter)
    surface = defaultdict(Counter)
    ids = set(patent_ids) if patent_ids else None
    for p in corpus["patents"]:
        if ids is not None and p["id"] not in ids:
            continue
        for r in effective_relations(p, reviews):
            s, t = base_term(r["source"]), base_term(r["target"])
            if s == t:
                continue
            node_patents[s].add(p["id"])
            node_patents[t].add(p["id"])
            surface[s][r["source"]] += 1
            surface[t][r["target"]] += 1
            edge_patents[(s, t)].add(p["id"])
            edge_rel[(s, t)][r["relation"]] += 1
    keep = [n for n, ps in node_patents.items() if len(ps) >= min_patents]
    keep = set(sorted(keep, key=lambda n: -len(node_patents[n]))[:top_n])
    nodes = [{"id": n, "patents": sorted(node_patents[n]), "count": len(node_patents[n]),
              "surfaces": [s for s, _ in surface[n].most_common(8)]} for n in keep]
    edges = [{"source": s, "target": t, "count": len(ps), "relation": edge_rel[(s, t)].most_common(1)[0][0],
              "patents": sorted(ps)}
             for (s, t), ps in edge_patents.items() if s in keep and t in keep]
    return nodes, edges


def layout_network(nodes, edges, seed=0):
    import networkx as nx

    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"])
    for e in edges:
        w = e["count"]
        if g.has_edge(e["source"], e["target"]):
            g[e["source"]][e["target"]]["weight"] += w
        else:
            g.add_edge(e["source"], e["target"], weight=w)
    if not len(g):
        return {}
    k = 2.4 / math.sqrt(max(len(g), 1))
    return nx.spring_layout(g, k=k, iterations=120, seed=seed, weight="weight")


def patents_with_node(corpus, term, reviews=None):
    """基本語 term がSAOに現れる特許と、そのSAO・本文中の出現表記を返す。"""
    out = []
    for p in corpus["patents"]:
        rels = [r for r in effective_relations(p, reviews)
                if base_term(r["source"]) == term or base_term(r["target"]) == term]
        if rels:
            surf = sorted({x for r in rels for x in (r["source"], r["target"]) if base_term(x) == term},
                          key=len, reverse=True)
            out.append({"patent": p, "relations": rels, "surfaces": surf})
    return out


def highlight_colored(text, colors):
    """本文中の構成要素を、構成要素ごとの色（colors: 名前→{"fill","border"}）でマークした HTML。
    「前記」「複数の」などが付いた書き方にも当たるよう、表示名（display_node）で探す。"""
    import html as _html

    names = sorted({w for w in colors if w}, key=len, reverse=True)
    if not names:
        return _html.escape(text).replace("\n", "<br>")
    pat = re.compile("|".join(re.escape(w) for w in names))
    out, last = [], 0
    for m in pat.finditer(text):
        c = colors[m.group(0)]
        out.append(_html.escape(text[last:m.start()]))
        out.append(f'<mark style="background:{c["fill"]};border-bottom:2px solid {c["border"]};padding:0 2px;'
                   f'border-radius:3px">{_html.escape(m.group(0))}</mark>')
        last = m.end()
    out.append(_html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


def highlight(text, words):
    """本文中の words をマーカーで強調した HTML を返す。"""
    import html as _html

    esc = _html.escape(text)
    for w in sorted({w for w in words if w}, key=len, reverse=True):
        esc = esc.replace(_html.escape(w), "\u0000" + _html.escape(w) + "\u0001")
    esc = esc.replace("\u0000", '<mark style="background:#fde68a;padding:0 2px;border-radius:3px">')
    esc = esc.replace("\u0001", "</mark>")
    return esc.replace("\n", "<br>")


# ---------------------------------------------------------------------------
# 類似性（SAOトリプルのTF-IDF）
# ---------------------------------------------------------------------------


def sao_tokens(relations):
    toks = []
    for r in relations:
        s, t = base_term(r["source"]), base_term(r["target"])
        rel = "有する" if is_has(r["relation"]) else re.sub(r"(される|させる|する|され|して)$", "", r["relation"])
        toks += ["T:%s→%s→%s" % (s, rel, t), "P:%s→%s" % (s, t), "N:" + s, "N:" + t]
    return toks


def similarity_matrix(corpus, reviews=None):
    """全特許のSAOをTF-IDFベクトルにし、コサイン類似度の行列を返す。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [sao_tokens(effective_relations(p, reviews)) for p in corpus["patents"]]
    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, min_df=1)
    X = vec.fit_transform(docs)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)
    return S


def similarity_explain(rel_a, rel_b):
    """2件のSAOの共通部分（基本語のトリプル・組・ノード）。"""
    ta, tb = set(sao_tokens(rel_a)), set(sao_tokens(rel_b))
    common = ta & tb
    return {
        "共通のSAO": sorted(x[2:] for x in common if x.startswith("T:")),
        "共通の組": sorted(x[2:] for x in common if x.startswith("P:")),
        "共通の構成要素": sorted(x[2:] for x in common if x.startswith("N:")),
    }


# ---------------------------------------------------------------------------
# 集計（ダッシュボード・バブル・ヒートマップ）
# ---------------------------------------------------------------------------


def status_counts(corpus, reviews=None):
    c = Counter()
    for p in corpus["patents"]:
        for r in p["relations"]:
            c[r["status"]] += 1
        c[STATUS_REJECT] += p.get("n_rejected", 0)
    confirmed = sum(sum(1 for r in rs if r.get("keep", True)) for rs in (reviews or {}).values())
    return {
        "特許件数": len(corpus["patents"]),
        STATUS_ACCEPT: c[STATUS_ACCEPT], STATUS_REVIEW: c[STATUS_REVIEW], STATUS_REJECT: c[STATUS_REJECT],
        "抽出SAO": sum(1 for p in corpus["patents"] for r in p["relations"] if r.get("selected")),
        "人手確認済みの特許": len(reviews or {}), "確定SAO（人手確認済み）": confirmed,
    }


def company_year_bubble(corpus, reviews=None):
    rows = []
    for p in corpus["patents"]:
        if p.get("year") is None:
            continue
        rows.append({"企業": p.get("group", p["company"]), "出願年": p["year"],
                     "SAO数": len(effective_relations(p, reviews))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["企業", "出願年"]).agg(請求項数=("SAO数", "size"), 平均SAO数=("SAO数", "mean"),
                                              SAO数合計=("SAO数", "sum")).reset_index())


def company_tech_matrix(corpus, axis="FIサブクラス", reviews=None, top_tech=15, top_comp=12):
    rows = []
    comp_top = {c for c, _ in Counter(p["company"] for p in corpus["patents"]).most_common(top_comp)}
    for p in corpus["patents"]:
        if p["company"] not in comp_top:
            continue
        if axis == "FIサブクラス":
            techs = p.get("fi_sub") or []
        elif axis == "FIメイングループ":
            techs = p.get("fi_main") or []
        else:
            techs = sorted({base_term(x) for r in effective_relations(p, reviews)
                            for x in (r["source"], r["target"])})
        for t in techs:
            rows.append({"企業": p["company"], "技術": t, "特許番号": p["id"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    top = df.groupby("技術")["特許番号"].nunique().sort_values(ascending=False).head(top_tech).index
    m = df[df["技術"].isin(top)].groupby(["企業", "技術"])["特許番号"].nunique().unstack(fill_value=0)
    return m[list(top)]


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------


def export_excel(corpus, reviews=None, sim=None, top_k=5):
    """分析結果一式を1つのExcelファイル（bytes）にする。"""
    buf = io.BytesIO()
    feats = feature_table(corpus, reviews)
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([
            {"項目": k, "値": v} for k, v in status_counts(corpus, reviews).items()
        ] + [{"項目": "採用の下限確率", "値": corpus["bands"]["accept"]},
             {"項目": "選別のしきい値", "値": corpus["bands"]["threshold"]},
             {"項目": "要確認の下限確率", "値": corpus["bands"]["review_low"]},
             {"項目": "抽出手法", "値": corpus["meta"].get("method", "")}]).to_excel(xw, sheet_name="概要", index=False)
        pd.DataFrame([{
            "特許番号": p["id"], "発明の名称": p["title"], "出願人": p["applicant"], "企業": p["company"],
            "出願年": p["year"], "FI": p["fi"], "人手確認": p["id"] in (reviews or {}), "URL": p.get("url", ""),
        } for p in corpus["patents"]]).to_excel(xw, sheet_name="特許一覧", index=False)
        pd.DataFrame([{
            "特許番号": p["id"], "主語(S)": r["source"], "関係(A)": r["relation"], "目的語(O)": r["target"],
            "確率": round(float(r["prob"]), 4), "判定": r["status"], "選別モデルの選択": bool(r.get("selected")),
            "抽出元": origin_label(r.get("origin", "")),
        } for p in corpus["patents"] for r in p["relations"]]).to_excel(xw, sheet_name="SAO（AI判定）", index=False)
        if reviews:
            pd.DataFrame([{
                "特許番号": pid, "主語(S)": r["source"], "関係(A)": r["relation"], "目的語(O)": r["target"],
                "採用": r.get("keep", True), "元の判定": r.get("status", ""),
            } for pid, rs in reviews.items() for r in rs]).to_excel(xw, sheet_name="SAO（人手確認済み）", index=False)
        feats.to_excel(xw, sheet_name="構造的特徴", index=False)
        if sim is not None:
            ids = [p["id"] for p in corpus["patents"]]
            rows = []
            for i, pid in enumerate(ids):
                for j in np.argsort(-sim[i])[:top_k]:
                    rows.append({"特許番号": pid, "類似特許": ids[j], "類似度": round(float(sim[i, j]), 4)})
            pd.DataFrame(rows).to_excel(xw, sheet_name="類似特許", index=False)
    return buf.getvalue()


def relations_csv(corpus, reviews=None):
    rows = []
    for p in corpus["patents"]:
        for r in effective_relations(p, reviews):
            rows.append({"特許番号": p["id"], "発明の名称": p["title"], "企業": p["company"],
                         "主語(S)": r["source"], "関係(A)": r["relation"], "目的語(O)": r["target"],
                         "判定": "人手確認済み" if reviews and p["id"] in reviews else r.get("status", "")})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# 汎用のデータセット（任意の特許リストを読み込んで解析する）
# ---------------------------------------------------------------------------

# 実験13の選別モデルを532件の5分割交差検証で較正した帯（新しいデータにも同じ基準を使う）
# app.py と組で使う版。app.py 側の NEED_PIPELINE と一致しないときは、片方だけ差し替えたことを知らせる
PIPELINE_VERSION = "2026-09-25k"

DEFAULT_BANDS = {
    "accept": 0.57, "threshold": 0.3, "review_low": 0.2, "target_precision": 0.8,
    "stats": {"採用": {"精度": 0.801}, "要確認": {"精度": 0.3034}, "除外": {"精度": 0.0224},
              "正解の所在": {"採用": 0.3799, "要確認": 0.1907, "除外": 0.2288, "候補なし": 0.2005}},
}
METHOD_NAME = "最終方式（学習なし：LLMで構成要素を固定 → GiNZAの規則 → 構造の整理）"
METHOD_SCORE = ("学習データを使わない最終方式。LLMの呼び出しは1件あたり1回（構成要素の書き出し）。"
                "参考（532件・トリプル完全一致）：通常のGiNZAの規則 F1 42.3%。構成要素が正しく取れた場合の上限 47.6%"
                "（正解データの構成要素を与えた場合）。精度はローカルのOllamaで評価コマンドを実行して測る")
MODEL_METHOD_NAME = "実験14（区間内のノード拡張の組＋係り受け候補＋区間の主役の候補＋2段階選別）"
MODEL_METHOD_SCORE = ("比較用。532件の正解データで学習した選別モデル。トリプル完全一致 F1 56.6%（適合率 65.2%・再現率 50.0%）。"
                      "分野を丸ごと隠しても F1 の低下は0〜2ポイント")

GROUP_PALETTE = ["#3987e5", "#d95926", "#199e70", "#9b59d0", "#e0a100", "#2bb3c0", "#e0457b", "#7a8b2c"]
OTHER_COLOR = "#8a8880"

COLUMN_ALIASES = {
    "id": ["文献番号", "公開番号", "登録番号", "公報番号", "出願番号", "特許番号", "番号", "id", "ID", "patent_id"],
    "title": ["発明の名称", "名称", "タイトル", "title", "発明名称"],
    "applicant": ["出願人/権利者", "出願人／権利者", "出願人", "権利者", "出願人名", "applicant", "applicants"],
    "fi": ["FI", "FI分類", "FIコード", "fi", "fi_code"],
    "ipc": ["IPC", "国際特許分類", "ipc"],
    "date": ["出願日", "出願年月日", "application_date", "filing_date", "date"],
    "url": ["文献URL", "URL", "url"],
    "claim": ["請求項", "請求項1", "請求項１", "請求の範囲", "特許請求の範囲", "請求項本文", "クレーム", "claim",
              "claims", "text", "本文"],
}


def _norm_col(c):
    return str(c).strip().lower().replace(" ", "").replace("　", "")


def detect_columns(df):
    """表の列名から、番号・発明の名称・出願人・FI・出願日・請求項の列を自動で見つける。"""
    norm = {_norm_col(c): c for c in df.columns}
    out = {}
    for key, aliases in COLUMN_ALIASES.items():
        out[key] = None
        for a in aliases:
            if _norm_col(a) in norm:
                out[key] = norm[_norm_col(a)]
                break
    return out


def read_table(data, filename):
    """アップロードされた CSV / Excel を DataFrame にする（CSV は UTF-8・Shift_JIS の両方に対応）。"""
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    for enc in ("utf-8-sig", "cp932", "utf-16"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(io.BytesIO(data), encoding="utf-8", errors="replace")


_CLAIM_HEAD_RE = re.compile(r"【請求項\s*[0-9０-９]+\s*】")


def first_claim(text):
    """【請求項１】【請求項２】…のように複数の請求項が入っていたら、請求項1だけを取り出す。"""
    t = str(text or "").strip()
    heads = list(_CLAIM_HEAD_RE.finditer(t))
    if len(heads) >= 1:
        end = heads[1].start() if len(heads) >= 2 else len(t)
        return t[heads[0].end():end].strip()
    return t


_CORP_RE = re.compile(r"(株式会社|有限会社|合同会社|一般社団法人|国立大学法人|学校法人|独立行政法人|\(株\)|（株）|"
                      r"Co\.,?\s*Ltd\.?|Corporation|Inc\.?|CO\.,?\s*LTD\.?)", re.IGNORECASE)


def company_name(applicant):
    """筆頭出願人を、会社の種類（株式会社など）を除いた短い名前にする。"""
    first = re.split(r"[;；、,\n]", str(applicant or ""))[0].strip()
    short = _CORP_RE.sub("", first).strip(" 　・")
    return short or first or "不明"


def fi_parts(fi):
    """FI（例：H01L25/04@C,H01L23/46@Z）から、サブクラス・メイングループ・FI記号の一覧を作る。"""
    subs, mains, full = [], [], []
    for x in re.split(r"[,;；、\s]+(?=[A-HY][0-9]{2}[A-Z])", str(fi or "")):
        x = x.strip()
        m = re.match(r"^([A-HY][0-9]{2}[A-Z])\s*([0-9]+)(?:\s*/\s*([0-9]+))?", x)
        if m:
            subs.append(m.group(1))
            mains.append("%s%s/00" % (m.group(1), m.group(2)))
            full.append("%s%s/%s" % (m.group(1), m.group(2), m.group(3) or "00"))
    return list(dict.fromkeys(subs)), list(dict.fromkeys(mains)), list(dict.fromkeys(full))


def make_patent(pid, text, title="", applicant="", fi="", date="", url=""):
    subs, mains, _ = fi_parts(fi)
    ym = re.search(r"(19|20)\d{2}", str(date or ""))
    return {"id": str(pid), "title": str(title or ""), "applicant": str(applicant or ""),
            "company": company_name(applicant) if applicant else "不明", "fi": str(fi or ""),
            "fi_sub": subs, "fi_main": mains, "year": int(ym.group(0)) if ym else None,
            "filing_date": str(date or ""), "url": str(url or ""), "text": first_claim(text),
            "relations": [], "n_rejected": 0, "analyzed": False}


def patents_from_table(df, cols, limit=None):
    """表（DataFrame）と列の対応から、解析前の特許レコードのリストを作る。
    請求項の列が無い・空の行も読み込む（J-PlatPat の CSV には請求項が入っていないため）。
    その特許は text が空のままで、あとから請求項を追加して解析する。"""
    out, seen = [], set()
    for i, row in df.iterrows():
        text = row.get(cols["claim"]) if cols.get("claim") else None
        if text is None or (isinstance(text, float) and math.isnan(text)):
            text = ""
        if not str(text).strip() and not any(cols.get(k) for k in ("id", "title")):
            continue

        def g(k):
            v = row.get(cols[k]) if cols.get(k) else ""
            return "" if v is None or (isinstance(v, float) and math.isnan(v)) else v

        pid = str(g("id") or "行%d" % (i + 1))
        base, k = pid, 2
        while pid in seen:
            pid = "%s_%d" % (base, k)
            k += 1
        seen.add(pid)
        out.append(make_patent(pid, text, g("title"), g("applicant"), g("fi") or g("ipc"), g("date"), g("url")))
        if limit and len(out) >= limit:
            break
    return out


def norm_pid(x):
    """文献番号の照合用（全角・半角、空白、ハイフンの種類の違いをそろえる）。"""
    import unicodedata
    t = unicodedata.normalize("NFKC", str(x or "")).strip()
    t = re.sub(r"\s+", "", t)
    return re.sub(r"[‐‑‒–—―−ー－]", "-", t)


def claims_from_table(df, id_col, claim_col):
    """「文献番号」と「請求項」の列を持つ表から {正規化した文献番号: 請求項1} を作る。"""
    out = {}
    for _, row in df.iterrows():
        pid, text = row.get(id_col), row.get(claim_col)
        if pid is None or text is None or (isinstance(text, float) and math.isnan(text)):
            continue
        t = first_claim(text)
        if t:
            out[norm_pid(pid)] = t
    return out


def apply_analysis(patent, cands):
    """1件の解析結果（全候補と判定）を特許レコードに入れる。除外は件数だけ残す。"""
    keep = [c for c in cands if c["status"] != STATUS_REJECT]
    patent["relations"] = [{"source": c["source"], "relation": c["relation"], "target": c["target"],
                            "prob": c["prob"] if c["prob"] is not None else 1.0, "selected": bool(c["selected"]),
                            "status": c["status"], "origin": "+".join(c.get("srcs", [])) or c.get("origin", "")}
                           for c in keep]
    patent["n_rejected"] = len(cands) - len(keep)
    patent["analyzed"] = True
    return patent


def new_dataset(name, patents, bands=None):
    return {"meta": {"name": name, "method": METHOD_NAME, "method_key": "llm_select", "n": len(patents)},
            "bands": dict(bands or DEFAULT_BANDS), "patents": patents}


def assign_groups(corpus, top=7):
    """出願人（会社）ごとの色分け用グループ。件数の多い上位 top 社と「その他」。"""
    cnt = Counter(p["company"] for p in corpus["patents"])
    tops = [c for c, _ in cnt.most_common() if c not in ("不明", "その他")][:top]
    for p in corpus["patents"]:
        p["group"] = p["company"] if p["company"] in tops else "その他"
    corpus["groups"] = tops + (["その他"] if any(p["group"] == "その他" for p in corpus["patents"]) else [])
    return corpus


def group_colors(corpus):
    cols = {g: GROUP_PALETTE[i % len(GROUP_PALETTE)] for i, g in enumerate(g for g in corpus.get("groups", [])
                                                                          if g != "その他")}
    cols["その他"] = OTHER_COLOR
    return cols


def _embed(X, dim, seed=0):
    """高次元の特徴を dim 次元に落とす（件数が十分なら t-SNE、少なければ SVD）。"""
    from sklearn.decomposition import TruncatedSVD

    n = X.shape[0]
    k = max(1, min(50, n - 1, X.shape[1] - 1))
    Z = TruncatedSVD(k, random_state=seed).fit_transform(X) if n > 2 and X.shape[1] > 2 else X.toarray()
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    if Zn.shape[1] < dim:
        # 特徴が少なすぎる（例：どの特許にもまだSAOが無い）ときは足りない次元を0で埋める
        Zn = np.hstack([Zn, np.zeros((n, dim - Zn.shape[1]))])
    if n >= 12 and np.linalg.matrix_rank(Zn) >= 2:
        from sklearn.manifold import TSNE

        Y = TSNE(dim, perplexity=max(2.0, min(30.0, (n - 1) / 3.0)), init="pca", random_state=seed,
                 metric="cosine").fit_transform(Zn)
        method = "SVD(%d)→t-SNE(%d)" % (k, dim)
    else:
        Y = np.zeros((n, dim))
        Y[:, :min(dim, Zn.shape[1])] = Zn[:, :dim]
        # 全部同じ点に重なるときは、少しずつずらして見えるようにする
        Y += np.random.default_rng(seed).normal(0, 0.02, Y.shape)
        method = "SVD(%d)" % dim
    return Y, Zn, method


def layout_world(corpus, k_neighbors=6):
    """Patent World（発明の名称＋FI の近さで3次元に配置）の座標と近傍エッジを求める。"""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    ps = corpus["patents"]
    if len(ps) < 2:
        for p in ps:
            p["x"], p["y"], p["z"] = 0.0, 0.0, 0.0
        corpus["world_edges"], corpus["world_method"] = [], "―"
        return corpus
    titles = [p.get("title") or p["text"][:60] for p in ps]
    fis = [fi_parts(p.get("fi", ""))[0] + fi_parts(p.get("fi", ""))[1] + fi_parts(p.get("fi", ""))[2] for p in ps]
    X1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True).fit_transform(titles)
    parts = [X1]
    if any(fis):
        parts.append(TfidfVectorizer(analyzer=lambda x: x or ["FIなし"], sublinear_tf=True).fit_transform(fis) * 1.5)
    X = hstack(parts).tocsr()
    Y, Zn, method = _embed(X, 3)
    for p, (a, b, c) in zip(ps, Y):
        p["x"], p["y"], p["z"] = round(float(a), 4), round(float(b), 4), round(float(c), 4)
    S = Zn @ Zn.T
    edges, seen = [], set()
    k = min(k_neighbors, len(ps) - 1)
    for i in range(len(ps)):
        for j in np.argsort(-S[i])[1:k + 1]:
            key = (min(i, int(j)), max(i, int(j)))
            if key not in seen:
                seen.add(key)
                edges.append({"source": key[0], "target": key[1], "dist": round(float(1 - S[i, j]), 4)})
    corpus["world_edges"], corpus["world_method"] = edges, method
    return corpus


def layout_map(corpus, reviews=None):
    """類似性マップ（SAOの近さで2次元に配置）の座標を求める。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    ps = corpus["patents"]
    docs = [sao_tokens(effective_relations(p, reviews)) or ["SAOなし"] for p in ps]
    if len(ps) < 2:
        for p in ps:
            p["map_x"], p["map_y"] = 0.0, 0.0
        return corpus
    X = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True).fit_transform(docs)
    Y, _, method = _embed(X, 2)
    for p, (a, b) in zip(ps, Y):
        p["map_x"], p["map_y"] = round(float(a), 4), round(float(b), 4)
    corpus["map_method"] = method
    return corpus


def finalize_dataset(corpus):
    """グループ分け・Patent World・類似性マップの座標をまとめて作る。"""
    assign_groups(corpus)
    layout_world(corpus)
    layout_map(corpus)
    corpus["meta"]["n"] = len(corpus["patents"])
    return corpus


_OZ_CSS = """
  #sel-panel { right: 16px; bottom: 16px; width: 330px; display: none; line-height: 1.6; max-height: 70vh;
               overflow: auto; }
  #sel-panel.show { display: block; }
  #sel-panel h2 { font-size: 13.5px; margin: 0 0 6px 0; color: var(--ink-primary); }
  #sel-panel dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-size: 12.5px; }
  #sel-panel dt { color: var(--ink-muted); white-space: nowrap; }
  #sel-panel dd { margin: 0; color: var(--ink-primary); word-break: break-all; }
  #sel-panel button { margin-top: 8px; margin-right: 6px; }
  #sel-panel ol { margin: 6px 0 0 0; padding-left: 20px; font-size: 12.5px; }
  #sel-panel li { cursor: pointer; color: var(--ink-secondary); }
  #sel-panel li:hover { color: var(--ink-primary); text-decoration: underline; }
  #axis-note { left: 16px; top: 44px; font-size: 11.5px; color: var(--ink-muted); background: transparent;
               border: none; box-shadow: none; padding: 0; pointer-events: none; }
"""

_OZ_JS = r"""
  // ==== 追加：軸・クリックで詳細・近い特許だけ線で結ぶ ====
  edgeLines.visible = false;              // 近傍の線は常時は表示しない
  var AXIS_NAMES = __AXIS_NAMES__;
  function makeLabel(text, color) {
    var cv = document.createElement("canvas"); var ctx = cv.getContext("2d");
    var fs = 42; ctx.font = fs + "px sans-serif";
    var w = Math.ceil(ctx.measureText(text).width) + 20; cv.width = w; cv.height = fs + 20;
    ctx.font = fs + "px sans-serif"; ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, 10, cv.height / 2);
    var tex = new THREE.CanvasTexture(cv);
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false,
                                                         sizeAttenuation: false }));
    var h = 0.032; sp.scale.set(h * w / cv.height, h, 1);   // 画面上で一定の大きさ
    return sp;
  }
  // 3本の軸が見えるよう、少し斜めから見下ろす視点にする
  initialCamPos.set(camDist * 0.62, camDist * 0.42, camDist * 0.72);
  camera.position.copy(initialCamPos);
  var axisLen = typicalRadius * 0.95;
  var axisDefs = [[new THREE.Vector3(1, 0, 0), 0xe57373, "#ef9a9a"], [new THREE.Vector3(0, 1, 0), 0x81c784, "#a5d6a7"],
                  [new THREE.Vector3(0, 0, 1), 0x64b5f6, "#90caf9"]];
  axisDefs.forEach(function (d, k) {
    var g = new THREE.BufferGeometry().setFromPoints([d[0].clone().multiplyScalar(-axisLen), d[0].clone().multiplyScalar(axisLen)]);
    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: d[1], transparent: true, opacity: 0.45 })));
    var lab = makeLabel(AXIS_NAMES[k], d[2]); lab.position.copy(d[0].clone().multiplyScalar(axisLen * 1.08)); scene.add(lab);
  });

  var neighborLines = null, selected = null, showNb = false;
  var selPanel = document.getElementById("sel-panel");
  function neighborsOf(i) {
    var out = [];
    edges.forEach(function (e) {
      if (e.source === i) out.push([e.target, e.dist]);
      else if (e.target === i) out.push([e.source, e.dist]);
    });
    out.sort(function (a, b) { return a[1] - b[1]; });
    return out.slice(0, 6);
  }
  function clearNb() {
    if (neighborLines) { scene.remove(neighborLines); neighborLines.geometry.dispose(); neighborLines = null; }
    meshes.forEach(function (m) { m.scale.set(1, 1, 1); });
  }
  function drawNb(i) {
    clearNb();
    meshes[i].scale.set(2.0, 2.0, 2.0);
    if (!showNb) return;
    var nb = neighborsOf(i), pts = [];
    nb.forEach(function (x) {
      var a = nodes[i], b = nodes[x[0]];
      pts.push(a.px, a.py, a.pz, b.px, b.py, b.pz);
      meshes[x[0]].scale.set(1.5, 1.5, 1.5);
    });
    var g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
    neighborLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xffd54f, transparent: true, opacity: 0.95 }));
    scene.add(neighborLines);
  }
  function row(k, v) { return "<dt>" + k + "</dt><dd>" + escapeHtml(v || "―") + "</dd>"; }
  function select(i) {
    selected = i; var n = nodes[i];
    var html = "<h2>" + escapeHtml(n.title || "(名称不明)") + "</h2><dl>" + row("文献番号", n.id) + row("発明の名称", n.title) +
      row("出願人", n.applicant) + row("FI", n.fi) + row("出願日", n.date) + "</dl>" +
      '<button id="btn-nb">' + (showNb ? "近い特許の線を消す" : "近い特許を表示") + '</button><button id="btn-sel-close">閉じる</button>';
    if (showNb) {
      html += '<ol>' + neighborsOf(i).map(function (x) {
        var m = nodes[x[0]];
        return '<li data-i="' + x[0] + '">' + escapeHtml(m.title || m.id) + '（' + escapeHtml(m.applicant) + '）</li>';
      }).join("") + "</ol>";
    }
    selPanel.innerHTML = html; selPanel.classList.add("show");
    document.getElementById("btn-nb").addEventListener("click", function () { showNb = !showNb; select(selected); });
    document.getElementById("btn-sel-close").addEventListener("click", function () {
      selPanel.classList.remove("show"); selected = null; showNb = false; clearNb();
    });
    Array.prototype.forEach.call(selPanel.querySelectorAll("li[data-i]"), function (li) {
      li.addEventListener("click", function () { select(parseInt(li.getAttribute("data-i"), 10)); });
    });
    drawNb(i);
  }
  var downPos = null;
  renderer.domElement.addEventListener("pointerdown", function (ev) { downPos = [ev.clientX, ev.clientY]; });
  renderer.domElement.addEventListener("pointerup", function (ev) {
    if (!downPos || Math.abs(ev.clientX - downPos[0]) + Math.abs(ev.clientY - downPos[1]) > 5) return;
    var rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    var hits = raycaster.intersectObjects(meshes);
    if (hits.length > 0) { select(hits[0].object.userData.index); }
  });

"""


def oz_world_html(corpus, template):
    """Patent World（Three.js）の HTML に、このデータセットの点・エッジ・凡例を差し込む。
    ・近傍の線は常時は出さず、点をクリックして「近い特許を表示」を押したときだけ、その特許と
      近い特許（最大6件）を線で結ぶ。
    ・3本の軸に「技術特徴軸1〜3（次元圧縮の方法）」と名前を付ける。各軸に固有の技術的な意味はない。"""
    ps = corpus["patents"]
    colors = group_colors(corpus)
    keys = {g: "g%d" % i for i, g in enumerate(corpus.get("groups", []))}
    nodes = [{"id": p["id"], "title": p.get("title", ""), "applicant": p.get("applicant", ""), "fi": p.get("fi", ""),
              "date": p.get("filing_date", ""), "group": keys.get(p.get("group", "その他"), "other"),
              "x": p.get("x", 0.0), "y": p.get("y", 0.0), "z": p.get("z", 0.0)} for p in ps]
    data = {"nodes": nodes, "edges": corpus.get("world_edges", [])}
    i = template.index("window.__GRAPH_DATA__ = ") + len("window.__GRAPH_DATA__ = ")
    _, end = json.JSONDecoder().raw_decode(template[i:])
    html_out = template[:i] + json.dumps(data, ensure_ascii=False) + template[i + end:]
    gc = ",\n    ".join("%s: 0x%s" % (keys[g], colors.get(g, OTHER_COLOR)[1:]) for g in keys)
    html_out = re.sub(r"var GROUP_COLOR = \{[^}]*\};", "var GROUP_COLOR = {\n    %s,\n    other: 0x%s,\n  };"
                      % (gc, OTHER_COLOR[1:]), html_out, count=1)
    rows = "\n".join('  <div class="legend-row"><span class="legend-dot" style="background:%s"></span>%s</div>'
                     % (colors.get(g, OTHER_COLOR), g.replace("<", "&lt;")) for g in keys)
    html_out = re.sub(r'(<div id="legend" class="panel">\s*<h2>[^<]*</h2>).*?(</div>\s*<div id="info-panel")',
                      lambda m: m.group(1) + "\n" + rows + "\n" + m.group(2), html_out, count=1, flags=re.S)
    n_e = len(corpus.get("world_edges", []))
    method = corpus.get("world_method", "") or "次元圧縮"
    short = "UMAP" if "UMAP" in method else ("t-SNE" if "t-SNE" in method else ("SVD" if "SVD" in method else "次元圧縮"))
    axis_names = ["技術特徴軸%d（%s）" % (k, short) for k in (1, 2, 3)]
    html_out = re.sub(r"(<div id=\"title-bar\" class=\"panel\">\s*<h1>[^<]*</h1>\s*<span>)[^<]*(</span>)",
                      lambda m: m.group(1) + "発明の名称＋FI・%s・%d件　点をクリックすると詳細" % (method, len(ps))
                      + m.group(2), html_out, count=1)
    html_out = re.sub(r"<dt>特許件数</dt><dd>[^<]*</dd>", "<dt>特許件数</dt><dd>%d件</dd>" % len(ps), html_out)
    html_out = re.sub(r"<dt>エッジ数</dt><dd>[^<]*</dd>", "<dt>近傍の線</dt><dd>%d本（通常は非表示）</dd>" % n_e, html_out)
    html_out = re.sub(r"<dt>次元圧縮</dt><dd>[^<]*</dd>", "<dt>次元圧縮</dt><dd>%s</dd>" % method, html_out)
    html_out = re.sub(r'(<div id="info-panel" class="panel">.*?)<p>.*?</p>',
                      lambda m: m.group(1) + "<p>X・Y・Zは技術特徴軸1〜3（%s）。各軸そのものに固有の技術的意味はなく、"
                      "特許間の近さを3次元に配置したもの。発明の名称＋FIから得られる特徴が近い特許ほど近くに配置されます。"
                      "線は、次元圧縮の前の特徴空間で近い特許どうしを結ぶ。</p>" % short, html_out, count=1, flags=re.S)
    for old, new in (("Zoom In", "拡大"), ("Zoom Out", "縮小"), ("Reset View", "視点を戻す"), ("Rotate", "回転"),
                     ("Expand All", "全ての近傍線"), ("Collapse All", "線を消す"), (">Info<", ">データ概要<"),
                     ("Node Types", "凡例"), (">Help<", ">操作方法<")):
        html_out = html_out.replace(old, new, 1)
    html_out = html_out.replace("<li>点にマウスを合わせる：詳細表示</li>",
                                "<li>点にマウスを合わせる：名称を表示</li><li>点をクリック：詳細と「近い特許を表示」</li>", 1)
    # 表示名は「Patent World」（元の部品の題名「オズの世界」を置き換える）
    html_out = html_out.replace("<title>オズの世界</title>", "<title>Patent World</title>", 1)
    html_out = html_out.replace("<h1>オズの世界</h1>", "<h1>Patent World</h1>", 1)
    html_out = html_out.replace("</style>", _OZ_CSS + "</style>", 1)
    html_out = html_out.replace('<div id="tooltip"></div>',
                                '<div id="tooltip"></div>\n<div id="sel-panel" class="panel"></div>\n'
                                '<div id="axis-note" class="panel">X・Y・Z＝技術特徴軸1〜3（%s）。軸そのものに意味はなく、'
                                '発明の名称＋FIから得られる特徴が近い特許ほど近くに配置されます</div>' % short, 1)
    anchor = '  renderer.domElement.addEventListener("mousemove", onPointerMove);'
    html_out = html_out.replace(anchor, anchor + "\n" + _OZ_JS.replace("__AXIS_NAMES__", json.dumps(axis_names,
                                                                                                   ensure_ascii=False)), 1)
    return html_out


def sample_world_edges(corpus, template):
    """サンプル（532件）用：研究で作った Patent World の近傍エッジを、このデータの並び順に付け替える。"""
    i = template.index("window.__GRAPH_DATA__ = ") + len("window.__GRAPH_DATA__ = ")
    d, _ = json.JSONDecoder().raw_decode(template[i:])
    pos = {p["id"]: k for k, p in enumerate(corpus["patents"])}
    tid = [n["id"] for n in d["nodes"]]
    out = []
    for e in d["edges"]:
        a, b = pos.get(tid[e["source"]]), pos.get(tid[e["target"]])
        if a is not None and b is not None:
            out.append({"source": a, "target": b, "dist": e.get("dist", 0.0)})
    corpus["world_edges"] = out
    corpus["world_method"] = "SVD(50)→UMAP(3)"
    return corpus


# ---------------------------------------------------------------------------
# FIのレーダーチャート
# ---------------------------------------------------------------------------

FI_LEVELS = {"サブクラス（例：H01L）": 0, "メイングループ（例：H01L25/00）": 1, "FI記号（例：H01L25/04）": 2}


def fi_codes(patent, level=0):
    return fi_parts(patent.get("fi", ""))[level]


def fi_radar_data(corpus, by="出願人", level=0, groups=None, top_groups=4, top_fi=8, share=True):
    """FIを軸にしたレーダーチャート用のデータ。
    by: "出願人"（会社ごと）または "出願年"（年ごと）。値は、そのグループの特許のうち
    そのFIを持つものの割合（share=True、%）または件数。
    戻り値: (軸のFIの一覧, {グループ: [値...]}, {グループ: 件数})"""
    def key(p):
        return p["company"] if by == "出願人" else (str(p["year"]) if p.get("year") else "年不明")

    ps = corpus["patents"]
    if not groups:
        cnt = Counter(key(p) for p in ps)
        groups = [g for g, _ in cnt.most_common() if g not in ("不明", "年不明")][:top_groups]
    sel = [p for p in ps if key(p) in groups]
    fi_cnt = Counter(c for p in sel for c in set(fi_codes(p, level)))
    axes = [c for c, _ in fi_cnt.most_common(top_fi)]
    out, sizes = {}, {}
    for g in groups:
        gp = [p for p in sel if key(p) == g]
        sizes[g] = len(gp)
        vals = [sum(1 for p in gp if a in fi_codes(p, level)) for a in axes]
        out[g] = [round(100 * v / len(gp), 1) if (share and gp) else v for v in vals]
    return axes, out, sizes


# ---------------------------------------------------------------------------
# 表示用の整形（関係語の頭に残った助詞、「複数の」などの付いたノード名）
# ---------------------------------------------------------------------------

_LEAD_PARTICLE_RE = re.compile(r"^(には|では|とは|へは|からは|にも|とも|は|が|を|に|で|と|へ|も|の)+(?=[一-龥ァ-ヶー])")
_NODE_PREFIX_RE = re.compile(r"^(前記|当該|上記|複数の|少なくとも(一|１|1)つの|少なくとも(一|１|1)個の|一対の|１対の|"
                             r"各|それぞれの)+")


_CONJ_RULES = [(r"されている$", "される"), (r"られている$", "られる"), (r"([一-龥])している$", r"\1する"),
               (r"づき$", "づく"), (r"された$", "される"), (r"されて$", "される"), (r"され$", "される"), (r"られた$", "られる"),
               (r"られ$", "られる"), (r"れた$", "れる"), (r"([^さら])れ$", r"\1れる"), (r"した$", "する"),
               (r"([一-龥])し$", r"\1する"), (r"([一-龥])み$", r"\1む"), (r"([一-龥])え$", r"\1える"),
               (r"([一-龥])き$", r"\1く"), (r"([一-龥])ち$", r"\1つ"), (r"([一-龥])り$", r"\1る")]


def clean_relation(rel):
    """関係語を表示用に整える。「には有する」→「有する」（頭に残った助詞を取る）、
    「接続され」「接続された」→「接続される」、「含み」→「含む」（言い切りの形にそろえる）。"""
    r = str(rel or "").strip()
    out = _LEAD_PARTICLE_RE.sub("", r) or r
    if len(out) >= 2:
        for pat, rep in _CONJ_RULES:
            new = re.sub(pat, rep, out)
            if new != out:
                out = new
                break
    return out


def display_node(name):
    """図の上で同じものを1つの箱にまとめるための名前（「複数の多穴管」→「多穴管」）。"""
    s = str(name or "").strip()
    out = _NODE_PREFIX_RE.sub("", s)
    return out or s


def tidy_relations(rels):
    """図や一覧に出す前に、関係語とノード名を整え、同じ関係の重複を除く。"""
    out, seen = [], set()
    for r in rels:
        s, a, t = display_node(r["source"]), clean_relation(r["relation"]), display_node(r["target"])
        if s == t or (s, a, t) in seen:
            continue
        seen.add((s, a, t))
        out.append(dict(r, source=s, relation=a, target=t))
    return out


# ---------------------------------------------------------------- ワードクラウド（サーモグラフィー風）
# 温度計の色（アイアンボウ）：冷たい＝黒・紺・紫、熱い＝赤・橙・黄・白。よく出る語ほど熱く大きく描く。
THERMO_STOPS = [(0.00, (40, 20, 110)), (0.18, (95, 20, 150)), (0.36, (175, 25, 135)), (0.52, (230, 60, 60)),
                (0.68, (250, 125, 20)), (0.84, (255, 205, 40)), (1.00, (255, 255, 225))]


def thermo_color(v):
    """0〜1の値を、サーモグラフィーの色（#rrggbb）にする。"""
    v = min(max(float(v), 0.0), 1.0)
    for (a, ca), (b, cb) in zip(THERMO_STOPS, THERMO_STOPS[1:]):
        if v <= b:
            t = (v - a) / (b - a) if b > a else 0
            return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(ca, cb))
    return "#%02x%02x%02x" % THERMO_STOPS[-1][1]


_WC_NUMERIC_RE = re.compile(r"^[0-9０-９.．,，\-－~〜～]+")


def wordcloud_terms(corpus, target="構成要素", patent_ids=None, reviews=None, drop_title=True, min_count=2, top_n=80):
    """ワードクラウドに出す語。選んだ特許の中で多く出てくる順に top_n 語。
    target: 構成要素（番号を除いた部品名：「第1電極」→「電極」）／関係（言い切りの形の関係語）
    戻り値: [(語, 選んだ特許での件数, 全体での件数, 特化係数)]
    特化係数＝（選んだ特許のうちその語を含む割合）÷（データ全体での割合）。1が平均、2なら全体の2倍よく出る。"""
    def terms(p):
        rels = effective_relations(p, reviews)
        if target == "関係":
            out = {clean_relation(r["relation"]) for r in rels}
        else:
            out = {base_term(x) for r in rels for x in (r["source"], r["target"])}
            if drop_title and p.get("title"):
                out.discard(base_term(p["title"]))
        # 「4.0質量%」のような数値だけの語は除く
        return {t for t in out if t and len(t) >= 2 and not _WC_NUMERIC_RE.match(t)}

    all_ids = [p["id"] for p in corpus["patents"]]
    sel = set(patent_ids) if patent_ids is not None else set(all_ids)
    df_all, df_sel = Counter(), Counter()
    for p in corpus["patents"]:
        ts_ = terms(p)
        df_all.update(ts_)
        if p["id"] in sel:
            df_sel.update(ts_)
    n_all, n_sel = max(len(all_ids), 1), max(len(sel & set(all_ids)), 1)
    rows = [(t, c, df_all[t], (c / n_sel) / (df_all[t] / n_all)) for t, c in df_sel.items() if c >= min_count]
    rows.sort(key=lambda r: (-r[1], -r[3]))
    return rows[:top_n]


def wordcloud_heat(rows, color_by="件数"):
    """各語の「温度」（0〜1）。件数なら多いほど熱く、特化係数なら全体より偏って多いほど熱い
    （特化係数1＝平均で中ほどの赤紫、2倍で橙、3倍近くで白。平均より少ない語は紫〜紺）。"""
    if color_by == "件数":
        c = np.sqrt(np.array([r[1] for r in rows], dtype=float))
        lo, hi = float(c.min()), float(c.max())
        return [float((x - lo) / (hi - lo)) if hi - lo > 1e-9 else 1.0 for x in c]
    return [float(min(max(0.45 + 0.35 * math.log2(max(r[3], 1e-6)), 0.0), 1.0)) for r in rows]


def _text_width(t, fs):
    return sum(fs * (0.58 if ord(ch) < 0x3000 else 1.0) for ch in t)


def wordcloud_layout(rows, heat=None, width=900, height=520, min_fs=13, max_fs=62, seed=0):
    """語を中心から渦巻き状に、重ならないように置く（件数の多い語ほど大きく、先に中心近くへ）。"""
    if not rows:
        return []
    heat = heat or wordcloud_heat(rows)
    c = np.sqrt(np.array([r[1] for r in rows], dtype=float))
    lo, hi = float(c.min()), float(c.max())
    placed, boxes = [], []
    rng = np.random.default_rng(seed)
    for r, x, h_ in sorted(zip(rows, c, heat), key=lambda z: -z[1]):
        k = (x - lo) / (hi - lo) if hi - lo > 1e-9 else 1.0
        fs = min_fs + (max_fs - min_fs) * k
        w, h = _text_width(r[0], fs) + 6, fs * 1.18
        start = rng.uniform(0, 2 * np.pi)
        ok = False
        for step in range(4000):
            ang = start + 0.35 * step
            rad = 7.0 * np.sqrt(step)
            cx, cy = width / 2 + rad * np.cos(ang) * 1.45, height / 2 + rad * np.sin(ang)
            x0, y0 = cx - w / 2, cy - h / 2
            if x0 < 4 or y0 < 4 or x0 + w > width - 4 or y0 + h > height - 4:
                continue
            if all(x0 + w <= bx or bx + bw <= x0 or y0 + h <= by or by + bh <= y0 for bx, by, bw, bh in boxes):
                ok = True
                break
        if not ok:
            continue
        boxes.append((x0, y0, w, h))
        placed.append({"text": r[0], "count": r[1], "count_all": r[2], "ratio": r[3], "fs": fs,
                       "x": cx, "y": cy, "heat": h_})
    return placed


def wordcloud_svg(placed, width=900, height=520, legend=("少ない（冷）", "多い（熱）"), title=""):
    """サーモグラフィー風のSVG（暗い背景に、熱い色ほどぼんやり光る語）。語にマウスを重ねると値が出る。"""
    import html as _html
    grad = "".join(f'<stop offset="{int(a * 100)}%" stop-color="{thermo_color(a)}"/>' for a, _ in THERMO_STOPS)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height + 46}" width="100%" '
             'style="font-family:\'Noto Sans JP\',\'Yu Gothic\',Meiryo,sans-serif;display:block">',
             '<defs><radialGradient id="bg" cx="50%" cy="48%" r="75%"><stop offset="0%" stop-color="#1b1036"/>'
             '<stop offset="100%" stop-color="#05030c"/></radialGradient>'
             f'<linearGradient id="bar" x1="0" x2="1" y1="0" y2="0">{grad}</linearGradient>'
             '<filter id="glow" x="-30%" y="-60%" width="160%" height="220%"><feGaussianBlur stdDeviation="5"/></filter>'
             '</defs>',
             f'<rect width="{width}" height="{height + 46}" rx="10" fill="url(#bg)"/>']
    # 熱い語のまわりにぼんやりした光（サーモグラフィーのにじみ）
    for p in sorted(placed, key=lambda p: p["heat"]):
        if p["heat"] < 0.25:
            continue
        parts.append(f'<text x="{p["x"]:.1f}" y="{p["y"]:.1f}" font-size="{p["fs"]:.1f}" font-weight="800" '
                     f'text-anchor="middle" dominant-baseline="central" fill="{thermo_color(p["heat"])}" '
                     f'opacity="{0.25 + 0.45 * p["heat"]:.2f}" filter="url(#glow)">{_html.escape(p["text"])}</text>')
    for p in sorted(placed, key=lambda p: p["heat"]):
        parts.append(f'<text x="{p["x"]:.1f}" y="{p["y"]:.1f}" font-size="{p["fs"]:.1f}" '
                     f'font-weight="{700 if p["heat"] > 0.5 else 500}" text-anchor="middle" dominant-baseline="central" '
                     f'fill="{thermo_color(0.16 + 0.84 * p["heat"])}" style="cursor:default">'
                     f'<title>{_html.escape(p["text"])}：選んだ特許 {p["count"]}件／全体 {p["count_all"]}件'
                     f'（特化係数 {p["ratio"]:.2f}）</title>'
                     f'{_html.escape(p["text"])}</text>')
    y = height + 14
    parts.append(f'<rect x="{width - 290}" y="{y}" width="200" height="12" rx="3" fill="url(#bar)"/>')
    parts.append(f'<text x="{width - 298}" y="{y + 10}" font-size="12" fill="#cbd5e1" text-anchor="end">{legend[0]}</text>')
    parts.append(f'<text x="{width - 82}" y="{y + 10}" font-size="12" fill="#cbd5e1">{legend[1]}</text>')
    if title:
        parts.append(f'<text x="16" y="{y + 10}" font-size="13" fill="#e2e8f0">{_html.escape(title)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ===========================================================================
# 【統合】eval_translate_sao.py
# 名前の付け替え: main → main_eval
# ===========================================================================
"""
評価（旧 eval_translate_sao.py）
================================
532件の正解データに対して、実験13の抽出を5分割交差検証で評価する。

    python patent_pipeline.py --eval-mode exact --limit 532 --llm-cache llm_cache.json --out exp14_exact.json

--eval-mode exact（主指標：トリプル完全一致）／node（主語・目的語ごとの意味的一致）／
lenient（旧：トリプル全体の意味的類似度 0.75）／strict（旧：表記の完全一致）。
毎件 --out に保存するので、--resume で途中から再開できる。
"""
import argparse
import json
import sys
import time
from collections import Counter
import pathlib as _pathlib

ts = sys.modules[__name__]  # 統合後はこのファイル自身


def _aggregate(per_claim):
    """per_claimのリストからMICRO/MACRO集計を計算する（再開時も毎回この
    リストから計算し直すので、二重カウントの心配がない）。"""
    n = len(per_claim) or 1
    total_tp = sum(c["正解数"] for c in per_claim)
    total_pred = sum(c["システム抽出数"] for c in per_claim)
    total_gold = sum(c["正解データ数"] for c in per_claim)
    micro_p = total_tp / total_pred if total_pred else 0.0
    micro_r = total_tp / total_gold if total_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    macro_p = sum(c["precision"] for c in per_claim) / n
    macro_r = sum(c["recall"] for c in per_claim) / n
    macro_f1 = sum(c["f1"] for c in per_claim) / n
    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
    }


_FALLBACK_TYPES_FOR_TABLE = {
    "claim_title_ginza", "ginza_has_fallback", "attribute",
    # 【LLM無言／LLM矛盾の分離】claim_title_ginza_conflict / ginza_has_fallback_conflict は、
    # 同じ(source, target)についてLLMが既に別の（同義語ではない）関係を出している
    # ケース。無言タイプ（LLMが何も出していないケース）とTP/FP/Precisionを
    # 分けて見られるようにするため、表には別の行として並べる。
    "claim_title_ginza_conflict", "ginza_has_fallback_conflict",
}


def _aggregate_type_relation(per_claim):
    """全claimのtp_fp_by_type_relationを合算し、(type, 関係語)ごとの
    TP/FP/Precisionの表を作る。claim_title_ginza / ginza_has_fallback /
    attribute（GiNZA由来で語彙が限られたフォールバック）だけに絞る
    （llm_direct/translateはLLMの自由な言い換えで関係語のバリエーションが
    膨大になり、個別の採用/不採用ルールを作る対象として実用的ではないため）。
    Precisionが低い順（不採用候補が先）に並べて返す。"""
    combined = {}
    for c in per_claim:
        for tr_key, counts in c.get("tp_fp_by_type_relation", {}).items():
            combined.setdefault(tr_key, {"tp": 0, "fp": 0})
            combined[tr_key]["tp"] += counts["tp"]
            combined[tr_key]["fp"] += counts["fp"]

    rows = []
    for tr_key, counts in combined.items():
        type_name, _, relation = tr_key.partition("|")
        if type_name not in _FALLBACK_TYPES_FOR_TABLE:
            continue
        tp, fp = counts["tp"], counts["fp"]
        n = tp + fp
        rows.append({
            "type": type_name, "relation": relation, "tp": tp, "fp": fp,
            "precision": tp / n if n else 0.0,
        })
    rows.sort(key=lambda r: r["precision"])
    return rows


def _load_llm_cache(cache_path):
    """LLM生出力キャッシュを読み込む。ファイルが無い/壊れている場合は
    空のキャッシュから始める（キャッシュは常にベストエフォートで、
    失敗しても評価自体は最初からOllamaを呼んで進められる）。"""
    if cache_path is None:
        return None
    p = _pathlib.Path(cache_path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cache", {})
    except Exception as e:  # noqa: BLE001
        print(f"--llm-cache: {p} の読み込みに失敗したため、空のキャッシュから始めます（{e}）")
        return {}


def _save_llm_cache(cache_path, llm_cache):
    """LLM生出力キャッシュを保存する（1件処理するたびに呼び、途中終了しても
    それまでにOllamaを呼んだ分は失わずに再利用できるようにする）。"""
    if cache_path is None:
        return
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"cache": llm_cache}, f, ensure_ascii=False, indent=1)


def _lenient_match_details(pp, predicted, gold, semantic_threshold, use_semantic):
    """evaluate_triples_lenientと同じ手順・同じ順序で対応付けを行い、
    どの抽出がどの正解に、どの方法（表記正規化／意味的類似度）と類似度で
    対応付いたかを記録する（評価結果そのものは変えない。呼び出し側で
    正解数が公式の評価と一致することを確認している）。"""
    n = pp._normalize_node_text_lenient

    def rel_match(p_rel, g_rel):
        if p_rel == g_rel:
            return True
        pn = pp._normalize_relation_for_match(p_rel)
        gn = pp._normalize_relation_for_match(g_rel)
        if pn == gn or pn in gn or gn in pn:
            return True
        return pp._relation_synonym_match(p_rel, g_rel)

    matched = {}
    matched_gold = set()
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(gold):
            if gi in matched_gold:
                continue
            if (n(p["source"]) == n(g["source"]) and n(p["target"]) == n(g["target"])
                    and rel_match(p["relation"], g["relation"])):
                matched[pi] = (gi, "normalized", None)
                matched_gold.add(gi)
                break

    remaining_pred = [pi for pi in range(len(predicted)) if pi not in matched]
    remaining_gold = [gi for gi in range(len(gold)) if gi not in matched_gold]
    if use_semantic and remaining_pred and remaining_gold:
        try:
            model = pp._get_embed_model()

            def text(r):
                return pp._triple_to_text((n(r["source"]), r["relation"], n(r["target"])))

            emb_p = model.encode([text(predicted[i]) for i in remaining_pred], normalize_embeddings=True)
            emb_g = model.encode([text(gold[i]) for i in remaining_gold], normalize_embeddings=True)
            sim = emb_p @ emb_g.T
            cands = [(sim[a, b], a, b) for a in range(sim.shape[0]) for b in range(sim.shape[1])]
            cands.sort(key=lambda x: -x[0])
            used_b = set()
            for s, a, b in cands:
                if s < semantic_threshold:
                    break
                pi, gi = remaining_pred[a], remaining_gold[b]
                if pi in matched or b in used_b or gi in matched_gold:
                    continue
                matched[pi] = (gi, "semantic", float(s))
                matched_gold.add(gi)
                used_b.add(b)
        except Exception:  # noqa: BLE001 -- 公式評価側と同じく、使えなければ意味的一致なしで扱う
            pass

    return [
        {"pred": predicted[pi], "gold": gold[gi], "method": method, "similarity": s}
        for pi, (gi, method, s) in sorted(matched.items())
    ]


def _save(out_path, per_claim, args, elapsed, done):
    agg = _aggregate(per_claim)
    type_relation_table = _aggregate_type_relation(per_claim)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_mode": args.eval_mode,
                "semantic_threshold": (args.semantic_threshold if args.eval_mode == "lenient"
                                       else args.node_threshold if args.eval_mode == "node" else None),
                "micro": agg["micro"],
                "macro": agg["macro"],
                "type_relation_table": type_relation_table,
                "per_claim": per_claim,
                "elapsed_sec": elapsed,
                "model": args.model,
                "done": done,  # False=まだ途中（--resumeで再開可能）, True=全件完了
            },
            f, ensure_ascii=False, indent=1,
        )


def retrain_with_extra(args, pp):
    """【学習データの追加】他分野などの正解データ（--claims-file / --gold-file）の候補と正解ラベルを、
    532件の学習データ（sao_selector14_train.npz）に足した新しい学習データを作る。
    ・候補と特徴量は、アプリと同じ実験14の方法で作る（LLMの出力は --llm-cache に保存・再利用）
    ・ラベルはトリプル完全一致（主指標と同じ）
    ・2段目の構造特徴に使う1段目の確率は、532件だけで学習したモデルで求める（追加した請求項を
      学習に使っていないモデルの確率なので、532件側の交差検証の確率と同じ扱いになる）
    できたファイルを sao_selector14_train.npz に置き換えると、アプリの選別モデルが追加分も学習する。"""
    import numpy as np
    pass  # （統合済み）import node_match_eval
    pass  # （統合済み）import sao_selector
    pass  # （統合済み）import sao_selector12
    pass  # （統合済み）import sao_selector14

    with open(args.claims_file, encoding="utf-8") as f:
        claims = json.load(f)
    with open(args.gold_file, encoding="utf-8") as f:
        gold_all = json.load(f)
    base = np.load(TRAIN_FILE14, allow_pickle=False)
    X1b, Yb, idsb, X2b = base["X1"], base["Y"], base["ids"], base["X2"]
    old = set(idsb.tolist())
    X1f = X1b.astype(np.float64).copy()
    X1f[:, KEPT_INDEX] = 0.0
    print(f"532件の学習データ: 候補 {len(Yb):,} 件。1段目のモデルを学習しています…")
    m1 = _model().fit(X1f, Yb)
    llm_cache = _load_llm_cache(args.llm_cache)
    n = pp._normalize_node_text_lenient
    add_X1, add_X2, add_Y, add_ids = [], [], [], []
    for k, c in enumerate(claims):
        cid = c["id"]
        if cid in old or cid not in gold_all or not c.get("text"):
            continue
        try:
            info = build_candidates14(ts, pp, c["text"], llm_cache=llm_cache, claim_id=cid,
                                                   model=args.model, host=args.host)
        except Exception as e:  # noqa: BLE001
            print(f"[{cid}] 候補を作れなかったためスキップ: {e}")
            continue
        if not info["cands"]:
            continue
        x1 = claim_features14(pp, info)
        x1[:, KEPT_INDEX] = 0.0
        p1 = m1.predict_proba(x1)[:, 1]
        x2 = structural_features(pp, info, p1)
        G = [(n(g["source"]), g["relation"], n(g["target"])) for g in gold_all[cid]]
        y = [int(any(n(cc["source"]) == gs and n(cc["target"]) == gt and rel_match(pp, cc["relation"], gr)
                     for gs, gr, gt in G)) for cc in info["cands"]]
        add_X1.append(x1.astype(X1b.dtype))
        add_X2.append(x2.astype(X2b.dtype))
        add_Y.append(np.array(y, dtype=Yb.dtype))
        add_ids.append(np.array([cid] * len(y)))
        print(f"  {k + 1}/{len(claims)} {cid}: 候補 {len(y)} 件（正解 {sum(y)} 件）", flush=True)
        _save_llm_cache(args.llm_cache, llm_cache)
    if not add_Y:
        print("追加できる請求項がありませんでした（532件と同じ番号・正解なし・請求項なしは除きます）。")
        return
    save = {k: base[k] for k in base.files if not k.startswith("X2_fold")}
    save["X1"] = np.vstack([X1b] + add_X1)
    save["X2"] = np.vstack([X2b] + add_X2)
    save["Y"] = np.concatenate([Yb] + add_Y)
    ids_all = np.concatenate([idsb.astype(str)] + add_ids)
    save["ids"] = ids_all.astype("<U%d" % max(13, max(len(x) for x in ids_all)))
    out = args.out if args.out.endswith(".npz") else "sao_selector14_train_plus.npz"
    np.savez_compressed(out, **save)
    print(f"保存しました: {out}（追加した請求項 {len(add_Y)} 件・候補 {sum(len(y) for y in add_Y):,} 件）。"
          "アプリで使うときは sao_selector14_train.npz という名前に置き換えてください。")


def main_eval():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-dir", default=".",
        help="patent_pipeline.py（evaluate_triples等）が置いてあるディレクトリ",
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="claims_532_for_gold.json / gold_sao_532_merged.json が置いてあるディレクトリ",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--claims-file", default=None,
                        help="評価する請求項のJSON（既定: data-dir の claims_532_for_gold.json）。"
                             "アプリの「正解データとして書き出す」で作った他分野の請求項を指定できる")
    parser.add_argument("--gold-file", default=None,
                        help="正解SAOのJSON（既定: data-dir の gold_sao_532_merged.json）")
    parser.add_argument("--method", default="a2", choices=["a2", "model", "llm-select"],
                        help="a2（既定・最終方式）: 構成要素をLLMで固定したGiNZAの規則／model: 学習済み選別モデル（実験14・比較用）／"
                             "llm-select: 学習なしの試作（GiNZAの規則＋LLMの抽出→LLMによる追加）")
    parser.add_argument("--pool", default="wide", choices=["standard", "wide"],
                        help="--method llm-select の候補の範囲。standard: GiNZAの規則＋LLM／wide（既定）: Recall重視で、"
                             "係り受け・区間の主役・題名・同一ノードの結合の規則の候補も加える")
    parser.add_argument("--with-translate", action="store_true",
                        help="--method llm-select で、英訳経由の抽出（方式C）の候補も加える（LLMの呼び出しが増える）")
    parser.add_argument("--translate-cache", default="translate_cache.json",
                        help="--with-translate の英訳経由の抽出結果を保存・再利用するファイル")
    parser.add_argument("--components", default="ginza", choices=["ginza", "llm"],
                        help="--method llm-select の土台。ginza: 通常のGiNZAの規則／llm: 構成要素をLLMで先に取り出し、"
                             "GiNZAはそれを分割せずに規則で関係を取る（LLMの呼び出しが1件あたり1回増える）")
    parser.add_argument("--votes", type=int, default=1, choices=[1, 2],
                        help="--method llm-select の③で、LLMに2回聞いて一致したものだけ追加するなら 2（精度重視・時間は約2倍）")
    parser.add_argument("--sample", type=int, default=0,
                        help="全体から等間隔にN件を選んで評価する（予備実験用。例：--sample 50）")
    parser.add_argument("--select-cache", default="select_cache.json",
                        help="--method llm-select の選別のLLMの出力を保存・再利用するファイル")
    parser.add_argument("--retrain", action="store_true",
                        help="--claims-file / --gold-file の正解データを532件の学習データに足した、新しい学習データ"
                             "（--out に .npz の名前を指定。既定 sao_selector14_train_plus.npz）を作る")
    parser.add_argument("--external", action="store_true",
                        help="532件すべてで学習した1つの選別モデルで評価する（他分野など、学習に使っていない"
                             "請求項の評価用）。指定しない場合、532件の請求項は交差検証の分割モデルで評価する")
    parser.add_argument("--host", default=None)
    parser.add_argument(
        "--mode", default="selected12", choices=["selected12"],
        help="抽出方法。実験13（主指標 F1 56.0%%）のみ（名前は互換のため selected12 のまま）。",
    )
    parser.add_argument("--backend", default="ollama", choices=["ollama", "deepl"],
                         help="--mode translate専用。翻訳エンジン。"
                              "deeplはDEEPL_API_KEY環境変数（または--deepl-key）が必要")
    parser.add_argument("--deepl-key", default=None, help="DeepL APIキー（省略時はDEEPL_API_KEY環境変数）")
    parser.add_argument("--limit", type=int, default=10, help="評価するクレーム数（先頭からN件）")
    parser.add_argument(
        "--format", default=None,
        choices=["順次列挙形式", "構成要素列挙形式", "ジェプソン的形式"],
        help="classify_claim_format()でこの表現形式に分類されたクレームだけに絞り込む"
             "（--limitは絞り込んだ後の件数に適用される）。省略時は絞り込まない。",
    )
    parser.add_argument("--out", default="eval_translate_result.json")
    parser.add_argument("--debug", action="store_true", help="タグ付き原文・英訳・抽出結果を表示する")
    parser.add_argument(
        "--eval-mode", default="lenient", choices=["lenient", "strict", "node", "exact"],
        help="exact: 【主指標】トリプル完全一致（主語・目的語は表記正規化後に完全一致、関係は"
             "表記正規化・漢字部分一致・同義語。意味的類似度は使わない。node_match_eval.py）。"
             "node: 主語どうし・目的語どうしを比べる評価（node_match_eval.py、閾値は"
             "--node-threshold）。"
             "lenient（デフォルト）: 表記正規化＋意味的類似度による緩い評価"
             "（evaluate_triples_lenient）。strict: 従来のsource/target完全一致"
             "（evaluate_triples）。GiNZA初期0.277→改良0.421と比較する場合はstrictを使う。",
    )
    parser.add_argument(
        "--semantic-threshold", type=float, default=0.75,
        help="--eval-mode lenientでの、埋め込みモデルによる意味的一致とみなす"
             "コサイン類似度の閾値（0〜1）。",
    )
    parser.add_argument(
        "--node-threshold", type=float, default=0.9,
        help="--eval-mode nodeでの、主語どうし・目的語どうしの埋め込み類似度の閾値。",
    )
    parser.add_argument(
        "--no-semantic", action="store_true",
        help="--eval-mode lenientで、埋め込みモデルを使わず表記正規化のみで評価する"
             "（sentence-transformersのインストール・モデルダウンロードが不要になる）。",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="--outに既存の結果ファイルがあれば読み込み、そこに含まれるクレームIDは"
             "スキップして続きから実行する。532件のような長時間の実行を、"
             "途中で中断されても失わずに再開するためのオプション。",
    )
    parser.add_argument(
        "--save-details", action="store_true",
        help="評価方法の検証用。各クレームについて、全抽出結果（predicted）、"
             "どの抽出がどの正解にどの方法（表記正規化／意味的類似度）と類似度で"
             "一致したか（match_details）、厳格評価の結果（strict）も--outに保存する。"
             "評価の数値そのものは変わらない。",
    )
    parser.add_argument(
        "--verify-risky-ginza", action="store_true",
        help="【実験2：条件付きGiNZA（検証型）】--mode llm_direct専用。1請求項内で"
             "ginza_has_fallback|有する候補が--risk-threshold件以上生成された"
             "場合だけ、その候補群をLLMに個別確認させ、確認できなかった候補を"
             "除外する。それ未満の通常のケースは変更しない。省略時（デフォルト）"
             "は実験1のbaselineと完全に同じ挙動。",
    )
    parser.add_argument(
        "--risk-threshold", type=int, default=_DEFAULT_RISK_THRESHOLD,
        help=f"--verify-risky-ginza時の閾値（1請求項あたりのginza_has_fallback|"
             f"有する候補数）。デフォルト{ts._DEFAULT_RISK_THRESHOLD}は、532件"
             f"baselineの分析（gh_n>=16の13件がPrecision25.3%%で候補総数の約4割を"
             f"占めていた）に基づく。",
    )
    parser.add_argument(
        "--verify-fanout", action="store_true",
        help="【実験8：ginza_has_fallback|有するのfan-out型検証】--mode llm_direct"
             "専用。--verify-risky-ginza（gh_n＝1請求項あたりの候補総数）とは別の、"
             "より細かい粒度の指標。ginza_has_fallback|有するの候補を同じsourceで"
             "グループ化し、1つのsourceが--fanout-risk-threshold件以上の別々の"
             "targetと繋がっている場合だけ、そのグループをLLMに個別確認させる"
             "（例：「係合爪の各々」が「第１部材」「係合爪」「半導体装置」等、"
             "6つの無関係なtargetに繋がってしまうようなケースを狙う）。"
             "--verify-risky-ginzaと独立に指定でき、併用もできる（両方の基準の"
             "いずれかに該当する候補が1回のLLM呼び出しでまとめて検証される）。"
             "省略時（デフォルト）は実験1〜7のbaselineと完全に同じ挙動。",
    )
    parser.add_argument(
        "--fanout-risk-threshold", type=int, default=_DEFAULT_FANOUT_RISK_THRESHOLD,
        help=f"--verify-fanout時の閾値（同一source内での、別々のtarget数）。"
             f"デフォルト{ts._DEFAULT_FANOUT_RISK_THRESHOLD}は、raw候補×gold_sao_"
             f"532_merged.jsonのシミュレーション分析（fan-out=1で55.9%%、2で"
             f"66.5%%、3で53.1%%、4で49.2%%、5で27.3%%、6以上で20.9%%）に基づく。",
    )
    parser.add_argument(
        "--verify-extra-risky", action="store_true",
        help="【実験3：検証型の対象拡張】--mode llm_direct専用。"
             "ginza_has_fallback|有する（--verify-risky-ginza）以外にも、"
             "532件baselineの精査で見つかった危険な(type,関係語)——"
             "claim_title_ginza|有する（候補--claim-title-risk-threshold件以上）、"
             "attribute|の（常に）、llm_direct側の「超える」「方向」"
             "「より小さい」「位置する」（常に）——を同じLLM再確認の仕組みで"
             "検証する。--verify-risky-ginzaと独立に指定でき、併用もできる。",
    )
    parser.add_argument(
        "--claim-title-risk-threshold", type=int, default=_DEFAULT_CLAIM_TITLE_RISK_THRESHOLD,
        help=f"【実験5：既存GiNZAルールの閾値調整】--verify-extra-risky時の、"
             f"claim_title_ginza|有するの検証閾値（1請求項あたりの候補数）。"
             f"デフォルト{ts._DEFAULT_CLAIM_TITLE_RISK_THRESHOLD}は実験3の値。"
             f"実験4後の残存FP分析で、閾値のすぐ下（候補数10〜13件）にも"
             f"Precision72.0%%程度の危険な塊が残っていることが判明したため、"
             f"10まで下げて適用範囲を広げられるようにした。",
    )
    parser.add_argument(
        "--verify-cache", default=None,
        help="--verify-risky-ginza / --verify-extra-risky時のLLM検証呼び出し用"
             "キャッシュファイル（--llm-cacheとは別ファイルで管理する）。"
             "仕組みは--llm-cacheと同じ。",
    )
    parser.add_argument(
        "--filter-invalid-targets", action="store_true",
        help="【実験4：target無条件フィルタ】--mode llm_direct専用。type・relationを"
             "問わず、targetがts._INVALID_TARGET_WORDS（「複数」「互い」等、"
             "gold_sao_532_merged.json全10,497件中に一件も出現しないと確認済みの"
             "18語）に完全一致する関係を一律除去する。LLM呼び出し不要"
             "（Ollama非依存の後処理のみ）。--verify-risky-ginza / "
             "--verify-extra-risky とは独立に併用できる。デフォルト（省略時）は"
             "実験1〜3のbaselineと完全に同じ挙動。",
    )
    parser.add_argument(
        "--filter-redundant-root-ownership", action="store_true",
        help="【実験7：クレームタイトルによる二重所有の除去】--mode llm_direct専用。"
             "sourceがクレームタイトル相当の構成要素（claim_title_ginza判定に使う"
             "ものと同じ）である「有する」系関係のうち、同じtargetを既に別の"
             "（より具体的な）構成要素が所有しているものを、過大包摂として一律"
             "除去する。LLM呼び出し不要（Ollama非依存の後処理のみ）。実験4の"
             "targetフィルタとは異なる根拠（target自体ではなく既存の所有関係との"
             "突き合わせ）に基づく決定的フィルタで、--filter-invalid-targets等"
             "とは独立に併用できる。デフォルト（省略時）は実験1〜6のbaselineと"
             "完全に同じ挙動。",
    )
    parser.add_argument(
        "--llm-cache", default=None,
        help="【実験2以降のためのLLM生出力キャッシュ】--mode llm_direct専用。"
             "指定したパスのJSONファイルにクレームIDごとのOllama生出力（GiNZA"
             "補完前）を保存し、次回以降は同じクレームでOllamaを呼ばずに再利用する。"
             "GiNZA側のロジック（_apply_ginza_fallback_and_normalize等）だけを"
             "変更した実験を、Ollamaの再呼び出しなしで再評価できるようにするため。"
             "ファイルが存在しない場合は空のキャッシュから始めて新規作成し、既存の"
             "場合は追記（既存エントリは上書きされない限り保持）する。--out（評価"
             "結果）とは別ファイルで管理する。省略時（デフォルト）は今まで通り"
             "毎クレーム必ずOllamaを呼ぶ（挙動は変わらない）。",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.pipeline_dir)
    pp = sys.modules[__name__]  # 統合後はこのファイル自身

    if args.retrain:
        retrain_with_extra(args, pp)
        return
    data_dir = _pathlib.Path(args.data_dir)
    with open(args.claims_file or (data_dir / "claims_532_for_gold.json"), encoding="utf-8") as f:
        claims = json.load(f)
    with open(args.gold_file or (data_dir / "gold_sao_532_merged.json"), encoding="utf-8") as f:
        gold_all = json.load(f)

    if args.format:
        claims = [c for c in claims if pp.classify_claim_format(c["text"]) == args.format]
        print(f"表現形式「{args.format}」に絞り込み: {len(claims)}件")

    if args.sample:
        # 先頭からではなく、全体から等間隔に選ぶ（会社・年・分野が偏らないように。毎回同じ請求項になる）
        step = len(claims) / args.sample
        claims = [claims[int(k * step)] for k in range(min(args.sample, len(claims)))]
        print(f"全体から等間隔に {len(claims)} 件を選びました（予備実験用）")
    claims = claims[: args.limit] if args.limit else claims

    out_path = _pathlib.Path(args.out)
    per_claim = []
    done_ids = set()
    if args.resume and out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            per_claim = prev.get("per_claim", [])
            done_ids = {c["id"] for c in per_claim}
            print(f"--resume: {out_path} から完了済み{len(done_ids)}件を読み込みました。続きから実行します。")
        except Exception as e:  # noqa: BLE001 -- 読み込み失敗時は最初からでも進められるようにする
            print(f"--resume: {out_path} の読み込みに失敗したため、最初から実行します（{e}）")
            per_claim = []
            done_ids = set()

    targets = [c for c in claims if c["id"] not in done_ids]
    n_target = len(targets)
    t0 = time.time()
    n_done_this_run = 0

    llm_cache = _load_llm_cache(args.llm_cache)
    if llm_cache is not None:
        print(f"--llm-cache: {args.llm_cache}（既存{len(llm_cache)}件を読み込み）")
    verify_cache = _load_llm_cache(args.verify_cache)
    if verify_cache is not None:
        print(f"--verify-cache: {args.verify_cache}（既存{len(verify_cache)}件を読み込み）")
    if args.verify_risky_ginza:
        print(f"--verify-risky-ginza有効（閾値: gh_n>={args.risk_threshold}）")
    if args.verify_fanout:
        print(f"--verify-fanout有効（閾値: 同一source内のtarget数>={args.fanout_risk_threshold}）")
    extra_risk_rules = _build_extra_risk_rules(args.claim_title_risk_threshold) if args.verify_extra_risky else None
    if args.verify_extra_risky:
        rules_desc = ", ".join(f"{r['type']}|{r['relation']}(閾値{r['threshold']})" for r in extra_risk_rules)
        print(f"--verify-extra-risky有効: {rules_desc}")
    if args.filter_invalid_targets:
        words_desc = "、".join(sorted(_INVALID_TARGET_WORDS))
        print(f"--filter-invalid-targets有効: target完全一致で無条件除去する語 = {words_desc}")
    if args.filter_redundant_root_ownership:
        print("--filter-redundant-root-ownership有効: クレームタイトルによる二重所有を除去")

    # 抽出は実験14（区間内のノード拡張の組＋係り受け候補＋区間の主役の候補＋2段階選別）のみ。交差検証の分割ごとに、その分割を
    # 除いた請求項だけで学習した選別モデルを使う（評価する請求項を学習に使わない）
    pass  # （統合済み）import sao_selector
    pass  # （統合済み）import sao_selector14
    base532 = data_dir / "claims_532_for_gold.json"
    fold_of, fold_selector, fold_threshold = {}, [], []
    if not args.external and base532.exists():
        folds = cv_folds({c["id"]: c["text"] for c in json.load(open(base532, encoding="utf-8"))})
        fold_of = {cid: k for k, f in enumerate(folds) for cid in f}
    if args.method == "a2":
        pass  # （統合済み）import llm_select
        fold_of = {}
        select_cache = _load_llm_cache(args.select_cache) or {}
        ls_modes = {}
        print(f"最終方式（A2：構成要素をLLMで固定したGiNZAの規則）で評価します。LLMの出力の保存先: {args.select_cache}")
    if args.method == "llm-select":
        pass  # （統合済み）import llm_select
        fold_of = {}
        select_cache = _load_llm_cache(args.select_cache) or {}
        translate_cache = (_load_llm_cache(args.translate_cache) or {}) if args.with_translate else None
        ls_modes = {}
        print(f"学習なしの方法（GiNZAの規則＋LLMの抽出→LLMによる選別→ルールで整理）で評価します。"
              f"候補の範囲: {args.pool}／土台: {args.components}／確認の回数: {args.votes}／"
              f"選別の出力の保存先: {args.select_cache}")
    if args.method == "model" and any(c["id"] in fold_of for c in claims):
        import numpy as _np
        _train = _np.load(TRAIN_FILE14, allow_pickle=False)
        fold_threshold = [float(t) for t in _train["fold_thresholds"]]
        print("交差検証の5分割ごとに選別モデルを学習しています…")
        # 2段目の構造特徴は、その分割の学習用請求項だけで作ったもの（X2_fold{k}）を使う
        fold_selector = [Selector14(exclude_ids=set(f), fold=k) for k, f in enumerate(folds)]
        print(f"  分割ごとのしきい値: {fold_threshold}")
    selector_all = None
    if args.method == "model" and any(c["id"] not in fold_of for c in claims):
        print("学習に使っていない請求項は、532件すべてで学習した選別モデルで評価します…")
        selector_all = Selector14()
    if args.eval_mode in ("node", "exact"):
        pass  # （統合済み）import node_match_eval

    for c in targets:
        cid = c["id"]
        ls_variants = None
        try:
            extra = {}
            if args.method == "a2":
                info, judged, _raw = analyze_claim_a2(ts, pp, c["text"], cache=select_cache,
                                                                 model=args.model, host=args.host)
                ls_modes[info.get("mode")] = ls_modes.get(info.get("mode"), 0) + 1
                if info.get("mode") != "llm":
                    print(f"  注意: {cid} は {str(_raw)[:120]}")
                predicted = [{"source": cc["source"], "relation": cc["relation"], "target": cc["target"],
                              "type": "a2"} for cc, j in zip(info["cands"], judged) if j["selected"]]
                ls_variants = {
                    "A 通常のGiNZAの規則": [cc for cc in info["cands"] if any(x.startswith("G:") for x in cc["srcs"])],
                    "A2 最終方式（構成要素をLLMで固定）": predicted,
                }
                _save_llm_cache(args.select_cache, select_cache)
            elif args.method == "llm-select":
                info, judged, _raw = analyze_claim_ls(ts, pp, c["text"], llm_cache=llm_cache,
                                                              select_cache=select_cache, claim_id=cid,
                                                              model=args.model, host=args.host, pool=args.pool,
                                                              with_translate=args.with_translate,
                                                              translate_cache=translate_cache, votes=args.votes,
                                                              components=args.components)
                if info.get("component_error"):
                    print(f"  注意: {cid} の構成要素の取り出しに失敗: {info['component_error'][:120]}")
                if info.get("translate_error"):
                    print(f"  注意: {cid} の英訳経由の抽出に失敗: {info['translate_error'][:120]}")
                ls_modes[info.get("mode")] = ls_modes.get(info.get("mode"), 0) + 1
                if info.get("mode") != "llm":
                    print(f"  注意: {cid} は {_raw[:120]}")
                predicted = [{"source": cc["source"], "relation": cc["relation"], "target": cc["target"],
                              "type": "llm_select"} for cc, j in zip(info["cands"], judged) if j["selected"]]
                # 同じ候補から、方式ごとの出力も作って比べる（LLMの呼び出しは増えない）
                _cs = info["cands"]
                ls_variants = {
                    "A GiNZAの規則だけ": [cc for cc in _cs if any(x.startswith("G:") for x in cc["srcs"])],
                    "B LLMの直接抽出（タグ付き日本語）": [cc for cc in _cs
                                                  if "LLM" in families(cc["srcs"])],
                }
                if args.components == "llm":
                    ls_variants["A2 構成要素をLLMで固定したGiNZAの規則"] = [
                        cc for cc in _cs if any(x.startswith("GF:") for x in cc["srcs"]) or "ST:工程" in cc["srcs"]]
                if args.with_translate:
                    ls_variants["C 英訳経由（タグ付き→英訳）"] = [cc for cc in _cs
                                                          if "英訳" in families(cc["srcs"])]
                ls_variants["D 学習なし方式（本体：規則＋LLMの追加＋整理）"] = predicted
                ls_variants["参考：候補すべて（Recallの上限）"] = _cs
                _save_llm_cache(args.select_cache, select_cache)
                if translate_cache is not None:
                    _save_llm_cache(args.translate_cache, translate_cache)
            elif cid in fold_of:
                selector = fold_selector[fold_of[cid]]
                thr = fold_threshold[fold_of[cid]]
                if selector.fold_max_per_pair:
                    extra["max_per_pair"] = selector.fold_max_per_pair[fold_of[cid]]
            else:
                selector, thr = selector_all, selector_all.threshold
            if args.method == "model":
                _, predicted = analyze_claim_selected14(
                    ts, pp, selector, c["text"], threshold=thr,
                    llm_cache=llm_cache, claim_id=cid, model=args.model, host=args.host, **extra)
        except Exception as e:  # noqa: BLE001 -- 1件の失敗で全体を止めない
            print(f"[{cid}] エラーのためスキップ: {e}")
            continue
        gold = gold_all[cid]
        if args.eval_mode == "exact":
            metrics = evaluate_triples_exact(pp, predicted, gold)
            # 補助指標（緩い一致）の正解数も記録する（主指標は完全一致のまま）
            metrics["緩い一致の正解数"] = {str(lv): loose_match_count(pp, predicted, gold, lv)
                                      for lv in LOOSE_LEVELS}
            # 構造評価（ノードの同一視の規則を正解と抽出の両方にかけてから、部分一致で比べる）
            metrics["構造評価"] = structure_counts(pp, predicted, gold)
            if args.method in ("llm-select", "a2") and ls_variants:
                metrics["方式の比較"] = {
                    name: dict(structure_counts(pp, pv, gold),
                               exact=evaluate_triples_exact(pp, pv, gold)["正解数"],
                               n_pred_raw=len(pv), n_gold_raw=len(gold))
                    for name, pv in ls_variants.items()}
        elif args.eval_mode == "node":
            metrics = evaluate_triples_node(pp, predicted, gold, theta=args.node_threshold)
        elif args.eval_mode == "lenient":
            metrics = pp.evaluate_triples_lenient(
                predicted, gold, lenient_relation_match=True,
                semantic_threshold=args.semantic_threshold,
                use_semantic=not args.no_semantic,
            )
        else:
            metrics = pp.evaluate_triples(predicted, gold, lenient_relation_match=True)
        # unmatched_gold/unmatched_pred は従来 --debug 時のみ画面表示していたが、
        # 532件規模では画面出力を後から見返すのは非現実的なので、常に --out の
        # JSONへ保存する（誤抽出の傾向を、実行後にファイルから集計できるようにするため）。
        keep_keys = [
            "precision", "recall", "f1", "正解数", "システム抽出数", "正解データ数",
            "unmatched_gold", "unmatched_pred",
        ]
        if "正解内訳" in metrics:
            keep_keys.append("正解内訳")
        if "緩い一致の正解数" in metrics:
            keep_keys.append("緩い一致の正解数")
        if "構造評価" in metrics:
            keep_keys.append("構造評価")
        if "方式の比較" in metrics:
            keep_keys.append("方式の比較")

        # 【type×関係語ごとのTP/FP内訳】translate_sao.pyの各関係には、どの処理が
        # 作ったかを示す"type"（llm_direct / claim_title_ginza /
        # ginza_has_fallback / attribute / translate 等）が付いている。
        # 「GiNZAフォールバックを丸ごとON/OFFする」のではなく、「(type, 関係語)の
        # 組み合わせごとに、これまでの実測でPrecision（TP/(TP+FP)）が高ければ
        # 採用し、低ければ不採用にする」という、より粒度の細かい判断をしたい、
        # というユーザーの提案に基づく。
        # predicted（今回のシステム予測全体、type付き）からunmatched_pred
        # （FP）をマルチセット差分すれば、残りが必ずTPになるので、追加の
        # Ollama呼び出しなしで(type, 関係語)単位のTP/FPを集計できる。
        def _rel_key(r):
            return (r.get("source"), r.get("relation"), r.get("target"), r.get("type"))

        pred_counter = Counter(_rel_key(r) for r in predicted)
        unmatched_pred_counter = Counter(_rel_key(r) for r in metrics.get("unmatched_pred", []))
        matched_counter = pred_counter - unmatched_pred_counter  # 残り＝TP

        tp_fp_by_type_relation = {}
        for key, cnt in matched_counter.items():
            tr_key = f"{key[3] or 'unknown'}|{key[1]}"
            tp_fp_by_type_relation.setdefault(tr_key, {"tp": 0, "fp": 0})
            tp_fp_by_type_relation[tr_key]["tp"] += cnt
        for key, cnt in unmatched_pred_counter.items():
            tr_key = f"{key[3] or 'unknown'}|{key[1]}"
            tp_fp_by_type_relation.setdefault(tr_key, {"tp": 0, "fp": 0})
            tp_fp_by_type_relation[tr_key]["fp"] += cnt
        if tp_fp_by_type_relation:
            keep_keys.append("tp_fp_by_type_relation")
            metrics = {**metrics, "tp_fp_by_type_relation": tp_fp_by_type_relation}

        entry = {"id": cid, **{k: v for k, v in metrics.items() if k in keep_keys}}
        if args.save_details:
            # 評価方法の検証用：全抽出結果、対応付けの詳細（方法・類似度）、
            # 厳格評価（evaluate_triples）の結果を同時に保存する。
            entry["predicted"] = predicted
            if args.eval_mode == "lenient":
                details = _lenient_match_details(
                    pp, predicted, gold, args.semantic_threshold, not args.no_semantic)
                if len(details) != metrics["正解数"]:
                    print(f"  [{cid}] 警告: 対応付けの記録（{len(details)}件）が公式の正解数"
                          f"（{metrics['正解数']}件）と一致しません")
                entry["match_details"] = details
            strict = pp.evaluate_triples(predicted, gold, lenient_relation_match=True)
            entry["strict"] = {k: strict[k] for k in ("precision", "recall", "f1", "正解数",
                                                       "システム抽出数", "正解データ数")}
        per_claim.append(entry)
        n_done_this_run += 1
        elapsed_run = time.time() - t0
        avg = elapsed_run / n_done_this_run
        remaining = n_target - n_done_this_run
        eta_min = avg * remaining / 60
        print(f"[{cid}] precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
              f"(正解{metrics['正解数']}/抽出{metrics['システム抽出数']}/正解データ{metrics['正解データ数']}) "
              f"[{n_done_this_run}/{n_target}件 経過{elapsed_run / 60:.1f}分 残り目安{eta_min:.1f}分]")
        if args.debug:
            for g in metrics["unmatched_gold"]:
                print("    見逃し:", g["source"], g["relation"], g["target"])
            for p in metrics["unmatched_pred"]:
                print("    誤抽出:", p["source"], p["relation"], p["target"])
        # 1件ごとに保存する（PCのスリープ・強制終了・ネットワーク切断等で
        # 途中終了しても、ここまでの結果は --resume で失わずに再開できる）。
        _save(out_path, per_claim, args, elapsed_run, done=False)
        _save_llm_cache(args.llm_cache, llm_cache)
        _save_llm_cache(args.verify_cache, verify_cache)

    elapsed = time.time() - t0
    _save(out_path, per_claim, args, elapsed, done=True)
    _save_llm_cache(args.llm_cache, llm_cache)
    _save_llm_cache(args.verify_cache, verify_cache)
    agg = _aggregate(per_claim)

    print("\n=== 集計 ===")
    if args.method == "a2":
        print(f"構成要素をLLMで取り出せた件数: {ls_modes.get('llm', 0)}／LLMを呼べず通常のGiNZAの規則で代用: "
              f"{ls_modes.get('rules', 0)}")
        if ls_modes.get("rules"):
            print("  ※ LLMを呼べなかった請求項があります。この結果は最終方式の正しい評価ではありません。"
                  "Ollama を起動して（ollama serve）もう一度実行してください（--resume で続きから）。")
    if args.method == "llm-select":
        print(f"LLMによる選別が使えた件数: {ls_modes.get('llm', 0)}／③が失敗し規則で代用: {ls_modes.get('both', 0)}"
              f"／GiNZAの規則だけ: {ls_modes.get('rules', 0)}")
        if ls_modes.get("both") or ls_modes.get("rules"):
            print("  ※ LLMを呼べなかった請求項があります。この結果は「学習なし（LLMによる選別）」の正しい評価では"
                  "ありません。Ollama を起動して（ollama serve）もう一度実行してください。")
    print(f"件数: {len(per_claim)}（今回の実行で処理: {n_done_this_run}件）  "
          f"所要時間（今回の実行分）: {elapsed:.1f}秒  評価方法: {args.eval_mode}"
          + ("" if args.eval_mode in ("strict", "exact")
             else f"（主語・目的語ごとの類似度の閾値={args.node_threshold}）" if args.eval_mode == "node"
             else f"（閾値={args.semantic_threshold}, 意味的類似度={'無効' if args.no_semantic else '有効'}）"))
    print(f"MICRO precision={agg['micro']['precision']:.4f} recall={agg['micro']['recall']:.4f} f1={agg['micro']['f1']:.4f}")
    print(f"MACRO precision={agg['macro']['precision']:.4f} recall={agg['macro']['recall']:.4f} f1={agg['macro']['f1']:.4f}")
    if args.eval_mode == "exact":
        rows = [m for m in per_claim if "緩い一致の正解数" in m]
        if rows:
            n_pred = sum(m["システム抽出数"] for m in rows)
            n_gold = sum(m["正解データ数"] for m in rows)
            print("補助指標（緩い一致。① 厳密評価は上の完全一致）:")
            for lv, name in LOOSE_LEVELS.items():
                tp = sum(m["緩い一致の正解数"][str(lv)] for m in rows)
                P = tp / n_pred if n_pred else 0.0
                R = tp / n_gold if n_gold else 0.0
                F = 2 * P * R / (P + R) if P + R else 0.0
                print(f"  {name}: precision={P:.4f} recall={R:.4f} f1={F:.4f}")
        rows = [m for m in per_claim if "構造評価" in m]
        if rows:
            tot = {k: sum(m["構造評価"][k] for m in rows) for k in rows[0]["構造評価"]}
            print("② 構造評価（ノードの同一視の規則を正解と抽出の両方にかけ、主語・目的語は部分一致で比べる）:")
            for key, name in (("struct", "構造F1（関係は完全一致と同じ規則）"),
                              ("skeleton", "骨組みF1（関係名・向きを問わない）")):
                P, R, F = prf_counts(tot[key], tot["n_pred"], tot["n_gold"])
                print(f"  {name}: precision={P:.4f} recall={R:.4f} f1={F:.4f}")
            g = tot["n_gold"] or 1
            print(f"  要素別の一致率（正解のSAOのうち）: 主語 {tot['comp_s'] / g:.4f}／関係 {tot['comp_r'] / g:.4f}"
                  f"／目的語 {tot['comp_o'] / g:.4f}")
        rows = [m for m in per_claim if "方式の比較" in m]
        if rows:
            print(f"\n=== 方式の比較（同じ {len(rows)} 件・同じ候補から。数字は％） ===")
            print(f"{'方式':<34}{'抽出数/件':>8}{'完全一致 P':>10}{'R':>7}{'F1':>7}{'構造 P':>9}{'R':>7}{'F1':>7}")
            for name in rows[0]["方式の比較"]:
                t = {k: sum(m["方式の比較"][name][k] for m in rows if name in m["方式の比較"])
                     for k in rows[0]["方式の比較"][name]}
                eP, eR, eF = prf_counts(t["exact"], t["n_pred_raw"], t["n_gold_raw"])
                sP, sR, sF = prf_counts(t["struct"], t["n_pred"], t["n_gold"])
                print(f"{name:<34}{t['n_pred_raw'] / len(rows):>8.1f}{100 * eP:>10.1f}{100 * eR:>7.1f}{100 * eF:>7.1f}"
                      f"{100 * sP:>9.1f}{100 * sR:>7.1f}{100 * sF:>7.1f}")
            if args.method == "llm-select":
                print("（A・B・C は選別の前の各方式の出力そのもの。D は規則の結果に、LLM が候補から選んだ関係を追加し、"
                      "ルールで整理したもの）")
            else:
                print("（A は構成要素を固定しない通常のGiNZAの規則、A2 は構成要素をLLMで固定したGiNZAの規則＋構造の整理）")

    # 【(type, 関係語)ごとのPrecision表】claim_title_ginza / ginza_has_fallback /
    # attribute（＝GiNZA由来のフォールバックで、LLMの自由な言い換えと違って
    # 語彙が限られており、「この(type, 関係語)は採用する/しない」という
    # ルールを現実的に作れる）についてだけ表示する。llm_direct/translateは
    # LLMの自由な言い換えで関係語のバリエーションが膨大になり、個別の
    # ON/OFFルールを作る対象として実用的ではないため表示対象から外す
    # （JSON側にはtype_relation_tableとして全type分を保存済み）。
    rows = _aggregate_type_relation(per_claim)
    if rows:
        print("\n=== GiNZAフォールバック系の (type, 関係語) 別 TP/FP/Precision ===")
        print("（Precisionが低いものは、その(type, 関係語)だけ不採用にする候補）")
        print(f"{'type':<20} {'relation':<20} {'TP':>5} {'FP':>5} {'Precision':>10}")
        for r in rows:
            print(f"{r['type']:<20} {r['relation']:<20} {r['tp']:>5} {r['fp']:>5} {r['precision']:>10.3f}")

    print(f"保存しました: {out_path}")


# ===========================================================================
# 元のモジュール名で呼べるようにする名前空間
# （app.py からは pp.sao_selector12.Selector() のように使う）
# ===========================================================================
_THIS = sys.modules[__name__]

en_relation_rules = _types.SimpleNamespace(ACOMP_LABELS=ACOMP_LABELS, ACTIVE_PREP_VERBS=ACTIVE_PREP_VERBS, ACTIVE_VERB_LABELS=ACTIVE_VERB_LABELS, CAPABLE_OF_GERUND_LABELS=CAPABLE_OF_GERUND_LABELS, CONFIGURE_XCOMP_LABELS=CONFIGURE_XCOMP_LABELS, CONSIST_OF_VERBS=CONSIST_OF_VERBS, HAS_VERBS=HAS_VERBS, PASSIVE_ADVMOD_OVERRIDES=PASSIVE_ADVMOD_OVERRIDES, PASSIVE_VERB_LABELS=PASSIVE_VERB_LABELS, PREP_NOUN_PATTERNS=PREP_NOUN_PATTERNS, REVERSED_PASSIVE_VERB_LABELS=REVERSED_PASSIVE_VERB_LABELS, SURFACE_WORDS=SURFACE_WORDS, TAG_RE=TAG_RE, _LazyNLP=_LazyNLP, _load_nlp=_load_nlp, _nlp_instance=_nlp_instance, _scan_passive_targets=_scan_passive_targets, _verb_key=_verb_key, conj_chain=conj_chain, dedup=dedup, extract_relations=extract_relations, extract_relations_from_text=extract_relations_from_text, is_tag=is_tag, nlp=nlp_en)
translate_sao = _THIS  # 関数の差し替え（_ollama_chat など）がそのまま効くよう、このファイル自身
node_match_eval = _types.SimpleNamespace(IDENTITY_TAILS=IDENTITY_TAILS, LOOSE_LEVELS=LOOSE_LEVELS, _IDENTITY_REST_RE=_IDENTITY_REST_RE, _KANJI_NUM=_KANJI_NUM, _NUM_RE=_NUM_RE, _align=_align, _loose_node=_loose_node, canon_pair=canon_pair, evaluate_triples_exact=evaluate_triples_exact, evaluate_triples_node=evaluate_triples_node, identity_map=identity_map, loose_match_count=loose_match_count, node_score=node_score, numbers=numbers, prf_counts=prf_counts, rel_match=rel_match, structure_counts=structure_counts)
nested_graph = _types.SimpleNamespace(COMPONENT_PALETTE=COMPONENT_PALETTE, DEEPSEA_BG=DEEPSEA_BG, DEEPSEA_DIM=DEEPSEA_DIM, DEEPSEA_NODE=DEEPSEA_NODE, DEEPSEA_ROOT=DEEPSEA_ROOT, FONT=FONT, HAS_RELATIONS=HAS_RELATIONS, RELATION_COLORS=RELATION_COLORS, RELATION_GROUP_NAMES=RELATION_GROUP_NAMES, TREE_LINE=TREE_LINE, TREE_STYLES=TREE_STYLES, WRAP=WRAP, _esc=_esc, _tree_levels=_tree_levels, component_colors=component_colors, relation_color=relation_color, relation_group=relation_group, relations_to_flat_dot=relations_to_flat_dot, relations_to_nested_dot=relations_to_nested_dot, relations_to_tree_dot=relations_to_tree_dot, relations_to_tree_html=relations_to_tree_html, tree_cross_relations=tree_cross_relations)
claim_segmenter = _types.SimpleNamespace(_COMPOSE_ONLY_RE=_COMPOSE_ONLY_RE, _COORD_SPLIT_RE=_COORD_SPLIT_RE, _DISTRIB_RE=_DISTRIB_RE, _ENZAI_NAME=_ENZAI_NAME, _JEPSON_RE=_JEPSON_RE, _NEW_TOPIC_RE=_NEW_TOPIC_RE, _ensure_enzai_component=_ensure_enzai_component, _enzai=_enzai, _split_line=_split_line, distribute=distribute, segment_relations=segment_relations, split_claim=split_claim, to_sentence=to_sentence)
dep_pairs = _types.SimpleNamespace(ARG_DEPS=ARG_DEPS, CASES=CASES, _case_of=_case_of, _comp_map=_comp_map, _component_of=_component_of, _coordinated=_coordinated, _dependency_pairs=_dependency_pairs, _is_pred=_is_pred, _label=_label, dependency_pairs=dependency_pairs, pairs_from_doc=pairs_from_doc)
seg_pairs = _types.SimpleNamespace(HAS_VERBS=HAS_VERBS_sp, _tree_dist=_tree_dist, pairs_from_segment=pairs_from_segment, segment_pairs=segment_pairs)
sao_selector = _types.SimpleNamespace(FORMATS=FORMATS, HERE=HERE, INVALID=INVALID, KEPT_INDEX=KEPT_INDEX, SIMP=SIMP, SIMPLIFIED=SIMPLIFIED, SRC_KEYS=SRC_KEYS, build_candidates=build_candidates9, claim_features=claim_features9, cv_folds=cv_folds, rel_group=rel_group)
sao_selector10 = _types.SimpleNamespace(HERE=HERE, SEG_KEYS=SEG_KEYS, build_candidates=build_candidates10, claim_features=claim_features10)
sao_selector11 = _types.SimpleNamespace(HERE=HERE, build_candidates=build_candidates11, claim_features=claim_features11, merge_occurrences=merge_occurrences)
sao_selector12 = _types.SimpleNamespace(CASE_KEYS=CASE_KEYS, HAS=HAS, HERE=HERE, Selector=Selector12, TRAIN_FILE=TRAIN_FILE12, _is_has=_is_has, _model=_model, build_candidates=build_candidates12, canon=canon, claim_features=claim_features12, cv_folds=cv_folds, select=select12, structural_features=structural_features12)
sao_selector13 = _types.SimpleNamespace(HERE=HERE, Selector=Selector13, TRAIN_FILE=TRAIN_FILE13, add_segment_candidates=add_segment_candidates, analyze_claim_selected=analyze_claim_selected13, build_candidates=build_candidates13, canon=canon13, claim_features=claim_features13, cv_folds=cv_folds, select=select13, structural_features=structural_features13)
node_pairs = _types.SimpleNamespace(FORMAL=FORMAL, HAS_LABELS=HAS_LABELS, LEAD=LEAD, MAX_NODES=MAX_NODES, NOUNISH=NOUNISH, OWNER_LABELS=OWNER_LABELS, QTY_RE=QTY_RE, _chains=_chains, _coord_partner=_coord_partner, _units=_units, keep_pair=keep_pair, node_pair_candidates=node_pair_candidates, pairs_in_segment=pairs_in_segment, segment_nodes=segment_nodes)
sao_selector14 = _types.SimpleNamespace(HAS_LIKE=HAS_LIKE, HAS_REL=HAS_REL, HERE=HERE, KINDS=KINDS, Selector=Selector14, TRAIN_FILE=TRAIN_FILE14, _pair_origin=_pair_origin, add_has_variants=add_has_variants, add_node_pair_candidates=add_node_pair_candidates, analyze_claim_selected=analyze_claim_selected14, build_candidates=build_candidates14, canon=canon14, claim_features=claim_features14, select=select14, structural_features=structural_features)
struct_extra = _types.SimpleNamespace(CONJ=CONJ, HAS_STEMS=HAS_STEMS, LEAD_RE=LEAD_RE, NOT_ITEM_RE=NOT_ITEM_RE, STEP_END_RE=STEP_END_RE, _clean=_clean, _np_left=_np_left, _span_text=_span_text, coord_expansions=coord_expansions, doc_text=doc_text, find_lists=find_lists, step_candidates=step_candidates)
comp_first = _types.SimpleNamespace(BARE_STEPS=BARE_STEPS, COMPONENT_SYSTEM=COMPONENT_SYSTEM, NOUNISH=NOUNISH_cf, _nounish=_nounish, _pattern=_pattern, extract_components_llm=extract_components_llm, forced_relations=forced_relations, guard_names=guard_names, llm_component_relations=llm_component_relations, parse_components=parse_components)
llm_select = _types.SimpleNamespace(BARE_STEPS=BARE_STEPS_ls, LLM_SRCS=LLM_SRCS, MAX_PAIRS_PER_CALL=MAX_PAIRS_PER_CALL, MAX_VARIANTS=MAX_VARIANTS, NON_NODES=NON_NODES, POOLS=POOLS, SELECT_SYSTEM=SELECT_SYSTEM, _GA_FIX_RE=_GA_FIX_RE, _HAS_RELS=_HAS_RELS, _ITEM_RE=_ITEM_RE, _QUANT_PREFIX_RE=_QUANT_PREFIX_RE, _add=_add, _chat_cached=_chat_cached, _family=_family, analyze_claim=analyze_claim_ls, analyze_claim_a2=analyze_claim_a2, build_rule_and_llm_candidates=build_rule_and_llm_candidates, clean_node=clean_node, families=families, grounded=grounded, group_pairs=group_pairs, is_base=is_base, judge=judge, normalize_candidates=normalize_candidates, parse_answer=parse_answer, parse_selection=parse_selection, select_prompt=select_prompt, select_with_llm=select_with_llm, structure_fixes=structure_fixes, tidy=tidy, translate_relations=translate_relations)
platform_core = _types.SimpleNamespace(COLUMN_ALIASES=COLUMN_ALIASES, CORPUS_FILE=CORPUS_FILE, CORPUS_NAME=CORPUS_NAME, DEFAULT_BANDS=DEFAULT_BANDS, FI_LEVELS=FI_LEVELS, GROUP_PALETTE=GROUP_PALETTE, HAS_WORDS=HAS_WORDS, HERE=HERE, METHOD_NAME=METHOD_NAME, METHOD_SCORE=METHOD_SCORE, MODEL_METHOD_NAME=MODEL_METHOD_NAME, MODEL_METHOD_SCORE=MODEL_METHOD_SCORE, OTHER_COLOR=OTHER_COLOR, PIPELINE_VERSION=PIPELINE_VERSION, RADAR_AXES=RADAR_AXES, STATUS_ACCEPT=STATUS_ACCEPT, STATUS_ORDER=STATUS_ORDER, STATUS_REJECT=STATUS_REJECT, STATUS_REVIEW=STATUS_REVIEW, THERMO_STOPS=THERMO_STOPS, _CLAIM_HEAD_RE=_CLAIM_HEAD_RE, _CONJ_RULES=_CONJ_RULES, _CORP_RE=_CORP_RE, _LEAD_PARTICLE_RE=_LEAD_PARTICLE_RE, _NODE_PREFIX_RE=_NODE_PREFIX_RE, _NUM=_NUM, _NUMERIC_RE=_NUMERIC_RE, _ORD_RE=_ORD_RE, _ORIGIN=_ORIGIN, _OZ_CSS=_OZ_CSS, _OZ_JS=_OZ_JS, _PREFIX_RE=_PREFIX_RE, _SUFFIX_RE=_SUFFIX_RE, _TAIL_RE=_TAIL_RE, _WC_NUMERIC_RE=_WC_NUMERIC_RE, _embed=_embed, _longest_path=_longest_path, _norm_col=_norm_col, _text_width=_text_width, apply_analysis=apply_analysis, assign_groups=assign_groups, base_term=base_term, build_network=build_network, claims_from_table=claims_from_table, classify=classify, clean_relation=clean_relation, company_name=company_name, company_tech_matrix=company_tech_matrix, company_year_bubble=company_year_bubble, detect_columns=detect_columns, display_node=display_node, effective_relations=effective_relations, export_excel=export_excel, feature_table=feature_table, fi_codes=fi_codes, fi_parts=fi_parts, fi_radar_data=fi_radar_data, finalize_dataset=finalize_dataset, find_corpus_file=find_corpus_file, first_claim=first_claim, group_colors=group_colors, highlight=highlight, highlight_colored=highlight_colored, is_has=is_has, layout_map=layout_map, layout_network=layout_network, layout_world=layout_world, load_corpus=load_corpus, make_patent=make_patent, new_dataset=new_dataset, norm_pid=norm_pid, origin_label=origin_label, oz_world_html=oz_world_html, patents_from_table=patents_from_table, patents_with_node=patents_with_node, percentile_scores=percentile_scores, read_table=read_table, relations_csv=relations_csv, review_table=review_table, reviews_from_csv=reviews_from_csv, reviews_to_csv=reviews_to_csv, sample_world_edges=sample_world_edges, sao_tokens=sao_tokens, similarity_explain=similarity_explain, similarity_matrix=similarity_matrix, status_counts=status_counts, structural_features=claim_structure_features, table_to_review=table_to_review, thermo_color=thermo_color, tidy_relations=tidy_relations, wordcloud_heat=wordcloud_heat, wordcloud_layout=wordcloud_layout, wordcloud_svg=wordcloud_svg, wordcloud_terms=wordcloud_terms)
eval_translate_sao = _types.SimpleNamespace(_FALLBACK_TYPES_FOR_TABLE=_FALLBACK_TYPES_FOR_TABLE, _aggregate=_aggregate, _aggregate_type_relation=_aggregate_type_relation, _lenient_match_details=_lenient_match_details, _load_llm_cache=_load_llm_cache, _save=_save, _save_llm_cache=_save_llm_cache, main=main_eval, retrain_with_extra=retrain_with_extra, ts=ts)


if __name__ == "__main__":
    # 評価（旧 eval_translate_sao.py）：python patent_pipeline.py --mode selected12 --eval-mode exact ...
    main_eval()
