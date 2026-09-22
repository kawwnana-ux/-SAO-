# -*- coding: utf-8 -*-
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
    import translate_sao as ts
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

import en_relation_rules as err

DEFAULT_MODEL = "qwen2.5:7b"

_TAG_OPEN = "《"
_TAG_CLOSE = "》"
# 《C1》が本来の形だが、翻訳モデルが《》の代わりに〈〉や<>、{}を使って
# しまうことがあるため、フォールバックとして主要な括弧パターンも許容する。
_BRACKET_TAG_RE = re.compile(r"[《〈<{\[]\s*(C\d+)\s*[》〉>}\]]")


def _load_pipeline(pipeline_dir=None):
    if pipeline_dir:
        sys.path.insert(0, pipeline_dir)
    import patent_pipeline as pp  # noqa: WPS433 (遅延import: pipeline_dirを先に通す必要があるため)
    return pp


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
        # 「（ａ）」「（ｂ）」の工程ラベル除外は、extract_patent_components_general
        # 側（patent_pipeline.py）で一元的に対応済み（_is_paren_step_label）。
        # ここでは重複対応しない。
        start_char = doc[c["start"]].idx
        end_tok = doc[c["end"]]
        end_char = end_tok.idx + len(end_tok.text)
        spans.append((start_char, end_char, c["text"]))

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
        options={"temperature": 0, "num_predict": _OLLAMA_MAX_OUTPUT_TOKENS},
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
    doc = err.nlp(clean)
    rels = []
    for sent in doc.sents:
        rels.extend(err.extract_relations(sent.as_doc()))
    return err.dedup(rels)


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


def analyze_claim_translate(text, pipeline_dir=None, model=DEFAULT_MODEL, host=None, pp=None,
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
_INVALID_TARGET_WORDS = {
    "複数", "互い", "こと", "もの", "場合", "状態", "様子",
    "全体", "一部", "両方", "それぞれ", "いずれか", "各々",
    "これ", "それ", "あれ", "ここ", "そこ",
    "一方", "他方",
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
        if len(idx) >= rule["threshold"]:
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
        extra_risk_rules=extra_risk_rules,
        model=model, host=host, verify_cache=verify_cache, claim_id=claim_id, debug=debug,
        filter_invalid_targets=filter_invalid_targets,
        filter_redundant_root_ownership=filter_redundant_root_ownership,
    )
