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
import glob

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
    writable_model_path = "/tmp/ja_ginza_model_copy"
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

FONT_PATH = "/tmp/NotoSansJP-Regular.ttf"
if not os.path.exists(FONT_PATH):
    os.system(
        f'curl -sL -o {FONT_PATH} '
        '"https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"'
    )
fm.fontManager.addfont(FONT_PATH)
FONT_PROP = fm.FontProperties(fname=FONT_PATH)

RELATION_WORDS = {
    # 基本方向・位置
    "間", "側", "上", "下", "内部", "外部",
    "周囲", "近傍", "前", "後", "間隔",
    # 部位（上下前後・中央系）
    "上部", "下部", "底部", "前部", "後部", "左側", "右側",
    "頂部", "頂上", "中央部", "中心部", "側部", "隅部", "角部",
    "一側", "他側", "一端", "他端",
    # 端・先端系
    "上端", "下端", "先端", "基端", "端部", "末端", "終端",
    "左端", "右端", "前端", "後端", "頂点",
    # 面
    "外周面", "内周面", "外面", "内面", "表面", "裏面", "上面", "下面",
    "側面", "底面", "天面", "端面", "接触面", "対向面",
    # 周辺・中間
    "中央", "中間", "周辺", "周縁", "周辺部", "縁", "縁部",
    "外周", "内周",
    # 方向
    "前方", "後方", "上方", "下方", "左方", "右方",
    "内側", "外側", "上側", "下側",
    "水平方向", "垂直方向", "長手方向", "幅方向", "厚さ方向", "径方向", "軸方向",
}

# 「有する」と同じ意味で使われる動詞（「Ａを備える」「Ａを具備する」等）
HAS_LEMMAS = {"有する", "備える", "具備する"}

# 「ことを特徴とする」のような決まり文句に出てくる、実在の構成要素ではない
# 一般的な語（構成要素としては登録しない）
GENERIC_NOUNS = {"こと", "もの", "とき", "場合", "特徴", "ため", "下記", "上記"}


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
    """
    for prefix in ("前記", "該"):
        if phrase.startswith(prefix) and phrase != prefix:
            phrase = phrase[len(prefix):]
    return phrase


# ============================================================
# ① 構成要素抽出
# ============================================================

def extract_patent_components_general(doc):
    components = []
    i = 0
    while i < len(doc):
        token = doc[i]

        if token.text in ("前記", "該", "うち", "乃至") or not token.text.strip():
            i += 1
            continue

        if _is_generic_relation_word_bigram(doc, i):
            i += 2
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
            while i < len(doc) and doc[i].pos_ in {"NOUN", "PROPN"}:
                if _is_generic_relation_word(doc[i]) or doc[i].text in ("前記", "該", "うち", "乃至") or not doc[i].text.strip():
                    break
                words.append(doc[i].text)
                i += 1
            end = i - 1
            phrase = _normalize_component_text("".join(words))
            if phrase not in RELATION_WORDS and phrase not in GENERIC_NOUNS:
                components.append({"text": phrase, "start": start, "end": end})
            continue

        if token.pos_ in {"NOUN", "PROPN"}:
            if _is_counter_word(token):
                i += 1
                continue
            start = i
            words = [token.text]
            i += 1
            while i < len(doc) and doc[i].pos_ in {"NOUN", "PROPN"}:
                if _is_generic_relation_word(doc[i]) or doc[i].text in ("前記", "該", "うち", "乃至") or not doc[i].text.strip():
                    break
                words.append(doc[i].text)
                i += 1
            end = i - 1
            phrase = _normalize_component_text("".join(words))
            if phrase not in RELATION_WORDS and phrase not in GENERIC_NOUNS:
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
    return unique_components


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
    """
    if token.dep_ != "obl":
        return False
    for child in token.children:
        if child.dep_ == "case":
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
        prefix = (child_comp["text"] if child_comp is not None else child.text) + "の"
        break

    return prefix + base


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

        if verb.lemma_ in HAS_LEMMAS:
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
                # 動詞自身に直接の目的語(obj)がない場合（受身形など）だけ、
                # 従来通り動詞連鎖を遡って構成要素を探す
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
                relations.append({
                    "source": source["text"],
                    "relation": label,
                    "target": target["text"],
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
                seen_path = set()
                while cursor.i not in seen_path:
                    seen_path.add(cursor.i)
                    if cursor.i == verb.i:
                        break
                    if cursor.dep_ == "acl":
                        crosses_acl = True
                        break
                    if cursor.head.i == cursor.i:
                        break
                    cursor = cursor.head
                if crosses_acl:
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
    COMPARISON_ADJ = {"小さい", "大きい", "高い", "低い", "長い", "短い", "多い", "少ない", "広い", "狭い"}
    for adj in doc:
        if adj.pos_ != "ADJ" or adj.lemma_ not in COMPARISON_ADJ:
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
        if source_comp is None or source_comp["text"] == target_comp["text"]:
            continue

        relations.append({
            "source": source_comp["text"],
            "relation": f"より{adj.text}",
            "target": target_comp["text"],
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

        head_component = find_component_by_token(components, verb.head.i)
        used_fallback = head_component is None
        if head_component is None:
            # 係り先が構成要素でない場合（別の動詞に連なっている等）は、
            # さらに上まで遡って構成要素を探す
            fallback = find_target_component_from_verb(components, verb)
            head_component = fallback[0] if fallback else None
        if head_component is None:
            continue

        if _is_passive(verb):
            # 受身：headが受け手（target）。「に」で係る語などが動作主（source）。
            source_candidates = []
            for child in verb.children:
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
                if source is None or source["text"] == head_component["text"]:
                    continue
                relations.append({
                    "source": source["text"],
                    "relation": verb.text,
                    "target": head_component["text"],
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

                # 「Ａ及びＢを含み」のように、objの前に「及び」等で
                # 繋がれた項目がある場合、それはGiNZAの解析上、objの
                # nmod修飾語として表れる。見逃さないよう、objの前に
                # 「及び」「又は」等のcc（等位接続）子がある場合は、
                # そのnmod修飾語も対象に加える。
                obj_candidates = [child]
                for nm in child.children:
                    if nm.dep_ == "nmod" and nm.i < child.i:
                        has_cc_between = any(
                            gc.dep_ == "cc" and nm.i < gc.i < child.i for gc in child.children
                        )
                        if has_cc_between:
                            obj_candidates.append(nm)

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
                        if not _is_list_item_component(doc, comp):
                            continue
                        if comp["text"] == effective_source["text"]:
                            continue
                        relations.append({
                            "source": effective_source["text"],
                            "relation": verb.text,
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

def _is_list_item_component(doc, comp):
    """
    「Ａと、Ｂと、Ｃと、…を有する（備える）」のような並列列挙で、
    その構成要素の直後に「と、」（区切りの格助詞＋読点）が来ているかどうかを
    判定する。「Ａと通信する」のような、単に「と」で係る場合（直後が
    読点でない）は対象外にする。

    「第１のトランジスタから第６のトランジスタと、…を有し」のような、
    範囲の始点側（「から」が直後に来る場合）も対象に含める
    （「乃至」は解析前に「から」へ正規化されるため、同じ扱いになる）。
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

    relations = []
    for verb in doc:
        if verb.lemma_ not in HAS_LEMMAS or verb.pos_ != "VERB":
            continue

        subj_token = None
        obj_token = None
        for child in verb.children:
            if child.dep_ == "nsubj" and subj_token is None:
                subj_token = child
            if child.dep_ == "obj" and obj_token is None:
                obj_token = child

        head_component = find_component_by_token(components, verb.head.i)

        targets = []
        owner = None

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
            if c["end"] < verb.i and _is_list_item_component(doc, c)
        ]
        early_list_targets = list(all_list_targets)
        if early_list_targets:
            nearest_end = max(c["end"] for c in early_list_targets)
            if verb.i - nearest_end > 15:
                early_list_targets = []

        if subj_token is not None:
            owner = (
                find_component_by_token(components, subj_token.i)
                or find_referenced_component(components, subj_token)
            )
            if obj_token is not None:
                t = (
                    find_component_by_token(components, obj_token.i)
                    or find_referenced_component(components, obj_token)
                )
                if t is not None:
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
            else:
                owner = root_component
            targets = [c for c in early_list_targets if c["text"] != owner["text"]]
        elif _find_nearest_topic_before_text(doc, components, verb) is not None:
            # 明示的なnsubjが見つからなくても、テキスト上に「Ｘは、」という
            # 主題が近くにあれば、そちらを所有者として優先する
            # （GiNZAが超長文で、離れた場所にある「Ｘは」を正しく
            #  この動詞のnsubjとして結びつけられないことがあるため）。
            owner = _find_nearest_topic_before_text(doc, components, verb)
            list_targets = [
                c for c in components
                if c["end"] < verb.i and _is_list_item_component(doc, c) and c["text"] != owner["text"]
            ]
            targets = list_targets
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
            if all_list_targets:
                targets = all_list_targets
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
                targets = [c for c in all_list_targets if c["text"] != owner["text"]]
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
            })
            added.add(s)
        return relations + extra

    owners = set(r["source"] for r in has_edges)
    all_targets = set(r["target"] for r in relations)
    roots = [o for o in owners if o not in all_targets]

    if len(roots) > 1 and doc is not None and components is not None:
        # 根の候補が複数ある場合、文末の語（＝発明の名称）に一致する
        # ものを優先する（例：「基板」と「ロードポート」の両方が候補に
        # なってしまっても、実際の根は文末の「ロードポート」であるため）。
        last_i = len(doc) - 1
        while last_i > 0 and doc[last_i].pos_ == "PUNCT":
            last_i -= 1
        claim_title = find_component_by_token(components, last_i)
        if claim_title is not None and claim_title["text"] in roots:
            roots = [claim_title["text"]]

    root = roots[0] if roots else next(iter(owners))

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
    #    ノードは、根の直接の子として補って繋ぐ
    incoming2 = {}
    for r in simplified:
        incoming2.setdefault(r["target"], []).append(r)

    extra = []
    added = set()
    for r in simplified:
        if r["type"] not in ("positional", "direct"):
            continue
        s = r["source"]
        if s == root or s in added:
            continue
        if s not in incoming2:
            extra.append({"source": root, "relation": "有する", "target": s, "type": "has"})
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
    copula = extract_copula_relations(doc, components)
    comparison = extract_comparison_relations(doc, components)
    direct = extract_direct_relations(doc, components)
    has = extract_has_relations(doc, components)

    final_relations = combine_all_relations(
        positional + location + installation + boundary,
        direct + contact + capability + composition + attribute + copula + comparison,
        has,
    )
    return components, final_relations, doc


def analyze_claim(text):
    """単文形式の請求項テキストを渡すと (構成要素リスト, 関係リスト) を返す"""
    components, final_relations, doc = _extract_raw_relations(text)
    final_relations = _simplify_hierarchy(final_relations, doc, components)
    return components, final_relations


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


def build_graphviz(final_relations, title=None, theme="deepsea"):
    """
    analyze_claim()等が返した関係リストを、Graphvizのdotエンジンで
    階層型に自動レイアウトしたグラフとして組み立てる。

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
        g.node(
            n,
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
    return {
        "numbers": sorted(set(nums)),
        "match_start": m.start(),
        "match_end": m.end(),
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
        all_relations.extend(relations)

    seen_keys = set()
    unique_relations = []
    for r in all_relations:
        key = (r["source"], r["relation"], r["target"], r["type"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_relations.append(r)

    # 全部合体させたあとで、最後にもう1回だけ階層整理をかける
    # （独立請求項由来の「有する」エッジが十分にあるので、
    #  正しい根はそこから見つかる）
    final_relations = _simplify_hierarchy(unique_relations)

    full_text = resolve_dependent_claim(claim_number, claim_texts, prefer_parent=prefer_parent)
    return (None, final_relations), full_text


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


def build_keyword_frequency(database, ids=None, kind="both"):
    """
    database（build_patent_database() または build_abstract_database() の
    戻り値）から、指定したid群（省略時は全件）のキーワード出現頻度を
    集計する。「出現した件数」で数える（1件の中で同じ語が何度出ても1件とする）。
    """
    from collections import Counter

    counter = Counter()
    target_ids = set(ids) if ids is not None else None
    for entry in database:
        if target_ids is not None and entry["id"] not in target_ids:
            continue
        keywords = _entry_keywords(entry, kind=kind)
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

def build_semantic_map(database, n_components=2):
    """
    database（build_patent_database()の戻り値、doc_embeddingを含む）から、
    2次元（または指定した次元数）の座標を計算する。

    戻り値: [{"id":.., "text":.., "x":.., "y":..}, ...]
    """
    import numpy as np
    from sklearn.decomposition import PCA

    if len(database) < 2:
        raise ValueError("2件以上のデータが必要です")

    embeddings = np.array([e["doc_embedding"] for e in database])
    n_comp = min(n_components, len(database) - 1, embeddings.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(embeddings)

    points = []
    for entry, xy in zip(database, coords):
        points.append({
            "id": entry["id"],
            "text": entry["text"],
            "x": float(xy[0]),
            "y": float(xy[1]) if n_comp > 1 else 0.0,
        })
    return points, pca.explained_variance_ratio_


def plot_semantic_map(points, explained_variance=None, groups=None, title="意味的俯瞰マップ", theme="deepsea"):
    """
    build_semantic_map() の結果を、散布図として描画する。

    groups: {id: グループ名, ...} を渡すと、グループごとに色分けする
            （例：自社 vs 競合の比較地図にする場合）。
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

    for p in points:
        ax.annotate(str(p["id"]), (p["x"], p["y"]), fontsize=8, fontproperties=FONT_PROP,
                    color=fg, xytext=(4, 4), textcoords="offset points")

    ax.set_title(title, fontproperties=FONT_PROP, fontsize=16, color=fg)
    if explained_variance is not None and len(explained_variance) >= 2:
        ax.set_xlabel(f"第1主成分（寄与率 {explained_variance[0]*100:.1f}%）", fontproperties=FONT_PROP, color=fg)
        ax.set_ylabel(f"第2主成分（寄与率 {explained_variance[1]*100:.1f}%）", fontproperties=FONT_PROP, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(grid)
    ax.grid(True, color=grid, alpha=0.3)
    plt.tight_layout()
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
        rid = row.get("文献番号") or row.get("出願番号") or row.get("公開番号") or f"行{i+1}"
        records.append({
            "id": rid,
            "出願番号": row.get("出願番号", ""),
            "出願日": _parse_date(row.get("出願日", "")),
            "公知日": _parse_date(row.get("公知日", "")),
            "発明の名称": row.get("発明の名称", ""),
            "出願人": _split_applicants(row.get("出願人/権利者", "")),
            "FI": _split_fi(row.get("FI", "")),
            "要約": row.get("要約", ""),
            "公開番号": row.get("公開番号", ""),
            "ステージ": row.get("ステージ", ""),
            "文献URL": row.get("文献URL", ""),
        })
    return records


def build_full_database(metadata_records, show_progress=True):
    """
    load_patent_metadata_csv() の結果から、要約のSAO解析
    （build_abstract_database相当）とメタデータを1つにまとめた
    データベースを作る。ATLAS・MEGA・Saturn V・Explorer・COREの
    すべてに共通して使える、一番リッチな形式。
    """
    abstract_records = [(r["id"], r["要約"]) for r in metadata_records if r.get("要約")]
    abs_db = build_abstract_database(abstract_records, show_progress=show_progress)
    abs_db_by_id = {e["id"]: e for e in abs_db}

    database = []
    for r in metadata_records:
        entry = dict(r)
        abs_entry = abs_db_by_id.get(r["id"])
        if abs_entry is not None:
            entry["sections"] = abs_entry["sections"]
            entry["keywords"] = abs_entry["keywords"]
            entry["doc_embedding"] = abs_entry["doc_embedding"]
            entry["text"] = abs_entry["text"]
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


def plot_applicant_fi_bubble(database, top_applicants=10, top_fi=10, title="出願人×FI バブルチャート", theme="deepsea"):
    """
    出願人 × FI の組み合わせごとの件数を、対数スケールのバブルの
    大きさで表す散布図（バブルチャート）にする。
    """
    from collections import Counter

    applicant_counter = Counter()
    fi_counter = Counter()
    for entry in database:
        applicant_counter.update(set(entry.get("出願人") or []))
        fi_counter.update(set(entry.get("FI") or []))

    top_applicant_names = [a for a, _ in applicant_counter.most_common(top_applicants)]
    top_fi_names = [f for f, _ in fi_counter.most_common(top_fi)]

    pair_counter = Counter()
    for entry in database:
        for a in set(entry.get("出願人") or []):
            if a not in top_applicant_names:
                continue
            for f in set(entry.get("FI") or []):
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
