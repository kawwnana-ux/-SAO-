# -*- coding: utf-8 -*-
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


nlp = _LazyNLP()

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
    doc = nlp(english_text)
    return dedup(extract_relations(doc))
