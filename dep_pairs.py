# -*- coding: utf-8 -*-
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

import claim_segmenter as CS

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
    with CS._enzai(pp, True):
        return _dependency_pairs(pp, text)


def _dependency_pairs(pp, text):
    res = []
    try:
        res += pairs_from_doc(pp, pp.nlp(pp._clean_claim_text(text)), "全体")
    except Exception:  # noqa: BLE001
        pass
    for seg in CS.split_claim(pp, text):
        sent = CS.to_sentence(seg)
        if sent is None:
            continue
        for s in CS.distribute(sent):
            try:
                res += pairs_from_doc(pp, pp.nlp(pp._clean_claim_text(s)), "区間")
            except Exception:  # noqa: BLE001
                continue
    return res
