# -*- coding: utf-8 -*-
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
from pathlib import Path

import numpy as np

import dep_pairs as D
import sao_selector as S
import sao_selector11 as S11

HERE = Path(__file__).resolve().parent
TRAIN_FILE = HERE / "sao_selector12_train.npz"
CASE_KEYS = D.CASES + ["連体", "は(継承)", "-"]
HAS = {"有する", "備える", "具備する", "含む", "含める", "の"}


def _is_has(rel):
    return rel in HAS or any(h in rel for h in ("有する", "備える", "具備する", "含む"))


def build_candidates(ts, pp, text, own=False, owner_canon=False, **kw):
    """実験11の候補に、係り受けに基づく候補（DEP）を加える。
    owner_canon=True: 候補の「X 有する Y」から持ち主Xを求め、「XのY」と「Y」を同じ組とみなして
      重複を除く（選別時）。採用した実験12（dep）はこれを使わない設定で学習したので、既定は False。
    own=True: 持ち主つきノード「XのY」の候補（OWN）も作る。532件の検証で、OWN候補は75,464件中
      正解が340件（0.4%）しかなく、選別モデルが選んだものはすべて誤りだったため、既定では作らない。"""
    info = S11.build_candidates(ts, pp, text, **kw)
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

    for d in D.dependency_pairs(pp, text):
        add(d["source"], d["relation"].translate(S.SIMP), d["target"], "DEP", dep=d)

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


def claim_features(pp, info):
    X = S11.claim_features(pp, info)
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


def structural_features(pp, info, p):
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


def select(pp, info, prob, threshold, max_per_pair=2):
    """確率の高い順に採用。同じ組（「XのY」と「Y」は同じノードとみなす）では最大 max_per_pair 件まで。
    2件目は、既に採った関係と同義でないものだけ。max_per_pair は交差検証の内側で選ぶ。"""
    import node_match_eval as NM

    n = pp._normalize_node_text_lenient
    cm = canon(info)
    out, chosen = [], defaultdict(list)
    for i in np.argsort(-prob):
        if prob[i] < threshold:
            break
        c = info["cands"][i]
        k = frozenset((n(cm.get(c["source"], c["source"])), n(cm.get(c["target"], c["target"]))))
        if any(NM.rel_match(pp, c["relation"], r) or NM.rel_match(pp, r, c["relation"]) for r in chosen[k]):
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


class Selector:
    """2段階の選別モデル。学習データ（npz）には1段目の特徴量X1・ラベルY・請求項ID・
    学習時の1段目確率（交差検証の外側で求めたもの）P1・2段目の構造特徴X2を保存してある。"""

    def __init__(self, exclude_ids=None, train_file=TRAIN_FILE, fold=None):
        """exclude_ids を除いて学習する（交差検証の評価用）。fold を渡すと、2段目の構造特徴に
        「その分割の学習用請求項だけで作った1段目の確率」から求めたもの（X2_fold{k}）を使い、
        交差検証（make_train12.py）と同じモデルを再現する。"""
        d = np.load(train_file, allow_pickle=False)
        keep = np.ones(len(d["Y"]), bool)
        if exclude_ids:
            keep = ~np.isin(d["ids"], np.array(sorted(exclude_ids)))
        x2_key = "X2_fold%d" % fold if fold is not None and ("X2_fold%d" % fold) in d.files else "X2"
        X1, Y, X2 = d["X1"][keep], d["Y"][keep], d[x2_key][keep]
        self.fold_thresholds = [float(t) for t in d["fold_thresholds"]] if "fold_thresholds" in d.files else None
        X1 = X1.copy()
        X1[:, S.KEPT_INDEX] = 0.0
        self.m1 = _model().fit(X1, Y)
        self.m2 = _model().fit(np.hstack([X1, X2]), Y)
        self.threshold = float(d["threshold"])
        self.max_per_pair = int(d["max_per_pair"]) if "max_per_pair" in d.files else 2
        self.fold_max_per_pair = [int(x) for x in d["fold_max_per_pair"]] if "fold_max_per_pair" in d.files else None

    def predict(self, pp, info):
        if not len(info["cands"]):
            return np.zeros(0)
        X1 = claim_features(pp, info)
        p1 = self.m1.predict_proba(X1)[:, 1]
        X2 = structural_features(pp, info, p1)
        return self.m2.predict_proba(np.hstack([X1, X2]))[:, 1]


def analyze_claim_selected(ts, pp, selector, text, threshold=None, max_per_pair=None, **kw):
    info = build_candidates(ts, pp, text, **kw)
    prob = selector.predict(pp, info)
    return info, select(pp, info, prob, selector.threshold if threshold is None else threshold,
                        selector.max_per_pair if max_per_pair is None else max_per_pair)
