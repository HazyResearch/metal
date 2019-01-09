"""Exports list of labeling functions: `LFs` that can then be applied to data."""

import bz2
import re

from six.moves.cPickle import load
from snorkel.lf_helpers import (
    get_tagged_text,
    rule_regex_search_before_A,
    rule_regex_search_before_B,
    rule_regex_search_btw_AB,
    rule_regex_search_btw_BA,
    rule_regex_search_tagged_text,
)

with bz2.BZ2File("data/ctd.pkl.bz2", "rb") as ctd_f:
    ctd_unspecified, ctd_therapy, ctd_marker = load(ctd_f)


def cand_in_ctd_unspecified(c):
    return 1 if c.get_cids() in ctd_unspecified else 0


def cand_in_ctd_therapy(c):
    return 1 if c.get_cids() in ctd_therapy else 0


def cand_in_ctd_marker(c):
    return 1 if c.get_cids() in ctd_marker else 0


def LF_in_ctd_unspecified(c):
    return -1 * cand_in_ctd_unspecified(c)


def LF_in_ctd_therapy(c):
    return -1 * cand_in_ctd_therapy(c)


def LF_in_ctd_marker(c):
    return cand_in_ctd_marker(c)


# List to parenthetical
def ltp(x):
    return "(" + "|".join(x) + ")"


def LF_induce(c):
    return (
        1
        if re.search(
            r"{{A}}.{0,20}induc.{0,20}{{B}}", get_tagged_text(c), flags=re.I
        )
        else 0
    )


causal_past = ["induced", "caused", "due"]


def LF_d_induced_by_c(c):
    return rule_regex_search_btw_BA(
        c, ".{0,50}" + ltp(causal_past) + ".{0,9}(by|to).{0,50}", 1
    )


def LF_d_induced_by_c_tight(c):
    return rule_regex_search_btw_BA(
        c, ".{0,50}" + ltp(causal_past) + " (by|to) ", 1
    )


def LF_induce_name(c):
    return 1 if "induc" in c.chemical.get_span().lower() else 0


causal = ["cause[sd]?", "induce[sd]?", "associated with"]


def LF_c_cause_d(c):
    return (
        1
        if (
            re.search(
                r"{{A}}.{0,50} " + ltp(causal) + ".{0,50}{{B}}",
                get_tagged_text(c),
                re.I,
            )
            and not re.search(
                "{{A}}.{0,50}(not|no).{0,20}" + ltp(causal) + ".{0,50}{{B}}",
                get_tagged_text(c),
                re.I,
            )
        )
        else 0
    )


treat = [
    "treat",
    "effective",
    "prevent",
    "resistant",
    "slow",
    "promise",
    "therap",
]


def LF_d_treat_c(c):
    return rule_regex_search_btw_BA(c, ".{0,50}" + ltp(treat) + ".{0,50}", -1)


def LF_c_treat_d(c):
    return rule_regex_search_btw_AB(c, ".{0,50}" + ltp(treat) + ".{0,50}", -1)


def LF_treat_d(c):
    return rule_regex_search_before_B(c, ltp(treat) + ".{0,50}", -1)


def LF_c_treat_d_wide(c):
    return rule_regex_search_btw_AB(c, ".{0,200}" + ltp(treat) + ".{0,200}", -1)


def LF_c_d(c):
    return 1 if ("{{A}} {{B}}" in get_tagged_text(c)) else 0


def LF_c_induced_d(c):
    return (
        1
        if (
            ("{{A}} {{B}}" in get_tagged_text(c))
            and (
                ("-induc" in c[0].get_span().lower())
                or ("-assoc" in c[0].get_span().lower())
            )
        )
        else 0
    )


def LF_improve_before_disease(c):
    return rule_regex_search_before_B(c, "improv.*", -1)


pat_terms = ["in a patient with ", "in patients with"]


def LF_in_patient_with(c):
    return (
        -1
        if re.search(ltp(pat_terms) + "{{B}}", get_tagged_text(c), flags=re.I)
        else 0
    )


uncertain = ["combin", "possible", "unlikely"]


def LF_uncertain(c):
    return rule_regex_search_before_A(c, ltp(uncertain) + ".*", -1)


def LF_induced_other(c):
    return rule_regex_search_tagged_text(c, "{{A}}.{20,1000}-induced {{B}}", -1)


def LF_far_c_d(c):
    return rule_regex_search_btw_AB(c, ".{100,5000}", -1)


def LF_far_d_c(c):
    return rule_regex_search_btw_BA(c, ".{100,5000}", -1)


def LF_risk_d(c):
    return rule_regex_search_before_B(c, "risk of ", 1)


def LF_develop_d_following_c(c):
    return (
        1
        if re.search(
            r"develop.{0,25}{{B}}.{0,25}following.{0,25}{{A}}",
            get_tagged_text(c),
            flags=re.I,
        )
        else 0
    )


procedure, following = ["inject", "administrat"], ["following"]


def LF_d_following_c(c):
    return (
        1
        if re.search(
            "{{B}}.{0,50}"
            + ltp(following)
            + ".{0,20}{{A}}.{0,50}"
            + ltp(procedure),
            get_tagged_text(c),
            flags=re.I,
        )
        else 0
    )


def LF_measure(c):
    return (
        -1
        if re.search("measur.{0,75}{{A}}", get_tagged_text(c), flags=re.I)
        else 0
    )


def LF_level(c):
    return (
        -1
        if re.search("{{A}}.{0,25} level", get_tagged_text(c), flags=re.I)
        else 0
    )


def LF_neg_d(c):
    return (
        -1
        if re.search(
            "(none|not|no) .{0,25}{{B}}", get_tagged_text(c), flags=re.I
        )
        else 0
    )


WEAK_PHRASES = [
    "none",
    "although",
    "was carried out",
    "was conducted",
    "seems",
    "suggests",
    "risk",
    "implicated",
    "the aim",
    "to (investigate|assess|study)",
]

WEAK_RGX = r"|".join(WEAK_PHRASES)


def LF_weak_assertions(c):
    return -1 if re.search(WEAK_RGX, get_tagged_text(c), flags=re.I) else 0


def LF_ctd_marker_c_d(c):
    return LF_c_d(c) * cand_in_ctd_marker(c)


def LF_ctd_marker_induce(c):
    return (
        LF_c_induced_d(c) or LF_d_induced_by_c_tight(c)
    ) * cand_in_ctd_marker(c)


def LF_ctd_therapy_treat(c):
    return LF_c_treat_d_wide(c) * cand_in_ctd_therapy(c)


def LF_ctd_unspecified_treat(c):
    return LF_c_treat_d_wide(c) * cand_in_ctd_unspecified(c)


def LF_ctd_unspecified_induce(c):
    return (
        LF_c_induced_d(c) or LF_d_induced_by_c_tight(c)
    ) * cand_in_ctd_unspecified(c)


def LF_closer_chem(c):
    # Get distance between chemical and disease
    chem_start, chem_end = (
        c.chemical.get_word_start(),
        c.chemical.get_word_end(),
    )
    dis_start, dis_end = c.disease.get_word_start(), c.disease.get_word_end()
    if dis_start < chem_start:
        dist = chem_start - dis_end
    else:
        dist = dis_start - chem_end
    # Try to find chemical closer than @dist/2 in either direction
    sent = c.get_parent()
    # closest_other_chem = float("inf")
    for i in range(dis_end, min(len(sent.words), dis_end + dist // 2)):
        et, cid = sent.entity_types[i], sent.entity_cids[i]
        if et == "Chemical" and cid != sent.entity_cids[chem_start]:
            return -1
    for i in range(max(0, dis_start - dist // 2), dis_start):
        et, cid = sent.entity_types[i], sent.entity_cids[i]
        if et == "Chemical" and cid != sent.entity_cids[chem_start]:
            return -1
    return 0


def LF_closer_dis(c):
    # Get distance between chemical and disease
    chem_start, chem_end = (
        c.chemical.get_word_start(),
        c.chemical.get_word_end(),
    )
    dis_start, dis_end = c.disease.get_word_start(), c.disease.get_word_end()
    if dis_start < chem_start:
        dist = chem_start - dis_end
    else:
        dist = dis_start - chem_end
    # Try to find chemical disease than @dist/8 in either direction
    sent = c.get_parent()
    for i in range(chem_end, min(len(sent.words), chem_end + dist // 8)):
        et, cid = sent.entity_types[i], sent.entity_cids[i]
        if et == "Disease" and cid != sent.entity_cids[dis_start]:
            return -1
    for i in range(max(0, chem_start - dist // 8), chem_start):
        et, cid = sent.entity_types[i], sent.entity_cids[i]
        if et == "Disease" and cid != sent.entity_cids[dis_start]:
            return -1
    return 0


LFs = [
    LF_c_cause_d,
    LF_c_d,
    LF_c_induced_d,
    LF_c_treat_d,
    LF_c_treat_d_wide,
    LF_closer_chem,
    LF_closer_dis,
    LF_ctd_marker_c_d,
    LF_ctd_marker_induce,
    LF_ctd_therapy_treat,
    LF_ctd_unspecified_treat,
    LF_ctd_unspecified_induce,
    LF_d_following_c,
    LF_d_induced_by_c,
    LF_d_induced_by_c_tight,
    LF_d_treat_c,
    LF_develop_d_following_c,
    LF_far_c_d,
    LF_far_d_c,
    LF_improve_before_disease,
    LF_in_ctd_therapy,
    LF_in_ctd_marker,
    LF_in_patient_with,
    LF_induce,
    LF_induce_name,
    LF_induced_other,
    LF_level,
    LF_measure,
    LF_neg_d,
    LF_risk_d,
    LF_treat_d,
    LF_uncertain,
    LF_weak_assertions,
]
