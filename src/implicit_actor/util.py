import csv
import json
from typing import Tuple, Iterable, Set, List

from spacy.tokens import Token, Span, Doc

# Constants based on textacy's constants
SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
ACTIVE_VOICE_SUBJ_DEPS = {"agent", "csubj", "expl", "nsubj"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd", "pobj"}
AUX_DEPS = {"aux", "auxpass", "neg"}
NOMINAL_SUBJ_DEPS = {"agent", "expl", "nsubj", "nsubjpass"}
CLAUSAL_SUBJ_DEPS = {"csubj", "csubjpass"}

# Adapted from Mayfield Electronic Handbook of Technical & Scientific Writing
# by Leslie C. Perelman, Edward Barrett, and James Paradis, available at
# https://www.mit.edu/course/21/21.guide/cnj-sub.htm#:~:text=Some%20common%20subordinating%20conjunctions%20are,whereas%2C%20whether%2C%20and%20while.
CONDITIONAL_SUBORDINATE_CONJUNCTIONS = {"if", "unless", "when", "where", "while"}


def get_main_verbs_of_sent(sent: Span) -> List[List[Token]]:
    """
    Based on 'get_main_verbs_of_sent' in textacy, but also groups verbs linked by conjunctions

    Return the main (non-auxiliary) verbs in a sentence.
    """

    verbs = [
        set([tok] + get_conjuncts(tok, {"VERB", "AUX"})) for tok in sent if
        tok.pos_ in {"VERB", "AUX"} and tok.dep_ not in AUX_DEPS
    ]

    verbs_out = []

    for vg in verbs:
        for vo in verbs_out:
            if not vo.isdisjoint(vg):
                vo.update(vg)
                break
        else:
            verbs_out.append(vg)

    return [list(x) for x in verbs_out]


def extract_prepositions(verb: Token):
    """
    Finds all children and children's children for which there is a direct path of prep/agent
    dependencies to the root.
    :param verb: The root from which to search for prepositions.
    :return: A list of prepositions.
    """
    out = []
    preps = [verb]
    # Tree search for all paths of prepositions as to catch constructs with multiple prepositions.
    while preps:
        p = preps.pop()
        new_preps = [tok for tok in p.rights if tok.dep_ in {"prep", "agent", "acomp"}]
        preps.extend(new_preps)
        out.extend(new_preps)
    return out


def get_objects_of_verb_consider_preposition(verbs: List[Token]) -> List[Token]:
    """
    Adapted from 'get_objects_of_verb' from textacy.

    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """

    objs = []
    for verb in verbs:
        verb_and_prep = [verb] + extract_prepositions(verb)
        objs.extend(tok for v in verb_and_prep for tok in v.rights if tok.dep_ in OBJ_DEPS)
        objs.extend(tok for tok in verb.rights if tok.dep_ in {"acomp", "advmod"})
        objs.extend(tok for tok in verb.rights if tok.dep_ == "xcomp" and tok.pos_ != "VERB")
        objs.extend(tok for obj in objs for tok in get_conjuncts(obj, {obj.pos_}))

    return objs


def get_conjuncts(tok: Token, allowed_pos: Set[str] = None) -> List[Token]:
    """
    Adapted from '_get_conjuncts' from textacy.

    We also treat appositions as conjunctions. Though technically not correct,
    this helps with enumerations.

    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """

    return [right for right in tok.rights if
            right.dep_ in {"conj", "appos"} and (not allowed_pos or right.pos_ in allowed_pos)]


def is_acl_without_subj(tok: Token):
    """
    Check if a phrase is an adnominal phrase without a subject (and thus the subject should be taken
    from higher up)
    """
    return tok.dep_ == "acl" and not any(x.dep_ in SUBJ_DEPS for x in tok.children)


def get_noun_chunk(token: Token) -> Span:
    """
    Gets the noun chunk the token is contained in.
    """
    if not token.doc.has_annotation("DEP"):
        return token.doc[token.i: token.i + 1]

    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return chunk
    return token.doc[token.i: token.i + 1]


def load_gold_standard(file_name='./data/evaluation/gold_standard.csv') -> Iterable[
    Tuple[str, str, str, List[str], List[str]]]:
    """
    Loads the gold standard.
    """
    with open(file_name, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for source, inp, gs, impl_subject, target in reader:
            with open(f"./data/external{source}", 'r', encoding="utf-8") as source_file:
                source_txt = source_file.read().replace("\n", " ").replace("  ", " ")
            yield source_txt, inp, gs, json.loads(impl_subject), json.loads(target)


def search_for_head(tok: Token):
    """
    Used to find the predicate of an object/subject.
    """
    head = tok
    while head.dep_ in AUX_DEPS | OBJ_DEPS | NOMINAL_SUBJ_DEPS | {"prep"}:
        head = head.head
    return head


def search_for_head_block_nouns(tok: Token):
    """
    Used to find the predicate of an object/subject.
    """
    head = tok
    while head.dep_ in AUX_DEPS | OBJ_DEPS | NOMINAL_SUBJ_DEPS | {"prep"} and (
            head.head is None or head.head.pos_ != "NOUN"):
        head = head.head
    return head


def has_explicit_subject(token: Token):
    """
    Determines if a potential detection should be discarded as it already has an explicit subject.
    """
    return any(c.dep_ in ACTIVE_VOICE_SUBJ_DEPS for c in token.children)


def find_conj_head(token: Token) -> Token:
    """
    Finds the head of a group of conjuncts.
    """
    head = token
    while head.dep_ == "conj":
        head = head.head
    return head


def dependency_trees_equal(doc1: Doc, doc2: Doc):
    """
    Checks if two docs are equivalent in terms of their dependency structure.

    Note: This is not actually a technically correct implementation, but it works good enough for our purposes
    """

    sents1 = list(doc1.sents)
    sents2 = list(doc2.sents)

    if len(sents1) != len(sents2):
        return False

    for s1, s2 in zip(sents1, sents2):
        toks1 = sorted((x for x in s1), key=lambda x: (x.text, x.dep_))
        toks2 = sorted((x for x in s2), key=lambda x: (x.text, x.dep_))
        if not all(x.dep_ == y.dep_ and x.text.lower() == y.text.lower() for x, y in zip(toks1, toks2)):
            return False

    return True
