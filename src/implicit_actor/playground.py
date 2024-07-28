import itertools

import spacy
from bs4 import BeautifulSoup
from spacy import displacy

from implicit_actor.insertion.DefaultInserter import DefaultInserter
from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
from implicit_actor.missing_subject_detection.ImperativeDetector import ImperativeDetector
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, \
    ImplicitSubjectType
from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector


# from spacy import displacy

# from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
# from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
# from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
# from implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector


# from dotenv import load_dotenv
# from nltk.stem import PorterStemmer
# from spacy import displacy

# from src.implicit_actor.insertion.pattern.inflect import lexeme
# from src.implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector


# load_dotenv()


def main():
    """
    This is just a file for messing around mostly with the dependency parser.
    Nothing of value can be found here.
    """

    nlp = spacy.load("en_core_web_lg")

    txt = """The contract shall specify the rules, policies and procedures for the provision of services by the Registry and the conditions according to which the Commission is to supervise the organisation, administration and management of the .eu TLD by the Registry."""

    doc = nlp(txt)

    for tok in doc:
        print(tok, tok.lemma_)



    # print(PassiveDetector().detect(doc[:]))
    # print(GerundDetector().detect(doc[:]))
    # print(NominalizedGerundWordlistDetector().detect(doc[:]))
    # print(ImperativeDetector().detect(doc[:]))

    # c = DefinitionCandidateExtractor().extract(doc)
    # print(c)

    #
    #
    # matcher = Matcher(nlp.vocab)
    #
    # pattern = [
    #     {"TEXT": "‘"},
    #     {"IS_ALPHA": True, "OP": "+"},
    #     {"TEXT": "’"},
    # ]
    #
    # # Add the pattern to the matcher
    # matcher.add("DEFINITION", [pattern], greedy="FIRST")
    #
    # # Apply the matcher to the doc
    # matches = matcher(doc)
    #
    # # Extract and print the matched spans
    # for match_id, start, end in matches:
    #     print(start, end)
    #     span = doc[start:end]
    #     print(span.text)
    #     print("---")

    # print("-" * 10)
    # for tok in doc:
    #     print(tok.text, tok)
    #     print("-")

    # Omitting the verb from the sentence is also possible. Once omitted, it is no longer present.

    # txt = "That period shall be extended by two months at the initiative of the European Parliament or of the Council."

    # doc = nlp(txt)

    # print(CandidateExtractorImpl().extract(doc))

    # print(doc.ents)

    # stemmer = PorterStemmer()
    #
    # print(stemmer.stem("eating"))
    #
    # similarity_nlp = spacy.load("en_core_web_lg")
    # t1 = "The setup of your account starts with Blizzard checking whether you have a battle.net account."
    # t2 = "The setup of your account starts with you checking whether you have a battle.net account."
    #
    # print(similarity_nlp(t1).similarity(similarity_nlp(t2)))

    # for tok in doc:
    #     print(tok.text, tok.dep_, tok.tag_, tok.lemma_, tok.pos_)

    # print(lexeme("be"))

    displacy.serve(doc, style="dep")
    # for source, inp, gs, impl_subjects, targets in list(load_gold_standard())[38:]:
    #     print(gs)


if __name__ == "__main__":
    main()
