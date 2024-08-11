import re
from collections import Counter
from collections import defaultdict
from typing import Optional, Set

import requests
import unicodedata
from bs4 import BeautifulSoup
from django.http import FileResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from explicit_information.definitions import find_definitions, get_annotations, \
    any_definition_in_text, get_dictionary
from explicit_information.relations import noun_relations, build_tree, get_hyponymy, get_meronymy, get_synonymy
from implicit_actor.ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.implicit_subject_pipeline_factory import create_implicit_subject_pipeline
from implicit_actor.util import get_noun_chunk
from presentation.forms import FormCELEX, FormDefinition
from presentation.graph import create_bar_chart, construct_default_graph, construct_relation_graph

site = ""
celex = ""
pipeline: Optional[ImplicitSubjectPipeline] = None
reg_title = ""
definitions = list(tuple())
relations = ""
annotations = {}
regulation_with_annotations = ""
done_date = ""
regulation_body = ""
frequency_articles = {}
sentences_set = {}
sentences_with_implicit_actors: Set[str] = set()
articles_set = {}
articles_set_and_frequency = {}
implicit_actor_counter = Counter()


def index(request):
    if request.method == 'POST':
        form = FormCELEX(request.POST)

        if form.is_valid():
            global celex
            celex = form.cleaned_data['number']
            global pipeline
            pipeline = create_implicit_subject_pipeline(form.cleaned_data['filters'])
            global site
            site = load_document(celex)
            return HttpResponseRedirect('result/')
    else:
        form = FormCELEX()
    return render(request, 'index.html', {'form': form})


def original_document(request):
    global site
    return HttpResponseRedirect(site)


def load_document(celex):
    # https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679&from=EN
    new_url = 'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:' + celex + '&from=EN'
    return new_url


def result(request):
    extract_text(site)

    create_bar_chart(most_frequent_definitions(), 'presentation/static/top.png', 'Definitions', '# of hits',
                     'Top Most Frequent Definitions')
    create_bar_chart(dict(implicit_actor_counter.most_common(5)), 'presentation/static/implicit_actor_frequency.png',
                     'Implicit Actors', '# of hits', 'Frequency Diagram')

    context_dict = {'site': site, 'celex': celex, 'definitions': definitions,
                    'num_def': len(annotations.keys()),
                    'title': reg_title, 'date': done_date, 'path': '/top.png',
                    'implicit_actor_freq': '/implicit_actor_frequency.png'}

    return render(request, 'result.html', context_dict)


# for testing purposes of assignment of annotations
def annotations_page(request):
    context_dict = {'body': regulation_body}
    return render(request, 'annotations.html', context_dict)


# for relation graph
def graph(request):
    defin = get_dictionary()
    image_path = 'static/graph.png'
    definition_frequency_image_path = 'static/frequency.png'
    if request.method == 'POST':
        form = FormDefinition(request.POST)
        if form.is_valid():
            current_def = form.cleaned_data['definition']

            # construct a graph depending on the relation
            relation = form.cleaned_data['relation']
            if relation == 'meronymy':
                construct_relation_graph(get_meronymy(), defin, current_def, image_path)
            elif relation == 'synonymy':
                construct_relation_graph(get_synonymy(), defin, current_def, image_path)
            else:
                construct_relation_graph(get_hyponymy(), defin, current_def, image_path)

            create_bar_chart(get_freq_dict(current_def), definition_frequency_image_path, 'Articles', '# of hits',
                             'Frequency Diagram')

            return render(request, 'html',
                          {'form': form, 'definitions': defin, 'image_path': 'src/graph.png',
                           'freq': 'src/frequency.png'})
    else:
        # if the user enters no definition: create a default graph and an empty diagram
        form = FormDefinition()
        construct_default_graph(image_path)
        create_bar_chart(dict(), definition_frequency_image_path, 'Articles', '# of hits', 'Frequency Diagram')
    return render(request, 'graph.html', {'form': form, 'definitions': defin,
                                          'image_path': 'graph.png', 'freq': 'frequency.png'})


def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    global definitions
    definitions = find_definitions(soup)

    for script in soup.find_all('script'):
        script.decompose()

    div_re = re.compile(r"^art_\d+$")

    definitions_header = soup.find(text="Definitions")
    definitions_article = definitions_header.find_parent(
        'div', id=lambda x: x and div_re.fullmatch(x),
    )

    implicit_actor_counter.clear()

    # TODO refactor
    soup.smooth()
    context = definitions_article.get_text()
    for article in soup.find_all('div', id=lambda x: x and div_re.fullmatch(x)):
        # context = article.get_text()
        for paragraph in article.findChildren('p', class_="oj-normal"):
            old_text = paragraph.get_text().replace("\u00A0", " ")
            new_text = pipeline.apply(old_text, context + "\n\n\n\n" + old_text)  # + "\n\n\n\n" + old_text
            for c in pipeline.last_selected_candidates():
                noun_chunk = get_noun_chunk(c)
                implicit_actor_counter[noun_chunk.text] += 1
            new_html = BeautifulSoup(new_text, 'html.parser')
            paragraph.clear()
            paragraph.append(new_html)
    soup.smooth()

    global reg_title
    reg_title = find_title(soup)
    global done_date
    done_date = soup.find(string=re.compile("Done at"))
    global annotations
    annotations = get_annotations()
    global sentences_set

    global articles_set
    add_annotations_to_the_regulation(soup)
    global regulation_body
    regulation_body = soup.body
    global relations
    relations = "\n".join(noun_relations(definitions))
    # uncomment for evaluation purposes
    # compare_sentences()
    # compare_definitions_and_relations()


def add_annotations_to_the_regulation(soup):
    global sentences_set
    sentences_set.clear()
    global articles_set
    articles_set.clear()
    global articles_set_and_frequency
    articles_set_and_frequency.clear()
    article = ""
    # case if a regulation has div for each article
    if soup.find("div", id="001") is not None:
        for div in soup.find_all("div"):
            div.unwrap()

    for sentence in soup.find_all("p"):
        if check_if_article(sentence.text):
            article = sentence.text
            create_an_article(article)

        # for (key, value) in definitions:
        # capitalized = key[0].islower() and key.capitalize() in sentence.text
        #
        # if not (key in sentence.text or capitalized):
        #     continue

        # text = sentence.text
        # sentence.clear()
        # only one definition of a type per sentence
        definitions_in_sentence: Set[str] = set()

        # create copy as we will change the elements during iteration
        for element in list(sentence.strings):
            text = element.text

            # TODO find out what this line does...
            # if not check_more_definitions_in_text(key, sentence.text, capitalized):
            # sort by the starting index
            defs = sorted(any_definition_in_text(text), key=lambda x: x[2])
            start_index = 0

            new_elements = []

            for (k, v, start, end) in defs:
                if k in definitions_in_sentence:
                    continue
                definitions_in_sentence.add(k)

                new_elements.append(text[start_index:start])
                tag = create_new_tag(soup, text, k, v, start, end)

                new_elements.append(tag)

                start_index = end
                sent = text.replace("\n\n", "\n").strip()
                sent = unicodedata.normalize("NFKD", sent)

                if k not in get_dictionary().keys():
                    k = k[0].lower() + k[1:]

                if k not in sentences_set:
                    sentences_set[k] = set()
                sentences_set[k].add(sent)
                if k not in articles_set:
                    articles_set[k] = set()
                articles_set[k].add(article)
                if k not in articles_set_and_frequency:
                    articles_set_and_frequency[k] = list()
                if len(article) != 0:
                    articles_set_and_frequency[k].append(article)

                global frequency_articles
                if len(article) != 0:
                    frequency_articles[article].add((k, sent))
            new_elements.append(text[start_index:])
            element.replace_with(*new_elements)


def check_if_article(text):
    if len(text) > 11 or not text.__contains__("Article"):
        return False
    new_text = text.replace("Article", "").strip()
    if new_text.isdigit():
        return True
    return False


def create_an_article(article):
    global frequency_articles
    frequency_articles[article] = set()  # set of tuples (definition, sentence)


# returns a dictionary where a key is an article and a value is a set of tuples (legal definition, number of hits)
def count_article_frequency():
    global frequency_articles
    result_dict = {}
    for key, value in frequency_articles.items():
        counts = defaultdict(int)
        for (k, v) in value:
            counts[k] += 1
        result_dict[key] = [(k, count) for k, count in counts.items()]
    return result_dict


# creates a dictionary of a form Article #: # of hits for the given definition
def get_freq_dict(definition):
    counters = count_article_frequency()
    filtered_dict = {key: {(w, num) for w, num in value if w == definition} for key, value in counters.items()}
    result_dict = {}
    for article in filtered_dict.keys():
        for k, v in filtered_dict[article]:
            if k == definition:
                a = article.replace("Article ", "")
                result_dict[a] = v
    return result_dict


# can be adjusted depending on the processed document
def find_title(s):
    start_class = s.find("p", string=re.compile("REGULATION"))
    if start_class is None:
        return ""
    end_class = s.find("p", string=re.compile("THE EUROPEAN PARLIAMENT AND THE COUNCIL"))
    title = str(start_class.text)
    for element in start_class.next_siblings:
        if element == end_class:
            break
        title = title + " " + element.text
    return title


def create_new_tag(soup, text, key, value, start, end):
    new_tag = soup.new_tag('span')
    new_tag["style"] = "background-color: yellow;"
    new_tag["data-tooltip"] = key + ' ' + value
    new_tag.string = text[start:end]
    return new_tag


def most_frequent_definitions():
    def_list = dict()
    sorted_def = sorted(sentences_set.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_definitions = [definition[0] for definition in sorted_def[:5]]
    for d in top_five_definitions:
        def_list[d] = len(sentences_set[d])
    return def_list


def calculate_the_frequency(key):
    counter = Counter(articles_set_and_frequency[key])
    repeated_elements = [(element, count) for element, count in counter.items()]
    articles = "Definition " + key + " can be found in: Article "
    for (element, count) in repeated_elements:
        num = re.findall(r'\d+', element)
        articles = articles + "".join(num) + "; "
    return articles.replace(" ; ", " ")


def cut_tag(tag):
    new_string = str(tag)
    start = new_string.find(">")
    end = new_string.rfind("<")
    return new_string[start + 1:end]


def get_sentences():
    return sentences_set


# create a txt. file to download with all definitions and their explanations
def download_definitions_file(request):
    with open("presentation/output/file.txt", "w", encoding="utf-8") as file:
        for key, value in annotations.items():
            file.write(key + " " + value + "\n")
    response = FileResponse(open("presentation/output/file.txt", 'rb', encoding="utf-8"))
    response['Content-Disposition'] = 'attachment; filename="file.txt"'
    return response


# create a txt. file to download with all sentences with definitions
def download_sentences(request):
    with open("presentation/output/sentences.txt", "w", encoding="utf-8") as file:
        for key in sentences_set:
            file.write("Definition: " + key + "\n")
            file.write("Total number of text segments including definition: " + str(len(sentences_set[key])) + "\n\n")
            file.write(calculate_the_frequency(key) + "\n\n")
            for sent in sentences_set[key]:
                file.write("\t\t" + sent + "\n\n")
            file.write("\n\n")
    response = FileResponse(open("presentation/output/sentences.txt", 'rb', encoding="utf-8"))
    response['Content-Disposition'] = 'attachment; filename="sentences.txt"'
    return response


# create an HMTL file to download with annotations
def download_annotations(request):
    context_dict = {'body': str(regulation_body)}
    response = render(request, 'annotations.html', context_dict)
    response['Content-Disposition'] = 'attachment; filename="annotated_page.html"'
    return response


# create a text file to download with all semantic relations listed
def download_relations(request):
    # TODO fix this file location
    with open("presentation/output/relations.txt", "w", encoding="utf-8") as file:
        file.write(relations)
        file.write("\n\n")
        file.write("Hyponymy Tree: \n")
        hyponymy = get_hyponymy()
        keys = set(hyponymy.keys())
        values = set().union(*hyponymy.values())
        roots = keys.difference(values)
        for root in roots:
            file.write(build_tree(root))
    response = FileResponse(open("presentation/output/relations.txt", 'rb', encoding="utf-8"))
    response['Content-Disposition'] = 'attachment; filename="relations.txt"'
    return response
