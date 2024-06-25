import requests
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
import re


def get_synsets(word):
    return wn.synsets(word)


def get_wikipedia_url(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "format": "json"
    }
    response = requests.get(url, params=params).json()
    pages = response.get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "missing" not in page:
            if "disambiguation" in page.get("pageprops", {}):
                return handle_disambiguation(title)
            return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return None


def handle_disambiguation(title):
    disambiguation_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}_disambiguation"
    response = requests.get(disambiguation_url)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    possible_links = [a['href'] for a in links if 'wiki' in a['href']]
    
    for possible_link in possible_links:
        possible_link_title = possible_link.split('/wiki/')[1]
        possible_link_title = possible_link_title.replace('_', ' ')
        possible_link_url = f"https://en.wikipedia.org{possible_link}"
        possible_link_context = get_wikipedia_vocab(possible_link_title)
        if possible_link_context:
            return possible_link_url
    return None


def get_wikipedia_redirects(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "redirects",
        "rdlimit": "max",
        "format": "json"
    }
    response = requests.get(url, params=params).json()
    redirects = response.get("query", {}).get("pages", {})
    redirect_titles = []
    for _, page in redirects.items():
        if "redirects" in page:
            for redirect in page["redirects"]:
                redirect_titles.append(redirect["title"])
    return redirect_titles


def get_wikipedia_links(wikipedia_url):
    response = requests.get(wikipedia_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    filter_keywords = ['Special:', 'login', 'special', 'index.php', 'ISBN (identifier)', 'BookSources', 'Doi (identifier)', 'identifier', 
                       'Category', 'main page', 'ISSN (identifier)', 'Wikipedia:', 'Help:', 'Template:', 'File:', 'Special:', 
                       'Portal:', 'User:', 'PMID (identifier)', 'S2CID (identifier)', 'Bibcode (identifier)', 'OCLC (identifier)',
                       'ArXiv (identifier)', 'Hdl (identifier)', 'Encyclopedia', 'Routledge', 'Wayback Machine',
                       'Encyclopædia', 'University', 'Template:', 'commons:']
    links = [a.get('title') for a in soup.find_all('a', href=True) if a.get('title') and not any(keyword in a['href'] for keyword in filter_keywords)]
    return links


def get_synset_vocab(synset, depth=0, max_depth=2, visited=None):
    if visited is None:
        visited = set()
    if depth > max_depth or synset in visited:
        return set()
    visited.add(synset)
    context = set()
    for lemma in synset.lemmas():
        context.add(lemma.name())
    for relation in [synset.hypernyms, synset.hyponyms, synset.member_holonyms, synset.member_meronyms]:
        for related_synset in relation():
            context.update(get_synset_vocab(related_synset, depth + 1, max_depth, visited))
    context.update(set(re.findall(r'\w+', synset.definition())))
    return context


def get_wikipedia_vocab(title):
    wikipedia_url = get_wikipedia_url(title)
    if wikipedia_url:
        links = get_wikipedia_links(wikipedia_url)
        context = set(re.findall(r'\w+', title))
        context.update(set(links))
        return context
    return set()


def compute_score(synset_context, wiki_context):
    return len(synset_context & wiki_context) + 1


def map_to_wiki(word):
    synsets = get_synsets(word)
    mappings = {}
    synset_contexts = {synset: get_synset_vocab(synset) for synset in synsets}

    for synset in synsets:
        title = synset.name().split('.')[0]
        wiki_context = get_wikipedia_vocab(title)
        if wiki_context:
            score = compute_score(synset_contexts[synset], wiki_context)
            mappings[synset] = {'title': title, 'score': score, 'url': get_wikipedia_url(title)}
        else:
            redirects = get_wikipedia_redirects(title)
            for redirect in redirects:
                wiki_context = get_wikipedia_vocab(redirect)
                if wiki_context:
                    score = compute_score(synset_contexts[synset], wiki_context)
                    mappings[synset] = {'title': redirect, 'score': score, 'url': get_wikipedia_url(redirect)}
                    break
    
    if len(mappings) > 1:
        total_score = sum(mapping['score'] for mapping in mappings.values())
        for synset in mappings:
            mappings[synset]['probability'] = mappings[synset]['score'] / total_score
    
    return mappings


def print_mappings_and_wiki_relations(words, filename):
    with open(filename, "w") as file:
        for word in words:
            file.write(f"Word: {word}\n")
            mappings = map_to_wiki(word)
            for synset, mapping in mappings.items():
                file.write(f"  Synset: {synset.name()}\n")
                file.write(f"    Definition: {synset.definition()}\n")
                file.write(f"    Wikipedia URL: {mapping['url']}\n")
                file.write("\n")


def print_detailed_relations(words, filename):
    with open(filename, "w") as file:
        for word in words:
            file.write(f"Word: {word}\n")
            mappings = map_to_wiki(word)
            for synset, mapping in mappings.items():
                file.write(f"  Synset: {synset.name()}\n")
                file.write(f"    Definition: {synset.definition()}\n")
                file.write(f"    Wikipedia URL: {mapping['url']}\n")
                if 'probability' in mapping:
                    file.write(f"    Probability: {mapping['probability']}\n")
                file.write(f"    Similar Concepts from Wikipedia:\n")
                similar_concepts = get_wikipedia_links(mapping['url'])
                for concept in similar_concepts:
                    file.write(f"      - {concept}\n")
                file.write("\n")


def main():
    astrophysics_words = [
        'planet', 'galaxy', 'comet', 'asteroid', 'supernova', 'nebula', 'telescope', 'satellite', 'constellation',
        'meteor', 'universe', 'black hole', 'quasar', 'cosmos', 'gravity', 'light year', 'orbit', 'astronaut', 'spacecraft',
        'observatory', 'eclipse', 'pulsar', 'solar system', 'moon', 'jupiter', 'mars', 'earth',
        'exoplanet', 'cosmology', 'interstellar', 'astrobiology', 'astrophysics'
    ]
    output_file = "nlp_lab_wordnet/wordnet++_output_Sirbu_Oana-Adriana.txt"
    print_mappings_and_wiki_relations(astrophysics_words, output_file)
    
    chosen_words = ['cosmos', 'comet', 'supernova']
    detailed_output_file = "nlp_lab_wordnet/wordnet++_detailed_output_Sirbu_Oana-Adriana.txt"
    print_detailed_relations(chosen_words, detailed_output_file)


if __name__ == "__main__":
    main()