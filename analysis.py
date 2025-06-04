from collections import Counter
from math import log2
from pathlib import Path
from pretty_print import format_prop, show_list, show_posts
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd

def data_exploration(all_posts):
    liberal_posts = all_posts[all_posts.political_lean == 'Liberal']
    conservative_posts = all_posts[all_posts.political_lean == 'Conservative']

    print(
        format_prop("liberal posts", len(liberal_posts), len(all_posts)) + ' from',
        show_list(
            liberal_posts.subreddit.value_counts().items(),
            (20, 5),
        ),
        '',
        'Top scoring posts from liberal',
        show_posts(liberal_posts.nlargest(5, 'score')),
        '',
        '='*80,
        '',
        format_prop("conservative posts", len(conservative_posts), len(all_posts)) + ' from',
        show_list(
            conservative_posts.subreddit.value_counts().items(),
            (20, 5),
        ),
        '',
        'Top scoring posts from conservative',
        show_posts(conservative_posts.nlargest(5, 'score')),
        sep='\n'
    )

def words_analysis(posts):
    liberal_posts = all_posts[all_posts.political_lean == 'Liberal']
    conservative_posts = all_posts[all_posts.political_lean == 'Conservative']

    nltk.download('punkt_tab')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    liberal_word_count = count_words(liberal_posts, lemmatizer)
    conservative_word_count = count_words(conservative_posts, lemmatizer)
    liberal_word_freq = to_freq(liberal_word_count)
    conservative_word_freq = to_freq(conservative_word_count)
    combined_word_freq = to_freq(liberal_word_count + conservative_word_count)
    # if a word is not present in one of the corpus, we cheat a bit and pretend
    # it’s present one time in the corpus. This is to avoid computation issues
    # caused by non-existing words (crash / division by zero). If we had a
    # bigger corpus, we could expect any word to show up at least once. Also,
    # since we are only interested in comparing the word frequencies between
    # the two groups, this cheat should not add too much distorsion.
    liberal_fallback_freq = 1 / liberal_word_count.total()
    conservative_fallback_freq = 1 / conservative_word_count.total()

    liberal_voc = set(liberal_word_freq.keys())
    conservative_voc = set(conservative_word_freq.keys())
    all_voc = liberal_voc | conservative_voc

    lc_voc_log_ratio = {
        word: log2(liberal_word_freq.get(word, liberal_fallback_freq))
            - log2(conservative_word_freq.get(word, conservative_fallback_freq))
        for word in all_voc
    }
    def idf(word):
        return -log2(combined_word_freq[word])
    stats = [
        (
            word,
            ratio,
            idf(word),
        ) for word, ratio in lc_voc_log_ratio.items()
    ]
    voc_stats = pd.DataFrame(
        stats,
        columns=['word', 'ratio', 'idf']
    )
    voc_stats['score'] = voc_stats.ratio * voc_stats.idf

    voc_stats.score.plot(kind='hist', bins=np.arange(voc_stats.score.min(), voc_stats.score.max()+1))
    plt.savefig('vocabulary_score_distribution.png')
    plt.show()

    top_liberal_voc = voc_stats.nlargest(50, 'score')
    top_conservative_voc = voc_stats.nsmallest(50, 'score')

    def present_top_words(top_words):
        return show_list(
            [(i.word, i.score) for i in top_words.itertuples()],
            widths=(25, 15),
            formats=('', '.2f'),
            headers=('Word', 'Score'),
            with_index=True
        )

    print(
        "Top words used more by liberals:",
        present_top_words(top_liberal_voc),
        "",
        "Top words used more by conservatives:",
        present_top_words(top_conservative_voc),
        sep='\n'
    )

    word_score_explorer(voc_stats)

def word_score_explorer(voc_stats):
    PROMPT = 'Write a word or a score value'
    print(PROMPT)
    while True:
        query = input('> ')
        try:
            l, h = map(float, query.split(' '))
        except ValueError:
            res = voc_stats[voc_stats.word == query]
        except EOFError:
            return
        else:
            res = voc_stats[(l <= voc_stats.score) & (voc_stats.score <= h)]
        print(res)


def count_words(posts, lemmatizer):
    counter = Counter()
    for post in posts.itertuples():
        tokens = extract_tokens(post.title, lemmatizer)
        counter.update(tokens)
    return counter

def to_freq(counts: Counter):
    total = counts.total()
    freq = Counter({
        word: c / total
        for word, c in counts.items()
    })
    return freq

def extract_tokens(text: str, lemmatizer):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) > 1]
    return lemmatized_tokens

def data_cleaning(posts):
    # remove urls
    URL_REGEX = r'https?://\S+'
    total_count = posts.shape[0]
    url_in_title = posts.title.str.contains(URL_REGEX, regex=True)
    url_in_selftext = posts.text.str.contains(URL_REGEX, regex=True)
    total_with_urls = (url_in_title | url_in_selftext).sum()
    print(f"{total_with_urls} posts with urls ({100*total_with_urls/total_count:.2f} %)")
    posts.title = posts.title.replace(URL_REGEX, '', regex=True)
    posts.text = posts.text.replace(URL_REGEX, '', regex=True)
    # normalize titles
    posts.title = posts.title.map(lambda p: p.strip().lower())

def build_network(posts):
    shared_posts = posts.groupby('title').filter(lambda g: len(g) > 1).groupby('title')

    graph = nx.Graph()
    graph.add_nodes_from(
        (item.subreddit, dict(political_lean=item.political_lean))
        for item in posts[['subreddit', 'political_lean']].itertuples()
    )

    edges = Counter()
    for group_name, group in shared_posts:
        shared_subs = group['subreddit']
        for subs in itertools.combinations(shared_subs, 2):
            key = tuple(sorted(subs))
            edges[key] += 1

    graph.add_edges_from((n1, n2, {'weight': w}) for (n1, n2), w in edges.items())
    print("Network graph computed.")
    return graph


if __name__ == '__main__':
    all_posts = pd.read_csv(Path(__file__).parent / 'data/reddit_posts.csv').rename(
        columns={
            'Title': 'title',
            'Political Lean': 'political_lean',
            'Score': 'score',
            'Id': 'id',
            'Subreddit': 'subreddit',
            'URL': 'url',
            'Num of Comments': 'comments_count',
            'Text': 'text',
            'Date Created': 'date_created',
        }
    )
    data_cleaning(all_posts)
    network = build_network(all_posts)
    nx.write_gexf(network, 'duplicate_posts_network.gexf')
    data_exploration(all_posts)
    words_analysis(all_posts)
