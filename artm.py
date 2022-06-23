# https://towardsdatascience.com/hierarchical-topic-modeling-with-bigartm-library-6f2ff730689f
# Hierarchical Topic Modeling

import logging
import string
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
from sklearn.feature_extraction.text import CountVectorizer

import artm
from parse_json import test

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

lemmatizer = WordNetLemmatizer()
# ps = PorterStemmer()

min_ngram_repetitions = 0

cachedStopWords = [
    "fig",
    "figure",
    "et",
    "al",
    "table",
    "data",
    "analysis",
    "analyze",
    "study",
    "method",
    "result",
    "conclusion",
    "author",
    "find",
    "found",
    "show",
    "perform",
    "demonstrate",
    "evaluate",
    "discuss",
    "google",
    "scholar",
    "pubmed",
    "web",
    "science",
    "crossref",
    "supplementary",
    "(fig.)",
    "(figure",
    "fig.",
    "al.",
    "did",
    "thus,",
    "…",
    "" "",
    "interestingly",
    "and/or",
    "author",
] + list(esw)


def lemmatize_article(sentence):
    sentence = word_tokenize(sentence)
    res = ""
    for word in sentence:
        word = lemmatizer.lemmatize(word)
        res += word + " "
    return res


def remove_stop_words(sentence):
    return " ".join([word for word in sentence.split() if word not in cachedStopWords])


def remove_short(sentence):
    return " ".join([word for word in sentence.split() if len(word) >= 3])


def remove_digits(sentence):
    return " ".join([i for i in sentence.split() if not i.isdigit()])


def preprocess(all_texts):
    all_texts = list(map(lambda x: x.lower(), all_texts))
    all_texts = list(
        map(lambda x: x.translate(str.maketrans(
            "", "", string.punctuation)), all_texts)
    )
    logger.info("Lemmatizing")
    all_texts = list(map(lambda x: lemmatize_article(x), all_texts))
    logger.info("Removing whitespaces")
    all_texts = list(map(lambda x: x.strip(), all_texts))
    logger.info("Removing stop words")
    all_texts = list(map(lambda x: remove_stop_words(x), all_texts))
    logger.info("Removing short words")
    all_texts = list(map(lambda x: remove_short(x), all_texts))
    logger.info("Removing digits")
    all_texts = list(map(lambda x: remove_digits(x), all_texts))
    return all_texts


def get_ngrams(text, n):
    n_grams = ngrams(word_tokenize(text), n)
    return [" ".join(grams) for grams in n_grams]


def print_measures(model_artm):
    print(
        "Sparsity Phi ARTM:{}".format(
            model_artm.score_tracker["SparsityPhiScore"].last_value
        )
    )
    print(
        "Sparsity Theta ARTM:{}".format(
            model_artm.score_tracker["SparsityThetaScore"].last_value
        )
    )
    print(
        "Perplexity ARTM: {}".format(
            model_artm.score_tracker["PerplexityScore"].last_value
        )
    )

    ig, axs = plt.subplots(1, 3, figsize=(30, 5))

    for idx, score, y_label in zip(
        range(3),
        ["PerplexityScore", "SparsityPhiScore", "SparsityThetaScore"],
        ["ARTM perplexity", "ARTM Phi sparsity", "ARTM Theta sparsity"],
    ):
        axs[idx].plot(
            range(model_artm.num_phi_updates),
            model_artm.score_tracker[score].value,
            "r--",
            linewidth=2,
        )
        axs[idx].set_xlabel("Iterations count")
        axs[idx].set_ylabel(y_label)
        axs[idx].grid(True)


def get_articles_on_theme(dataset, topic, num_topics):
    theta = np.array(model_artm.get_theta("topic_{}".format(topic)).iloc[0]).theta[
        theta <= 0.05
    ] = 0
    idx = np.nonzero(theta)[0]
    articles = zip(idx, theta[idx])
    articles = sorted(articles, key=lambda x: x[1], reverse=True)
    articles = [x[0] for x in articles]
    return dataset.iloc[articles].PaperText[:num_topics]


def main():

    # PRE PROCESSING ###

    # df = pd.read_csv("./papers.csv")
    # all_texts = df.paper_text
    all_texts = test()
    all_texts = preprocess([all_texts])
    print(all_texts)

    bigrams = []
    for article in all_texts:
        bigrams += list(
            map(
                lambda x: x[0],
                list(
                    filter(
                        lambda x: x[1] >= min_ngram_repetitions,
                        Counter(get_ngrams(article, 2)).most_common(),
                    )
                ),
            )
        )

    bigrams = list(
        filter(lambda x: "package" not in x and "document" not in x, bigrams)
    )
    bigrams = list(
        map(
            lambda x: x[0],
            (
                list(
                    filter(
                        lambda x: x[1] >= min_ngram_repetitions,
                        Counter(bigrams).most_common(),
                    )
                )
            ),
        )
    )

    print(len(bigrams))
    print(bigrams[:5])
    topic_names = bigrams[:5]

    # ARTM MODEL ###
    features = 6

    n_wd_bigrams = np.empty((len(bigrams), len(all_texts)))

    for i in range(len(bigrams)):
        for j in range(len(all_texts)):
            n_wd_bigrams[i][j] = all_texts[j].count(bigrams[i])

    cv = CountVectorizer(max_features=features, stop_words="english")
    n_wd = np.array(cv.fit_transform(all_texts).todense()).T
    vocabulary = cv.get_feature_names()

    n_wd = np.concatenate((n_wd, n_wd_bigrams))
    vocabulary += bigrams
    dictionary = Dictionary(vocabulary)

    model_artm = artm.ARTM(
        topic_names=topic_names,
        cache_theta=True,
        scores=[
            artm.PerplexityScore(name="PerplexityScore",
                                 dictionary=dictionary),
            artm.SparsityPhiScore(name="SparsityPhiScore"),
            artm.SparsityThetaScore(name="SparsityThetaScore"),
            artm.TopicKernelScore(
                name="TopicKernelScore", probability_mass_threshold=0.3
            ),
            artm.TopTokensScore(name="TopTokensScore", num_tokens=8),
        ],
        regularizers=[
            artm.SmoothSparseThetaRegularizer(name="SparseTheta", tau=-0.4),
            artm.DecorrelatorPhiRegularizer(name="DecorrelatorPhi", tau=2.5e5),
        ],
    )

    model_artm.num_document_passes = 4
    model_artm.initialize(dictionary)
    model_artm.fit_offline(batch_vectorizer=bv, num_collection_passes=20)

    print_measures(model_artm)
    return

    for topic_name in model_artm.topic_names:
        print(
            topic_name
            + ": "
            + model_artm.score_tracker["TopTokensScore"].last_tokens[topic_name]
        )

    get_articles_on_theme(df, 8, 5)
    topic_names = ["topic_{}".format(i) for i in range(50)]

    model_artm1 = artm.ARTM(
        topic_names=topic_names,
        cache_theta=True,
        scores=[
            artm.PerplexityScore(name="PerplexityScore",
                                 dictionary=dictionary),
            artm.SparsityPhiScore(name="SparsityPhiScore"),
            artm.SparsityThetaScore(name="SparsityThetaScore"),
            artm.TopicKernelScore(
                name="TopicKernelScore", probability_mass_threshold=0.3
            ),
            artm.TopTokensScore(name="TopTokensScore", num_tokens=12),
        ],
        regularizers=[
            artm.SmoothSparseThetaRegularizer(name="SparseTheta", tau=-0.4),
            artm.SmoothSparsePhiRegularizer(name="SparsePhi", tau=-0.25),
            artm.DecorrelatorPhiRegularizer(name="DecorrelatorPhi", tau=2.5e5),
        ],
        seed=243,
    )  # seed is required for heirarchy

    model_artm1.num_document_passes = 4
    model_artm1.set_parent_model(
        parent_model=model_artm, parent_model_weight=0.75)
    model_artm1.initialize(dictionary)

    model_artm1.fit_offline(batch_vectorizer=bv, num_collection_passes=12)

    subt = pd.DataFrame(model_artm1.get_parent_psi())
    subt.columns = ["topic_{}".format(i) for i in range(10)]
    subt.index = ["subtopic_{}".format(i) for i in range(50)]

    def subtopics_wrt_topic(topic_number, matrix_dist):
        return matrix_dist.iloc[:, topic_number].sort_values(ascending=False)[:5]

    subtopics_wrt_topic(0, subt)


if __name__ == "__main__":
    main()
