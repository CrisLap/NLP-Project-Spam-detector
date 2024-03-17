import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases

def lda_preprocessing(emails):
    """
    Perform preprocessing for Latent Dirichlet Allocation (LDA) on a corpus of emails.

    Args:
        emails (list): A list of strings representing email documents.

    Returns:
        tuple: A tuple containing the following elements:
            - cleaned_tokens_with_bigrams_trigrams (list): A list of tokenized emails with bigrams and trigrams.
            - dictionary (gensim.corpora.Dictionary): A dictionary mapping words to their integer ids.
            - corpus (list of list of tuple): A list of Bag-of-Words representations of the documents.

    This function preprocesses the emails by tokenizing them, creating bigrams and trigrams,
    building a dictionary, filtering extreme cases, and creating a corpus of Bag-of-Words representations.

    """
    # Clean the tokens
    cleaned_tokens = [[word for word in email.split() if len(word) > 1] for email in emails]
    
    # Generate bigrams and trigrams
    bigram_model = Phrases(cleaned_tokens, min_count=20)
    trigram_model = Phrases(bigram_model[cleaned_tokens], min_count=20)
    cleaned_tokens_with_bigrams_trigrams = [trigram_model[bigram_model[token]] for token in cleaned_tokens]
    
    # Create a dictionary of terms
    dictionary = Dictionary(cleaned_tokens_with_bigrams_trigrams)
    print(f"Dictionary size before filtering: {len(dictionary)}")
    
    # Filter extremes
    dictionary.filter_extremes(no_below=5, no_above=0.95)
    print(f"Dictionary size after filtering: {len(dictionary)}")
    
    # Convert documents to bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in cleaned_tokens_with_bigrams_trigrams]
    
    return cleaned_tokens_with_bigrams_trigrams, dictionary, corpus


def plot_coherence_values(texts, corpus, dictionary, topics):
    """
    Plot the coherence scores against different numbers of topics for Latent Dirichlet Allocation (LDA) models.

    Parameters:
    texts (list of list of str): Preprocessed texts, where each element is a list of words representing a document.
    corpus (gensim.corpora.MmCorpus): Gensim corpus object.
    dictionary (gensim.corpora.Dictionary): Gensim dictionary object.
    topics (list of int): List of numbers of topics to evaluate.

    Returns:
    None

    This function trains multiple LDA models with different numbers of topics and evaluates their coherence scores.
    It then plots the coherence scores against the number of topics using seaborn line plot.

    Note:
    This function requires the following libraries:
    - tqdm
    - gensim.models.ldamulticore.LdaMulticore
    - gensim.models.CoherenceModel
    - seaborn
    - matplotlib.pyplot

    Example usage:
    plot_coherence_values(texts, corpus, dictionary, topics)
    """
    
    coherences = []

    for num_topics in tqdm(topics, 
                           desc="Training and evaluating LDA models",
                           unit="num_topics"):
        lda = LdaMulticore(corpus=corpus, 
                           id2word=dictionary, 
                           num_topics=num_topics, 
                           passes=10)
        coherence_model_lda = CoherenceModel(model=lda,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence='c_v')
        coherence = coherence_model_lda.get_coherence()
        coherences.append(coherence)

    sns.set(style="darkgrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=topics, y=coherences, marker='o')
    plt.xticks(topics)

    plt.title('Coherence Score vs. Number of Topics', fontsize=18)
    plt.xlabel('Number of Topics', fontsize=14)
    plt.ylabel('Coherence Score', fontsize=14)

    plt.show()