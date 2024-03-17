import re
import string
import contractions
import spacy
import nltk
from nltk.corpus import stopwords
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS

nltk.download('stopwords')
stop_words = list(set(STOPWORDS.union(stopwords.words('english'))))
nlp = spacy.load('en_core_web_trf', disable=['parser'])

def show_barplot(x: str, df: pd.DataFrame):
    """
    Display a bar plot showing the distribution of a categorical variable in the DataFrame.

    Args:
        x (str): The name of the categorical variable to visualize.
        df (pd.DataFrame): The pandas DataFrame containing the data.

    Returns:
        None

    This function creates a bar plot to visualize the distribution of a categorical variable in the DataFrame.
    It calculates the percentage of each category and displays it on top of each bar.
    The plot is displayed using seaborn.barplot.
    """

    # Calculate percentage distribution
    df_percent = df[x].value_counts(normalize=True).mul(100).reset_index()
    df_percent.columns = [x, 'percent']

    # Plot
    sns.set(style="darkgrid")
    ax = sns.barplot(x=x, y='percent', data=df_percent, palette=["blue", "red"], alpha=0.8)

    # Add percentage labels on top of bars
    for p in ax.patches:
        txt = f"{p.get_height():.2f}%"
        txt_x = p.get_x() + p.get_width() / 2
        txt_y = p.get_height()
        ax.annotate(txt, (txt_x, txt_y), ha='center', va='bottom')

    # Set plot labels and title
    plt.title('Label Distribution', fontsize=18)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.ylim(0, 100)  # Limit y-axis to 0-100 for percentage representation
    plt.show()

def _data_cleaner(sentence):
    """
    Clean a sentence by performing the following operations:
    1. Convert the sentence to lowercase.
    2. Remove English stopwords.
    3. Expand contractions.
    4. Remove the prefix "Subject:" (if present).
    5. Remove punctuation.
    6. Remove digits.
    7. Tokenize the sentence using spaCy and lemmatize each token.
    8. Remove extra whitespaces.

    Args:
    - sentence (str): Input sentence to be cleaned.

    Returns:
    - str: Cleaned sentence.
    """
    # Convert the sentence to lowercase
    sentence = sentence.lower()

    # Remove English stopwords
    sentence = ' '.join(word for word in sentence.split() if word not in stop_words)

    # Expand contractions
    sentence = contractions.fix(sentence)

    # Remove the prefix "Subject:" (if present)
    to_remove = "subject:"
    if sentence.startswith(to_remove):
        sentence = sentence.removeprefix(to_remove)

    # Remove punctuation
    for c in string.punctuation:
        sentence = sentence.replace(c, " ")

    # Remove digits
    sentence = re.sub('\d', ' ', sentence)

    # Tokenize the sentence using spaCy and lemmatize each token
    document = nlp(sentence)
    sentence = ' '.join(token.lemma_ for token in document)

    # Remove extra whitespaces
    sentence = re.sub(' +', ' ', sentence).strip()

    return sentence

def clean_data(corpus):
    """
    Clean the text data in the corpus.

    Args:
        corpus (list): A list of strings representing text documents.

    Returns:
        list: A list of cleaned text documents.

    This function iterates through each document in the corpus and applies a cleaning function to it.
    The cleaning process may include removing special characters, punctuation, and stopwords.
    The progress of the cleaning process is displayed using tqdm.

    """
    cleaned_corpus = []
    
    for document in tqdm(corpus, desc="Cleaning data", unit="document"):
        cleaned_corpus.append(_data_cleaner(document))
    
    return cleaned_corpus

def show_wordclouds(x, text_col, df):
    """
    Display word clouds for each category in the DataFrame.

    Args:
        x (str): The name of the column containing category labels.
        text_col (str): The name of the column containing text data.
        df (pd.DataFrame): The pandas DataFrame containing the data.

    Returns:
        None

    This function generates and displays word clouds for each category in the DataFrame.
    It creates a subplot for each category and generates a word cloud using the text data associated with that category.
    The word clouds are displayed using Matplotlib's imshow function.
    """
    
    labels = df[x].unique()
    
    plt.figure(figsize=(20,10))
    count=1
    for label in labels:
        label_text = ' '.join(df[df['label']==label][text_col])
        wordcloud = WordCloud(width=1500, 
                              height=1200,
                              random_state=0,
                              background_color ='black', 
                              margin=1,
                              stopwords = stop_words,
                              ).generate(label_text)
        plt.subplot(1,2,count)
        plt.axis("off")
        plt.title("Label: " + label,fontsize=18)
        plt.tight_layout(pad=3)
        plt.imshow(wordcloud,interpolation='bilinear')
        count=count+1
    plt.show()