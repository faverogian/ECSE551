import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#### PREPROCESSING HELPER FUNCTIONS ####

# Replace words with their lemmings
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_verbs(text):
    return [lemmatizer.lemmatize(word, pos='v') for word in text]

def lemmatize_nouns(text):
    return [lemmatizer.lemmatize(word) for word in text]

def prep_data(df):
    # Convert all characters to lowercase
    df['body'] = df['body'].str.lower()

    # Align encodings
    df['body'] = df['body'].str.replace('“', '"')
    df['body'] = df['body'].str.replace('”', '"')
    df['body'] = df['body'].str.replace('’', "'")
    df['body'] = df['body'].str.replace('‘', "'")
    df['body'] = df['body'].str.replace('—', '-')
    df['body'] = df['body'].str.replace('–', '-')
    df['body'] = df['body'].str.replace('\n', ' ')
    df['body'] = df['body'].str.replace('/', ' ')
    df['body'] = df['body'].str.replace('#x200b', ' ')
    df['body'] = df['body'].str.replace('-', ' ')
    df['body'] = df['body'].str.replace('%', ' ')

    # Remove basic punctuation
    translator = str.maketrans('', '', '<>"°œ!\()*+,.:;=?[\\]^_`{|}~1234567890')
    df['body'] = df['body'].str.translate(translator)

    # Replace accented characters with unaccented characters
    translator = str.maketrans('àáâãäåçèéêëìíîïñòóôõöùúûüýÿ', 'aaaaaaceeeeiiiinooooouuuuyy')
    df['body'] = df['body'].str.translate(translator)

    df['body'] = df['body'].apply(lambda x: x.replace("'", " "))

    df['body'] = df['body'].apply(lambda x: lemmatize_nouns(x.split()))
    df['body'] = df['body'].apply(lambda x: lemmatize_verbs(x))

    # Reconcatenate the words into a string
    df['body'] = df['body'].apply(lambda x: ' '.join(x))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['body'] = df['body'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    stop_words = set(stopwords.words('french'))
    df['body'] = df['body'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    return df

def get_term_freq(df, subreddit, vocab):
    # Get term frequency for each word in the vocabulary
    term_freq = {}
    for word in vocab:
        term_freq[word] = 0

    for body in df[df['subreddit'] == subreddit]['body']:
        for word in body.split():
            term_freq[word] += 1

    return term_freq

def get_highest_freq(subreddit, df):
    subreddit_df = df[df['subreddit'] == subreddit]
    sub_word_index = {}
    
    for row in subreddit_df['body']:
        for word in row.split():
            if word in sub_word_index:
                sub_word_index[word] += 1
            else:
                sub_word_index[word] = 1

    # Sort the dictionary by value
    sub_word_index = dict(sorted(sub_word_index.items(), key=lambda item: item[1], reverse=True))
    return list(sub_word_index.items())
    
def remove_uncommon_words(subreddit, df, threshold=2):
    subreddit_df = df[df['subreddit'] == subreddit]
    
    # Build a vocabulary of words and how many samples they appear in
    vocab = {}
    for row in subreddit_df['body']:
        for word in row.split():
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

    # Remove words that appear in few samples
    for word in list(vocab):
        if vocab[word] < threshold:
            del vocab[word]

    # Remove all words that are not in the vocabulary
    subreddit_df['body'] = subreddit_df['body'].apply(lambda x: ' '.join([word for word in x.split() if word in vocab]))

    return subreddit_df

def build_vocab(df):
    vocab = []
    for row in df['body']:
        for word in row.split():
            if word not in vocab:
                vocab.append(word)
    return vocab

def get_mutual_information(term_freq_df, class_count):
     # Convert term_freq_df to a numpy array
     term_freq_array = term_freq_df.to_numpy()

     # Calculate the joint probabilities of terms with classes
     eps = 1e-10

     P_TC = (term_freq_array + eps) / np.array(class_count).reshape(1, -1)
     P_T_notC = (np.array(class_count).reshape(1, -1) - term_freq_array + eps) / np.array(class_count).reshape(1, -1)

     # Calculate the marginal probabilities
     total_samples = 540
     P_T = np.sum(term_freq_array, axis=1) / total_samples
     P_C = class_count / total_samples

     # Calculate the mutual information
     MI = np.sum(P_TC * np.log2((P_TC + eps) / (P_T[:, np.newaxis] * P_C + eps)), axis=1) + \
          np.sum(P_T_notC * np.log2((P_T_notC + eps) / ((1 - P_T[:, np.newaxis]) * P_C + eps)), axis=1)

     return MI

#### TESTING HELPER FUNCTIONS ####

# Create function to plot a confusion matrix
def plot_conf_mat(conf_mat):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.heatmap(conf_mat,
                     annot=True,  # Annotate the boxes
                     cbar=False,
                     fmt='g',
                     cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()