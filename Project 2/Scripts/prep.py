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

#### Vocab ####

mtl_associations = [
    "canadiens",
    "bagel",
    "poutine",
    "mtl",
    "yul",
    "muc",
    "stm",
    "udem",
    "mcgill",
    "chum",
    "quebec",
    "qc",
    "fleurdelisé",
    "carnaval",
    "chateau",
    "vieuxquebec",
    "lacsaintjean",
    "gaspe",
    "peninsula",
    "saint",
    "jean",
    "baptiste",
    "bonhomme"
    ]
tor_associations = [
    "jays",
    "raptors",
    "cntower",
    "the6ix",
    "tdot",
    "scarborough",
    "etobicoke",
    "vaughan",
    "yorkville",
    "annex",
    "kensington",
    "distillery",
    "ford",
    "tory",
    "drake",
    "ttc",
    "uoft",
    "tor",
    "ryerson",
    "york",
    "leafs",
]
london_associations = [
    "uk",
    "britain",
    "british",
    "england",
    "queen",
    "king",
    "brit",
    "mate",
    "soho",
    "bbc",
    "flat"
]
paris_associations = [
    "france",
    "eiffel",
    "louvre",
    "macron",
    "fr"
]

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

    chars_to_replace = '<>"°œ!\\()*+,.:;=?[\\]^_`{|}~1234567890'
    translator = str.maketrans(chars_to_replace, ' ' * len(chars_to_replace))
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

def word_replacement(df):
    # Replace words in df with their associated words
    df['body'] = df['body'].apply(lambda x: ' '.join([word if word not in mtl_associations else 'montreal' for word in x.split()]))
    df['body'] = df['body'].apply(lambda x: ' '.join([word if word not in tor_associations else 'toronto' for word in x.split()]))
    df['body'] = df['body'].apply(lambda x: ' '.join([word if word not in london_associations else 'london' for word in x.split()]))
    df['body'] = df['body'].apply(lambda x: ' '.join([word if word not in paris_associations else 'paris' for word in x.split()]))
    
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
     total_samples = np.sum(class_count)
     P_T = np.sum(term_freq_array, axis=1) / total_samples
     P_C = class_count / total_samples

     # Calculate the mutual information
     MI = np.sum(P_TC * np.log2((P_TC + eps) / (P_T[:, np.newaxis] * P_C + eps)), axis=1) + \
          np.sum(P_T_notC * np.log2((P_T_notC + eps) / ((1 - P_T[:, np.newaxis]) * P_C + eps)), axis=1)

     return MI

def remove_common_words(df, subreddits, thresh):
    # Build vocabulary of words
    vocab = build_vocab(df)

    # Get the frequency of each word in the vocabulary (how many samples it appears in)
    subdict = {}
    for subreddit in subreddits:
        subdict[subreddit] = get_highest_freq(subreddit, df)
    for key in subdict:
        subdict[key] = [item[0] for item in subdict[key]]

    # Get aggregate index of words in the vocabulary
    agg_index = {}
    for word in vocab:
        agg_index[word] = 0
        for key in subdict:
            if word in subdict[key]:
                agg_index[word] += subdict[key].index(word)
            else:
                agg_index[word] += len(subdict[key])
    agg_index = dict(sorted(agg_index.items(), key=lambda item: item[1], reverse=False))

    # Remove these most common words from the vocabulary
    for word in list(agg_index)[:thresh]:
        vocab.remove(word)
    
    vocab.extend(['montreal', 'toronto', 'london', 'paris'])

    df['body'] = [' '.join(word for word in sample.split() if word in vocab) for sample in df['body']]

    return df, vocab

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

def generate_kaggle_submission(kaggle_test, kaggle_test_pred):
    kaggle_test_dict = {
    'id': kaggle_test['id'],
    'subreddit': kaggle_test_pred
    }

    kaggle_test_df = pd.DataFrame(kaggle_test_dict)

    return kaggle_test_df

def mutual_info_transform(df, thresh):
    vocab = build_vocab(df)

    classes = df['subreddit'].unique()
    class_counts = df['subreddit'].value_counts().to_numpy()

    term_freq = {}
    for subreddit in classes:
        term_freq[subreddit] = get_term_freq(df, subreddit, vocab)
        
    # Make a dataframe of the term frequencies
    term_freq_df = pd.DataFrame.from_dict(term_freq, orient='index')
    term_freq_df = term_freq_df.transpose()

    # Create a dataframe of the mutual information
    MI = get_mutual_information(term_freq_df, class_counts)
    MI_df = pd.DataFrame(MI, columns=['MI'])
    MI_df['word'] = list(vocab)
    MI_df = MI_df.sort_values(by=['MI'], ascending=False)

    # Create a list of the top words based on MI
    MI_N = thresh
    MI_df_top = MI_df.head(MI_N)
    top_words = MI_df_top['word'].tolist()
    top_words.extend(['montreal', 'toronto', 'london', 'paris'])

    # Create a new dataframe with only the top words
    top_df = df.copy()
    top_df['body'] = top_df['body'].apply(lambda x: ' '.join([word for word in x.split() if word in top_words]))

    # Remove samples with no words
    top_df = top_df[top_df['body'] != '']

    return top_df