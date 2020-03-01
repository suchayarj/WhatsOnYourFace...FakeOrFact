import unicodedata
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk import WordNetLemmatizer, pos_tag
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

stopwords_ = set(stopwords.words('english'))

#Clean & normalize text
def cleanText(wordSeries):
        tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
        def remove_accents(input_str):
            nfkd_form = unicodedata.normalize('NFKD', input_str)
            only_ascii = nfkd_form.encode('ASCII', 'ignore')
            return only_ascii.decode()
        def remove_punctuation(text):
            return text.translate(tbl)
        wordSeries = wordSeries.apply(lambda x: remove_punctuation(x))#remove punctuation
        wordSeries = wordSeries.apply(lambda x: ''.join([i for i in x if not i.isdigit()]))#remove digits
        wordSeries = wordSeries.apply(lambda x: x.lower())#lower cases
        wordSeries = wordSeries.apply(lambda x: x.replace('<br >', ' '))#remove html
        wordSeries = wordSeries.apply(lambda x: x.replace('<br>', ' '))#remove html
        wordSeries = wordSeries.apply(lambda x: x.replace('\n', ' '))#remove html
        wordSeries = wordSeries.apply(lambda x: x.replace('\n\n', ' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('$', ' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('>', ' '))
        wordSeries = wordSeries.apply(lambda x: remove_accents(x))
        wordSeries = wordSeries.apply(lambda x: x.replace('`', ''))#remove extra punctuation
        
        return wordSeries
        ## example df.text --> cleanText(df.text)

#Tokenize, remove stopwords, stem -->return text ready to be vectorized
def cleanText2(wordSeries):
        wordSeries = wordSeries.apply(word_tokenize)#tokenize the word
        #remove stopwords
        wordSeries = wordSeries.apply(lambda x: [item for item in x if item not in stopwords_ ])
        #stem
        stemmer = SnowballStemmer("english")
        wordSeries = wordSeries.apply(lambda x: [stemmer.stem(y) for y in x])
        wordSeries = wordSeries.apply(lambda x:' '.join([y for y in x ]))#detokenized
        
        return wordSeries

def vectorize(wordSeries, max_feat, ngram=1):
    '''
    input
    wordSeries: Detokenized wordSeries
    max_features: maximum features wanted (int)
    ngram_range: (n,n) 

    output
    vector ready to be used in MLP model
    '''
    tfidf = TfidfVectorizer(max_features = max_feat, ngram_range = (1,ngram))
    doc_tfidf_matrix = tfidf.fit_transform(X).todense()
    vector = pd.DataFrame(doc_tfidf_matrix, X = tfidf.get_feature_names())
    return vector


