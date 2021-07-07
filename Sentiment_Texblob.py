import nltk
import pandas as pd
from nltk.tokenize import ToktokTokenizer
import re
from textblob import TextBlob
import preprocessor as p

# Character Removing of tweets
def remove_special_characters(text):
  # define the pattern to keep
  pat = r'[^a-zA-z\'\s]'
  return re.sub(pat, '', text)
# function for stemming
def get_stem(text):
   stemmer = nltk.porter.PorterStemmer()
   text = ' '.join([stemmer.stem(word) for word in text.split()])
   return text

# function for lemmatization
lemmatizer = nltk.WordNetLemmatizer()
def get_lem(text):
   word_list = nltk.word_tokenize(text)
   lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
   return lemmatized_output

# function to remove stopwords
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
   # convert sentence into token of words
   tokens = tokenizer.tokenize(text)
   tokens = [token.strip() for token in tokens]
   # check in lowercase
   t = [token for token in tokens if token.lower() not in stopword_list]
   text = ' '.join(t)
   return text


df = pd.read_csv('Purchasing Power Tweet Data.csv',engine='python')
processed_text=[]
s=[]
for d in (df.text):
  text=p.clean(str(d))
  clean_text=remove_special_characters(text)
  stemmed_text=get_stem(clean_text)
  lemma_text=get_lem(stemmed_text)
  final_text=remove_stopwords(lemma_text)
  #Extra Preprocessing
  pat = r'[^a-zA-z\s]'
  final_text=re.sub(pat, '', final_text)
  final_text=' '.join([w for w in final_text.split() if len(w) > 1])
  processed_text.append(final_text)
  S=TextBlob(final_text)
  if S.sentiment.polarity>0:
     s.append(1)
  elif S.sentiment.polarity<0:
      s.append(-1)
  else:
      s.append(0)

dict = {'Processed_Tweet': processed_text, 'Sentiment': s}
df = pd.DataFrame(dict)
df.to_csv("Processed_Dataset.csv")
