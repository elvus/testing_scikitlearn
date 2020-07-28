import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np

df=pd.read_csv('Consumer_Complaints.csv')
col=['Product', 'Consumer complaint narrative']
df=df[col]
df=df[pd.notnull(df['Consumer complaint narrative'])]
df.columns = ['Product', 'Consumer_complaint_narrative']
df['category_id']=df['Product'].factorize()[0]
category_id_df=df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)
df.head()

tfidf=TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2), stop_words='english')

features=tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.category_id
features.shape

N=2
for Product, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    features_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in features_names if len(v.split(' '))==1]
    bigrams = [v for v in features_names if len(v.split(' '))==2]
    print("# '{}':".format(Product))
    print(" . Most correlated unigrams:\n. {}".format('\n.'.join(unigrams[-N:])))
    print(" . Most correlated bigrams:\n. {}".format('\n.'.join(bigrams[-N:])))