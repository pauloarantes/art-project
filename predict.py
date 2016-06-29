import models
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import cPickle as pickle
from itertools import izip

# Loading data
df = models.load_and_add_purchase_data()

# Saving original user ids for later
ids = df.index

# Adding information
merged_df = models.preprocess_purchases_and_join_with(df)

# Loading model
with open('model.pkl', 'r') as f:
    model = pickle.load(f)

# Scaling features
merged_df.num_sessions = scale(merged_df.num_sessions)
merged_df.total_artists_followed = scale(merged_df.total_artists_followed)
merged_df.total_artworks_favorited = scale(merged_df.total_artworks_favorited)
merged_df.total_artworks_shared = scale(merged_df.total_artworks_shared)

# Removing y label and defining X matrix for probability of purchase prediction
y = merged_df.pop('purchased').values
X = merged_df.values

# Creating Dataframe
probs = pd.DataFrame(columns=('id', 'purchase_prob'))

# Predicting probabilities of purchase for each user based on trained model
for user_id, prob in izip(ids[:20], model.predict_proba(X)):
    probs.loc[user_id] = [user_id, round(prob[1]*100, 4)]

# Sorting probabilities by importance and exporting to csv
probs = probs.sort_values('purchase_prob', ascending=False)
probs.to_csv('purchase_probs.csv')
