import numpy as np
import pandas as pd
import difflib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('movies.csv')

# printing the first 5 rows of the dataframe
movies_data.head()

# number of rows and columns in the data frame
movies_data.shape

# selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','overview','director']
print(selected_features)

# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

# combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['overview']+' '+movies_data['director']

print(combined_features)

# converting the text data to feature vectors

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

print(feature_vectors)

#creating and saving the feature vector
with open('vectorizer.pkl', 'wb') as f:
  pickle.dump(vectorizer, f)

with open('feature_vectors.pkl', 'wb') as f:
  pickle.dump(feature_vectors, f)

print("Pickel files created")








