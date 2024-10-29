import numpy as np
import pandas as pd
import difflib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import render_template



#loading the vectors
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('feature_vectors.pkl', 'rb') as f:
    feature_vectors = pickle.load(f)

movies_data = pd.read_csv('movies.csv')

def recommended_movies(movie_name):
# getting the similarity scores using cosine similarity

  similarity = cosine_similarity(feature_vectors)

  list_of_all_titles = movies_data['title'].tolist()

  find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

  # close_match = find_close_match[0]
  if find_close_match:
        close_match = find_close_match[0]
  else:
        print(f"No close match found for movie: {movie_name}")
        return []
        # error = f"No close match found for movie: {movie_name}"
        # return render_template("results.html", error=error)  # Pass the error to the template

 

  index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

  similarity_score = list(enumerate(similarity[index_of_the_movie]))

  sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
  #print('Movies suggested for you : \n')

  movie_list = []
  i = 1
  for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if (i<20):
      movie_list.append(title_from_index)
      i += 1
    else:
       break

  return(movie_list)