from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from backend import recommended_movies

# Load saved models
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('feature_vectors.pkl', 'rb') as f:
    feature_vectors = pickle.load(f)

# Load the movies data
movies_data = pd.read_csv('movies.csv')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serves the HTML file

@app.route('/rec', methods=['POST'])
def recommend_movies():
    # Get the movie name from the POST request
    movie_name = request.form['movie_name']
    movie_list = recommended_movies(movie_name)
    # return render_template("results.html", movie_list = movie_list)
    if not movie_list:
        error = f"No close match found for movie: {movie_name}"
        return render_template("results.html", error=error)
    
    return render_template("results.html", movie_list=movie_list)


if __name__ == '__main__':
    app.run(debug=True)
