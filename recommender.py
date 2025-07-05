import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load the data
movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits_df = pd.read_csv("tmdb_5000_credits.csv")

# Merge using 'title'
movies_df = movies_df.merge(credits_df, on='title')

# Helper to extract top 3 cast members
def get_top_cast(cast_json):
    try:
        cast = ast.literal_eval(cast_json)
        return ' '.join([actor['name'] for actor in cast[:3]])
    except:
        return ''

# Helper to extract director
def get_director(crew_json):
    try:
        crew = ast.literal_eval(crew_json)
        for person in crew:
            if person['job'] == 'Director':
                return person['name']
        return ''
    except:
        return ''

# Helper to get genres as text
def get_genres(genres_json):
    try:
        genres = ast.literal_eval(genres_json)
        return ' '.join([g['name'] for g in genres])
    except:
        return ''

# Extract useful features
movies_df['cast'] = movies_df['cast'].apply(get_top_cast)
movies_df['director'] = movies_df['crew'].apply(get_director)
movies_df['genres'] = movies_df['genres'].apply(get_genres)

# Fill only text columns
for col in ['overview', 'cast', 'director', 'genres']:
    movies_df[col] = movies_df[col].fillna('')

# Combine all features
movies_df['combined'] = (
    movies_df['title'] + ' ' +
    movies_df['overview'] + ' ' +
    movies_df['cast'] + ' ' +
    movies_df['director'] + ' ' +
    movies_df['genres']
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Indexing by title
movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'].str.lower().str.strip())

# Suggest similar title if not found
def suggest_title(input_title):
    titles = movies_df['title'].str.lower().str.strip().tolist()
    matches = get_close_matches(input_title.lower().strip(), titles, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Recommend similar movies
def recommend(movie_title, num_recommendations=5):
    movie_title = movie_title.lower().strip()

    if movie_title not in indices:
        suggestion = suggest_title(movie_title)
        if suggestion:
            print(f"Did you mean: '{suggestion}'?")
        return "Movie not found in dataset."

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    movie_indices = [i[0] for i in sim_scores]

    recommendations = []
    for i in movie_indices:
        row = movies_df.iloc[i]
        recommendations.append(f"{row['title']} (Rating: {row['vote_average']}, Genres: {row['genres']})")

    return recommendations

# ---- Run ----
movie = input("Enter a movie title: ")
results = recommend(movie)

print("\nRecommended movies:")
if isinstance(results, list):
    for m in results:
        print(f"- {m}")
else:
    print(results)
