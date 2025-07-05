import pandas as pd
import ast
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ------------------- TMDB API Config -------------------
TMDB_API_KEY = "649e07fe94678f01b46c9ff1695b8702"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

import time

def fetch_poster(title, retries=3, delay=1):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results")
                if results:
                    poster_path = results[0].get("poster_path")
                    if poster_path:
                        return f"{TMDB_IMAGE_BASE}{poster_path}"
            break  # If status code is bad or not retryable, exit early
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay)


# ------------------- Data Loading -------------------

@st.cache_data
def load_data():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits_df = pd.read_csv("tmdb_5000_credits.csv")
    movies_df = movies_df.merge(credits_df, on='title')

    def get_top_cast(cast_json):
        try:
            cast = ast.literal_eval(cast_json)
            return ' '.join([actor['name'] for actor in cast[:3]])
        except:
            return ''

    def get_director(crew_json):
        try:
            crew = ast.literal_eval(crew_json)
            for person in crew:
                if person['job'] == 'Director':
                    return person['name']
            return ''
        except:
            return ''

    def get_genres(genres_json):
        try:
            genres = ast.literal_eval(genres_json)
            return ' '.join([g['name'] for g in genres])
        except:
            return ''

    movies_df['cast'] = movies_df['cast'].apply(get_top_cast)
    movies_df['director'] = movies_df['crew'].apply(get_director)
    movies_df['genres'] = movies_df['genres'].apply(get_genres)

    for col in ['overview', 'cast', 'director', 'genres']:
        movies_df[col] = movies_df[col].fillna('')

    movies_df['combined'] = (
        movies_df['title'] + ' ' +
        movies_df['overview'] + ' ' +
        movies_df['cast'] + ' ' +
        movies_df['director'] + ' ' +
        movies_df['genres']
    )

    return movies_df

@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def suggest_title(input_title, titles):
    matches = get_close_matches(input_title.lower().strip(), titles, n=1, cutoff=0.6)
    return matches[0] if matches else None

def recommend_multiple(movie_titles, num, min_rating, df, cosine_sim):
    indices = pd.Series(df.index, index=df['title'].str.lower().str.strip())
    movie_titles = [m.strip().lower() for m in movie_titles.split(',') if m.strip()]
    missing = []
    idxs = []

    for title in movie_titles:
        if title in indices:
            idxs.append(indices[title])
        else:
            suggestion = suggest_title(title, df['title'].str.lower().str.strip().tolist())
            missing.append((title, suggestion))

    if not idxs:
        return [], missing

    mean_vector = cosine_sim[idxs].mean(axis=0)
    sim_scores = list(enumerate(mean_vector))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in sim_scores:
        if i in idxs:
            continue
        row = df.iloc[i]
        if row['vote_average'] >= min_rating:
            recommendations.append({
                'title': row['title'],
                'rating': row['vote_average'],
                'genres': row['genres']
            })
            if len(recommendations) >= num:
                break

    return recommendations, missing

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

df = load_data()
cosine_sim = compute_similarity(df)

movie_input = st.text_input("Enter movie titles (comma separated):")
num_recs = st.slider("Number of recommendations", 1, 20, 5)
min_rating = st.slider("Minimum IMDb rating", 0.0, 10.0, 6.0, 0.1)

if st.button("Recommend"):
    if movie_input.strip() == "":
        st.warning("Please enter at least one movie title.")
    else:
        results, missing = recommend_multiple(movie_input, num_recs, min_rating, df, cosine_sim)

        if missing:
            for wrong, suggestion in missing:
                st.warning(f"Movie '{wrong}' not found. Did you mean: **{suggestion or 'unknown'}**?")

        if not results:
            st.info("No suitable recommendations found with that minimum rating.")
        else:
            st.subheader("Recommended Movies:")
            for r in results:
                poster_url = fetch_poster(r['title'])

                cols = st.columns([1, 4])
                with cols[0]:
                    if poster_url:
                        st.image(poster_url, width=120)
                    else:
                        st.text("No image")
                with cols[1]:
                    st.markdown(f"**🎥 {r['title']}**  \n⭐ {r['rating']}  \n🎭 {r['genres']}")
                    st.markdown("---")
