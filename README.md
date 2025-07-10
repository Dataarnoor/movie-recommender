# 🎬 Movie Recommender App

This is a content-based movie recommendation system built using Python, Streamlit, and TMDB API. It recommends movies based on the title(s) you provide, analyzing plot summaries, cast, crew, and genres using both lexical (TF-IDF) and semantic (Sentence-BERT) similarity. It also displays live posters fetched from TMDB.

## 🔍 What's New (NLP Upgrade)

- Semantic Search with Sentence-BERT: Goes beyond keywords by using contextual embeddings of movie plot summaries.
- Multi-Movie Matching: Input multiple movie titles to get thematically blended recommendations.
- Improved Robustness: Fuzzy title matching and API retry logic for better UX.

## 🔧 Features

- Multi-movie similarity-based recommendations using TF-IDF and SBERT
- Semantic search on plot summaries using Sentence-BERT
- Filter recommendations by minimum IMDb rating
- Real-time movie poster fetching via TMDB API
- Interactive, clean UI using Streamlit
- Robust fuzzy matching and retry logic

## 🔗 Demo

GitHub Repository: https://github.com/Dataarnoor/movie-recommender

## 📁 File Structure

movie-recommender/
- app.py                  # Main Streamlit app
- utils.py                # Core logic and helper functions
- requirements.txt        # Python dependencies
- runtime.txt             # Python version for deployment
- README.md               # Project documentation
- tmdb_5000_movies.csv    # Movie metadata (from Kaggle)
- tmdb_5000_credits.csv   # Credits metadata (cast & crew)

## 📅 Datasets Used

- tmdb_5000_movies.csv
- tmdb_5000_credits.csv

These files are from the TMDB Movie Dataset on Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata  
Place them in the root folder.

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Dataarnoor/movie-recommender.git
cd movie-recommender
```
### 2. Install Dependencies
```bash
Copy
Edit
pip install -r requirements.txt
Note: If sentencepiece or sentence-transformers fail on certain platforms, use Python 3.10 or install CMake and pkg-config.
```
### 3. Get Your TMDB API Key
```bash
Sign up at https://www.themoviedb.org/
Generate a v3 API key
For local setup, add the following to app.py:
Copy
Edit
TMDB_API_KEY = "your_api_key_here"
```
### ▶️ Running the App
```bash
streamlit run app.py
Then open http://localhost:8501 in your browser.
```

### 📚 Example Usage
- Enter one or more movie titles (comma-separated): Inception, Interstellar

- Apply a minimum IMDb rating filter (e.g., 7.5) and get intelligent recommendations with real-time posters.

### 🛡️ License
MIT License. See LICENSE file.
