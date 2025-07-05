🎮 Movie Recommender App

This is a content-based movie recommendation system built using Python, Streamlit, and TMDB API. It recommends movies based on the title(s) you provide, analyzing plot summaries, cast, crew, and genres. It also displays live posters fetched from TMDB.

🔧 Features

✅ Multi-movie similarity-based recommendations

✅ Filter by minimum IMDb rating

✅ Real-time poster fetching using TMDB API

✅ Interactive UI with Streamlit

✅ Error handling with fuzzy title matching

🚀 Demo

Live on Streamlit Cloud (replace with actual URL after deployment)

📂 File Structure

movie-recommender/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
├── tmdb_5000_movies.csv    # Movie metadata
├── tmdb_5000_credits.csv   # Credits metadata (cast & crew)

📅 Datasets

tmdb_5000_movies.csv from Kaggle

tmdb_5000_credits.csv (included in the same dataset)

Place these files in the project directory.

🛠️ Setup Instructions

1. Clone the Repository

git clone https://github.com/Dataarnoor/movie-recommender.git
cd movie-recommender

2. Install Dependencies

pip install -r requirements.txt

3. Get Your TMDB API Key

Sign up at TMDB

Get your v3 API Key

For local testing, add this line to app.py:

TMDB_API_KEY = "your_api_key_here"

For deployment, use Streamlit Secrets (see below)

4. Run the App Locally

streamlit run app.py

Then go to http://localhost:8501

🚩 Deployment (Streamlit Cloud)

Push the project to a GitHub repo

Go to Streamlit Cloud and log in with GitHub

Click "New App" and select the repo

Add secrets under Advanced Settings:

TMDB_API_KEY = "your_api_key_here"

In app.py, access it with:

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

Deploy and share your app link!

📄 requirements.txt

streamlit
pandas
scikit-learn
requests

📚 Example Usage

Enter one or more movie titles, comma separated:

Inception, Interstellar

Apply a minimum IMDb rating filter (e.g., 7.0) and get recommendations with posters and genre info.

🛡️ License

MIT License. See LICENSE file.

