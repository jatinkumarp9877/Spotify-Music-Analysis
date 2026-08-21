import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Spotify Music Analysis",
    page_icon="🎵",
    layout="wide"
)

# Load dataset
df = pd.read_csv("spotify.csv")

# Title
st.title("🎵 Spotify Music Analysis Dashboard")
st.write("Exploring Spotify songs, artists, audio features and popularity.")

# Sidebar
st.sidebar.header("Dashboard Filters")

# Year filter
min_year = int(df["year"].min())
max_year = int(df["year"].max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_year,
    max_year,
    (min_year, max_year)
)

filtered_df = df[
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
]

# Main statistics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Songs",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Average Popularity",
    f"{filtered_df['popularity'].mean():.2f}"
)

col3.metric(
    "Average Energy",
    f"{filtered_df['energy'].mean():.2f}"
)

col4.metric(
    "Average Danceability",
    f"{filtered_df['danceability'].mean():.2f}"
)

st.divider()

# Popularity distribution
st.subheader("Popularity Distribution")

fig, ax = plt.subplots()

ax.hist(filtered_df["popularity"], bins=20)

ax.set_xlabel("Popularity")
ax.set_ylabel("Number of Songs")
ax.set_title("Distribution of Song Popularity")

st.pyplot(fig)

# Top songs
st.subheader("Top 10 Most Popular Songs")

top_songs = (
    filtered_df
    .sort_values("popularity", ascending=False)
    [["name", "artists", "popularity"]]
    .head(10)
)

st.dataframe(
    top_songs,
    use_container_width=True
)

# Audio features
st.subheader("Audio Features")

features = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness"
]

feature_means = filtered_df[features].mean()

fig2, ax2 = plt.subplots()

feature_means.sort_values().plot(
    kind="barh",
    ax=ax2
)

ax2.set_xlabel("Average Value")
ax2.set_title("Average Audio Features")

st.pyplot(fig2)

# Correlation
st.subheader("Correlation with Popularity")

correlations = (
    filtered_df[
        features + ["popularity"]
    ]
    .corr()["popularity"]
    .drop("popularity")
    .sort_values(ascending=False)
)

st.bar_chart(correlations)

# Footer
st.divider()

st.write(
    "Spotify Music Analysis Project is created by JATIN."
)