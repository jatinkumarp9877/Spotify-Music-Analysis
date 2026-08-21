## 1. Project Overview

This project analyzes Spotify music data using Python and Data Science techniques.

The main goal is to understand music listening trends and investigate which audio characteristics are associated with song popularity.

The project also uses Machine Learning models to predict song popularity.

---

## 2. Objectives

The main objectives of this project are:

- Analyze the Spotify dataset.
- Explore music characteristics such as energy, danceability and acousticness.
- Identify frequently appearing artists.
- Study trends in song popularity over time.
- Analyze relationships between audio features and popularity.
- Build Machine Learning models to predict song popularity.
- Compare Linear Regression and Random Forest models.

---

## 3. Dataset

The dataset contains approximately 170,653 songs.

Important columns include:

- `name` — Song name
- `artists` — Artist(s)
- `year` — Release year
- `popularity` — Spotify popularity score
- `danceability` — How suitable a track is for dancing
- `energy` — Intensity of the track
- `acousticness` — Acoustic characteristics
- `valence` — Musical positivity
- `tempo` — Beats per minute
- `loudness` — Overall loudness
- `speechiness` — Presence of spoken words
- `instrumentalness` — Likelihood that a track contains no vocals
- `liveness` — Presence of a live audience

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 5. Data Analysis

The project includes:

- Data inspection
- Missing-value analysis
- Duplicate-value analysis
- Artist analysis
- Popularity analysis
- Year-wise analysis
- Decade-wise analysis
- Song duration analysis
- Correlation analysis
- Audio feature analysis

---

## 6. Machine Learning

Two regression models were tested:

### Linear Regression

R² Score:

0.444

### Random Forest Regression

R² Score:

0.583

Random Forest performed better than Linear Regression.

---

## 7. Important Finding

The Random Forest model identified acousticness as the most important feature among the selected audio features.

Other important features included:

- Speechiness
- Loudness
- Valence
- Danceability

---

## 8. Conclusion

The analysis shows that different audio characteristics have different relationships with song popularity.

The Random Forest model performed better than Linear Regression for predicting popularity using the selected audio features.

However, popularity is influenced by many factors beyond audio characteristics, so the model cannot perfectly predict popularity.

---

## 9. Project Graphs

Important visualizations are stored in the `graphs` folder:

- Correlation with popularity
- Feature importance
- Actual vs predicted popularity
- Machine Learning model comparison

- 👨‍💻 Created By
JATIN KUMAR

Spotify Music Analysis Project

This project was created as a Data Science project using Python,
Data Analysis, Data Visualization, and Machine Learning.
