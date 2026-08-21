# Spotify Listening Trend Analysis

## 1. Introduction

Music streaming platforms generate large amounts of data about songs,
artists, and audio characteristics.

This project analyzes Spotify music data using Python and Data Science
techniques to understand music trends and investigate factors related to
song popularity.

---

## 2. Problem Statement

The objective of this project is to analyze Spotify music data and
understand how different characteristics of songs are related to their
popularity.

The project also attempts to predict song popularity using Machine
Learning techniques.

---

## 3. Objectives

The main objectives are:

- Analyze the Spotify dataset.
- Clean and inspect the data.
- Analyze song popularity.
- Study music trends over time.
- Analyze artists and songs.
- Study audio features such as danceability, energy, valence,
  and acousticness.
- Build Machine Learning models.
- Compare Linear Regression and Random Forest.
- Identify important features for predicting popularity.

---

## 4. Dataset

The dataset contains Spotify song information including:

- Song name
- Artists
- Release year
- Popularity
- Danceability
- Energy
- Acousticness
- Valence
- Tempo
- Loudness
- Speechiness
- Instrumentalness
- Liveness

The dataset contains approximately 170,653 songs.

---

## 5. Tools and Technologies

The project was developed using Python.

Libraries used:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 6. Data Analysis

The following analyses were performed:

- Dataset inspection
- Missing-value analysis
- Duplicate-value analysis
- Artist analysis
- Popularity analysis
- Year-wise analysis
- Decade-wise analysis
- Song duration analysis
- Audio feature analysis
- Correlation analysis

---

## 7. Machine Learning

Two Machine Learning regression models were developed:

### Linear Regression

R² Score: 0.444

### Random Forest Regression

R² Score: 0.583

Random Forest performed better than Linear Regression.

---

## 8. Feature Importance

The Random Forest model identified the following important features:

1. Acousticness
2. Speechiness
3. Loudness
4. Valence
5. Danceability

Acousticness was the most important feature in the model.

---

## 9. Results

The Random Forest model achieved:

- MAE: approximately 10.72
- MSE: approximately 199.72
- R² Score: approximately 0.583

The Linear Regression model achieved:

- MAE: approximately 13.11
- MSE: approximately 265.95
- R² Score: approximately 0.444

Therefore, Random Forest performed better.

---

## 10. Conclusion

The project shows that audio characteristics can be used to predict
song popularity to some extent.

Random Forest performed better than Linear Regression for this dataset.

However, audio features alone cannot explain all aspects of song
popularity because popularity can also depend on factors such as
artist popularity, marketing, social trends, playlists, and listener
behavior.

---

## 11. Future Scope

Future improvements could include:

- Using more recent Spotify data.
- Including genre information.
- Including artist popularity.
- Using playlist information.
- Testing additional Machine Learning algorithms.
- Hyperparameter tuning.
- Building a recommendation system.
- Creating an interactive dashboard.
