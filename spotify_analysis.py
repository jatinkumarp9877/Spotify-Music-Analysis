# ============================================================
# Spotify Listening Trend Analysis
# ============================================================

# This project analyzes Spotify music data using Python.
# It explores music trends and predicts song popularity
# using Machine Learning.










import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("graphs", exist_ok=True)

print("Spotify Music Analysis Project Started!")

# Load the dataset
df = pd.read_csv("spotify.csv")

# Display first 5 rows
print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# Step 3: Data Cleaning

print(df.isnull().sum())
print(df.duplicated().sum())

# Step 4: Check unique values

print("\nNumber of unique values in each column:")
print(df.nunique())
print("\nYear range:")
print(df["year"].min(), "to", df["year"].max())
print("\nTop 10 Most Popular Songs:")
print(df.nlargest(10, "popularity")[["name", "artists", "popularity"]])

# Step 5: Most Popular Artists

artist_popularity = (
    df.groupby("artists")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Artists by Average Popularity:")
print(artist_popularity)

# Step 6: Visualize Top Artists

plt.figure(figsize=(10, 6))

artist_popularity.sort_values().plot(kind="barh")

plt.title("Top 10 Artists by Average Popularity")
plt.xlabel("Average Popularity")
plt.ylabel("Artists")

plt.tight_layout()
plt.show()

# Step 7: Most Frequently Appearing Artists

import ast

df["artists_list"] = df["artists"].apply(ast.literal_eval)

artists_exploded = df.explode("artists_list")

top_artists = artists_exploded["artists_list"].value_counts().head(10)

print("\nTop 10 Most Frequently Appearing Artists:")
print(top_artists)

# Step 8: Visualize Most Frequently Appearing Artists

plt.figure(figsize=(10, 6))

top_artists.sort_values().plot(kind="barh")

plt.title("Top 10 Most Frequently Appearing Artists")
plt.xlabel("Number of Songs")
plt.ylabel("Artists")

plt.tight_layout()
plt.show()
# Step 9: Danceability vs Popularity

plt.figure(figsize=(10, 6))

plt.scatter(
    df["danceability"],
    df["popularity"],
    alpha=0.3
)

plt.title("Danceability vs Popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity")

plt.tight_layout()
plt.show()

# Step 10: Energy vs Popularity

plt.figure(figsize=(10, 6))

plt.scatter(
    df["energy"],
    df["popularity"],
    alpha=0.3
)

plt.title("Energy vs Popularity")
plt.xlabel("Energy")
plt.ylabel("Popularity")

plt.tight_layout()
plt.show()

# Step 11: Correlation with Popularity

features = [
    "valence",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo"
]

correlations = df[features + ["popularity"]].corr()["popularity"].sort_values(
    ascending=False
)

print("\nCorrelation of Audio Features with Popularity:")
print(correlations)
# Step 9: Correlation of Audio Features with Popularity

correlation = df[
    ["energy", "loudness", "danceability", "tempo",
     "valence", "liveness", "speechiness",
     "instrumentalness", "acousticness", "popularity"]
].corr()["popularity"].sort_values(ascending=False)

print("\nCorrelation of Audio Features with Popularity:")
print(correlation)

# Remove popularity itself for the graph
correlation_features = correlation.drop("popularity")

# Create bar chart
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
correlation_features.plot(kind="bar")

plt.title("Correlation of Audio Features with Popularity")
plt.xlabel("Audio Features")
plt.ylabel("Correlation with Popularity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphs/correlation_popularity.png", dpi=300)
plt.show()

# Step 10: Average Popularity by Year

yearly_popularity = (
    df.groupby("year")["popularity"]
    .mean()
    .sort_index()
)

print("\nAverage Popularity by Year:")
print(yearly_popularity)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(yearly_popularity.index, yearly_popularity.values)

plt.title("Average Song Popularity by Year")
plt.xlabel("Year")
plt.ylabel("Average Popularity")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Most Popular Song in Each Year

most_popular_by_year = df.loc[
    df.groupby("year")["popularity"].idxmax()
]

print("\nMost Popular Song in Each Year:")
print(
    most_popular_by_year[
        ["year", "name", "artists", "popularity"]
    ].sort_values("year")
)
# Step 12: Top 10 Artists by Number of Songs

top_artists = (
    artists_exploded["artists_list"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Artists by Number of Songs:")
print(top_artists)

# Plot Top 10 Artists

plt.figure(figsize=(10, 6))

top_artists.sort_values().plot(kind="barh")

plt.title("Top 10 Artists by Number of Songs")
plt.xlabel("Number of Songs")
plt.ylabel("Artist")

plt.tight_layout()
plt.show()

# Step 13: Average Popularity by Decade

df["decade"] = (df["year"] // 10) * 10

decade_popularity = (
    df.groupby("decade")["popularity"]
    .mean()
    .sort_index()
)

print("\nAverage Popularity by Decade:")
print(decade_popularity)

# Plot Average Popularity by Decade

plt.figure(figsize=(10, 6))

decade_popularity.plot(kind="bar")

plt.title("Average Song Popularity by Decade")
plt.xlabel("Decade")
plt.ylabel("Average Popularity")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Step 14: Music Features by Decade

decade_features = df.groupby("decade")[
    ["danceability", "energy", "valence", "acousticness"]
].mean()

print("\nAverage Music Features by Decade:")
print(decade_features)

#danceability
plt.figure(figsize=(10, 6))

plt.plot(decade_features.index, decade_features["danceability"], marker="o")

plt.title("Danceability Trend by Decade")
plt.xlabel("Decade")
plt.ylabel("Average Danceability")

plt.grid(True)
plt.tight_layout()
plt.show()

#energy
plt.figure(figsize=(10, 6))

plt.plot(decade_features.index, decade_features["energy"], marker="o")

plt.title("Energy Trend by Decade")
plt.xlabel("Decade")
plt.ylabel("Average Energy")

plt.grid(True)
plt.tight_layout()
plt.show()

#valence
plt.figure(figsize=(10, 6))

plt.plot(decade_features.index, decade_features["valence"], marker="o")

plt.title("Music Mood (Valence) Trend by Decade")
plt.xlabel("Decade")
plt.ylabel("Average Valence")

plt.grid(True)
plt.tight_layout()
plt.show()

#acousticness

plt.figure(figsize=(10, 6))

plt.plot(decade_features.index, decade_features["acousticness"], marker="o")

plt.title("Acousticness Trend by Decade")
plt.xlabel("Decade")
plt.ylabel("Average Acousticness")

plt.grid(True)
plt.tight_layout()
plt.show()

# Step 15: Song Duration Analysis

# Convert milliseconds to minutes
df["duration_minutes"] = df["duration_ms"] / 60000

print("\nAverage Song Duration:")
print(df["duration_minutes"].mean())

print("\nShortest Song:")
print(df["duration_minutes"].min())

print("\nLongest Song:")
print(df["duration_minutes"].max())

# Song Duration Distribution

plt.figure(figsize=(10, 6))

plt.hist(df["duration_minutes"], bins=50)

plt.title("Distribution of Song Duration")
plt.xlabel("Song Duration (minutes)")
plt.ylabel("Number of Songs")

plt.tight_layout()
plt.show()

# Step 16: Song Duration vs Popularity

plt.figure(figsize=(10, 6))

plt.scatter(
    df["duration_minutes"],
    df["popularity"],
    alpha=0.3
)

plt.title("Song Duration vs Popularity")
plt.xlabel("Song Duration (minutes)")
plt.ylabel("Popularity")

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

X = df[features]
y = df["popularity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData preprocessing for Machine Learning completed!")
# Step 18: Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("\nLinear Regression Model trained successfully!")
# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)
# Evaluate Linear Regression Model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Step 19: Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

# Create the model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Model trained successfully!")

# Evaluate Random Forest

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Performance:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)

# Step 20: Feature Importance

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Feature Importance Graph

plt.figure(figsize=(10, 6))

feature_importance.sort_values().plot(kind="barh")

plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Audio Feature")

plt.tight_layout()
plt.savefig("graphs/feature_importance.png", dpi=300)
plt.show()

# Step 21: Actual vs Predicted Popularity

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    rf_pred,
    alpha=0.3
)

plt.title("Actual vs Predicted Song Popularity")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")

plt.tight_layout()
plt.savefig("graphs/actual_vs_predicted.png", dpi=300)
plt.show()

# Step 22: Compare Machine Learning Models

models = ["Linear Regression", "Random Forest"]
r2_scores = [r2, rf_r2]

plt.figure(figsize=(8, 6))

plt.bar(models, r2_scores)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("R² Score")

plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("graphs/model_comparison.png", dpi=300)
plt.show()



# Step 23: Final Project Summary

print("\n" + "=" * 50)
print("       SPOTIFY LISTENING TREND ANALYSIS")
print("=" * 50)

print("\nDataset:")
print("Total Songs:", len(df))
print("Total Columns:", len(df.columns))
print("Year Range:", df["year"].min(), "-", df["year"].max())

print("\nTop 10 Most Frequently Appearing Artists:")
print(top_artists)

print("\nBest Machine Learning Model:")
print("Linear Regression R2:", round(r2, 3))
print("Random Forest R2:", round(rf_r2, 3))

print("\nMost Important Audio Features:")
print(feature_importance.head(5))

print("\nFinal Conclusion:")
print(
    "Random Forest performed better than Linear Regression "
    "for predicting song popularity."
)

# Step 27: Final Model Comparison

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        mae,
        rf_mae
    ],
    "MSE": [
        mse,
        rf_mse
    ],
    "R2 Score": [
        r2,
        rf_r2
    ]
})

print("\nFinal Model Comparison:")
print(results)
# ============================================================
# Step 28: Final Project Conclusion
# ============================================================

print("\n" + "=" * 60)
print("              FINAL PROJECT CONCLUSION")
print("=" * 60)

print("\n1. Dataset Analysis")
print("Total number of songs:", len(df))
print("Year range:", df["year"].min(), "-", df["year"].max())

print("\n2. Machine Learning Results")
print("Linear Regression R2 Score:", round(r2, 3))
print("Random Forest R2 Score:", round(rf_r2, 3))

print("\n3. Best Model")

if rf_r2 > r2:
    print("Random Forest performed better than Linear Regression.")
else:
    print("Linear Regression performed better than Random Forest.")

print("\n4. Most Important Audio Feature")
print("Most important feature:",
      feature_importance.index[0])

print("Importance:",
      round(feature_importance.iloc[0], 3))

print("\n5. Overall Conclusion")
print(
    "Audio features can be used to predict song popularity, "
    "although they cannot explain all factors that influence popularity."
)

print("=" * 60)