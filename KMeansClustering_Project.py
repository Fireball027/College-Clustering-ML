import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix


# Load Data
df = pd.read_csv('College_Data', index_col=0)
print(df.head())    # Show first few rows
print(df.info())    # Check data types and missing values
print(df.describe())    # Summary statistics


# Exploratory Data Analysis (EDA)
# Scatterplot: Room & Board vs Graduation Rate
sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', fit_reg=False, palette='coolwarm', height=6, aspect=1)
plt.title('Room & Board vs Graduation Rate')
plt.show()

# Scatterplot: Out-of-State Tuition vs Full-Time Undergrad Students
sns.lmplot(x='Outstate', y='F.Undergrad', data=df, hue='Private', fit_reg=False, palette='coolwarm', height=6, aspect=1)
plt.title('Out-of-State Tuition vs Full-Time Undergrad')
plt.show()

# Histogram: Out-of-State Tuition by Private/Public
g = sns.FacetGrid(df, hue='Private', palette='coolwarm', height=6, aspect=1.5)
g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)
g.add_legend()
plt.title('Distribution of Out-of-State Tuition')
plt.show()

# Histogram: Graduation Rate by Private/Public
g = sns.FacetGrid(df, hue='Private', palette='coolwarm', height=6, aspect=1.5)
g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
g.add_legend()
plt.title('Distribution of Graduation Rates')
plt.show()

# Fix Graduation Rate > 100%
print("\nColleges with Graduation Rate > 100%:")
print(df[df['Grad.Rate'] > 100])

# Fix incorrect Grad Rate
df.loc[df['Grad.Rate'] > 100, 'Grad.Rate'] = 100

# Confirm Fix
print("\nUpdated Graduation Rate:")
print(df[df['Grad.Rate'] > 100])


# K-Means Clustering
# Create and train KMeans model
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private', axis=1))

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)


# Model Evaluation
# Function to convert 'Private' column into numeric labels
def convertor(private):
    return 1 if private == 'Yes' else 0


# Create 'Cluster' column based on Private status
df['Cluster'] = df['Private'].apply(convertor)

# Show data with new column
print("\nUpdated DataFrame with Cluster Labels:")
print(df.head())

# Print confusion matrix & classification report
print("\nConfusion Matrix:")
print(confusion_matrix(df['Cluster'], kmeans.labels_))

print("\nClassification Report:")
print(classification_report(df['Cluster'], kmeans.labels_))
