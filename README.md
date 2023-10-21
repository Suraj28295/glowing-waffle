# glowing-waffle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Your data
data = {
    'Questions': ['question 1', 'question 2', 'question 3', 'question 4'],
    'Weights': [0.25, 0.1, 0.2, 0.45],
    'C1': [1, 4,8,11],
    'C2': [2,4,7,10],
    'C3': [2, 5,8,12],
    'C4': [2, 4,9,11],
    'C5': [1,6,7,10],
    'C6': [3, 5, 8,12],
    'C7': [3, 6, 7, 12],
    'C8': [3, 4, 9,11]
}

df = pd.DataFrame(data)

# Drop the questions and weights columns
df_customers = df.drop(['Questions', 'Weights'], axis=1)

# Transpose the dataframe to have one row per customer
df_customers = df_customers.T

# Convert the dataframe to a numpy array
X = df_customers.to_numpy()

# Apply k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Get the cluster labels for each customer
labels = kmeans.labels_

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Calculate the scores for each customer based on the weights and centroids
scores = []
weights = np.array(data['Weights'])
for i in range(len(labels)):
    # Get the centroid of the cluster that the customer belongs to
    centroid = centroids[labels[i]]
    # Calculate the score as the weighted sum of the absolute differences between the customer's answers and the centroid
    score = np.sum(weights * np.abs(X[i] - centroid))
    # Append the score to the list
    scores.append(score)

# Sort the customers by their scores in descending order
sorted_customers = sorted(zip(df_customers.index, scores), key=lambda x: x[1], reverse=True)

# Get the top three customers
top_customers = [x[0] for x in sorted_customers[:3]]

print(top_customers)

