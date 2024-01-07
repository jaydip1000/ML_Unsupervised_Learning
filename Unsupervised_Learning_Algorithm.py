import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_species(dataset):
    # Remove the "Species" column and store it separately
    species = dataset["Species"]
    dataset = dataset.drop("Species", axis=1)
    return dataset, species

def k_means_clustering(dataset, k=3, max_iters=100, plot=True):
    # Initialize centroids randomly
    centroids = dataset.sample(k).to_numpy()

    for _ in range(max_iters):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(dataset.to_numpy()[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([dataset[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Visualize the clustering results
    if plot:
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=labels, cmap='viridis', edgecolors='k', marker='o', label='Clustered')
        plt.title('K-Means Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

    return labels

def principal_component_analysis(dataset, plot=True):
    # Center the data
    centered_data = dataset - dataset.mean()

    # Compute covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Project data onto the first three principal components
    pca_result = centered_data.dot(eigenvectors[:, :3])

    # Visualize the PCA results
    if plot:
        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pca_result.iloc[:, 0], pca_result.iloc[:, 1], pca_result.iloc[:, 2], c='blue', marker='o', label='Projected Data')
        ax1.set_title('PCA - First 3 Principal Components')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_zlabel('Principal Component 3')
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.bar(range(1, len(eigenvalues) + 1), eigenvalues)
        ax2.set_title('Eigenvalues')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Eigenvalue')

        plt.tight_layout()
        plt.show()

    return pca_result

'''# Example usage:
# Assuming you have a dataset stored in a Pandas DataFrame called 'df'
# and the last column "Species" should be removed for both algorithms

# Example:
# df = pd.read_csv("your_dataset.csv")
# features, species = remove_species(df)

# Perform K-Means Clustering
k_means_labels = k_means_clustering(features)

# Perform Principal Component Analysis
pca_result = principal_component_analysis(features)'''

