from sklearn.cluster import KMeans
import numpy as np
# Define the training vectors
X = np.array([[4, 3],
              [4, 4],
              [4, 5],
              [0, 4],
              [8, 0],
              [3, 4],
              [5, 4]])

# Define the initial cluster centers
initial_centers = np.array([[4, 3],  # μ1 = x1
                            [0, 4],  # μ2 = x4
                            [8, 0]]) # μ3 = x5

# Create a KMeans instance with 3 clusters, using the initial centers provided
kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, max_iter=500)

# Fit the KMeans algorithm to the data
kmeans.fit(X)


# The number of iterations required for convergence
print(kmeans.n_iter_)

a = np.array([np.sqrt(1),
          np.sqrt(4),
          np.sqrt(4),
          np.sqrt(6),
          np.sqrt(12),
          np.sqrt(6),
          np.sqrt(4),
          np.sqrt(12),
          np.sqrt(12),
          np.sqrt(4),
          np.sqrt(1),
          np.sqrt(4),
          np.sqrt(6),
          np.sqrt(4),
          np.sqrt(1)])
b = np.array([np.sqrt(1),
          np.sqrt(4),
          -np.sqrt(4),
          np.sqrt(6),
          -np.sqrt(12),
          np.sqrt(6),
          np.sqrt(4),
          -np.sqrt(12),
          np.sqrt(12),
          -np.sqrt(4),
          np.sqrt(1),
          -np.sqrt(4),
          np.sqrt(6),
          -np.sqrt(4),
          np.sqrt(1)])
print(np.linalg.norm(a-b))

# # [1,2,2,6,12,6,2,12,12,2,1,2,6,2,1]
# # [1,2,2,6,12,6,2,12,12,2,1,2,6,2,1]


# def sqrt_corrected_fourth_order_polynomial_transform(x):
#     # Including the square-rooted binomial coefficients for each term
#     return np.array([
#         1,
#         2*x[0],
#         2*x[1],
#         np.sqrt(6)*x[0]**2,
#         np.sqrt(12)*x[0]*x[1],
#         np.sqrt(6)*x[1]**2,
#         2*x[0]**3,
#         np.sqrt(12)*x[0]**2*x[1],
#         np.sqrt(12)*x[0]*x[1]**2,
#         2*x[1]**3,
#         x[0]**4,
#         2*x[0]**3*x[1],
#         np.sqrt(6)*x[0]**2*x[1]**2,
#         2*x[0]*x[1]**3,
#         x[1]**4
#     ])

# # Apply the square-rooted corrected transformation to each XOR vector
# sqrt_corrected_representative_vectors = np.array([
#     sqrt_corrected_fourth_order_polynomial_transform(x) for x in xor_vectors
# ])

# # Recalculate pairwise distances with the square-rooted corrected representative vectors
# sqrt_corrected_pairwise_distances = np.array([
#     [np.linalg.norm(a-b) for b in sqrt_corrected_representative_vectors] for a in sqrt_corrected_representative_vectors
# ])

# sqrt_corrected_representative_vectors, sqrt_corrected_pairwise_distances