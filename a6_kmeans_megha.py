
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(1)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load data
# 
# Load data and preprocess.


X_all = np.load(open('X_train.npy', 'rb'))

X1 = X_all[:163, :]
X2 = X_all[163:327, :]
X3 = X_all[327:491, :]

X = np.concatenate((X1, X2, X3), axis=0)
X = np.transpose(X, (1,2,3,0)).reshape(-1, X.shape[0])
print('Shape of X:', X.shape)


# ---
# ## Analysis

pca = PCA(n_components=2)
pca.fit(X.T)
X_proj = pca.transform(X.T)

# Plot
fig = plt.figure()
plt.scatter(X_proj[:,0], X_proj[:,1])
plt.show()



# Initialize centroids
def init_centroids(X, k):
    """
    Args:
    X -- data, shape (n, m)
    k -- number of clusters
    
    Return:
    centroids -- k randomly picked data points as initial centroids, shape (n, k)
    """
    assert(k > 1)
    np.random.seed(1)
    
    centroids = X[:, np.random.choice(X.shape[1],k,replace=False)]
    
    return centroids


# Compute distances
def compute_distances(X, centroids):
    """
    Args:
    X -- data, shape (n, m)
    centroids -- shape (n, k)
    
    Return:
    distances -- shape (k, m)
    """
    
    centroids_expanded = np.expand_dims(centroids, axis=1)
    S = X.T - centroids_expanded.T
    distances = np.sqrt(np.sum(S**2, axis=2))
    
    return distances


# Find the closest centroid for each data point
def cloeset_centroid(distances):
    """
    Args:
    distances -- numpy array of shape (k, m), output of compute_distances()
    
    Return:
    indices -- numpy array of shape (1, m)
    """
    indices = np.argmin(distances, axis=0)
    
    return indices


# Update centroids
def update_centroids(X, closest_indices, centroids):
    """
    Args:
    X -- data, shape (n, m)
    cloesest_indices -- output of closest_centroid()
    centroids -- old centroids positions
    
    Return:
    new_centroids -- new centroids positions, shape (n, k)
    """
    new_centroids = np.zeros_like(centroids)
    for j in range(centroids.shape[1]):
        points = X[:, closest_indices==j]
        new_centroids[:, j] = np.mean(points, axis=1)
    
    
    assert(centroids.shape == new_centroids.shape)
    
    return new_centroids


# K-means
def kmeans(X, k):
    """
    Args:
    X -- data, shape (n, m)
    k -- number of clusters
    
    Return:
    closest_indices -- final assignment of clusters to each data point, shape (1, m)
    centroids -- final positions of centroids
    """
    centroids = init_centroids(X, k)
    
    old_centroids = None
    while not np.array_equal(old_centroids, centroids):
        # Backup centroids
        old_centroids = np.copy(centroids)
        
        # Compute distances
        distances = compute_distances(X, old_centroids)
        
        # Find cloeset centroid
        closest_indices = cloeset_centroid(distances)
        
        # Update centroids
        centroids = update_centroids(X, closest_indices, centroids)
    
    return closest_indices, centroids


# Evaluate Task 5
closest_indices, centroids = kmeans(X, 3)

print('closest_indices[:10]', closest_indices[:10])
print('closest_indices[70:80]', closest_indices[70:80])
print('closest_indices[140:150]', closest_indices[140:150])
print('closest_indices[210:220]', closest_indices[210:220])

# ------------------------------------

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X.T)
X_proj = pca.transform(X.T)

fig = plt.figure(figsize=(12, 3.5))

closest_indices, centroids = kmeans(X, 3)
fig.add_subplot(1, 2, 1)
plt.scatter(X_proj[closest_indices==0, 0], X_proj[closest_indices==0, 1])
plt.scatter(X_proj[closest_indices==1, 0], X_proj[closest_indices==1, 1])
plt.scatter(X_proj[closest_indices==2, 0], X_proj[closest_indices==2, 1])
plt.title('Clustering result of k=3')

closest_indices, centroids = kmeans(X, 4)
fig.add_subplot(1, 2, 2)
plt.scatter(X_proj[closest_indices==0, 0], X_proj[closest_indices==0, 1])
plt.scatter(X_proj[closest_indices==1, 0], X_proj[closest_indices==1, 1])
plt.scatter(X_proj[closest_indices==2, 0], X_proj[closest_indices==2, 1])
plt.scatter(X_proj[closest_indices==3, 0], X_proj[closest_indices==3, 1])
plt.title('Clustering result of k=4')

plt.show()



