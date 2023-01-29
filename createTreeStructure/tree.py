from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Extract SIFT descriptors for all images
all_descriptors = []
for img in images:
    kps, descs = sift.detectAndCompute(img, None)
    all_descriptors.append(descs)

# Stack all descriptors vertically
all_descriptors = np.vstack(all_descriptors)

# Perform clustering
agg_clustering = AgglomerativeClustering(n_clusters=10)
agg_clustering.fit(all_descriptors)

# Create dendrogram
Z = linkage(all_descriptors, 'ward')
dendrogram(Z)