# Clustering
- Unsupervised maching learning
## Flat Clustering
- Given a number k, the algorithm attempts to group data points into 2 clusters
## Heirarchical Clustering
### K-means
- steps:
  1. Take data set and choose k cluster centers (centroids) randomly
     - Can take the first k values in the feature set (after shuffle)
  2. Calculate distance of feature set from each centroid using euclidian distance
  3. Take all feature sets and take the mean (find center of all of them)
  4. These means become the new centroids
  5. Repeat steps 2-4 until centoroids are no longer moving or moving very little
- Pros:
  - Easy to implement
- Cons:
  - K-means attempts to group everything into equally sized clusters
    - Think of a micky mouse shaped data set. The ears, despite being smaller, will result in clusters the same size as mickey's face
  - Scaling... every point must be compared to all other points
    - However, no training required after intitial train
