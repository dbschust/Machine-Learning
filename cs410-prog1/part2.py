import numpy as np
import cv2
import matplotlib.pyplot as plt


DATASET = "510_cluster_dataset.txt"
IMG1 = "Kmean_img1.jpg"
IMG2 = "Kmean_img2.jpg"
MAX_ITERATIONS = 100
K = 4  #number of clusters
r = 10  #number of times to perform k-means (lowest error result of r runs gets displayed)


def main():
   #part i: dataset 
   X = load_dataset(DATASET)
   for k in 2,3,4:
      labels, means, error = perform_cluster(X, r, k)
      plot_results(X, k, labels, means, error)

   #part ii: images
   for img in IMG1, IMG2:
      X, h, w = load_image(img)
      for k in 5,10:
         labels, means, error = perform_cluster(X, 1, k)
         save_image(labels, means, h, w, img, k)


def save_image(labels, means, h, w, img, k):
   """
   function to reshape the image back to its original form and save it to a file
   """
   clustered_pixels = means[labels].astype(np.uint8)
   output_img = clustered_pixels.reshape((h, w, 3))
   output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
   cv2.imwrite(f"{img}_k={k}.jpg", output_bgr)


def load_image(filename):
   """
   function to load image from file, convert to RGB color scheme, and reshape it into a 2-d
   array for use in the K-means algorithm
   """
   img = cv2.imread(filename)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   h, w, c = img.shape
   reshaped = img.reshape((-1, c))
   return reshaped, h, w  


def perform_cluster(X, r, k):
   """
   function to perform the K-means clustering algorithm r times and return the results
   for the iteration with the lowest sum-squared error
   """
   min_error = float("inf")
   best_labels = []
   best_means = []
   for i in range(r):
      np.random.seed(i)
      labels, means = k_means(X, k)
      error = calc_sse(X, labels, means)
      if error < min_error:
         best_labels = labels
         best_means = means
         min_error = error
   return best_labels, best_means, min_error


def k_means(X, k):
   """
   function to run the K-means algorithm on data matrix X, with k centroids (means).
   initializes k random means from points in the dataset X, then runs until convergence
   or max iterations are reached
   """
   h,w = X.shape
   means = np.zeros(shape=(k,w))
   for i in range(k):
      means[i] = X[np.random.randint(0,h)]
   
   converged = False
   iter = 0
   while not converged and iter < MAX_ITERATIONS:
      #labels = calc_nearest_means(X, k, means)
      labels = calc_nearest_means_fast(X, means)
      means, converged = update_means(X, k, labels, means)
      iter += 1

   return labels, means


def update_means(X, k, labels, means):
   """
   function to update means to be the center of a cluster.  if means have not changed
   appreciably, sets converged to True to end the algorithm
   """
   converged = False
   new_means = np.zeros_like(means)
   for i in range(k):
      cluster = X[labels == i]
      new_means[i] = np.mean(cluster, axis=0)
   if np.allclose(means, new_means, atol=1e-6):
      converged = True 

   return new_means, converged


def calc_nearest_means_fast(X, means):
   """
   fast version of calculating nearest means for data points using numpy broadcasting.
   needed for part ii for running the algorithm on the images
   """
   distances = np.sum((X[:, np.newaxis, :] - means[np.newaxis, :, :]) ** 2, axis=2)
   labels = np.argmin(distances, axis=1)
   return labels


def calc_nearest_means(X, k, means):
   """
   my original slower function of calculating nearest means for data points.
   effective for the part i dataset, but not efficient enough for use on higher resolution
   images in part ii
   """
   labels = []
   for x in X:
      min_dist = float("inf")
      closest_mean = -1

      for i in range(k):
         mean = means[i]
         #dist_x = x[0] - mean[0]
         #dist_y = x[1] - mean[1]
         #distance_sq = dist_x**2 + dist_y**2
         distance_sq = np.sum((x - mean)**2) #updated version to improve runtime

         if distance_sq < min_dist:
            min_dist = distance_sq
            closest_mean = i

      labels.append(closest_mean)
   return np.array(labels)


def calc_sse(X, labels, means):
   """
   function to calculate the total sum-squared error for current clusters
   """
   ss_error = 0.0
   for i in range(len(X)):
      cluster = labels[i]
      dist = X[i] - means[cluster]
      ss_error += np.dot(dist, dist)
   return ss_error 


def plot_results(X, k, labels, means, min_error):
   """
   function to plot the results of K-means on a 2-d graph
   """
   colors = plt.cm.get_cmap("tab10", k)

   for i in range(k):
      cluster_points = X[labels == i]
      plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                  s=10, color=colors(i), label=f"Cluster {i}")

   plt.scatter(means[:, 0], means[:, 1], 
               c='black', s=150, marker='X', label='Centroids')

   plt.title(f"K-Means Clustering Result, sum-squared error={min_error:.2f}")
   plt.xlabel("Feature 1")
   plt.ylabel("Feature 2")
   plt.legend()
   plt.grid(True)
   plt.show()


def load_dataset(filename):
   """
   load a dataset as a numpy array
   """
   return np.loadtxt(filename)


if __name__ == "__main__":
   main()