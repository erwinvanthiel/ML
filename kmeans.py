import matplotlib.pyplot as plt
import numpy as np
import torch
np.random.seed(111)

def euclidean_distance(p1,p2):
	return np.linalg.norm(p1 - p2)

def kmeans_cluster(k, datax, datay):
	# initialize
	num_samples = datax.shape[0]
	indices = [np.random.randint(0, num_samples) for _ in range(k)]
	meansx = datasetx[indices]
	meansy = datasety[indices]
	clusters = np.zeros(num_samples, dtype=int)
	change = True

	# optimization loop
	while change:
		plot_clusters(datax,datay,clusters,k)
		change = False
		# cluster loop
		for i in range(num_samples):
			# for each sample determine cluster
			distances = []
			for j in range(k):
				distances.append(euclidean_distance(np.array([datax[i], datay[i]]), np.array([meansx[j], meansy[j]])))

			new_cluster = distances.index(min(distances))
			if clusters[i] != new_cluster:
				change = True
			clusters[i] = new_cluster

		# update cluster means
		for j in range(k):
			meansx[j] = np.average(datasetx[np.where(clusters == j)])
			meansy[j] = np.average(datasety[np.where(clusters == j)])

		print(calculate_cluster_score(datasetx, datasety, meansx, meansy, clusters))

	return clusters

def plot_clusters(datasetx, datasety, clusters, k):
	for i in range(k):
		groupx = datasetx[np.where(clusters == i)]
		groupy = datasety[np.where(clusters == i)]
		plt.scatter(groupx, groupy)
	plt.show()

def calculate_cluster_score(datasetx, datasety, meansx, meansy, clusters):
	total = 0
	for i in range(datasetx.shape[0]):
		p1 = np.array([datasetx[i], datasety[i]])
		p2 = np.array([meansx[clusters[i]], meansy[clusters[i]]])
		total += euclidean_distance(p1, p2)
	return total

# basketballers
group1height = np.linspace(200, 225, 10)
group1weight = (group1height * 0.45) + np.random.normal(0, 5, 10)

# bodybuilders
group2height = np.linspace(170, 190, 10)
group2weight = (group2height * 0.70) + np.random.normal(0, 5, 10)

# footballers
group3height = np.linspace(180, 200, 80)
group3weight = (group3height * 0.45) + np.random.normal(0, 5, 80)

datasetx = np.concatenate((group1height, group2height, group3height))
datasety = np.concatenate((group1weight, group2weight, group3weight))

plt.scatter(group1height, group1weight)
plt.scatter(group2height, group2weight)
plt.scatter(group3height, group3weight)
plt.show()



clusters = kmeans_cluster(5, datasetx, datasety)

