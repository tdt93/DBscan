import os
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import pairwise_distances

# import the dataset
os.chdir("..")
os.chdir("..")
path = os.getcwd()
new_path = os.path.join(path, "dataset", "text_files", "history")
os.chdir(new_path)
file_list = []
text_list = []
count = 0

for file in os.listdir():
    if file.endswith(".txt"):
        file_list.append(file)

# write text content into list
for file in file_list:
    if count < 10:
        text_file = open(file, "r")
        for line in text_file:
            line = line.rstrip()
            if line != "":
                line = line.replace("(", "").replace(")", "").replace(":", "")\
                            .replace("[", "").replace("]", "").replace(",", "")
                text_list = text_list + line.split(".")
        text_file.close()
        count += 1
    else:
        break
# print(text_list)

# for word in text_list:
vect = TfidfVectorizer(stop_words="english")
tfidf = vect.fit_transform(text_list)
# X = tfidf.toarray()

distance_array = pairwise_distances(tfidf, metric='cosine')
X = distance_array
# show word encoded to number
# print(vect.vocabulary_)

# apply DBSCAN
db_scan = DBSCAN(eps=2.5, min_samples=3).fit(X)
labels = db_scan.labels_
print(labels)

# identify the core points
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db_scan.core_sample_indices_] = True
print(core_samples_mask)

# calculate number of clusters
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(metrics.silhouette_score(X, labels))
n_noise = list(labels).count(-1)

# plotting: Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=10,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()


