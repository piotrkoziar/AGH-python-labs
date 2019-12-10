import math
import pandas as pd
import numpy as np
import prepare_data as p_d
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

def eucl_dist(x,y):
    # x [x1,x2]; y [y1, y2]
    [x1, x2] = x
    [y1, y2] = y
    # print(x, " ", y)
    result = math.sqrt(((x1-y1)**2)+((x2-y2)**2))
    return result

def proximity_matrix(X):
    length = len(X[:,0])
    prox_dict = dict()
    prox = np.full((length, length), np.inf)
    for i in range(length):
        for j in range(i):
            if i == j:
                continue

            prox[i][j] = eucl_dist(X[i], X[j])
        # create dictionary: key: number of proximity matrix row
        # value: cluster 
        prox_dict[i] = (i,)

    return prox, prox_dict

# n - proximity row number, in which we update the matrix
def dist_to_cluster(prox, pmin, n):
    # distance between cluster and other cluster == the closest distance between points from both clusters
    possible_dist = np.full((1, len(pmin)), np.inf)
    for i in range(len(pmin)):
        # if pmin[i] > n, then we need prox[pmin[i]][n]. Otherwise, we need prox[n][pmin[i]].
        if pmin[i] > n:
            possible_dist[0][i] = prox[pmin[i]][n]
        else:
            possible_dist[0][i] = prox[n][pmin[i]]
    dist = possible_dist.min()
    return dist

def agglomerative(prox, prox_dict):
    # distance between cluster and other cluster == the closest distance between points from both clusters
    glob_high = 0
    glob_high_dist = 0
    cluster_merge_table = []
    dist_diff_list = []
    dist_diff = 0
    nr_of_iter = len(prox) - 1
    for i in range(nr_of_iter):
        # print("Proximity matrix in {} iteration".format(i))
        # print(prox)
        pmin = np.unravel_index(np.argmin(prox, axis=None), prox.shape)
        
        entry = (prox_dict[pmin[0]], prox_dict[pmin[1]])
        cluster_merge_table.append(entry)
        # print("Pmin: {}".format(pmin))
        # print("glob_high: {}".format(glob_high))
        # print("dict:")
        # p_d.print_dict(prox_dict)
        dist = prox[pmin]
        if i == 0:
            dist_diff = dist
        else:
            dist_diff = dist - dist_diff_list[-1]

        dist_diff_list.append(dist_diff)

        if dist_diff > glob_high:
            glob_high = dist_diff
            glob_high_dist = dist

        # which row will represent the new cluster
        cl_row_nr = min(pmin)
        # which row will be deleted
        cl_row_del = max(pmin)
        # update dictionary (row-cluster mapping)
        prox_dict[cl_row_nr] = (prox_dict[cl_row_nr] + prox_dict[cl_row_del])
        # get the distance between the new cluster and other clusters
        for j in range(len(prox)):
            if j == cl_row_nr or j == cl_row_del:
                continue
            # print(prox)
            dist_to_new = dist_to_cluster(prox, pmin, j)

            if j > cl_row_nr:
                prox[j][cl_row_nr] = dist_to_new
            else:
                prox[cl_row_nr][j] = dist_to_new
        
            if cl_row_del < j:
                # align row
                for z in range(cl_row_del, j):
                    prox[j][z] = prox[j][z + 1]
        # delete row from prox (this row was merged to cl_row_nr)
        prox = np.delete(prox, cl_row_del, 0)
        # align dictionary
        for key in range(cl_row_del, len(prox)):
            prox_dict[key] = prox_dict[key+1] 
        del prox_dict[len(prox)]
    return cluster_merge_table, dist_diff_list, glob_high, glob_high_dist

# cl_order - dictionary where: key: single-point cluster, value: index on the x-axis.
# tp - tuple representing content of cluster. Can be single- or multi- point.
def dendrogram_get_x_position(tp, cl_order):
    # cl_order determines where on the x-axis single-point cluster is located.
    # For multi-point clusters, get the x-axis middle point lying between points from the cluster. 
    sum = 0
    tp_len = len(tp)
    for i in range(tp_len):
        point_nr = tp[i]
        sum += cl_order[point_nr]
    if tp_len > 1:
        return sum/tp_len
    return sum            

# cluster_merge_table - history of cluster merging: from first to last merge.
# entry in cluster_merge_table is tuple with two tuples.
# single_point_clusters - list with initial one-point clusters. One-point cluster includes number of point from data matrix.
# This list must have specific order: clusters that merge with each other must be neighbours (if not, the dendrogram can be less readable).
# dist_diff_list - list of delta distance value between clusters during each merge
def dendrogram(single_point_clusters, cluster_merge_table, dist_diff_list, threshold, draw):
    clusters = []
    cl_to_update = []
    cl_order = dict()

    if draw == True:    
        # assign distance from 0 on x-axis to the single-point clusters.
        for cl in range(len(single_point_clusters)):
            cl_order[single_point_clusters[cl]] = cl
        for cl in single_point_clusters:
            px = dendrogram_get_x_position((cl,), cl_order)
            cl_to_update.append(px)

    # current "one" level on the plot
    diff_one = 0
    for i in range(len(dist_diff_list)):
        
        # current "zero" level on the plot. Initially == 0
        diff_zero = diff_one
        # update diff_one
        diff_one = dist_diff_list[i] + diff_zero
        print("diff_zero: {}, diff_one: {}".format(diff_zero, diff_one))

        if draw == True:
        # draws vertical lines from diff_zero to diff_one for each x-axis coordinate in cl_to_update.
            # draw vertical lines to the next level
            for cl in cl_to_update:
                plt.plot((cl, cl), (diff_zero, diff_one), 'r-')                
                if threshold < diff_one and threshold > diff_zero:
                    clusters.append(clleft)
                    clusters.append(clright)
    

        # get info about the merge (which clusters, which points was in these clusters).
        cl_merge_entry = cluster_merge_table[i]
        clleft = cl_merge_entry[0]
        clright = cl_merge_entry[1]
        print("clleft: {}, clright: {}".format(clleft, clright))

        # check if clusters are above 
        if threshold < diff_one and threshold > diff_zero:
            clusters.append(clleft)
            clusters.append(clright)

        if draw == True:
            # get x-axis coords for clusters.
            p_xm1 = dendrogram_get_x_position(clleft, cl_order)
            p_xm2 = dendrogram_get_x_position(clright, cl_order)
            # cluster horizontal lines will no longer be updated (they are presented as one merged cluster from now)
            # Note that in the first iteration there are only one-point clusters in the update list. 
            cl_to_update.remove(p_xm1)
            cl_to_update.remove(p_xm2)
            # find a place where the horizontal line will start and add it to the update list.
            p_xm3 = dendrogram_get_x_position(clleft + clright, cl_order)
            cl_to_update.append(p_xm3)
            plt.subplot(2, 1, 2)
            plt.plot((p_xm1, p_xm2), (diff_one, diff_one), 'r-')

    # reference
    # plt.figure(figsize=(10, 7))
    # plt.title("Customer Dendograms")
    # dend = sch.dendrogram(sch.linkage(X, method='ward'))
    # cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    # cluster.fit_predict(X)
    # print(cluster.labels_)
    # plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')

    return clusters

def clusters_draw(clusters, X):
    label = [0] * len(X)
    for cl_nr in range(len(clusters)):
        for nr in clusters[cl_nr]:
            label[nr] = cl_nr
    print(label)
    plt.subplot(2, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=label, cmap='rainbow')#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

def hiclust(X):
    prox, prox_dict = proximity_matrix(X)
    # print(prox)
    cluster_merge_table, dist_diff_list, glob_high, glob_high_dist = agglomerative(prox, prox_dict)
    p_d.print_dict(prox_dict)
    print(glob_high)
    print(glob_high_dist)
    for i in dist_diff_list: print(i)
    for i in cluster_merge_table: print(i)

    threshold = glob_high_dist - glob_high/2
    # threshold = threshold/2

    clusters = dendrogram(prox_dict[0], cluster_merge_table, dist_diff_list, threshold, True)

    print(threshold)
    for i in clusters: print(i)

    clusters_draw(clusters, X)

    plt.show()


p_d.prepare_data("./test/france", 20)
# p_d.prepare_data(path, n)
data = pd.DataFrame.from_dict(p_d.globaldict)
X = data.iloc[[0, 1], :].values
# print(X)
X = X.transpose()
print(X.shape)
# X - ([number_of_data] x 2) data matrix, each row corresponds to a single-point (with [x, y] coords) cluster.
X = X[0:1000]
print(X)
# plt.scatter(X[:,0],X[:,1])
hiclust(X, n)