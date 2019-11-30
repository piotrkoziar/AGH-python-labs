import prepare_data as p_d
import math

import pandas as pd
import numpy as np
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

def agglomerative(prox, prox_dict, glob_high):
    # find minimum in the proximity matrix
    # pmin = np.unravel_index(np.argmin(prox, axis=None), prox.shape)
    # print(pmin)
    # dist = prox[pmin]
    # print(dist)
    # if dist > glob_high:
        # glob_high = dist
# 
    # distance between cluster and other cluster == the closest distance between points from both clusters
    # y = min(prox[4][3], prox[3][0])
    # print(y)
    # y = dist_to_cluster(prox, pmin, 3)
    # print(y)
    
    cluster_merge_table = []
    dist_diff_list = []
    dist_diff = 0
    nr_of_iter = len(prox) - 1
    for i in range(nr_of_iter):
        print("Proximity matrix in {} iteration".format(i))
        print(prox)
        pmin = np.unravel_index(np.argmin(prox, axis=None), prox.shape)
        
        entry = (prox_dict[pmin[0]], prox_dict[pmin[1]])
        cluster_merge_table.append(entry)
        print("Pmin: {}".format(pmin))
        print("glob_high: {}".format(glob_high))
        print("dict:")
        p_d.print_dict(prox_dict)
        dist = prox[pmin]
        if i == 0:
            dist_diff = dist
        else:
            dist_diff = dist - dist_diff_list[-1]

        dist_diff_list.append(dist_diff)

        if dist_diff > glob_high:
            glob_high = dist_diff

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
    return cluster_merge_table, dist_diff_list, glob_high

# X - matrix with prepared data
def dendrogram_get_points_from_tuple(tp, X):
    # on dendrogram, x axis = x coordinate from X matrix
    # get the x-axis middle point lying between points from the tuple 
    sum = 0
    tp_len = len(tp)
    for i in range(tp_len):
        point_nr = tp[i]
        # get x coordinates of the point
        sum += (X[point_nr])[0]
    if tp_len > 1:
        return sum/tp_len
    
    return sum

def dendrogram_vertical_update(cl_to_update, lvl_zero, lvl_one):
    for cl in cl_to_update:
        plt.plot((cl, cl), (lvl_zero, lvl_one), 'r-')        
            

print("start")
p_d.prepare_data("./sport", 100)
# p_d.print_dict(p_d.globaldict)
# eucl_dist(p_d.globaldict["the"], p_d.globaldict["a"])

data = pd.DataFrame.from_dict(p_d.globaldict)
X = data.iloc[[0, 1], :].values
# print(X)
X = X.transpose()
print(X.shape)
X = X[0:15]
print(X)
plt.scatter(X[:,0],X[:,1])
prox, prox_dict = proximity_matrix(X)
# print(prox)
glob_high = 0
cluster_merge_table, dist_diff_list, glob_high = agglomerative(prox, prox_dict, glob_high)
p_d.print_dict(prox_dict)
for i in dist_diff_list: print(i)
for i in cluster_merge_table: print(i)

x = np.linspace(0, len(dist_diff_list) + 1)

fig, ax = plt.subplots()

cl_to_update = []
for cl in prox_dict[0]:
    px = dendrogram_get_points_from_tuple((cl,), X)
    cl_to_update.append(px)

# current "one" level on the plot
diff_one = 0
for i in range(len(dist_diff_list)):
    
    # current "zero" level on the plot. Initially == 0
    diff_zero = diff_one
    # update diff_one
    diff_one = dist_diff_list[i] + diff_zero

    # draw vertical lines to the next level
    dendrogram_vertical_update(cl_to_update, diff_zero, diff_one)    

    p_ym1 = diff_zero
    p_ym2 = diff_zero

    # entry in cluster_merge_table is tuple with two tuples.
    cl_merge_entry = cluster_merge_table[i]
    clleft = cl_merge_entry[0]
    clright = cl_merge_entry[1]

    print("clleft: {}, clright: {}".format(clleft, clright))

    p_xm1 = dendrogram_get_points_from_tuple(clleft, X)
    p_xm2 = dendrogram_get_points_from_tuple(clright, X)

    cl_to_update.remove(p_xm1)
    cl_to_update.remove(p_xm2)

    p_xm3 = dendrogram_get_points_from_tuple(clleft + clright, X)
    cl_to_update.append(p_xm3)

    print("p_xmi: {}, p_xma: {}".format(p_xm1, p_xm2))
    print("diff_zero: {}, diff_one: {}".format(diff_zero, diff_one))

    plt.plot((p_xm1, p_xm2), (diff_one, diff_one), 'r-')
    # plt.plot((p_xm1, p_xm1), (p_ym1, diff_one), 'r-')
    # plt.plot((p_xm2, p_xm2), (p_ym2, diff_one), 'r-')
    # ax.hlines(y=diff_one, xmin=p_xmi, xmax=p_xma, linewidth=2, color='r')
    # ax.axvline(x=p_xmi, ymin=diff_zero, ymax=diff_one, linewidth=2, color='r')
    # ax.axvline(x=p_xma, ymin=diff_zero, ymax=diff_one, linewidth=2, color='r')



# reference
# plt.figure(figsize=(10, 7))
# plt.title("Customer Dendograms")
# dend = sch.dendrogram(sch.linkage(X, method='ward'))
# cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(X)
# print(cluster.labels_)
# plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()