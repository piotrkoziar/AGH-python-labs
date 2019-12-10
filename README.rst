=============
hi_clustering
=============


Hierarchical Clustering

The script performs Hierarchical Clustering data classification.

Script prepares the data from the text files in the directory like that:
p_d.prepare_data(path_to_file, 20)

Prepared data is a directory where:
key = word (from the text file)
value = [x, y], where x = number of files in which the word has appeared
                      y = number of times the word has appeared

The dendrogram is prepared if draw==True

The result is plotted with the matplotlib.

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
