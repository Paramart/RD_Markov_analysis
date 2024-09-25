Hidden Markov state modeling belongs to interesting tools that can analyze kinetic states in molecular dynamics trajectories. This simple Python script was derived from PyEMMA 2.15.12 libraries. The original code in PyEMMA can evaluate cross-validated (10-fold) VAMP2 score for MSMs. After simple changes, this updated code can evaluate HMMs depending on the number of cluster centers used in k-means clustering. 

In this repository, you can find a file with example data (example_data.pickle) containing featurized trajectory lasting 100 ns (containing 1001 frames). The faturization is based on ten Euclidean distances between C-alpha atoms in the transmembrane domain of human orexin receptor 2. Although the script involves random splitting within cross-validation, the output plot should look like the figure bellow.

This code works well with Python 3.10.12, Numpy 1.26.4, Matplotlib 3.8.4, Deeptime 0.4.4, Scipy 1.13.1.

Please, in case you have comments or suggestions, let me know.

Rafael Dolezal
