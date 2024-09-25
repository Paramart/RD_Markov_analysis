Hidden Markov State Modeling (HMM) is a powerful tool for analyzing kinetic states in molecular dynamics trajectories. This Python script, derived from the PyEMMA 2.15.12 library, has been adapted to evaluate HMMs based on the number of cluster centers used in k-means clustering. The original PyEMMA code was designed to assess cross-validated (10-fold) VAMP2 scores for Markov State Models (MSMs).

In this repository, you will find an example dataset (example_data.pickle) containing a featurized trajectory that spans 100 ns (with 1001 frames). The features represent ten Euclidean distances between C-alpha atoms in the transmembrane domain of the human orexin receptor 2.

Although the script performs random splits during cross-validation, the resulting plot should resemble the example figure provided below.
This code has been tested with Python 3.10.12 and the following library versions: NumPy 1.26.4, Matplotlib 3.8.4, Deeptime 0.4.4, and SciPy 1.13.1.

If you have any comments or suggestions, feel free to reach out.

Rafael Dolezal
![HMM_VAMP2_score_clustering](https://github.com/user-attachments/assets/832a7345-5a5d-4988-84af-df5409840a22)
