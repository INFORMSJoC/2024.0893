import numpy as np

with open("out.txt", "w") as f:
    for i in (60, 62, 64, 66, 68):
        f.write(f"i = {i}")
        sensing_matrix = np.load(f"data/road_data/sensing_matrix_{i}_100.npy")
        f.write("nodes per group")
        f.write(str(list(sensing_matrix.sum(axis=0))))
        f.write(str(float(sensing_matrix.sum(axis=0).mean())))
        f.write("groups per node")
        f.write(str(list(sensing_matrix.sum(axis=1))))
        f.write(str(float(sensing_matrix.sum(axis=1).mean())))
        f.write("\n\n")
