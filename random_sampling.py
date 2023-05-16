import numpy as np

aa = np.arange(30)

for i in range(10):
    print(len(aa))
    sample_indices = np.random.choice(aa, 10, replace=False)
    print(aa, sample_indices)
    # delete the values of sampled_indices from aa
    aa = np.setdiff1d(aa, sample_indices)

"""
30
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
24 25 26 27 28 29] [16 27 10  7  4  8 20 13 17 28]

20
[ 0  1  2  3  5  6  9 11 12 14 15 18 19 21 22 23 24 25 26 29] [23  3 26 21  1 19 12  6 22  5]

10
[ 0  2  9 11 14 15 18 24 25 29] [25 18 29 11  2 24 14  9 15  0]

0
"""
