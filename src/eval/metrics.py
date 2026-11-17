import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering(y, labels):
    n = int(max(y.max(), labels.max()) + 1)
    counts = np.zeros((n, n), dtype=np.int64)
    np.add.at(counts, (labels, y), 1)
    row, col = linear_sum_assignment(-counts)
    return dict(acc=float(counts[row, col].sum() / len(y)),
                nmi=float(normalized_mutual_info_score(y, labels)),
                ari=float(adjusted_rand_score(y, labels)))
