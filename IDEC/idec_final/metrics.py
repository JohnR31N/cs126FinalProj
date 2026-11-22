import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate(train_y, train_cluster, test_y, test_cluster, flipped_cluster, orientation):
    counts = np.zeros((10, 10), dtype=int)
    np.add.at(counts, (train_cluster, train_y), 1)
    rows, mapping = linear_sum_assignment(-counts)
    assert np.array_equal(rows, np.arange(10))
    return {
        "train_acc": float(np.mean(mapping[train_cluster] == train_y)),
        "digit_acc": float(np.mean(mapping[test_cluster] == test_y)),
        "digit_nmi": float(normalized_mutual_info_score(test_y, test_cluster)),
        "digit_ari": float(adjusted_rand_score(test_y, test_cluster)),
        "stripe_nmi": float(normalized_mutual_info_score(orientation, test_cluster)),
        "flip_rate": float(np.mean(test_cluster != flipped_cluster)),
    }
