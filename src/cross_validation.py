class MyKFold:
    """隣接するグループを除く．"""
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        n = X.shape[0]
        group_samples = 10000
        group_count = (n + group_samples - 1) // group_samples
        groups = np.arange(group_count)

        def group_idx_to_idx(idx):
            return np.concatenate([
                np.arange(group_samples * g, min(n, group_samples * (g + 1)))
                for g in idx
            ])

        cv = KFold(self.n_splits)
        for train_index, test_index in cv.split(groups):
            # remove the group next to the test group
            train_index = train_index[~np.isin(train_index, [
                test_index[0] - 1, test_index[0] + 1,
                test_index[-1] - 1, test_index[-1] + 1,
            ])]
            yield group_idx_to_idx(train_index), group_idx_to_idx(test_index)

    def get_n_splits(self):
        return self.n_splits
    
    
class MyTimeSeriesFold:
    """隣接するグループを除く．"""
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        n = X.shape[0]
        group_samples = 10000
        group_count = (n + group_samples - 1) // group_samples
        groups = np.arange(group_count)

        def group_idx_to_idx(idx):
            return np.concatenate([
                np.arange(group_samples * g, min(n, group_samples * (g + 1)))
                for g in idx
            ])

        cv = TimeSeriesSplit(self.n_splits)
        for train_index, test_index in cv.split(groups):
            # remove the group next to the test group
            train_index = train_index[~np.isin(train_index, [
                test_index[0] - 1, test_index[0] + 1,
            ])]
            yield group_idx_to_idx(train_index), group_idx_to_idx(test_index)

    def get_n_splits(self):
        return self.n_splits