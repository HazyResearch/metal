class Featurizer(object):
    def fit(self, input):
        """
        Args:
            input: An iterable of raw data of the appropriate type to be
                featurized, where input[i] corresponds to item i.
        """
        raise NotImplementedError

    def transform(self, input):
        """
        Args:
            input: An iterable of raw data of the appropriate type to be
                featurized, where input[i] corresponds to item i.
        Returns:
            X: A Tensor of features of shape (num_items, ...)
        """
        raise NotImplementedError

    def fit_transform(self, input, **fit_kwargs):
        """Execute fit and transform in sequence."""
        self.fit(input, **fit_kwargs)
        X = self.transform(input)
        return X
