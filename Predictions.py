import numpy as np


# Object containing a series of inputs and allowing to associate it with a series of predicted outputs
class Predictions(object):
    def __init__(self, images, results=None):
        self.images = images
        self.results = results

    # Get the size of the predictions
    def __len__(self):
        return self.images.shape[0]

    # Access a specific element
    # Return image, result
    def __getitem__(self, i):
        return self.images[i], self.results[i]

    # Concatenate two Predictions together
    def __add__(self, other):
        # Do not set results if any of them is missing
        results = None if self.results is None or other.results is None \
             else np.concatenate((self.results, other.results))

        return Predictions(np.concatenate(self.images, other.images),
                           results)

    # Concatenate two Predictions together
    def __radd__(self, other):
        # Do not set results if any of them is missing
        results = None if self.results is None or other.results is None \
             else np.concatenate((self.results, other.results))

        return Predictions(np.concatenate((self.images, other.images)),
                           results)

    # Get the Predictions object containing the predictions corresponding to each index in indices
    def filter_by_indices(self, indices):
        return Predictions(np.asarray([self.images[i] for i in indices]),
                           np.asarray([self.results[i] for i in indices] if self.results is not None else None))


# Prediction object which also map the expected output to each input
class TestPredictions(Predictions):
    def __init__(self, images, expected, results=None):
        assert (images.shape[0] == expected.shape[0])
        super(TestPredictions, self).__init__(images, results)
        self.expected = expected
        self.correct = None
        self.incorrect = None

    # Access a specific element
    # Return image, result, expected
    def __getitem__(self, i):
        return self.images[i], self.results[i], self.expected[i]

    # Concatenate two Predictions together
    def __add__(self, other):
        # Do not set results if any of them is missing
        results = None if (self.results is None or other.results is None) \
             else np.concatenate((self.results, other.results))

        return TestPredictions(np.concatenate((self.images, other.images)),
                               results,
                               np.concatenate((self.expected, other.expected)))

    # Concatenate two Predictions together
    def __radd__(self, other):
        # Do not set results if any of them is missing
        results = None if self.results is None or other.results is None \
             else np.concatenate((self.results, other.results))

        return TestPredictions(np.concatenate((self.images, other.images)),
                               results,
                               np.concatenate((self.expected, other.expected)))

    # Get the Predictions object containing all the correct predictions
    def get_correct(self):
        # Can't test validity if no prediction done
        if self.results is None:
            pass  # TODO Handle not predicted error

        # Compute value if cache not set (first call)
        if self.correct is None:
            #  Get the indices of all correct predictions
            correct_indices = list(filter(lambda i: self.results[i] == self.expected[i],
                                          list(range(self.images.shape[0]))))
            #  Update the cache
            self.correct = self.filter_by_indices(correct_indices)  # Apply filter
            self.correct.correct = self.correct                     # Avoid (get_correct().get_correct()) caching

        # Return the cached result
        return self.correct

    # Get the Predictions object containing all the incorrect predictions
    def get_incorrect(self):
        # Can't test validity if no prediction done
        if self.results is None:
            pass  # TODO Handle not predicted error

        # Compute value if cache not set (first call)
        if self.incorrect is None:
            #  Get the indices of all incorrect predictions
            incorrect_indices = list(filter(lambda i: self.results[i] != self.expected[i],
                                            list(range(self.images.shape[0]))))
            #  Update the cache
            self.incorrect = self.filter_by_indices(incorrect_indices)

        return self.incorrect

    # Get the Predictions object containing the predictions corresponding to each index in indices
    def filter_by_indices(self, indices):
        return TestPredictions(np.asarray([self.images[i] for i in indices]),
                               np.asarray([self.results[i] for i in indices]) if self.results is not None else None,
                               np.asarray([self.expected[i] for i in indices]))
