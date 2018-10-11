import numpy


# Mapping object between labels and category weight
# TODO Add saving & loading option for Mapping
class CategoricalMapping(object):
    def __init__(self, labels):
        self.labels = labels               # Define a label set
        self.labels_qt = len(self.labels)  # Quantity of characters in the label set

        # The dictionary mapping the categorical value of each label in the set
        self.label_to_cat_translation_dict = {val: self.generate_cat_mapping(idx, self.labels_qt)
                                              for idx, val in enumerate(self.labels)}

    # Get the categorical representation of the element n°'pos' of 'size' categories
    @classmethod
    def generate_cat_mapping(cls, pos, size):
        # Handle wrong param values
        assert (size > 0)
        assert (0 <= pos < size)

        return [1.0 if i == pos else 0.0 for i in range(size)]

    # Convert a label to the corresponding category
    # TODO Handle not found error
    def to_category(self, label):
        return self.label_to_cat_translation_dict.get(label)

    # Convert a category to the corresponding label
    def to_label(self, cat):
        if type(cat) is not numpy.ndarray:
            cat = numpy.array(cat)  # Try to cast (solve potential list type errors]
        if len(cat) != self.labels_qt:
            return  # TODO Handle wrong input shape error

        return self.labels[int(numpy.argmax(cat))]