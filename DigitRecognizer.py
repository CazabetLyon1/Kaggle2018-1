import random
import numpy

from Tools.DataSet import DataSet
from Tools.Plotting import plot_images_test_results, plot_images_predictions
from InputFormatter import to_drawable_images
from Predictions import TestPredictions, Predictions

from InputFormatter import to_normalized_images as input_formatter
from Models import mlp1D1_initializer as model_initializer
from OutputFormatter import DecimalDigitMapping

numpy.random.seed(7)       # Fix a random seed for reproducibility <!Data augmentation will ruin that effort!>
plot_correct_img_qt = 4    # The amount of correct predictions to plot during visual_confirmation
plot_incorrect_img_qt = 4  # The amount of incorrect predictions to plot during visual_confirmation


# Plot some (random pool) of the predictions (param)
def visual_confirmation(predictions):
    if isinstance(predictions, TestPredictions):
        # Split corrections to correct/incorrect ones
        correct_predictions = predictions.get_correct()
        incorrect_predictions = predictions.get_incorrect()

        # Ensure valid pool size
        correct_qt = plot_correct_img_qt if plot_correct_img_qt < len(correct_predictions) \
            else len(correct_predictions)
        incorrect_qt = plot_incorrect_img_qt if plot_incorrect_img_qt < len(incorrect_predictions) \
            else len(incorrect_predictions)

        # Pick indices in pools
        correct_indices = random.sample(range(len(correct_predictions)), correct_qt)
        incorrect_indices = random.sample(range(len(incorrect_predictions)), incorrect_qt)

        # Pick predictions
        predictions = correct_predictions.filter_by_indices(correct_indices) \
                    + incorrect_predictions.filter_by_indices(incorrect_indices)

        # Plot predictions
        plot_images_test_results(to_drawable_images(predictions.images),
                                 predictions.expected,
                                 predictions.results)
    elif isinstance(predictions, Predictions):
        # Compute expected plot size
        indices_qt = plot_correct_img_qt + plot_incorrect_img_qt
        # Ensure valid pool size
        indices_qt = indices_qt if indices_qt < len(predictions) else len(predictions)
        # Pick indices in pool
        indices = random.sample(range(len(predictions)), indices_qt)
        # Pick predictions
        predictions = predictions.filter_by_indices(indices)

        # Plot predictions
        plot_images_predictions(to_drawable_images(predictions.images), predictions.results)

    else:
        pass  # TODO handle type error


def main():
    # load the dataset
    dataset = DataSet.load_from_csv('Data/challenge_digits', 0.2)
    pixels_qt = dataset.train_images.shape[1]

    output_mapping = DecimalDigitMapping()  # Create an output digits encoder/decoder

    # Encode output (the input is already in the good format)
    # TODO Find a way to spread the charge for output fct! (pool worker w\ multiprocessing? matrix solving w\ numpy?)
    encoded_dataset = dataset.encode(labels_encoding_fct=lambda labels: numpy.array([output_mapping.to_category(label)
                                                                                     for label in labels]))

    # Rebuild image for drawing later
    dataset = dataset.encode(images_encoding_fct=lambda images: input_formatter(images)*255)

    model = model_initializer(pixels_qt, output_mapping.labels_qt)  # build the model

    # Fit the model
    model.fit(encoded_dataset.train_images, encoded_dataset.train_labels,
              validation_data=(encoded_dataset.test_images, encoded_dataset.test_labels),
              epochs=50, batch_size=10000, verbose=1)

    # Evaluate the model
    scores = model.evaluate(encoded_dataset.test_images, encoded_dataset.test_labels, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    del output_mapping


if __name__ == '__main__':
    main()
