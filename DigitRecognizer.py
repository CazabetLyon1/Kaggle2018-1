import random
import numpy

from Tools.DataSet import DataSet
from InputFormatter import to_normalized_images as input_formatter
from Models import mlp1D1_initializer as model_initializer
from Tools.Plotting import plot_numbers_test_results

from Data.Challenge1 import DataIO                                  # Challenge1 data loading & saving
from OutputFormatter import DecimalDigitMapping as OutputMapping    # [0...9] Categorical encoder (for output mapping)

numpy.random.seed(7)  # Fix a random seed for reproducibility


def main():
    cat_mapping = OutputMapping()                               # Create an output encoder/decoder
    images, labels = DataIO.load_training_data()                # Load the data
    dataset = DataSet(*DataSet.split_set(images, labels, 0.2))  # Instantiate the DataSet


    # Encode output (the input is already in the good format)
    # TODO Find a way to spread the charge for output fct! (pool worker w\ multiprocessing? matrix solving w\ numpy?)
    encoded_dataset = dataset.encode(labels_encoding_fct=lambda labels: numpy.array([output_mapping.to_category(label)
                                                                                     for label in labels]))

    # Compute the quantity of pixels
    pixels_qt = encoded_dataset.train_images.shape[1] * encoded_dataset.train_images.shape[2]

    # Rebuild image for drawing later
    dataset = dataset.encode(images_encoding_fct=lambda images: input_formatter(images)*255)

    model = model_initializer(pixels_qt, cat_mapping.labels_qt)  # build the model

    # Fit the model
    model.fit(encoded_dataset.train_images, encoded_dataset.train_labels,
              validation_data=(encoded_dataset.test_images, encoded_dataset.test_labels),
              epochs=50, batch_size=10000, verbose=1)

    # Evaluate the model
    scores = model.evaluate(encoded_dataset.test_images, encoded_dataset.test_labels, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    # Execute a random test of 8 numbers and plot it
    test_random_numbers(model, cat_mapping, dataset, encoded_dataset, 8)

    del cat_mapping


def test_random_numbers(model, output_mapping, dataset, encoded_dataset, numbers_qt=8, plot_result=True):
    # Random selection to plot for final approval
    samples_qt = numbers_qt if numbers_qt <= len(dataset.test_images) else len(dataset.test_images)
    samples_indexes = random.sample(range(len(dataset.test_images)), samples_qt)

    # Predicting the random selection
    results = [output_mapping.to_label(model.predict(numpy.array([encoded_dataset.test_images[i]]))[0])
               for i in samples_indexes]

    if plot_result:
        # Plotting the random selection results
        plot_numbers_test_results([dataset.test_images[i] for i in samples_indexes],
                                  [output_mapping.to_label(encoded_dataset.test_labels[i]) for i in samples_indexes],
                                  results)
    else:
        # Print result in console
        print("Results of {0} random tests".format(samples_qt))
        for i, idx in enumerate(samples_indexes):
            print("Expected {0} | Predicted {1}".format(output_mapping.to_label(encoded_dataset.test_labels[idx]),
                                                        results[i]))


if __name__ == '__main__':
    main()
