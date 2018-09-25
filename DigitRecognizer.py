import random
import functools
import numpy
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


from Tools.DataSet import DataSet
from Tools.Plotting import plot_images_test_results, plot_images_predictions
from InputFormatter import to_drawable_images
from Predictions import TestPredictions, Predictions


# Customizable import statements
from Data.Challenge1 import DataIO                                  # Challenge1 data loading & saving
from InputFormatter import to_normalized_images as input_formatter  # Format inputs to 0<=(n, 28, 28,)<=1
from Models import cnn2D1_initializer as model_initializer          # Model used
from OutputFormatter import DecimalDigitMapping as OutputMapping    # [0...9] Categorical encoder (for output mapping)


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


# Fit a model to a dataset
def train_model(model, encoded_dataset, epochs=1, batch_size=64, with_lr_reduction=False, with_augmentation=False):
    # TODO Factorize
    if with_augmentation:
        # Data augmentation to prevent overfitting
        data_gen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=10,                    # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range=0.1,                       # Randomly zoom image
                width_shift_range=0.1,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,               # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,                # randomly flip images
                vertical_flip=False)                  # randomly flip images

        data_gen.fit(encoded_dataset.train_images)

        # Prepare the fitting functor
        prepared = functools.partial(model.fit_generator,
                                     data_gen.flow(encoded_dataset.train_images,
                                                   encoded_dataset.train_labels, batch_size=batch_size),
                                     validation_data=(encoded_dataset.test_images,
                                                      encoded_dataset.test_labels),
                                     epochs=epochs, verbose=1)

    else:
        # Prepare the fitting functor
        prepared = functools.partial(model.fit,
                                     encoded_dataset.train_images, encoded_dataset.train_labels,
                                     validation_data=(encoded_dataset.test_images, encoded_dataset.test_labels),
                                     epochs=epochs, batch_size=batch_size, verbose=1)

    if with_lr_reduction:
        # Set a learning rate annealer
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.000001)
        # Fit the model
        prepared(callbacks=[learning_rate_reduction])
    else:
        # Fit the model
        prepared()


# Predict a prepared Predictions object with a given network
def predict(model, output_mapping, predictions, log_result=True):
    # Do predictions
    predictions.results = numpy.array([output_mapping.to_label(result) for result in model.predict(predictions.images)])

    if not log_result:
        return  # Stop here if no log needed
    elif not isinstance(predictions, TestPredictions):
        return  # TODO Handle type error
    else:
        # Separate correct & incorrect predictions
        correct_predictions = predictions.get_correct()
        incorrect_predictions = predictions.get_incorrect()

        # Log predictions results
        print('{} predictions: {} correct, {} incorrect'.format(len(predictions),
                                                                len(correct_predictions),
                                                                len(incorrect_predictions)))


# Evaluate a model on a validation set
def evaluate(nn, test_images, test_labels):
    # Evaluate the network
    scores = nn.evaluate(test_images, test_labels, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))


# Process entry point
def main():
    cat_mapping = OutputMapping()                               # Create an output encoder/decoder
    images, labels = DataIO.load_training_data()                # Load the data
    dataset = DataSet(*DataSet.split_set(images, labels, 0.2))  # Instantiate the DataSet

    # Encode the DataSet
    encoded_dataset = dataset.encode(images_encoding_fct=lambda images: input_formatter(images),
                                     labels_encoding_fct=lambda labels: numpy.array([cat_mapping.to_category(label)
                                                                                     for label in labels]))

    # Compute the quantity of pixels
    pixels_qt = encoded_dataset.train_images.shape[1] * encoded_dataset.train_images.shape[2]

    #nn = load_model('Networks/main_network.h5')                             # Load the model
    nn = model_initializer(pixels_qt, cat_mapping.labels_qt)                # Compile the model
    train_model(nn, encoded_dataset, 50, 2048, True, True)                  # Train the model
    evaluate(nn, encoded_dataset.test_images, encoded_dataset.test_labels)  # Evaluate the model

    # Make predictions on the test data
    predictions = TestPredictions(encoded_dataset.test_images, dataset.test_labels)  # Load predictions
    predict(nn, cat_mapping, predictions)                                            # Make predictions
    visual_confirmation(predictions)                                                 # Plot sample of predictions

    # Make predictions on the challenge data
    predictions = Predictions(input_formatter(DataIO.load_prediction_data()))  # Load predictions
    predict(nn, cat_mapping, predictions, False)                               # Make predictions
    visual_confirmation(predictions)                                           # Plot sample of predictions

    DataIO.save_submission(predictions.results)  # Save the result into the correct submission format
    #nn.save("Networks/main_network.h5")          # Save the model


# Use python shell/script w/ import statement to avoid jumping to the process entry point
if __name__ == '__main__':
    main()
