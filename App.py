import random
import functools
import numpy

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model as k_load_model

from Tools.DataSet import DataSet
from Tools.Plotting import plot_images_test_results, plot_images_predictions
from InputFormatter import to_drawable_images
from Predictions import TestPredictions, Predictions
from Models.ModelIO import load_model, save_model
from Tools.CategorigalMapping import CategoricalMapping


# Customizable import statements
from Data.Challenge2 import DataIO                                  # Challenge2 data loading
from Data.Challenge2.DataIO import DrawingData                      # Import custom data class for distant pickle
from InputFormatter import to_normalized_images as input_formatter  # Format inputs to 0<=(n, 28, 28,)<=1
from Models.Challenge2.CNN2D_2 import Compiler                      # Import a compiler for the model

model_name = 'Challenge2/CNN2D_2'
network_name = 'Main'
load_network = True
save_network = True

numpy.random.seed(7)       # Fix a random seed for reproducibility <!Data augmentation will ruin that effort!>
plot_correct_img_qt = 4    # The amount of correct predictions to plot during visual_confirmation
plot_incorrect_img_qt = 4  # The amount of incorrect predictions to plot during visual_confirmation


# Plot some (random pool) of the predictions (param)
def visual_confirmation(predictions, cat_mapping):
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
                                 list(map(cat_mapping.to_label, predictions.expected)),
                                 list(map(cat_mapping.to_label, predictions.results)))
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
        plot_images_predictions(to_drawable_images(predictions.images), list(map(cat_mapping.to_label, predictions.results)))

    else:
        pass  # TODO handle type error


# Fit a mode2l to a dataset
def train_model(model, dataset, epochs=1, batch_size=64, lr_reduction=None):
    # TODO Factorize
    # Prepare the fitting functor
    prepared = functools.partial(model.fit_generator,
                                 epochs=epochs,
                                 generator=dataset.train_generator.__iter__(),
                                 validation_data=dataset.val_generator.__iter__(),
                                 steps_per_epoch=dataset.train_generator.steps,
                                 validation_steps=dataset.val_generator.steps,
                                 verbose=1)

    if lr_reduction is not None:
        # Set a learning rate annealer
        prepared(callbacks=[lr_reduction])
    else:
        # Fit the model
        prepared()


# Predict a prepared Predictions object with a given network
def predict(model, output_mapping, predictions, log_result=True):
    # Do predictions
    predictions.results =  model.predict(predictions.images)

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
def main(subset=0):
    cat_mapping = CategoricalMapping(DataIO.get_output_categories())  # Create an output encoder/decoder
    images, labels = DataIO.load_training_data(subset)                # Load the data
    dataset = DataSet(*DataSet.split_set(images, labels, 0.2))        # Instantiate the DataSet

    # Encode the DataSet
    encoded_dataset = dataset.encode(images_encoding_fct=input_formatter,
                                     labels_encoding_fct=lambda labels: numpy.array([cat_mapping.to_category(label)
                                                                                     for label in labels]))

    load_name = network_name if load_network else None                      # Loading network or not ?
    if subset != 0:
        load_name = network_name+'-'+str(subset-1) if load_network else None  # Loading network or not ?

    nn = load_model('Models/{}/'.format(model_name), load_name)             # Load the model
    nn = Compiler.compile_model(nn)                                         # Compile the model

    train_model(nn, encoded_dataset, 2, 1024)                               # Train the model
    evaluate(nn, encoded_dataset.test_images, encoded_dataset.test_labels)  # Evaluate the model

    if save_network:
        save_model(nn, 'Models/{}/'.format(model_name), network_name+'-'+str(subset))  # Save model + (opt) network

    # Make predictions on the Validation data
    predictions = TestPredictions(encoded_dataset.test_images, dataset.val_labels)  # Load predictions
    predict(nn, cat_mapping, predictions)                                            # Make predictions
    #visual_confirmation(predictions)                                                 # Plot sample of predictions

    # Make predictions on the challenge data
    #predictions = Predictions(input_formatter(DataIO.load_prediction_data()))  # Load predictions
    #predict(nn, cat_mapping, predictions, False)                               # Make predictions
    #visual_confirmation(predictions)                                           # Plot sample of predictions

    #DataIO.save_submission(predictions.results)  # Save the result into the correct submission format

# Use python shell/script w/ import statement to avoid jumping to the process entry point
if __name__ == '__main__':
    for i in range(0, 10):
        main(i)
