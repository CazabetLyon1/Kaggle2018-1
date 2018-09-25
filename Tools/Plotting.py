import math
import matplotlib.pyplot as plt

DEFAULT_MAX_COLUMN_QT = 4
TITLE_FONT_SIZE = 8.0

# Open new plot containing the list of images
def plot_images(images, max_column_qt=DEFAULT_MAX_COLUMN_QT):
    if len(images) <= 0:
        return  # TODO add error handling (no images)

    column_qt = len(images) if len(images) < max_column_qt else max_column_qt  # Compute the required qt of columns
    row_qt = int(math.ceil(len(images) / column_qt))                           # Compute the required qt of rows

    for i in range(len(images)):
        plt.subplot(row_qt, column_qt, i+1)               # Add a subplot
        plt.axis('off')                                   # Disable the drawing of the axes
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))  # Associate the subplot with a number

    plt.show()  # Show the plot


# Open new plot containing the list of images and their predicted value as title
def plot_images_predictions(images, predicted_values, max_column_qt=DEFAULT_MAX_COLUMN_QT):
    if len(images) <= 0:
        return  # TODO add error handling (no images)
    if not (len(images) == len(predicted_values)):
        return  # TODO add error handling (incoherent array sizes)

    column_qt = len(images) if len(images) < max_column_qt else max_column_qt  # Compute the required qt of columns
    row_qt = int(math.ceil(len(images) / column_qt))                           # Compute the required qt of rows

    for i in range(len(images)):
        subplot = plt.subplot(row_qt, column_qt, i+1)     # Add a subplot
        plt.axis('off')                                   # Disable the drawing of the axes
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))  # Set the image

        # Set both the prediction and the expectation as title. Set the color of the result green or red.
        subplot.set_title('Predicted {0}'.format(predicted_values[i]),
                          fontsize=TITLE_FONT_SIZE)

    plt.show()  # Show the plot


# Open new plot containing the list of images and both their expected & predicted values as title
def plot_images_test_results(images, expected_values, predicted_values, max_column_qt=DEFAULT_MAX_COLUMN_QT):
    if len(images) <= 0:
        return  # TODO add error handling (no images)
    if not (len(images) == len(expected_values) == len(predicted_values)):
        return  # TODO add error handling (incoherent array sizes)

    column_qt = len(images) if len(images) < max_column_qt else max_column_qt  # Compute the required qt of columns
    row_qt = int(math.ceil(len(images) / column_qt))                           # Compute the required qt of rows

    for i in range(len(images)):
        subplot = plt.subplot(row_qt, column_qt, i+1)     # Add a subplot
        plt.axis('off')                                   # Disable the drawing of the axes
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))  # Set the image

        # Set both the prediction and the expectation as title. Set the color of the result green or red.
        subplot.set_title('Expected {0} | Predicted {1}'.format(expected_values[i], predicted_values[i]),
                          color="green" if expected_values[i] == predicted_values[i] else "red",
                          fontsize=TITLE_FONT_SIZE)

    plt.show()  # Show the plot
