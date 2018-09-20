import math
import matplotlib.pyplot as plt


def plot_numbers_images(numbers, max_column_qt=4):
    if len(numbers) == 0:
        return  # TODO add error handling (no images)
    if len(numbers) > 4*9:
        return  # TODO add error handling (to much images)

    column_qt = len(numbers) if len(numbers) < max_column_qt else max_column_qt  # Compute the required qt of columns
    row_qt = int(math.ceil(len(numbers) / column_qt))                            # Compute the required qt of rows

    for y in range(len(numbers)):
        plt.subplot(row_qt, column_qt, y+1)                # Add a subplot
        plt.imshow(numbers[y], cmap=plt.get_cmap('gray'))  # Associate the subplot with a number

    plt.show()  # Show the plot
