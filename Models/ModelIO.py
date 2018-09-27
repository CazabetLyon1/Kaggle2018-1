from keras.models import model_from_json


# Load a model from it's directory (path) and optionally load it's weights (network_name)
def load_model(path, network_name=None):
    loaded_model_json = None                                    # Initialise empty var
    with open('{0}Model.json'.format(path), 'r') as json_file:  # Try to open file
        loaded_model_json = json_file.read()                    # Load data
        json_file.close()                                       # Free file

    if loaded_model_json is None:                               # Ensure data loaded correctly
        return  # TODO Handle file read error

    loaded_model = model_from_json(loaded_model_json)           # De-serialize model

    if network_name is not None:
        loaded_model.load_weights('{0}{1}.h5'.format(path, network_name))  # Load the network weights

    return loaded_model


# Save a model in the directory (path) and optionally save it's weights (network_name)
def save_model(model, path, network_name=None):
    model_json = model.to_json()                                # Serialize model to JSON
    with open('{0}Model.json'.format(path), 'w') as json_file:  # Open a clean json file at location
        json_file.write(model_json)                             # Save model in file
        json_file.close()                                       # Free file

    if network_name is not None:
        model.save_weights('{0}{1}.h5'.format(path, network_name))  # Save the network weights
