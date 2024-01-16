import constants
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets.mnist import load_data
import re
from typing import Dict, List, Optional, Text, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors


from keras.utils import plot_model

import matplotlib.pyplot as plt
from matplotlib import colors


def visualize(dataset):
  inputs, labels = next(iter(dataset))
  TITLES = [
    'Elevation',
    'Wind\ndirection',
    'Wind\nvelocity',
    'Min\ntemp',
    'Max\ntemp',
    'Humidity',
    'Precip',
    'Drought',
    'Vegetation',
    'Population\ndensity',
    'Energy\nrelease\ncomponent',
    'Previous\nfire\nmask',
    'Fire\nmask'
  ]
  # Number of rows of data samples to plot
  n_rows = 5
  # Number of data variables
  n_features = inputs.shape[3]
  # Variables for controllong the color map for the fire masks
  CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
  BOUNDS = [-1, -0.1, 0.001, 1]
  NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

  fig = plt.figure(figsize=(15,6.5))

  for i in range(n_rows):
    for j in range(n_features + 1):
      plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
      if i == 0:
        plt.title(TITLES[j], fontsize=13)
      if j < n_features - 1:
        plt.imshow(inputs[i, :, :, j], cmap='viridis')
      if j == n_features - 1:
        plt.imshow(inputs[i, :, :, -1], cmap=CMAP, norm=NORM)
      if j == n_features:
        plt.imshow(labels[i, :, :, 0], cmap=CMAP, norm=NORM)
      plt.axis('off')
  plt.tight_layout()
def center_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    
    central_fraction = sample_size / input_img.shape[0]
    input_img = tf.image.central_crop(input_img, central_fraction)
    output_img = tf.image.central_crop(output_img, central_fraction)
    return input_img, output_img

def random_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor]:

    combined = tf.concat([input_img, output_img], axis=2)
    combined = tf.image.random_crop(
        combined,
        [sample_size, sample_size, num_in_channels + num_out_channels])
    input_img = combined[:, :, 0:num_in_channels]
    output_img = combined[:, :, -num_out_channels:]
    return input_img, output_img

def _get_base_key(key: Text) -> Text:
    """Extracts the base key from the provided key.
    ​
    Earth Engine exports TFRecords containing each data variable with its
    corresponding variable name. In the case of time sequences, the name of the
    data variable is of the form 'variable_1', 'variable_2', ..., 'variable_n',
    where 'variable' is the name of the variable, and n the number of elements
    in the time sequence. Extracting the base key ensures that each step of the
    time sequence goes through the same normalization steps.
    The base key obeys the following naming pattern: '([a-zA-Z]+)'
    For instance, for an input key 'variable_1', this function returns 'variable'.
    For an input key 'variable', this function simply returns 'variable'.
    ​
    Args:
        key: Input key.
    ​
    Returns:
        The corresponding base key.
    ​
    Raises:
        ValueError when `key` does not match the expected pattern.
    """
    match = re.match(r'([a-zA-Z]+)', key)
    if match:
        return match.group(1)
    raise ValueError(
        'The provided key does not match the expected pattern: {}'.format(key))

def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    """Clips and rescales inputs with the stats corresponding to `key`.

    Args:
        inputs: Inputs to clip and rescale.
        key: Key describing the inputs.

    Returns:
        Clipped and rescaled input.

    Raises:
        ValueError if there are no data statistics available for `key`.
    """
    base_key = _get_base_key(key)
    if base_key not in constants.DATA_STATS:
        raise ValueError(
            'No data statistics available for the requested key: {}.'.format(key))
    min_val, max_val, _, _ = constants.DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))


def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    """Clips and normalizes inputs with the stats corresponding to `key`.

    Args:
        inputs: Inputs to clip and normalize.
        key: Key describing the inputs.

    Returns:
        Clipped and normalized input.

    Raises:
        ValueError if there are no data statistics available for `key`.
    """
    base_key = _get_base_key(key)
    if base_key not in constants.DATA_STATS:
        raise ValueError(
            'No data statistics available for the requested key: {}.'.format(key))
    min_val, max_val, mean, std = constants.DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs = inputs - mean
    return tf.math.divide_no_nan(inputs, std)


def _get_features_dict(
    sample_size: int,
    features: List[Text],
) -> Dict[Text, tf.io.FixedLenFeature]:
    """Creates a features dictionary for TensorFlow IO.

    Args:
        sample_size: Size of the input tiles (square).
        features: List of feature names.

    Returns:
        A features dictionary for TensorFlow IO.
    """
    sample_shape = [sample_size, sample_size]
    features = set(features)
    columns = [
        tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32)
        for _ in features
    ]
    return dict(zip(features, columns))


def _parse_fn(
    example_proto: tf.train.Example, data_size: int, sample_size: int,
    num_in_channels: int, clip_and_normalize: bool,
    clip_and_rescale: bool, random_crop: bool, center_crop: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reads a serialized example.

    Args:
        example_proto: A TensorFlow example protobuf.
        data_size: Size of tiles (square) as read from input files.
        sample_size: Size the tiles (square) when input into the model.
        num_in_channels: Number of input channels.
        clip_and_normalize: True if the data should be clipped and normalized.
        clip_and_rescale: True if the data should be clipped and rescaled.
        random_crop: True if the data should be randomly cropped.
        center_crop: True if the data should be cropped in the center.

    Returns:
        (input_img, output_img) tuple of inputs and outputs to the ML model.
    """
    if (random_crop and center_crop):
        raise ValueError('Cannot have both random_crop and center_crop be True')
    input_features, output_features = constants.INPUT_FEATURES, constants.OUTPUT_FEATURES
    feature_names = input_features + output_features
    features_dict = _get_features_dict(data_size, feature_names)
    features = tf.io.parse_single_example(example_proto, features_dict)

    if clip_and_normalize:
        inputs_list = [
            _clip_and_normalize(features.get(key), key) for key in input_features
        ]
    elif clip_and_rescale:
        inputs_list = [
            _clip_and_rescale(features.get(key), key) for key in input_features
        ]
    else:
        inputs_list = [features.get(key) for key in input_features]

    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])

    outputs_list = [features.get(key) for key in output_features]
    assert outputs_list, 'outputs_list should not be empty'
    outputs_stacked = tf.stack(outputs_list, axis=0)

    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    assert len(outputs_stacked.shape) == 3, ('outputs_stacked should be rank 3'
                                                'but dimensions of outputs_stacked'
                                                f' are {outputs_stacked_shape}')
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(
            input_img, output_img, sample_size, num_in_channels, 1)
    if center_crop:
        input_img, output_img = center_crop_input_and_output_images(
            input_img, output_img, sample_size)
    return input_img, output_img
def reshape_dataset(example):
            inputs = tf.transpose(example['inputs'], perm=[0, 3, 1, 2])
            masks = tf.transpose(example['masks'], perm=[0, 3, 1, 2])
            masks = tf.squeeze(masks, axis=[1])
            return {'labels': masks, 'pixel_values': inputs}
        

def get_dataset(file_pattern: Text, data_size: int, sample_size: int,
                batch_size: int, num_in_channels: int, compression_type: Text,
                clip_and_normalize: bool, clip_and_rescale: bool,
                random_crop: bool, center_crop: bool, transformer_shape: bool) -> tf.data.Dataset:
    """Gets the dataset from the file pattern.

    Args:
        file_pattern: Input file pattern.
        data_size: Size of tiles (square) as read from input files.
        sample_size: Size the tiles (square) when input into the model.
        batch_size: Batch size.
        num_in_channels: Number of input channels.
        compression_type: Type of compression used for the input files.
        clip_and_normalize: True if the data should be clipped and normalized, False
        otherwise.
        clip_and_rescale: True if the data should be clipped and rescaled, False
        otherwise.
        random_crop: True if the data should be randomly cropped.
        center_crop: True if the data shoulde be cropped in the center.

    Returns:
        A TensorFlow dataset loaded from the input file pattern, with features
        described in the constants, and with the shapes determined from the input
        parameters to this function.
    """
    if (clip_and_normalize and clip_and_rescale):
        raise ValueError('Cannot have both normalize and rescale.')
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: _parse_fn(  # pylint: disable=g-long-lambda
            x, data_size, sample_size, num_in_channels, clip_and_normalize,
            clip_and_rescale, random_crop, center_crop),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


    def filter_function(data_1, data_2):
        has_negative_one = tf.reduce_any(tf.equal(data_2, -1))
        return tf.math.logical_not(has_negative_one)

    dataset = dataset.filter(filter_function)
        
    dataset = dataset.map(
            lambda x, y: {'inputs': x, 'masks': y}
        )
    
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if (transformer_shape):
        dataset = dataset.map(reshape_dataset)
    else:
        dataset=dataset.map(
            lambda x, y: {'input_1': x, 'masks':y}
        )
    return dataset


#Eventually would like a function that takes in an input array of dimensions nxn,
#outputs an array that gives avg of each cell's neighbors:
def avg_neighbors(array_in):
    #Check input
    if array_in.shape[0] != array_in.shape[1]:
        raise Exception('Only square arrays make sense here, since you\'re analyzing square arrays.')
    #Maybe should also do type-checking, but leave it for now.

    #Prepare the output array:
    n = array_in.shape[0]
    array_out = np.zeros((n,n))

    #Guess who doesn't know how to do signal processing in Python...

    for i in range(n):
        for j in range(n):
            if i == 0:
                #Upper edge
                if j == 0:
                    #Upper left corner
                    sum_neighbors = array_in[i+1, j] + array_in[i, j+1] + array_in[i+1, j+1]
                    avg = sum_neighbors/3

                elif j == (n-1):
                    #Upper right corner
                    sum_neighbors = array_in[i, j-1] + array_in[i+1, j] + array_in[i+1, j-1]
                    avg = sum_neighbors/3

                else:
                    #Upper edge except corners
                    sum_neighbors = array_in[i, j-1] + array_in[i+1, j] + array_in[i, j+1] + array_in[i+1, j-1] + array_in[i+1, j+1]
                    avg = sum_neighbors/5

            elif i == (n-1):
                #Lower edge
                if j == 0:
                    #Lower left corner
                    sum_neighbors = array_in[i-1, j] + array_in[i, j+1] + array_in[i-1, j+1]
                    avg = sum_neighbors/3

                elif j == (n-1):
                    #Lower right corner
                    sum_neighbors = array_in[i, j-1] + array_in[i-1, j] + array_in[i-1, j-1]
                    avg = sum_neighbors/3

                else:
                    #Lower edge except corners
                    sum_neighbors = array_in[i, j-1] + array_in[i, j+1] + array_in[i-1, j] + array_in[i-1, j-1] + array_in[i-1, j+1]
                    avg = sum_neighbors/5

            else:
                if j == 0:
                    #Left edge except corners
                    sum_neighbors = array_in[i-1, j] + array_in[i+1, j] + array_in[i, j+1] + array_in[i-1, j+1] + array_in[i+1, j+1]
                    avg = sum_neighbors/5

                elif j == (n-1):
                    #Right edge except corners
                    sum_neighbors = array_in[i-1, j] + array_in[i+1, j] + array_in[i, j-1] + array_in[i-1, j-1] + array_in[i+1, j-1]
                    avg = sum_neighbors/5

                else:
                    #Not on any edge or corner
                    sum_neighbors = array_in[i, j+1] + array_in[i, j-1] + array_in[i-1, j] + array_in[i+1, j] + \
                                    array_in[i-1, j-1] + array_in[i-1, j+1] + array_in[i+1, j-1] + array_in[i+1, j+1]
                    avg = sum_neighbors/8


            array_out[i,j] = avg
            #/for loop body

    return array_out


def elim_uncertain(prev_fire_mask_batch):

    prev_masks_array = np.array(prev_fire_mask_batch)
    num_imgs, rows, cols = prev_masks_array.shape

    #Build the array of certain data AND SAVE THE INDICES
    first_find_flag = 1
    count = 0
    indices = []

    for img_num in range(num_imgs):
        fire_mask = prev_fire_mask_batch[img_num, :, :] #grab the "working fire mask" off the pile

        if (np.all( np.invert(fire_mask == -1) )): #If no missing data, condition is TRUE.
            count += 1
            indices.append(img_num)
            if first_find_flag == 1: #If you need to start the array
                certain_prev_fire_masks_batch = fire_mask
                first_find_flag = 0  #Remember to turn the flag off!
            else:
                certain_prev_fire_masks_batch = np.dstack((certain_prev_fire_masks_batch, fire_mask))

    return certain_prev_fire_masks_batch, indices

#Function to extract only the labels (i.e. current fire masks) from the certain observations
def extract_certain_labels(certain_indices, og_labels):

    for i, index in enumerate(certain_indices):
        if i == 0:
            extracted_labels = og_labels[index,:,:,:] #the og_labels dimensions are batch_size by sidelength by sidelength by 1
        else:
            #labels
            extracted_labels = np.concatenate((extracted_labels, og_labels[index,:,:,:]), axis=2)

    return extracted_labels

#Create multi-D array of neighbor fire values
def avg_neighbor_batch(batch_in):
    rows, cols, batch_size = batch_in.shape #ordering of dimensions here meant to be compatible with elim_uncertain and extract_certain_labels
    batch_out = np.zeros((rows, cols, batch_size))
    for i in range(batch_size):
        working_arr = batch_in[:,:,i]
        avgd_arr = avg_neighbors(working_arr)
        batch_out[:,:,i] =  avgd_arr
    #/for loop

    return batch_out