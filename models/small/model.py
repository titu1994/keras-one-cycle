from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
import math

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import add, concatenate
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.inception_v3 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

import tensorflow as tf

BASE_WEIGHT_PATH = ''
BASE_WEIGHT_PATH_V2 = ''


def relu6(x):
    return K.relu(x, max_value=6)


def MiniVGG(input_shape=None,
            dropout=0.,
            weight_decay=0.,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None,
            classes=10):
    """Mini VGG is a 3 layer small CNN network

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        dropout: dropout rate
        weight_decay: Weight decay factor.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    if K.backend() != 'tensorflow':
        raise RuntimeError('Only Tensorflow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if K.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top or weights)
    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
        channel_axis = -1
    else:
        row_axis, col_axis = (1, 2)
        channel_axis = 1

    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, bn_epsilon=1e-3, bn_momentum=0.99, weight_decay=weight_decay, block_id=1)
    x = _conv_block(x, 64, bn_epsilon=1e-3, bn_momentum=0.99, weight_decay=weight_decay, block_id=2)
    x = _conv_block(x, 96, bn_epsilon=1e-3, bn_momentum=0.99, weight_decay=weight_decay, block_id=3)

    if include_top:

        # Fast.ai's Concat Pooling
        a = GlobalAveragePooling2D()(x)
        b = GlobalMaxPooling2D()(x)

        x = concatenate([a, b], axis=channel_axis)

        x = Dropout(dropout, name='dropout')(x)
        x = Dense(classes, activation='softmax', name='conv_preds')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mini_vgg_%0.2f_%s')
    return model


# taken from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/conv_blocks.py
def _make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1), bn_epsilon=1e-3,
                bn_momentum=0.99, weight_decay=0., block_id=1):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        bn_epsilon: Epsilon value for BatchNormalization
        bn_momentum: Momentum value for BatchNormalization
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = _make_divisible(filters)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               kernel_initializer=initializers.he_normal(),
               kernel_regularizer=regularizers.l2(weight_decay),
               name='conv%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                           name='conv%d_bn' % block_id)(x)
    return Activation(relu6, name='conv%d_relu' % block_id)(x)


if __name__ == '__main__':
    import tensorflow as tf
    from keras import backend as K

    run_metadata = tf.RunMetadata()

    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)

        model = MiniVGG(input_tensor=tf.placeholder('float32', shape=(1, 32, 32, 3)))
        opt = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opt)

        opt = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        param_count = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opt)

        print('flops:', flops.total_float_ops)
        print('param count:', param_count.total_parameters)

        model.summary()
