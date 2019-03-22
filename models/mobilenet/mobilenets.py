"""MobileNet v1 models for Keras.
MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.
MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 16 models from the paper can be built, with ImageNet weights provided.
The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.75, 0.5 and 0.25.
For each of these `alpha` values, weights for 4 different input image sizes
are provided (224, 192, 160, 128).
The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------
The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------
The weights for all 16 models are obtained and translated
from Tensorflow checkpoints found at
https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md
# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
"""
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
from keras.layers import DepthwiseConv2D
from keras.layers import add
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
# changed the next 3 lines from keras.applications to keras_applications for Keras version 2.2.4
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.inception_v3 import preprocess_input
from keras_applications.imagenet_utils import decode_predictions
from keras import backend as K

import tensorflow as tf

BASE_WEIGHT_PATH = ''
BASE_WEIGHT_PATH_V2 = ''


def relu6(x):
    return K.relu(x, max_value=6)


def MiniMobileNetV2(input_shape=None,
                    alpha=1.0,
                    expansion_factor=6,
                    depth_multiplier=1,
                    dropout=0.,
                    weight_decay=0.,
                    include_top=True,
                    weights=None,
                    input_tensor=None,
                    pooling=None,
                    classes=10):
    """Instantiates the MobileNet architecture.
    MobileNet V2 is from the paper:
    - [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.
    To load a MobileNet model via `load_model`, import the custom
    objects `relu6` and `DepthwiseConv2D` and pass them to the
    `custom_objects` parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6,
                       'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        expansion_factor: controls the expansion of the internal bottleneck
            blocks. Should be a positive integer >= 1
        depth_multiplier: depth multiplier for depthwise convolution
            (also called the resolution multiplier)
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
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.35`, `0.50`, `0.75`, `1.0`, `1.3` and `1.4` only.')

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            raise ValueError('If imagenet weights are being loaded, '
                             'input must have a static square shape (one of '
                             '(06, 96), (128,128), (160,160), (192,192), or '
                             '(224, 224)).Input shape provided = %s' % (input_shape,))

    if K.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, bn_epsilon=1e-3, bn_momentum=0.99, weight_decay=weight_decay)
    x = _depthwise_conv_block_v2(x, 16, alpha, 1, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.99,
                                 weight_decay=weight_decay, block_id=1)

    x = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, block_id=2,
                                 bn_epsilon=1e-3, bn_momentum=0.99, weight_decay=weight_decay, strides=(2, 2))
    x = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.99,
                                 weight_decay=weight_decay, block_id=3)

    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, block_id=4,
                                 bn_epsilon=1e-3, bn_momentum=0.99, weight_decay=weight_decay)
    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.99,
                                 weight_decay=weight_decay, block_id=5)
    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.99,
                                 weight_decay=weight_decay, block_id=6)

    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, block_id=7,
                                 bn_epsilon=1e-3, bn_momentum=0.99, weight_decay=weight_decay, strides=(2, 2))
    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.99,
                                 weight_decay=weight_decay, block_id=8)
    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.99,
                                 weight_decay=weight_decay, block_id=9)
    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.99,
                                 weight_decay=weight_decay, block_id=10)

    if alpha <= 1.0:
        penultimate_filters = 1280
    else:
        penultimate_filters = int(1280 * alpha)

    x = _conv_block(x, penultimate_filters, alpha=1.0, kernel=(1, 1), bn_epsilon=1e-3, bn_momentum=0.99,
                    block_id=18)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (penultimate_filters, 1, 1)
        else:
            shape = (1, 1, penultimate_filters)

        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(classes, (1, 1),
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay),
                   padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((classes,), name='reshape_2')(x)
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
    model = Model(inputs, x, name='mobilenetV2_%0.2f_%s' % (alpha, rows))

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 1.3:
            alpha_text = '1_3'
        elif alpha == 1.4:
            alpha_text = '1_4'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '3_5'

        if include_top:
            model_name = 'mobilenet_v2_%s_%d_tf.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH_V2 + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        else:
            model_name = 'mobilenet_v2_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH_V2 + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    if old_data_format:
        K.set_image_data_format(old_data_format)
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


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), bn_epsilon=1e-3,
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
    filters = filters * alpha
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


def _depthwise_conv_block_v2(inputs, pointwise_conv_filters, alpha, expansion_factor,
                             depth_multiplier=1, strides=(1, 1), bn_epsilon=1e-3,
                             bn_momentum=0.99, weight_decay=0.0, block_id=1):
    """Adds a depthwise convolution block V2.
    A depthwise convolution V2 block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        expansion_factor: controls the expansion of the internal bottleneck
            blocks. Should be a positive integer >= 1
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        bn_epsilon: Epsilon value for BatchNormalization
        bn_momentum: Momentum value for BatchNormalization
        block_id: Integer, a unique identification designating the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)
    depthwise_conv_filters = _make_divisible(input_shape[channel_axis] * expansion_factor)
    pointwise_conv_filters = _make_divisible(pointwise_conv_filters * alpha)

    if depthwise_conv_filters > input_shape[channel_axis]:
        x = Conv2D(depthwise_conv_filters, (1, 1),
                   padding='same',
                   use_bias=False,
                   strides=(1, 1),
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay),
                   name='conv_expand_%d' % block_id)(inputs)
        x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                               name='conv_expand_%d_bn' % block_id)(x)
        x = Activation(relu6, name='conv_expand_%d_relu' % block_id)(x)
    else:
        x = inputs

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        depthwise_initializer=initializers.he_normal(),
                        depthwise_regularizer=regularizers.l2(weight_decay),
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                           name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               kernel_initializer=initializers.he_normal(),
               kernel_regularizer=regularizers.l2(weight_decay),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                           name='conv_pw_%d_bn' % block_id)(x)

    if strides == (2, 2):
        return x
    else:
        if input_shape[channel_axis] == pointwise_conv_filters:

            x = add([inputs, x])

    return x


if __name__ == '__main__':
    import tensorflow as tf
    from keras import backend as K

    run_metadata = tf.RunMetadata()

    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)

        model = MiniMobileNetV2(input_tensor=tf.placeholder('float32', shape=(1, 224, 224, 3)), alpha=1.0)
        opt = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opt)

        opt = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        param_count = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opt)

        print('flops:', flops.total_float_ops)
        print('param count:', param_count.total_parameters)

        model.summary()
