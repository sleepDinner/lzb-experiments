"""Compatibility shims for running the original IRIS0-SPAN code on Keras 2.15."""

import sys
import types


def apply_compat():
    import keras
    import tensorflow as tf
    from keras.layers import InputSpec, Layer

    try:
        from keras.src.layers.convolutional.base_conv import Conv as conv_base
    except Exception:
        from tensorflow.python.keras.layers.convolutional import Conv as conv_base

    if not hasattr(tf, "py_func"):
        tf.py_func = tf.compat.v1.py_func
    if not hasattr(tf.image, "resize_images"):
        tf.image.resize_images = tf.image.resize
    if not hasattr(tf, "extract_image_patches"):
        tf.extract_image_patches = tf.image.extract_patches
    if not getattr(tf.nn.conv2d, "_lzb_filter_compat", False):
        original_conv2d = tf.nn.conv2d

        def conv2d_compat(*args, **kwargs):
            if "filter" in kwargs and "filters" not in kwargs:
                kwargs["filters"] = kwargs.pop("filter")
            return original_conv2d(*args, **kwargs)

        conv2d_compat._lzb_filter_compat = True
        tf.nn.conv2d = conv2d_compat

    convolutional = types.ModuleType("keras.layers.convolutional")
    convolutional._Conv = conv_base
    sys.modules["keras.layers.convolutional"] = convolutional

    interfaces = types.ModuleType("keras.legacy.interfaces")
    interfaces.legacy_conv2d_support = lambda func: func
    legacy = types.ModuleType("keras.legacy")
    legacy.interfaces = interfaces
    sys.modules["keras.legacy"] = legacy
    sys.modules["keras.legacy.interfaces"] = interfaces

    engine = types.ModuleType("keras.engine")
    engine.InputSpec = InputSpec
    topology = types.ModuleType("keras.engine.topology")
    topology.Layer = Layer
    sys.modules["keras.engine"] = engine
    sys.modules["keras.engine.topology"] = topology

    return keras
