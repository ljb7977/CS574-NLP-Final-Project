"""
This module servers as a library of API checks used as assertions during
constructing the computational graph.
"""


from typing import List, Optional, Iterable

import tensorflow as tf

from neuralmonkey.logging import log, debug, warn
from neuralmonkey.dataset import Dataset
from neuralmonkey.runners.base_runner import BaseRunner


class CheckingException(Exception):
    pass


def check_dataset_and_coders(dataset: Dataset,
                             runners: Iterable[BaseRunner]) -> None:
    # pylint: disable=protected-access

    data_list = []
    for runner in runners:
        for c in runner.all_coders:
            if hasattr(c, "data_id"):
                data_list.append((c.data_id, c))
            elif hasattr(c, "data_ids"):
                data_list.extend([(d, c) for d in c.data_ids])
            else:
                warn(("Coder: {} does not have"
                      "a data attribute").format(c))

    debug("Found series: {}".format(str(data_list)), "checking")
    missing = []

    for (serie, coder) in data_list:
        if not dataset.has_series(serie):
            log("dataset {} does not have serie {}".format(
                dataset.name, serie))
            missing.append((coder, serie))

    if len(missing) > 0:
        formated = ["{} ({}, {}.{})" .format(serie, cod.name,
                                             cod.__class__.__module__,
                                             cod.__class__.__name__)
                    for cod, serie in missing]

        raise CheckingException("Dataset '{}' is mising series {}:"
                                .format(dataset.name, ", ".join(formated)))


def assert_shape(tensor: tf.Tensor,
                 expected_shape: List[Optional[int]]) -> None:
    """Check shape of a tensor.

    Args:
        tensor: Tensor to be chcecked.
        expected_shape: Expected shape where `None` means the same as in TF and
            `-1` means not checking the dimension.
    """

    shape_list = tensor.get_shape().as_list()

    if len(shape_list) != len(expected_shape):
        raise CheckingException(
            "Tensor '{}' with shape {} should have {} dimensions.".format(
                tensor.name, shape_list, len(expected_shape)))

    mismatching_dims = []
    for i, (real, expected) in enumerate(zip(shape_list, expected_shape)):
        if expected != -1 and real != expected:
            mismatching_dims.append(i)

    if mismatching_dims:
        expected_str = ", ".join(
            "?" if x == -1 else str(x) for x in expected_shape)
        raise CheckingException(
            ("Shape mismatch of {} in dimensions: {}. "
             "Shape was {}, but should be [{}]").format(
                 tensor.name,
                 ", ".join(str(d) for d in mismatching_dims),
                 shape_list, expected_str))


def assert_same_shape(tensor_a: tf.Tensor, tensor_b: tf.Tensor) -> None:
    """Check if two tensors have the same shape."""

    shape_a = tensor_a.get_shape().as_list()
    shape_b = tensor_b.get_shape().as_list()

    if len(shape_a) != len(shape_b):
        raise CheckingException(
            ("Tensor '{}' has {} dimensions and tensor '{}' has {} "
             "dimension, but should have the same shape.").format(
                 tensor_a.name, len(shape_a), tensor_b.name, len(shape_b)))

    mismatching_dims = []
    for i, (size_a, size_b) in enumerate(zip(shape_a, shape_b)):
        if size_a != size_b:
            mismatching_dims.append(i)

    if mismatching_dims:
        raise CheckingException(
            ("Shape mismatch of '{}' and '{}' in dimensions: {}. "
             "Shapes were {} and {}").format(
                 tensor_a.name, tensor_b.name,
                 ", ".join(str(d) for d in mismatching_dims),
                 shape_a, shape_b))
