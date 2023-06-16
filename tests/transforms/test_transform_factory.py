
import pytest
from src.transforms import TransformFactory
from src.transforms import BaseTransform


def test_transform_from_name():
    valid_name = "BaseTransform"
    transform = TransformFactory.transform_from_name(valid_name)
    assert isinstance(transform, BaseTransform)


def test_transform_from_name_when_invalid():
    invalid_name = "foo"
    with pytest.raises(ValueError):
        TransformFactory.transform_from_name(invalid_name)