# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import tensorflow as tf
import numpy as np

from tensorguard import ShapeError
from tensorguard import TensorGuard



def test_guard_raises_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3])
    with pytest.raises(ShapeError):
        tg.guard(a, "3, 2, 1")


def test_guard_infers_dimensions_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3])
    tg.guard(a, "A, B, C")
    assert tg.dims == {"A": 1, "B": 2, "C": 3}


def test_guard_infers_dimensions_complex_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3])
    tg.guard(a, "A, B*2, A+C")
    assert tg.dims == {"A": 1, "B": 1, "C": 2}


def test_guard_infers_dimensions_operator_priority_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 8])
    tg.guard(a, "A, B, A+C*2+1")
    assert tg.dims == {"A": 1, "B": 2, "C": 3}


def test_guard_raises_complex_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3])
    with pytest.raises(ShapeError):
        tg.guard(a, "A, B, B")


def test_guard_raises_inferred_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3])
    b = tf.ones([3, 2, 5])
    tg.guard(a, "A, B, C")
    with pytest.raises(ShapeError):
        tg.guard(b, "C, B, A")


def test_guard_ignores_wildcard_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3])
    tg.guard(a, "*, *, 3")
    assert tg.dims == {}


def test_guard_dynamic_shape_tensorflow():
    tg = TensorGuard()
    with pytest.raises(ShapeError):
        tg.guard([None, 2, 3], "C, B, A")

    tg.guard([None, 2, 3], "?, B, A")
    tg.guard([1, 2, 3], "C?, B, A")
    tg.guard([None, 2, 3], "C?, B, A")


def test_guard_ellipsis_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3, 4, 5])
    tg.guard(a, "...")
    tg.guard(a, "..., 5")
    tg.guard(a, "..., 4, 5")
    tg.guard(a, "1, ...")
    tg.guard(a, "1, 2, ...")
    tg.guard(a, "1, 2, ..., 4, 5")
    tg.guard(a, "1, 2, 3, ..., 4, 5")

    with pytest.raises(ShapeError):
        tg.guard(a, "1, 2, 3, 4, 5, 6,...")

    with pytest.raises(ShapeError):
        tg.guard(a, "..., 1, 2, 3, 4, 5, 6")


def test_guard_ellipsis_infer_dims_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3, 4, 5])
    tg.guard(a, "A, B, ..., C")
    assert tg.dims == {"A": 1, "B": 2, "C": 5}


#  ============ pytorch ==================

def test_guard_raises_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3])
    with pytest.raises(ShapeError):
        tg.guard(a, "3, 2, 1")


def test_guard_infers_dimensions_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3])
    tg.guard(a, "A, B, C")
    assert tg.dims == {"A": 1, "B": 2, "C": 3}


def test_guard_infers_dimensions_complex_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3])
    tg.guard(a, "A, B*2, A+C")
    assert tg.dims == {"A": 1, "B": 1, "C": 2}


def test_guard_infers_dimensions_operator_priority_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 8])
    tg.guard(a, "A, B, A+C*2+1")
    assert tg.dims == {"A": 1, "B": 2, "C": 3}


def test_guard_raises_complex_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3])
    with pytest.raises(ShapeError):
        tg.guard(a, "A, B, B")


def test_guard_raises_inferred_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3])
    b = torch.ones([3, 2, 5])
    tg.guard(a, "A, B, C")
    with pytest.raises(ShapeError):
        tg.guard(b, "C, B, A")


def test_guard_ignores_wildcard_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3])
    tg.guard(a, "*, *, 3")
    assert tg.dims == {}


def test_guard_dynamic_shape_pytorch():
    tg = TensorGuard()
    with pytest.raises(ShapeError):
        tg.guard([None, 2, 3], "C, B, A")

    tg.guard([None, 2, 3], "?, B, A")
    tg.guard([1, 2, 3], "C?, B, A")
    tg.guard([None, 2, 3], "C?, B, A")


def test_guard_ellipsis_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3, 4, 5])
    tg.guard(a, "...")
    tg.guard(a, "..., 5")
    tg.guard(a, "..., 4, 5")
    tg.guard(a, "1, ...")
    tg.guard(a, "1, 2, ...")
    tg.guard(a, "1, 2, ..., 4, 5")
    tg.guard(a, "1, 2, 3, ..., 4, 5")

    with pytest.raises(ShapeError):
        tg.guard(a, "1, 2, 3, 4, 5, 6,...")

    with pytest.raises(ShapeError):
        tg.guard(a, "..., 1, 2, 3, 4, 5, 6")


def test_guard_ellipsis_infer_dims_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3, 4, 5])
    tg.guard(a, "A, B, ..., C")
    assert tg.dims == {"A": 1, "B": 2, "C": 5}



# ================= numpy =======================


def test_guard_raises_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3])
    with pytest.raises(ShapeError):
        tg.guard(a, "3, 2, 1")


def test_guard_infers_dimensions_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3])
    tg.guard(a, "A, B, C")
    assert tg.dims == {"A": 1, "B": 2, "C": 3}


def test_guard_infers_dimensions_complex_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3])
    tg.guard(a, "A, B*2, A+C")
    assert tg.dims == {"A": 1, "B": 1, "C": 2}


def test_guard_infers_dimensions_operator_priority_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 8])
    tg.guard(a, "A, B, A+C*2+1")
    assert tg.dims == {"A": 1, "B": 2, "C": 3}


def test_guard_raises_complex_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3])
    with pytest.raises(ShapeError):
        tg.guard(a, "A, B, B")


def test_guard_raises_inferred_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3])
    b = np.ones([3, 2, 5])
    tg.guard(a, "A, B, C")
    with pytest.raises(ShapeError):
        tg.guard(b, "C, B, A")


def test_guard_ignores_wildcard_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3])
    tg.guard(a, "*, *, 3")
    assert tg.dims == {}


def test_guard_dynamic_shape_numpy():
    tg = TensorGuard()
    with pytest.raises(ShapeError):
        tg.guard([None, 2, 3], "C, B, A")

    tg.guard([None, 2, 3], "?, B, A")
    tg.guard([1, 2, 3], "C?, B, A")
    tg.guard([None, 2, 3], "C?, B, A")


def test_guard_ellipsis_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3, 4, 5])
    tg.guard(a, "...")
    tg.guard(a, "..., 5")
    tg.guard(a, "..., 4, 5")
    tg.guard(a, "1, ...")
    tg.guard(a, "1, 2, ...")
    tg.guard(a, "1, 2, ..., 4, 5")
    tg.guard(a, "1, 2, 3, ..., 4, 5")

    with pytest.raises(ShapeError):
        tg.guard(a, "1, 2, 3, 4, 5, 6,...")

    with pytest.raises(ShapeError):
        tg.guard(a, "..., 1, 2, 3, 4, 5, 6")


def test_guard_ellipsis_infer_dims_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3, 4, 5])
    tg.guard(a, "A, B, ..., C")
    assert tg.dims == {"A": 1, "B": 2, "C": 5}

# ========================= global =======================

def test_guard_raises_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3])
    with pytest.raises(ShapeError):
        tg.guard(a, "3, 2, 1")


def test_guard_infers_dimensions_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3])
    tg.guard(a, "A, B, C")
    assert tg.get_dims() == {"A": 1, "B": 2, "C": 3}


def test_guard_infers_dimensions_complex_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3])
    tg.guard(a, "A, B*2, A+C")
    assert tg.get_dims() == {"A": 1, "B": 1, "C": 2}, f'{tg.get_dims()}' + ' != {"A": 1, "B": 1, "C": 2}'


def test_guard_infers_dimensions_operator_priority_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 8])
    tg.guard(a, "A, B, A+C*2+1")
    assert tg.get_dims() == {"A": 1, "B": 2, "C": 3}


def test_guard_raises_complex_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3])
    with pytest.raises(ShapeError):
        tg.guard(a, "A, B, B")


def test_guard_raises_inferred_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3])
    b = np.ones([3, 2, 5])
    tg.guard(a, "A, B, C")
    with pytest.raises(ShapeError):
        tg.guard(b, "C, B, A")


def test_guard_ignores_wildcard_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3])
    tg.guard(a, "*, *, 3")
    assert tg.get_dims() == {}


def test_guard_dynamic_shape_global():
    import tensorguard as tg; tg.reset()
    with pytest.raises(ShapeError):
        tg.guard([None, 2, 3], "C, B, A")

    tg.guard([None, 2, 3], "?, B, A")
    tg.guard([1, 2, 3], "C?, B, A")
    tg.guard([None, 2, 3], "C?, B, A")


def test_guard_ellipsis_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3, 4, 5])
    tg.guard(a, "...")
    tg.guard(a, "..., 5")
    tg.guard(a, "..., 4, 5")
    tg.guard(a, "1, ...")
    tg.guard(a, "1, 2, ...")
    tg.guard(a, "1, 2, ..., 4, 5")
    tg.guard(a, "1, 2, 3, ..., 4, 5")

    with pytest.raises(ShapeError):
        tg.guard(a, "1, 2, 3, 4, 5, 6,...")

    with pytest.raises(ShapeError):
        tg.guard(a, "..., 1, 2, 3, 4, 5, 6")


def test_guard_ellipsis_infer_dims_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3, 4, 5])
    tg.guard(a, "A, B, ..., C")
    assert tg.get_dims() == {"A": 1, "B": 2, "C": 5}
