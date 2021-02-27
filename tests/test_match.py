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

import tensorflow as tf
import torch
import numpy as np
from tensorguard.guard import TensorGuard

# ======== tensorflow =================


def test_matches_basic_numerical_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3])
    assert tg.matches(a, "1, 2, 3")
    assert not tg.matches(a, "1, 2, 4")
    assert not tg.matches(a, "1, 2, 3, 4")
    assert not tg.matches(a, "1, 2")


def test_matches_ignores_spaces_tensorflow():
    tg = TensorGuard()
    a = tf.ones([1, 2, 3])
    assert tg.matches(a, "1,2,3")
    assert tg.matches(a, "1 ,  2, 3   ")
    assert tg.matches(a, "1,  2,3 ")


def test_matches_named_dims_tensorflow():
    tg = TensorGuard(dims={"N": 24, "Z": 16})
    z = tf.ones([24, 16])
    assert tg.matches(z, "N, Z")
    assert tg.matches(z, "24, Z")
    assert not tg.matches(z, "N, N")


def test_matches_wildcards_tensorflow():
    tg = TensorGuard()
    z = tf.ones([1, 2, 4, 8])
    assert tg.matches(z, "1, 2, 4, *")
    assert tg.matches(z, "*, *, *, 8")
    assert not tg.matches(z, "*")
    assert not tg.matches(z, "*, *, *")

# ================= pytorch ==================

def test_matches_basic_numerical_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3])
    assert tg.matches(a, "1, 2, 3")
    assert not tg.matches(a, "1, 2, 4")
    assert not tg.matches(a, "1, 2, 3, 4")
    assert not tg.matches(a, "1, 2")


def test_matches_ignores_spaces_pytorch():
    tg = TensorGuard()
    a = torch.ones([1, 2, 3])
    assert tg.matches(a, "1,2,3")
    assert tg.matches(a, "1 ,  2, 3   ")
    assert tg.matches(a, "1,  2,3 ")


def test_matches_named_dims_pytorch():
    tg = TensorGuard(dims={"N": 24, "Z": 16})
    z = torch.ones([24, 16])
    assert tg.matches(z, "N, Z")
    assert tg.matches(z, "24, Z")
    assert not tg.matches(z, "N, N")


def test_matches_wildcards_pytorch():
    tg = TensorGuard()
    z = torch.ones([1, 2, 4, 8])
    assert tg.matches(z, "1, 2, 4, *")
    assert tg.matches(z, "*, *, *, 8")
    assert not tg.matches(z, "*")
    assert not tg.matches(z, "*, *, *")


# ================== numpy ===================

def test_matches_basic_numerical_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3])
    assert tg.matches(a, "1, 2, 3")
    assert not tg.matches(a, "1, 2, 4")
    assert not tg.matches(a, "1, 2, 3, 4")
    assert not tg.matches(a, "1, 2")


def test_matches_ignores_spaces_numpy():
    tg = TensorGuard()
    a = np.ones([1, 2, 3])
    assert tg.matches(a, "1,2,3")
    assert tg.matches(a, "1 ,  2, 3   ")
    assert tg.matches(a, "1,  2,3 ")


def test_matches_named_dims_numpy():
    tg = TensorGuard(dims={"N": 24, "Z": 16})
    z = np.ones([24, 16])
    assert tg.matches(z, "N, Z")
    assert tg.matches(z, "24, Z")
    assert not tg.matches(z, "N, N")


def test_matches_wildcards_numpy():
    tg = TensorGuard()
    z = np.ones([1, 2, 4, 8])
    assert tg.matches(z, "1, 2, 4, *")
    assert tg.matches(z, "*, *, *, 8")
    assert not tg.matches(z, "*")
    assert not tg.matches(z, "*, *, *")


# ================ global =====================

def test_matches_basic_numerical_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3])
    assert tg.matches(a, "1, 2, 3")
    assert not tg.matches(a, "1, 2, 4")
    assert not tg.matches(a, "1, 2, 3, 4")
    assert not tg.matches(a, "1, 2")


def test_matches_ignores_spaces_global():
    import tensorguard as tg; tg.reset()
    a = np.ones([1, 2, 3])
    assert tg.matches(a, "1,2,3")
    assert tg.matches(a, "1 ,  2, 3   ")
    assert tg.matches(a, "1,  2,3 ")


def test_matches_named_dims_global():
    tg = TensorGuard(dims={"N": 24, "Z": 16})
    z = np.ones([24, 16])
    assert tg.matches(z, "N, Z")
    assert tg.matches(z, "24, Z")
    assert not tg.matches(z, "N, N")


def test_matches_wildcards_global():
    import tensorguard as tg; tg.reset()
    z = np.ones([1, 2, 4, 8])
    assert tg.matches(z, "1, 2, 4, *")
    assert tg.matches(z, "*, *, *, 8")
    assert not tg.matches(z, "*")
    assert not tg.matches(z, "*, *, *")
