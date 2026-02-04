# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This python module contains ShapeGuard."""
from copy import copy
from typing import Optional, List, Any, Union, Dict

from tensorguard import tools
from tensorguard.exception import ShapeError
from tensorguard.guard import TensorGuard

__version__ = "1.0.2"

__author__ = "Michele De Vita"
__author_email__ = "mik3dev@gmail.com"

__url__ = "https://github.com/Michedev/tensorguard"

from tensorguard.tools import ShapedTensor

__tg = TensorGuard()


def reset():
    """
    Reset global tensorguard
    """
    global __tg
    __tg = TensorGuard()


def matches(tensor: Union[ShapedTensor, List[int]], template: str) -> bool:
    """
    Return True if tensor shape matches template
    """
    return tools.matches(tensor, template, __tg.dims)


def guard(tensor: Union[ShapedTensor, List[int]], template: str) -> Union[ShapedTensor, List[int]]:
    """
    Check if tensor shape matches template. If not, raise ShapeError
    :param tensor: Tensor or list of integers
    :param template: Template string that should match tensor shape
    :type template: str
    :return: input tensor
    """
    inferred_dims = tools.guard(tensor, template, __tg.dims)
    __tg.dims.update(inferred_dims)
    return tensor


def reshape(tensor: Union[ShapedTensor, List[int]], template: str):
    return tools.reshape(tensor, template, __tg.dims)


def evaluate(template: str, **kwargs) -> List[Optional[int]]:
    local_dims = copy(__tg.dims)
    local_dims.update(kwargs)
    return tools.evaluate(template, local_dims)


def get_dims(template: Optional[str] = None) -> Union[Dict[str, int], List[Optional[int]]]:
    """
    If template is None return dictionary of all {token_shape: token_value}.
    If token template is provided returns the corresponding list of token values.

    Example:

    >>> import tensorguard as tg
    >>> tg.get_dims("B, C, H, W")
    [16, 3, 224, 224]
    >>> tg.get_dims()
    {'B': 32, 'C': 3, 'H': 224, 'W': 224}

    :param template: Optional template value
    :type template: 
    :return: 
    :rtype: 
    """
    if template is None:
        return __tg.dims
    else:
        return tools.evaluate(template, __tg.dims)


def get_dim(item: str) -> Any:
    """
    Return corresponding value to shape token. If not exists, raise KeyError.
    :param item:
    :type item:
    :return:
    :rtype:
    """
    try:
        return __tg.dims[item]
    except KeyError:
        raise KeyError(item)


def safe_get_dim(item: str) -> Any:
    """
    Return corresponding value to shape token. If not exists doesn't raise an Exception
    :param item: the shape token
    :type item: str
    :return: shape token value
    """
    if has_dim(item):
        return get_dim(item)


def has_dim(key: str) -> bool:
    return key in __tg.dims


def set_dim(key: str, value: Any):
    """
    Set dimension in the global tensorguard guardian.
    :param key: the key to set
    :type key: str
    :param value: the value corresponding to the key
    :type value:
    :return: None
    """
    __tg.dims[key] = value

def set_dims(**kwargs):
    """
    Set multiple dimensions in the global tensorguard guardian.

    Example:

    >>> import tensorguard as tg
    >>> tg.set_dims(B=16, W=32, H=32, C=3)
    """
    for key, value in kwargs.items():
        set_dim(key, value)

def del_dim(item: str):
    """
    Delete a shape token. If not exists, raise KeyError.
    :param key:
    :type key: str
    :return: None
    """
    try:
        del __tg.dims[item]
    except KeyError:
        raise KeyError(item)


def safe_del_dim(key: str):
    """
    Delete a key only if exists. If not exists does not raise exceptions.
    :param key:
    :type key: str
    :return: None
    """
    if has_dim(key):
        del_dim(key)


def clear_dims():
    """
    Remove all the shape tokens
    """
    keys = list(__tg.dims.keys())
    for k in keys:
        del_dim(k)

__all__ = (
    "TensorGuard",
    "__version__",
    "__author__",
    "__author_email__",
    "ShapeError",
    "guard",
    "matches",
    "reshape",
    "evaluate",
    "get_dim",
    "set_dim",
    "set_dims",
    "safe_get_dim",
    "has_dim",
    "del_dim",
    "safe_del_dim",
    "get_dims",
    "clear_dims",
)
