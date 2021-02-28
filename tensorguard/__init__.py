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

__version__ = "0.1.1"

__author__ = "Michele De Vita"
__author_email__ = "mik3dev@gmail.com"

__url__ = "https://github.com/Michedev/shapeguard"

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


def guard(tensor: Union[ShapedTensor, List[int]], template: str):
    inferred_dims = tools.guard(tensor, template, __tg.dims)
    __tg.dims.update(inferred_dims)
    return tensor


def reshape(tensor: Union[ShapedTensor, List[int]], template: str):
    return tools.reshape(tensor, template, __tg.dims)


def evaluate(template: str, **kwargs) -> List[Optional[int]]:
    local_dims = copy(__tg.dims)
    local_dims.update(kwargs)
    return tools.evaluate(template, local_dims)


def get_dims(item: Optional[str] = None) -> Union[Dict[str, int], List[Optional[int]]]:
    if item is None:
        return __tg.dims
    else:
        return tools.evaluate(item, __tg.dims)


def get_dim(item: str) -> Any:
    try:
        return __tg.dims[item]
    except KeyError:
        raise AttributeError(item)


def set_dim(key: str, value: Any):
    try:
        __tg.dims[key] = value
    except KeyError:
        raise AttributeError(key)


def del_dim(item: str):
    try:
        del __tg.dims[item]
    except KeyError:
        raise AttributeError(item)


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
    "del_dim",
    "get_dims"
)
