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

from shapeguard import tools
from shapeguard.exception import ShapeError
from shapeguard.guard import ShapeGuard

__version__ = "0.1.0"

__author__ = "Michele De Vita"
__author_email__ = "mik3dev@gmail.com"

__url__ = "https://github.com/Michedev/shapeguard"

from shapeguard.tools import ShapedTensor

__sg = ShapeGuard()


def reset():
    """
    Reset global shapeguard
    """
    global __sg
    __sg = ShapeGuard()


def matches(tensor: Union[ShapedTensor, List[int]], template: str) -> bool:
    """
    Return True if tensor shape matches template
    """
    return tools.matches(tensor, template, __sg.dims)


def guard(tensor: Union[ShapedTensor, List[int]], template: str):
    inferred_dims = tools.guard(tensor, template, __sg.dims)
    __sg.dims.update(inferred_dims)
    return tensor


def reshape(tensor: Union[ShapedTensor, List[int]], template: str):
    return tools.reshape(tensor, template, __sg.dims)


def evaluate(template: str, **kwargs) -> List[Optional[int]]:
    local_dims = copy(__sg.dims)
    local_dims.update(kwargs)
    return tools.evaluate(template, local_dims)


def get_dims(item: Optional[str] = None) -> Union[Dict[str, int], List[Optional[int]]]:
    if item is None:
        return __sg.dims
    else:
        return tools.evaluate(item, __sg.dims)


def get_dim(item: str) -> Any:
    try:
        return __sg.dims[item]
    except KeyError:
        raise AttributeError(item)


def set_dim(key: str, value: Any):
    try:
        __sg.dims[key] = value
    except KeyError:
        raise AttributeError(key)


def del_dim(item: str):
    try:
        del __sg.dims[item]
    except KeyError:
        raise AttributeError(item)


__all__ = (
    "ShapeGuard",
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
