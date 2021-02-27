# Tensor Guard

TensorGuard helps to guard against bad Tensor shapes in any tensor based library (e.g. Numpy, Pytorch, Tensorflow) using an intuitive symbolic-based syntax



## Basic Usage

```python
import numpy as np  # could be tensorflow or torch as well
import tensorguard as tg

# tensorguard = tg.TensorGuard()  #could be done in a OOP fashion
img = np.ones([64, 32, 32, 3])
flat_img = np.ones([64, 1024])
labels = np.ones([64])

# check shape consistency
tg.guard(img, "B, H, W, C")
tg.guard(labels, "B, 1")  # raises error because of rank mismatch
tg.guard(flat_img, "B, H*W*C")  # raises error because 1024 != 32*32*3

# guard also returns the tensor, so it can be inlined
mean_img = tg.guard(np.mean(img, axis=0), "H, W, C")

# more readable reshapes
flat_img = tg.reshape(img, 'B, H*W*C')

# evaluate templates
assert tg.get_dims('H, W*C+1') == [32, 97]

```


## Shape Template Syntax
The shape template mini-DSL supports many different ways of specifying shapes:

  * numbers: `"64, 32, 32, 3"`
  * named dimensions: `"B, width, height2, channels"`
  * wildcards: `"B, *, *, *"`
  * ellipsis: `"B, ..., 3"`
  * addition, subtraction, multiplication, division: `"B*N, W/2, H*(C+1)"`
  * dynamic dimensions: `"?, H, W, C"`  *(only matches `[None, H, W, C]`)*

### Original Repo link: https://github.com/Qwlouse/shapeguard
