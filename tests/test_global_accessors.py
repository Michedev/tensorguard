import numpy as np
import pytest


def test_reset():
    import tensorguard as tg
    tg.reset()
    tg.guard([3, 5, 4], "A, B, C")
    assert tg.get_dims() == {"A": 3, "B": 5, "C": 4}
    tg.reset()
    assert tg.get_dims() == {}


def test_get_dims():
    import tensorguard as tg
    tg.reset()
    assert tg.get_dims() == {}
    x = np.zeros([15, 4, 32])
    tg.guard(x, "B, C, W")
    assert tg.get_dims("B * 2, W/4") == [30, 8]


def test_get_dim():
    import tensorguard as tg
    tg.reset()
    x = np.zeros([32, 2, 5])
    tg.guard(x, "B, C, W")
    assert tg.get_dim("W") == 5
    with pytest.raises(AttributeError):
        tg.get_dim("W* 5")


def test_set_dim():
    import tensorguard as tg
    tg.reset()
    x = np.zeros([32, 2, 5])
    tg.guard(x, "B, C, W")
    assert tg.get_dim("W") == 5
    tg.set_dim("W", 10)
    assert tg.get_dim("W") == 10
    tg.set_dim("WF", 40)
    assert tg.get_dim("WF") == 40


def test_del_dim():
    import tensorguard as tg
    tg.reset()
    x = np.zeros([32, 2, 5])
    tg.guard(x, "B, C, W")
    assert tg.get_dim("W") == 5
    tg.del_dim("W")
    with pytest.raises(AttributeError):
        tg.get_dim("W")

