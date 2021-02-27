import numpy as np
import pytest


def test_reset():
    import shapeguard as sg
    sg.reset()
    sg.guard([3, 5, 4], "A, B, C")
    assert sg.get_dims() == {"A": 3, "B": 5, "C": 4}
    sg.reset()
    assert sg.get_dims() == {}


def test_get_dims():
    import shapeguard as sg
    sg.reset()
    assert sg.get_dims() == {}
    x = np.zeros([15, 4, 32])
    sg.guard(x, "B, C, W")
    assert sg.get_dims("B * 2, W/4") == [30, 8]


def test_get_dim():
    import shapeguard as sg
    sg.reset()
    x = np.zeros([32, 2, 5])
    sg.guard(x, "B, C, W")
    assert sg.get_dim("W") == 5
    with pytest.raises(AttributeError):
        sg.get_dim("W* 5")


def test_set_dim():
    import shapeguard as sg
    sg.reset()
    x = np.zeros([32, 2, 5])
    sg.guard(x, "B, C, W")
    assert sg.get_dim("W") == 5
    sg.set_dim("W", 10)
    assert sg.get_dim("W") == 10
    sg.set_dim("WF", 40)
    assert sg.get_dim("WF") == 40


def test_del_dim():
    import shapeguard as sg
    sg.reset()
    x = np.zeros([32, 2, 5])
    sg.guard(x, "B, C, W")
    assert sg.get_dim("W") == 5
    sg.del_dim("W")
    with pytest.raises(AttributeError):
        sg.get_dim("W")

