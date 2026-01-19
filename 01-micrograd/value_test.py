from value import Value


def test_add():
    v1 = Value(10)
    v2 = Value(20)
    result = v1 + v2
    assert result.data == 30


def test_sub():
    v1 = Value(10)
    v2 = Value(5)
    result = v1 - v2
    assert result.data == 5


def test_mult():
    v1 = Value(5)
    v2 = Value(4)
    result = v1 * v2
    assert result.data == 20


def test_div():
    v1 = Value(9)
    v2 = Value(3)
    result = v1 / v2
    assert result.data == 3


def test_pow():
    v1 = Value(2)
    result = v1**3
    assert result.data == 8


test_add()
test_sub()
test_mult()
test_div()
test_pow()
