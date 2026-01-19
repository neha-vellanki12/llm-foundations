from value import Value


def test_add():
    v1 = Value(10)
    v2 = Value(20)
    result = v1 + v2
    assert result == 30


def test_mult():
    v1 = Value(5)
    v2 = Value(4)
    result = v1 * v2
    assert result == 20


def test_pow():
    v1 = Value(2)
    v2 = Value(3)
    result = v1**v2
    assert result == 8
