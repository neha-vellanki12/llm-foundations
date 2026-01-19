import math


class Value:
    def __init__(self, data, _children=(), _op="", _label=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._op = _op
        self._prev = set(_children)
        self.label = _label

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        self._backward = backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        self._backward = backward
        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        print(t)
        out = Value(t, (self,), "tanh")

        def backward():
            self.grad += (1 - t**2) * out.grad

        self._backward = backward
        return out

    def backward(self):
        self.grad = 1
        visited = set()
        ts = []

        def visit(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    visit(child)
                ts.append(node)

        visit(self)
        for node in reversed(ts):
            node._backward()
