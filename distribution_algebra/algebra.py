#!/usr/bin/env python3

from collections.abc import Callable
from numbers import Number
from typing import Any, final


class Algebra(Number):
    ...


def Algebra_reverse_op(op_name: str) -> Callable[[Algebra, Any], Any]:
    def self_op_other(self: Algebra, other: Any) -> Any:
        return getattr(self, op_name)(other)
    return self_op_other


for magic_method in ["__add__", "__mul__", "__sub__", "__truediv__", "__pow__"]:
    # Set reverse operation to be just the operation acting on the arguments.
    setattr(Algebra, "__r" + magic_method.removeprefix("__"),
            final(Algebra_reverse_op(magic_method)))
