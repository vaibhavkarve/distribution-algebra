#!/usr/bin/env python3
"""This module defines the commutative algebra structure of probability distributions.

We define an `Algebra` class for enforcing the requisite structure via
abstract-method definitions for left-operators. Right-operators for
each left-operator is added on via `setattr` on the Algebra class (see function
`define_right_operators_for_Algebra_class`).

"""
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, final


class Algebra:
    """An abstract class for enforcing the definition of operations we care about.

    We specify left-operators as abstract methods in this class. Once
    the class is defined, we define the right-operators as being the
    swapped versions of their left-operators (see `Algebra_reverse_op`
    for details on how this is achieved).

    Note:
       This class is for internal use only.
    """
    @abstractmethod
    def __add__(self, other: Any) -> Any:
        """An abstract left-addition method."""
        ...

    @abstractmethod
    def __sub__(self, other: Any) -> Any:
        """An abstract left-subtraction method."""
        ...

    @abstractmethod
    def __mul__(self, other: Any) -> Any:
        """An abstract left-multiplication method."""
        ...

    @abstractmethod
    def __truediv__(self, other: Any) -> Any:
        """An abstract left-division method."""
        ...

    @abstractmethod
    def __pow__(self, other: Any) -> Any:
        """An abstract left-power (or left-exponentiation) method."""
        ...

    @abstractmethod
    def __neg__(self) -> Any:
        """An abstract negation (additive inverse) method."""
        ...


def Algebra_reverse_op(op_name: str) -> Callable[[Algebra, Any], Any]:
    """Define the right-operator method, given a left-operator name in `op_name`.

    Note:
       This function is for internal use only.
    """
    def self_op_other(self: Algebra, other: Any) -> Any:
        """Reverse operation is just the operation acting on the arguments."""
        return getattr(self, op_name)(other)
    return self_op_other

def define_right_operators_for_Algebra_class() -> None:
    """Define a right-operator method for each of `Algebra`'s magic methods.

    We additionally mark each right-operator as being `final` in order
    to guard against a user over-riding them and possibly breaking
    symmetry with the left-operators.

    """
    magic_left_operator_methods: list[str] = [
        "__add__", "__mul__", "__sub__", "__truediv__", "__pow__"]

    for magic_method in magic_left_operator_methods:
        right_operator_name: str = "__r" + magic_method.removeprefix("__")
        setattr(Algebra, right_operator_name, final(Algebra_reverse_op(magic_method)))

define_right_operators_for_Algebra_class()
