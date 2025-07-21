from typing import List


def valid(a: str) -> List[str]:
    """
    A valid G2P engine.
    """
    return a.split()


def upper(a: str) -> List[str]:
    """
    A valid G2P engine with very visible effects, for unit testing
    """
    return a.upper().split()


def multiple_arguments(a: int, b: int) -> List[str]:
    """
    A G2P engine with the wrong function's signature.
    """
    return ["a", "b"]


def not_a_string(a: int) -> List[str]:
    """
    A G2P engine with the wrong function's signature.
    """
    return ["a", "b"]


def not_a_list(a: str) -> int:
    """
    A G2P engine with the wrong function's signature.
    """
    return 42


not_a_function = "A g2p engine that is not a callable object."


class AClass:
    def __init__(self, a: str):
        self.a = a

    def __call__(self, a: str) -> List[str]:
        return a.split()


a_class_instance = AClass("x")


def implicit_signature(a):
    return a.split()
