from typing import List


def valid(a: str) -> List[str]:
    """
    A valid G2P engine.
    """
    return a.split()


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
