from typing import List


def valid(a: str) -> List[str]:
    """
    A valid G2P engine.
    """
    return a.split()


def g2p_test_upper(a: str) -> List[str]:
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


def dummy_g2p(normalized_input_text: str) -> List[str]:
    """
    A dummy G2P engine that actually does something, so we can unit test the 'everyvoice g2p' command.
    """
    return normalized_input_text.lower().split()
