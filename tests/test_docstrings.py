#!/usr/bin/env python3

import vizdoom as vzd


def test_docstrings(object):
    object_methods = [
        method_name
        for method_name in dir(object)
        if callable(getattr(object, method_name))
    ]

    for method in object_methods:
        assert method.__doc__ is not None, f"Method {method} has no docstring"


if __name__ == "__main__":
    test_docstrings(vzd.DoomGame)
