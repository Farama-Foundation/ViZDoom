#!/usr/bin/env python3


import re

import vizdoom as vzd


def _get_object_methods(object):
    object_methods = [
        method_name
        for method_name in dir(object)
        if callable(getattr(object, method_name)) and not method_name.startswith("_")
    ]
    return object_methods


def _check_object_docstrings(object):
    object_methods = _get_object_methods(object)

    for method in object_methods:
        method_doc = eval(f"object.{method}.__doc__")
        assert method_doc is not None, f"Method {method} has no docstring"

        # Check if there is correct signature in docstring (with proper argument names)
        m = re.search(r"arg[0-9]+", method_doc)
        assert m is None, f"Method {method} has arguments without names"


def test_docstrings():
    _check_object_docstrings(vzd.DoomGame)


if __name__ == "__main__":
    test_docstrings()
