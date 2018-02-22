#!/usr/bin/env python3

"""
Miscellaneous functions for dlcutils.
"""


def print_mixed(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    sequence = [str(arg) for arg in args]
    print(*sequence, sep=sep)
