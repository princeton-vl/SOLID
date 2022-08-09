import math
import numpy as np

from argparse import Action

_PARSERS = {}
_RENDERS = {}


def load_parsers(parser):
    for v in _PARSERS.values():
        v(parser)


def get_render_func(module):
    return _RENDERS[module]


def add_render_func(submodule):
    def add_func(func):
        _RENDERS[submodule] = func

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return add_func


def add_default_arguments(submodule):
    def add_arguments(func):
        def wrapper(parser, *args, **kwargs):
            parser = parser.add_parser(submodule)
            parser.add_argument("--res_w", default=640, type=int)
            parser.add_argument("--res_h", default=480, type=int)
            parser.add_argument("--scales", default=[1.0], type=float, action="append")
            parser.add_argument("--num_objects", default=9, type=int)
            parser.add_argument("--min_scale", default=1, type=float)
            parser.add_argument("--max_scale", default=1, type=float)
            parser.add_argument(
                "--scale_set",
                default="small",
                type=str,
                choices=[
                    "all",
                    "small",
                ],
            )
            parser.add_argument(
                "--min_thetax",
                default=-np.pi,
                type=float,
                action=ConvertDegreeToRadian,
            )
            parser.add_argument(
                "--max_thetax",
                default=np.pi,
                type=float,
                action=ConvertDegreeToRadian,
            )
            parser.add_argument(
                "--min_thetay",
                default=-np.pi,
                type=float,
                action=ConvertDegreeToRadian,
            )
            parser.add_argument(
                "--max_thetay",
                default=np.pi,
                type=float,
                action=ConvertDegreeToRadian,
            )
            parser.add_argument(
                "--min_thetaz",
                default=-np.pi,
                type=float,
                action=ConvertDegreeToRadian,
            )
            parser.add_argument(
                "--max_thetaz",
                default=np.pi,
                type=float,
                action=ConvertDegreeToRadian,
            )
            func(parser, *args, **kwargs)

        _PARSERS[submodule] = wrapper
        return wrapper
    return add_arguments


class ConvertDegreeToRadian(Action):
    def __call__(
        self,
        parser,
        namespace,
        values,
        option_string=None
    ):
        setattr(namespace, self.dest, math.radians(values))
