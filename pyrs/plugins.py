import io
import os
import ast
import functools
import math
import time
import random
import sys
import textwrap

from tempfile import NamedTemporaryFile
from typing import Callable, Dict, List, Tuple, Union

try:
    from argparse_dataclass import dataclass as ap_dataclass
    from argparse_dataclass import ArgumentParser
except:
    ArgumentParser = "ArgumentParser"
    ap_dataclass = "ap_dataclass"


class RustTranspilerPlugins:
    def visit_argparse_dataclass(self, node):
        fields = []
        for (
            declaration,
            typename_with_default,
        ) in node.declarations_with_defaults.items():
            typename, default_value = typename_with_default
            if typename == None:
                return None
            if default_value is not None and typename != "bool":
                default_value = self.visit(default_value)
                default_value = f', default_value = "{default_value}"'
            else:
                default_value = ""
            fields.append(
                f"#[structopt(short, long{default_value})]\npub {declaration}: {typename},"
            )
        fields = "\n".join(fields)
        self._usings.add("structopt::StructOpt")
        clsdef = "\n" + textwrap.dedent(
            f"""\
        #[derive(Debug, StructOpt)]
        #[structopt(name = "{self._module}", about = "Placeholder")]
        struct {node.name} {{
            {fields}
        }}
        """
        )
        return clsdef

    def visit_open(self, node, vargs):
        self._usings.add("std::fs::File")
        if len(vargs) > 1:
            self._usings.add("std::fs::OpenOptions")
            mode = vargs[1]
            opts = "OpenOptions::new()"
            is_binary = "b" in mode
            for c in mode:
                if c == "w":
                    if not is_binary:
                        self._usings.add("pylib::FileWriteString")
                    opts += ".write(true)"
                if c == "r":
                    if not is_binary:
                        self._usings.add("pylib::FileReadString")
                    opts += ".read(true)"
                if c == "a":
                    opts += ".append(true)"
                if c == "+":
                    opts += ".read(true).write(true)"
            node.result_type = True
            return f"{opts}.open({vargs[0]})"
        node.result_type = True
        return f"File::open({vargs[0]})"

    def visit_read(self, node, vargs):
        assert len(vargs) <= 1, "read() with more than one argument"
        if len(vargs) == 0:
            return "f.read_string()"
        self._usings.add("pylib::FileReadBytes")
        return f"std::str::from_utf8(&f.read_bytes({vargs[0]})?)"

    def visit_write(self, node, vargs):
        assert len(vargs) == 1, "write() can only have one argument"
        return f"f.write_string({vargs[0]})"

    def visit_named_temp_file(self, node, vargs):
        node.annotation = ast.Name(id="tempfile._TemporaryFileWrapper")
        node.result_type = True
        return "NamedTempFile::new()"

    def visit_textio_read(self, node, vargs):
        # TODO
        return None

    def visit_textio_write(self, node, vargs):
        # TODO
        return None

    def visit_ap_dataclass(self, cls):
        # Do whatever transformation the decorator does to cls here
        return cls

    def visit_range(self, node, vargs: List[str]) -> str:
        assert 0 < len(node.args) < 4, "range() call with unexpected arguments"
        if len(node.args) == 1:
            return "(0..{})".format(vargs[0])
        elif len(node.args) == 2:
            return "({}..{})".format(vargs[0], vargs[1])
        return "({}..{}).step_by({})".format(vargs[0], vargs[1], vargs[2])

    def visit_print(self, node, vargs: List[str]) -> str:
        placeholders = []
        for n in node.args:
            placeholders.append("{}")
        return 'println!("{0}",{1});'.format(" ".join(placeholders), ", ".join(vargs))

    def visit_exit(self, node, vargs) -> str:
        self._allows.add("unreachable_code")
        return f"std::process::exit({vargs[0]})"

    def visit_min_max(self, node, vargs, is_max: bool) -> str:
        self._usings.add("std::cmp")
        min_max = "max" if is_max else "min"
        self._typename_from_annotation(node.args[0])
        if hasattr(node.args[0], "container_type"):
            node.result_type = True
            return f"{vargs[0]}.iter().{min_max}()"
        else:
            all_vargs = ", ".join(vargs)
            return f"cmp::{min_max}({all_vargs})"

    @staticmethod
    def visit_cast(node, vargs, cast_to: str) -> str:
        return f"{vargs[0]} as {cast_to}"

    @staticmethod
    def visit_asyncio_run(node, vargs) -> str:
        return f"block_on({vargs[0]})"


# small one liners are inlined here as lambdas
SMALL_DISPATCH_MAP = {
    "str": lambda n, vargs: f"&{vargs[0]}.to_string()",
    "len": lambda n, vargs: f"{vargs[0]}.len()",
    "enumerate": lambda n, vargs: f"{vargs[0]}.iter().enumerate()",
    "sum": lambda n, vargs: f"{vargs[0]}.iter().sum()",
    "int": functools.partial(RustTranspilerPlugins.visit_cast, cast_to="i32"),
    "bool": lambda n, vargs: f"({vargs[0]} != 0)",
    "float": functools.partial(RustTranspilerPlugins.visit_cast, cast_to="f64"),
    # as usize below is a hack to pass comb_sort.rs. Need a better solution
    "floor": lambda n, vargs: f"{vargs[0]}.floor() as usize",
    "reversed": lambda n, vargs: f"{vargs[0]}.iter().rev()",
    "map": lambda n, vargs: f"{vargs[1]}.iter().map({vargs[0]})",
    "filter": lambda n, vargs: f"{vargs[1]}.into_iter().filter({vargs[0]})",
    "list": lambda n, vargs: f"{vargs[0]}.collect::<Vec<_>>()",
    "asyncio.run": RustTranspilerPlugins.visit_asyncio_run,
}

SMALL_USINGS_MAP = {
    "asyncio.run": "futures::executor::block_on",
}

DISPATCH_MAP = {
    "max": functools.partial(RustTranspilerPlugins.visit_min_max, is_max=True),
    "min": functools.partial(RustTranspilerPlugins.visit_min_max, is_max=False),
    "range": RustTranspilerPlugins.visit_range,
    "xrange": RustTranspilerPlugins.visit_range,
    "print": RustTranspilerPlugins.visit_print,
}

MODULE_DISPATCH_TABLE = {"tempfile.NamedTemporaryFile": "tempfile::NamedTempFile"}

DECORATOR_DISPATCH_TABLE = {ap_dataclass: RustTranspilerPlugins.visit_ap_dataclass}

CLASS_DISPATCH_TABLE = {ap_dataclass: RustTranspilerPlugins.visit_argparse_dataclass}

ATTR_DISPATCH_TABLE = {
    "temp_file.name": lambda self, node, value, attr: f"{value}.path()",
}

FuncType = Union[Callable, str]

FUNC_DISPATCH_TABLE: Dict[FuncType, Tuple[Callable, bool]] = {
    # Uncomment after upstream uploads a new version
    # ArgumentParser.parse_args: lambda node: "Opts::parse_args()",
    # HACKs: remove all string based dispatch here, once we replace them with type based
    "parse_args": (lambda self, node, vargs: "::from_args()", False),
    "f.read": (RustTranspilerPlugins.visit_read, True),
    "f.write": (RustTranspilerPlugins.visit_write, True),
    "f.close": (lambda self, node, vargs: "drop(f)", False),
    open: (RustTranspilerPlugins.visit_open, True),
    NamedTemporaryFile: (RustTranspilerPlugins.visit_named_temp_file, True),
    io.TextIOWrapper.read: (RustTranspilerPlugins.visit_textio_read, True),
    io.TextIOWrapper.read: (RustTranspilerPlugins.visit_textio_write, True),
    math.pow: (lambda self, node, vargs: f"{vargs[0]}.powf({vargs[1]})", False),
    math.sin: (lambda self, node, vargs: f"{vargs[0]}.sin()", False),
    math.cos: (lambda self, node, vargs: f"{vargs[0]}.cos()", False),
    math.tan: (lambda self, node, vargs: f"{vargs[0]}.tan()", False),
    math.asin: (lambda self, node, vargs: f"{vargs[0]}.asin()", False),
    math.acos: (lambda self, node, vargs: f"{vargs[0]}.acos()", False),
    math.atan: (lambda self, node, vargs: f"{vargs[0]}.atan()", False),
    time.time: (lambda self, node, vargs: "pylib::time()", False),
    random.seed: (
        lambda self, node, vargs: f"pylib::random::reseed_from_f64({vargs[0]})",
        False,
    ),
    random.random: (lambda self, node, vargs: "pylib::random::random()", False),
    os.unlink: (lambda self, node, vargs: f"std::fs::remove_file({vargs[0]})", True),
    sys.exit: (RustTranspilerPlugins.visit_exit, True),
}

FUNC_USINGS_MAP = {
    time.time: "pylib",
    random.seed: "pylib",
    random.random: "pylib",
}
