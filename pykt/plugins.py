import io
import os
import ast
import functools
import re
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


class KotlinTranspilerPlugins:
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
            return "(0..{}-1)".format(vargs[0])
        elif len(node.args) == 2:
            return "({}..{}-1)".format(vargs[0], vargs[1])
        return "({}..{}-1 step {})".format(vargs[0], vargs[1], vargs[2])

    def visit_print(self, node, vargs: List[str]) -> str:
        def _format(arg):
            if arg.isdigit():
                return arg
            if re.match(r"'.*'", arg) or re.match(r'".*"', arg):
                return arg[1:-1]
            else:
                return f"${arg}"

        vargs_str = " ".join([f"{_format(arg)}" for arg in vargs])
        return f'println("{vargs_str}")'

    def visit_min_max(self, node, vargs, is_max: bool) -> str:
        min_max = "max" if is_max else "min"
        self._usings.add(f"kotlin.math.{min_max}")
        self._typename_from_annotation(node.args[0])
        if hasattr(node.args[0], "container_type"):
            return f"maxOf({vargs[0]})"
        else:
            all_vargs = ", ".join(vargs)
            return f"{min_max}({all_vargs})"

    def visit_floor(self, node, vargs) -> str:
        self._usings.add("kotlin.math.floor")
        return f"floor({vargs[0]}).toInt()"


# small one liners are inlined here as lambdas
SMALL_DISPATCH_MAP = {
    "str": lambda n, vargs: f"{vargs[0]}.toString()",
    # TODO: strings use .length
    "len": lambda n, vargs: f"{vargs[0]}.size",
    "int": lambda n, vargs: f"{vargs[0]}.toInt()",
    "bool": lambda n, vargs: f"({vargs[0]} != 0)",
    "reversed": lambda n, vargs: f"{vargs[0]}.reversed()",
}

SMALL_USINGS_MAP: Dict[str, str] = {}

DISPATCH_MAP = {
    "max": functools.partial(KotlinTranspilerPlugins.visit_min_max, is_max=True),
    "min": functools.partial(KotlinTranspilerPlugins.visit_min_max, is_max=False),
    "range": KotlinTranspilerPlugins.visit_range,
    "xrange": KotlinTranspilerPlugins.visit_range,
    "print": KotlinTranspilerPlugins.visit_print,
    "floor": KotlinTranspilerPlugins.visit_floor,
}

MODULE_DISPATCH_TABLE: Dict[str, str] = {}

DECORATOR_DISPATCH_TABLE = {ap_dataclass: KotlinTranspilerPlugins.visit_ap_dataclass}

CLASS_DISPATCH_TABLE = {ap_dataclass: KotlinTranspilerPlugins.visit_argparse_dataclass}

ATTR_DISPATCH_TABLE = {
    "temp_file.name": lambda self, node, value, attr: f"{value}.path()",
}

FuncType = Union[Callable, str]

FUNC_DISPATCH_TABLE: Dict[FuncType, Tuple[Callable, bool]] = {
    # Uncomment after upstream uploads a new version
    # ArgumentParser.parse_args: lambda node: "Opts::parse_args()",
    # HACKs: remove all string based dispatch here, once we replace them with type based
    "parse_args": (lambda self, node, vargs: "::from_args()", False),
    "f.read": (lambda self, node, vargs: "f.read_string()", True),
    "f.write": (lambda self, node, vargs: f"f.write_string({vargs[0]})", True),
    "f.close": (lambda self, node, vargs: "drop(f)", False),
    open: (KotlinTranspilerPlugins.visit_open, True),
    NamedTemporaryFile: (KotlinTranspilerPlugins.visit_named_temp_file, True),
    io.TextIOWrapper.read: (KotlinTranspilerPlugins.visit_textio_read, True),
    io.TextIOWrapper.read: (KotlinTranspilerPlugins.visit_textio_write, True),
    os.unlink: (lambda self, node, vargs: f"std::fs::remove_file({vargs[0]})", True),
    sys.exit: (
        lambda self, node, vargs: f"kotlin.system.exitProcess({vargs[0]})",
        True,
    ),
}
