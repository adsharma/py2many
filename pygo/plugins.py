import ast
import functools
import io
import os
import sys
import textwrap

from tempfile import NamedTemporaryFile
from typing import Callable, Dict, List, Tuple, Union

try:
    from argparse_dataclass import dataclass as ap_dataclass
    from argparse_dataclass import ArgumentParser
except ImportError:
    ArgumentParser = "ArgumentParser"
    ap_dataclass = "ap_dataclass"


class GoTranspilerPlugins:
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
        self._usings.add('iter "github.com/hgfischer/go-iter"')
        if len(node.args) == 1:
            return f"iter.NewIntSeq(iter.Start(0), iter.Stop({vargs[0]})).All()"
        elif len(node.args) == 2:
            return (
                f"iter.NewIntSeq(iter.Start({vargs[0]}), iter.Stop({vargs[1]})).All()"
            )
        return f"iter.NewIntSeq(iter.Start({vargs[0]}), iter.Stop({vargs[1]}), iter.Step({vargs[2]})).All()"

    def visit_print(self, node, vargs: List[str]) -> str:
        placeholders = []
        for n in node.args:
            placeholders.append("%v")
        self._usings.add('"fmt"')
        placeholders_str = " ".join(placeholders)
        vargs_str = ", ".join(vargs)
        return f'fmt.Printf("{placeholders_str}\\n",{vargs_str})'

    def visit_min_max(self, node, vargs, is_max: bool) -> str:
        min_max = "math.Max" if is_max else "math.Min"
        self._usings.add('"math"')
        vargs_str = ", ".join(vargs)
        return f"{min_max}({vargs_str})"

    def visit_floor(self, node, vargs) -> str:
        self._usings.add('"math"')
        return f"math.Floor({vargs[0]})"

    def visit_exit(self, node, vargs) -> str:
        self._usings.add('"os"')
        return f"os.Exit({vargs[0]})"


# small one liners are inlined here as lambdas
SMALL_DISPATCH_MAP = {
    "str": lambda n, vargs: f"String({vargs[0]})",
    "bool": lambda n, vargs: f"({vargs[0]} != 0)",
}

SMALL_USINGS_MAP: Dict[str, str] = {}

DISPATCH_MAP = {
    "max": functools.partial(GoTranspilerPlugins.visit_min_max, is_max=True),
    "min": functools.partial(GoTranspilerPlugins.visit_min_max, is_max=False),
    "range": GoTranspilerPlugins.visit_range,
    "range_": GoTranspilerPlugins.visit_range,
    "xrange": GoTranspilerPlugins.visit_range,
    "print": GoTranspilerPlugins.visit_print,
    "floor": GoTranspilerPlugins.visit_floor,
}

MODULE_DISPATCH_TABLE: Dict[str, str] = {}

DECORATOR_DISPATCH_TABLE = {ap_dataclass: GoTranspilerPlugins.visit_ap_dataclass}

CLASS_DISPATCH_TABLE = {ap_dataclass: GoTranspilerPlugins.visit_argparse_dataclass}

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
    open: (GoTranspilerPlugins.visit_open, True),
    NamedTemporaryFile: (GoTranspilerPlugins.visit_named_temp_file, True),
    io.TextIOWrapper.read: (GoTranspilerPlugins.visit_textio_read, True),
    io.TextIOWrapper.read: (GoTranspilerPlugins.visit_textio_write, True),
    os.unlink: (lambda self, node, vargs: f"std::fs::remove_file({vargs[0]})", True),
    sys.exit: (GoTranspilerPlugins.visit_exit, True),
}
