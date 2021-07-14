import ast
from typing import Callable, Dict, List, Tuple, Union

from .inference import get_inferred_v_type, V_WIDTH_RANK


class VTranspilerPlugins:
    def visit_sys_argv(self, node: ast.Attribute) -> str:
        self._usings.add("os")
        return "os.args"

    def visit_range(self, node: ast.Call, vargs: List[str]) -> str:
        if len(node.args) == 1:
            return f"0..{vargs[0]}"
        elif len(node.args) == 2:
            return f"{vargs[0]}..{vargs[1]}"

        raise Exception(
            "encountered range() call with unknown parameters: range({})".format(vargs)
        )

    def visit_print(self, node: ast.Call, vargs: List[str]) -> str:
        args: List[str] = []
        total_args: List[str] = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                args.append(arg.value.replace("'", ""))
            elif get_inferred_v_type(arg) == "string":
                args.append(f"${{{self.visit(arg)}}}")
            else:
                if args:
                    total_args.append(f"'{' '.join(args)}'")
                    args = []
                total_args.append(f"({self.visit(arg)}).str()")
        if args:
            total_args.append(f"'{' '.join(args)}'")
        total_args = " + ' ' + ".join(total_args)

        return f"println({total_args})"

    def visit_bool(self, node: ast.Call, vargs: List[str]) -> str:
        if get_inferred_v_type(node.args[0]) in V_WIDTH_RANK:
            return f"{self.visit(node.args[0])} != 0"
        else:
            return f"({self.visit(node.args[0])}).bool()"

    def visit_min_max(self, node: ast.Call, vargs: List[str]) -> str:
        self._usings.add("arrays")
        func = "min" if node.func.id == "min" else "max"
        # Since python allows calling min/max on either a container or
        # multiple integers, we simply it to only a container when passing
        # it to the V side.
        if len(vargs) == 1:
            return f"arrays.{func}({vargs[0]})"
        else:
            return f"arrays.{func}([{', '.join(vargs)}])"


SMALL_DISPATCH_MAP: Dict[str, Callable] = {
    "str": lambda n, vargs: f"({vargs[0]}).str()",
    "floor": lambda n, vargs: f"int(floor({vargs[0]}))",
    "len": lambda n, vargs: f"{vargs[0]}.len",
    "sys.exit": lambda n, vargs: f"exit({vargs[0] if vargs else '0'})",
}

SMALL_USINGS_MAP: Dict[str, str] = {}

DISPATCH_MAP: Dict[str, Callable] = {}

MODULE_DISPATCH_TABLE: Dict[str, str] = {}

DECORATOR_DISPATCH_TABLE: Dict[str, Callable] = {}

CLASS_DISPATCH_TABLE: Dict[str, Callable] = {}

AttrType = Union[type, str]

ATTR_DISPATCH_TABLE: Dict[AttrType, Callable] = {
    "sys.argv": VTranspilerPlugins.visit_sys_argv,
}

FuncType = Union[Callable, str]

FUNC_DISPATCH_TABLE: Dict[FuncType, Tuple[Callable, bool]] = {
    range: (VTranspilerPlugins.visit_range, False),
    print: (VTranspilerPlugins.visit_print, False),
    bool: (VTranspilerPlugins.visit_bool, False),
    min: (VTranspilerPlugins.visit_min_max, False),
    max: (VTranspilerPlugins.visit_min_max, False),
}
