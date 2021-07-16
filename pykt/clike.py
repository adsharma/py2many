import ast

from .inference import KT_TYPE_MAP, KT_WIDTH_RANK

from py2many.analysis import get_id
from py2many.clike import CLikeTranspiler as CommonCLikeTranspiler


# allowed as names in Python but treated as keywords in Kotlin
kotlin_keywords = frozenset(
    [
        "abstract",
        "actual",
        "annotation",
        "as",
        "break",
        "by",
        "catch",
        "class",
        "companion",
        "const",
        "constructor",
        "continue",
        "crossinline",
        "data",
        "delegate",
        "do",
        "dynamic",
        "else",
        "enum",
        "expect",
        "external",
        "field",
        "file",
        "file",
        "final",
        "finally",
        "for",
        "fun",
        "get",
        "if",
        "import",
        "in",
        "infix",
        "init",
        "inline",
        "inner",
        "interface",
        "internal",
        "is",
        "lateinit",
        "noinline",
        "object",
        "open",
        "operator",
        "out",
        "override",
        "package",
        "param",
        "private",
        "property",
        "protected",
        "public",
        "receiver",
        "reified",
        "return",
        "sealed",
        "set",
        "setparam",
        "super",
        "suspend",
        "tailrec",
        "this",
        "throw",
        "try",
        "typealias",
        "typeof",
        "val",
        "var",
        "vararg",
        "when",
        "where",
        "while",
    ]
)


class CLikeTranspiler(CommonCLikeTranspiler):
    def __init__(self):
        super().__init__()
        self._type_map = KT_TYPE_MAP
        self._statement_separator = ""
        self._temp = 0

    def _get_temp(self):
        self._temp += 1
        return f"__tmp{self._temp}"

    def _check_keyword(self, name):
        if name in kotlin_keywords:
            return name + "_", True
        if name == "_":
            return self._get_temp(), True
        return name, False

    def visit_Name(self, node) -> str:
        node_id, changed = self._check_keyword(node.id)
        if changed:
            return node_id
        return super().visit_Name(node)

    def visit_BinOp(self, node) -> str:
        if isinstance(node.op, ast.Pow):
            return "pow({0}, {1})".format(self.visit(node.left), self.visit(node.right))

        left = self.visit(node.left)
        op = self.visit(node.op)
        right = self.visit(node.right)

        left_type = self._typename_from_annotation(node.left)
        right_type = self._typename_from_annotation(node.right)

        left_rank = KT_WIDTH_RANK.get(left_type, -1)
        right_rank = KT_WIDTH_RANK.get(right_type, -1)

        if left_rank > right_rank:
            right = f"{right}.to{left_type}()"
        elif right_rank > left_rank:
            left = f"{left}.to{right_type}()"

        # Multiplication and division binds tighter (has higher precedence) than addition and subtraction.
        # To visually communicate this we omit spaces when multiplying and dividing.
        if isinstance(node.op, (ast.Mult, ast.Div)):
            return f"({left}{op}{right})"
        else:
            return f"({left} {op} {right})"

    def visit_Compare(self, node) -> str:
        if isinstance(node.ops[0], ast.In):
            return self.visit_In(node)

        node_right = node.comparators[0]
        left_type = self._typename_from_annotation(node.left)
        right_type = self._typename_from_annotation(node_right)

        left = self.visit(node.left)
        op = self.visit(node.ops[0])
        right = self.visit(node_right)

        left_rank = KT_WIDTH_RANK.get(left_type, -1)
        right_rank = KT_WIDTH_RANK.get(right_type, -1)

        if left_rank > right_rank and right_rank != -1:
            right = f"{right}.to{left_type}()"
        elif right_rank > left_rank and left_rank != -1:
            left = f"{left}.to{right_type}()"

        return f"{left} {op} {right}"

    def visit_In(self, node) -> str:
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return f"{left} in {right}"

    def _recursive_expand(self, slice_value):
        if isinstance(slice_value, ast.Name):
            slice_value = get_id(slice_value)
        elif isinstance(slice_value, (ast.Tuple, ast.List)):
            slice_value = [self._recursive_expand(e) for e in slice_value.elts]
        return slice_value

    def _typename_from_annotation(self, node, attr="annotation") -> str:
        if hasattr(node, attr):
            typename = getattr(node, attr)
            if isinstance(typename, ast.Subscript):
                if get_id(typename.value) == "Callable":
                    slice_value = self._slice_value(typename)
                    slice_value = self._recursive_expand(slice_value)

                    def recursive_tuple(lst) -> tuple:
                        return (
                            tuple(recursive_tuple(x) for x in lst)
                            if isinstance(lst, list)
                            else lst
                        )

                    slice_value = recursive_tuple(slice_value)
                    if len(slice_value) == 2:
                        # Kotlin lambda syntax
                        args = ", ".join(self._map_types(slice_value[0]))
                        ret = self._map_type(slice_value[1])
                        return f"({args}) -> {ret}"
                    return f"{slice_value}"

        return super()._typename_from_annotation(node, attr)
