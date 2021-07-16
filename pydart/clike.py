import ast

from ctypes import c_int8, c_int16, c_int32, c_int64
from ctypes import c_uint8, c_uint16, c_uint32, c_uint64

from py2many.clike import CLikeTranspiler as CommonCLikeTranspiler


dart_type_map = {
    bool: "bool",
    int: "int",
    float: "double",
    bytes: "Uint8List",
    str: "String",
    c_int8: "int",
    c_int16: "int",
    c_int32: "int",
    c_int64: "int",
    c_uint8: "int",
    c_uint16: "int",
    c_uint32: "int",
    c_uint64: "int",
}

# allowed as names in Python but treated as keywords in Dart
dart_keywords = frozenset(
    [
        "assert",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "else",
        "enum",
        "extends",
        "false",
        "final",
        "finally",
        "for",
        "if",
        "in",
        "is",
        "new",
        "null",
        "rethrow",
        "return",
        "super",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "var",
        "void",
        "while",
        "with",
    ]
)


class CLikeTranspiler(CommonCLikeTranspiler):
    def __init__(self):
        super().__init__()
        self._type_map = dart_type_map

    def visit_Name(self, node) -> str:
        if node.id in dart_keywords:
            return node.id + "_"
        return super().visit_Name(node)

    def visit_BinOp(self, node) -> str:
        if isinstance(node.op, ast.Pow):
            return "pow({0}, {1})".format(self.visit(node.left), self.visit(node.right))

        left = self.visit(node.left)
        op = self.visit(node.op)
        right = self.visit(node.right)

        # Multiplication and division binds tighter (has higher precedence) than addition and subtraction.
        # To visually communicate this we omit spaces when multiplying and dividing.
        if isinstance(node.op, (ast.Mult, ast.Div)):
            if getattr(node, "use_integer_div", False):
                # Use integer division operator
                op = "~/"
            return f"({left}{op}{right})"
        else:
            return f"({left} {op} {right})"

    def visit_In(self, node) -> str:
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return "{0}.any({1})".format(right, left)
