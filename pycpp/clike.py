import ast

from ctypes import c_int8, c_int16, c_int32, c_int64
from ctypes import c_uint8, c_uint16, c_uint32, c_uint64

from py2many.clike import CLikeTranspiler as CommonCLikeTranspiler

pycpp_type_map = {
    bool: "bool",
    int: "int",
    float: "double",
    bytes: "byte[]",
    str: "std::string",
    c_int8: "int8_t",
    c_int16: "int16_t",
    c_int32: "int32_t",
    c_int64: "int64_t",
    c_uint8: "uint8_t",
    c_uint16: "uint16_t",
    c_uint32: "uint32_t",
    c_uint64: "uint64_t",
}

# Commented out keywords below so we don't break existing tests
# Need to revisit
cpp_keywords = frozenset(
    [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "atomic_cancel",
        "atomic_commit",
        "atomic_noexcept",
        # "auto",
        "bitand",
        "bitor",
        # "bool",
        "break",
        "case",
        "catch",
        # "char",
        # "char16_t",
        # "char32_t",
        # "char8_t",
        "class",
        "co_await",
        "compl",
        "concept",
        "const",
        "const_cast",
        "consteval",
        "constexpr",
        "constinit",
        "continue",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        # "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        # "int",
        # "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "reflexpr",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        # "short",
        # "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "synchronized",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        # "wchar_t",
        "while",
        "xor",
        "xor_eq",
    ]
)


class CLikeTranspiler(CommonCLikeTranspiler):
    def __init__(self):
        super().__init__()
        self._type_map = pycpp_type_map

    def _check_keyword(self, name):
        if name in cpp_keywords:
            return name + "_", True
        return name, False

    def visit_Name(self, node) -> str:
        node_id, changed = self._check_keyword(node.id)
        if changed:
            return node_id
        return super().visit_Name(node)

    def visit_BinOp(self, node) -> str:
        if isinstance(node.op, ast.Pow):
            return "std::pow({0}, {1})".format(
                self.visit(node.left), self.visit(node.right)
            )
        left = self.visit(node.left)
        if not isinstance(node.left, ast.Name) and not isinstance(
            node.left, ast.Constant
        ):
            left = f"({left})"
        right = self.visit(node.right)
        if not isinstance(node.right, ast.Name) and not isinstance(
            node.right, ast.Constant
        ):
            right = f"({right})"
        return " ".join([left, self.visit(node.op), right])

    def visit_In(self, node) -> str:
        self._usings.add("<algorithm>")
        left_str = self.visit(node.left)
        right = node.comparators[0]
        right_str = self.visit(right)

        _ = self._generic_typename_from_annotation(right)
        if hasattr(right, "generic_container_type"):
            container_type, _ = right.generic_container_type
            if container_type == "Dict":
                return f"({right_str}.find({left_str}) != {right_str}.end())"
        return f"(std::find({right_str}.begin(), {right_str}.end(), {left_str}) != {right_str}.end())"
