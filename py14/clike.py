import ast

from py2many.clike import CLikeTranspiler as CommonCLikeTranspiler

py14_type_map = {
    "bool": "bool",
    "int": "int",
    "float": "double",
    "bytes": "byte[]",
    "str": "string",
    "c_int8": "int8_t",
    "c_int16": "int16_t",
    "c_int32": "int32_t",
    "c_int64": "int64_t",
    "c_uint8": "uint8_t",
    "c_uint16": "uint16_t",
    "c_uint32": "uint32_t",
    "c_uint64": "uint64_t",
}


class CLikeTranspiler(CommonCLikeTranspiler):
    def __init__(self):
        self._type_map = py14_type_map

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            return "std::pow({0}, {1})".format(
                self.visit(node.left), self.visit(node.right)
            )
        return " ".join(
            [self.visit(node.left), self.visit(node.op), self.visit(node.right)]
        )

    def visit_In(self, node):
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return "{0}.contains({1})".format(right, left)

    def visit_Constant(self, node):
        if node.value is True:
            return "true"
        elif node.value is False:
            return "false"
        else:
            return str(node.value)
