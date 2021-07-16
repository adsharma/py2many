import ast
import textwrap

from .tracer import decltype
from .clike import CLikeTranspiler
from .plugins import (
    ATTR_DISPATCH_TABLE,
    CLASS_DISPATCH_TABLE,
    FUNC_DISPATCH_TABLE,
    MODULE_DISPATCH_TABLE,
    DISPATCH_MAP,
    SMALL_DISPATCH_MAP,
    SMALL_USINGS_MAP,
)


from py2many.analysis import add_imports, is_global, is_void_function, get_id
from py2many.ast_helpers import create_ast_block
from py2many.clike import _AUTO_INVOKED, class_for_typename
from py2many.context import add_variable_context, add_list_calls
from py2many.declaration_extractor import DeclarationExtractor
from py2many.exceptions import AstNotImplementedError
from py2many.inference import InferMeta
from py2many.scope import add_scope_context
from py2many.rewriters import PythonMainRewriter
from py2many.tracer import (
    defined_before,
    is_class_or_module,
    is_enum,
    is_list,
    is_self_arg,
)

from typing import List, Tuple

_AUTO = "auto()"


# TODO: merge this into py2many.cli.transpiler and fixup the tests
def transpile(source, headers=False, testing=False):
    """
    Transpile a single python translation unit (a python script) into
    C++ 14 code.
    """
    tree = ast.parse(source)
    rewriter = PythonMainRewriter("cpp")
    tree = rewriter.visit(tree)
    add_variable_context(tree, (tree,))
    add_scope_context(tree)
    add_list_calls(tree)
    add_imports(tree)

    transpiler = CppTranspiler()

    buf = []
    if testing:
        buf += ['#include "catch.hpp"']
        transpiler.use_catch_test_cases = True

    if headers:
        buf += transpiler._headers
        buf += transpiler._usings

    if testing or headers:
        buf.append("")  # Force empty line

    cpp = transpiler.visit(tree)
    return "\n".join(buf) + cpp


def generate_catch_test_case(node, body):
    funcdef = 'TEST_CASE("{0}")'.format(node.name)
    return funcdef + " {\n" + body + "\n}"


def generate_lambda_fun(node, body):
    params = ["auto {0}".format(param.id) for param in node.args.args]
    funcdef = "auto {0} = []({1})".format(node.name, ", ".join(params))
    return funcdef + " {\n" + body + "\n};"


class CppListComparisonRewriter(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self._temp = 0

    def _get_temp(self):
        self._temp += 1
        return f"__tmp{self._temp}"

    def visit_Compare(self, node):
        left = self.visit(node.left)
        ops = [self.visit(op) for op in node.ops]
        comparators = [self.visit(cmp) for cmp in node.comparators]
        right = comparators[0]
        if isinstance(right, ast.List):
            tmp = ast.Name(id=self._get_temp(), lineno=node.lineno)
            comparators[0] = tmp
            return create_ast_block(
                body=[
                    ast.Assign(targets=[tmp], value=right, lineno=node.lineno),
                    ast.Compare(left, ops, comparators, lineno=node.lineno),
                ],
                at_node=node,
            )

        return node


class CppTranspiler(CLikeTranspiler):
    NAME = "cpp"

    CONTAINER_TYPES = {
        "List": "std::vector",
        "Dict": "std::map",
        "Set": "std::set",
        "Optional": "std::optional",
    }

    def __init__(self, extension: bool = False, no_prologue: bool = False):
        super().__init__()
        # TODO: include only when needed
        self._headers = []
        self._usings = set([])
        self.use_catch_test_cases = False
        self._container_type_map = self.CONTAINER_TYPES
        self._extension = extension
        self._no_prologue = no_prologue
        self._dispatch_map = DISPATCH_MAP
        self._small_dispatch_map = SMALL_DISPATCH_MAP
        self._small_usings_map = SMALL_USINGS_MAP
        self._func_dispatch_table = FUNC_DISPATCH_TABLE
        self._attr_dispatch_table = ATTR_DISPATCH_TABLE
        self._main_signature_arg_names = ["argc", "argv"]

    def usings(self):
        usings = sorted(list(set(self._usings)))
        lint_exception = (
            "  // NOLINT(build/include_order)" if not self._no_prologue else ""
        )
        uses = "\n".join(f"#include {mod}{lint_exception}" for mod in usings)
        return uses

    def headers(self, meta: InferMeta):
        lint_exception = (
            "  // NOLINT(build/include_order)" if not self._no_prologue else ""
        )
        self._headers.append(f'#include "pycpp/runtime/sys.h"{lint_exception}')
        self._headers.append(f'#include "pycpp/runtime/builtins.h"{lint_exception}')
        if self.use_catch_test_cases:
            self._headers.append(f'#include "pycpp/runtime/catch.hpp"{lint_exception}')
        if meta.has_fixed_width_ints:
            self._headers.append("#include <stdint.h>")
        return "\n".join(self._headers)

    def visit_FunctionDef(self, node) -> str:
        body = "\n".join([self.visit(n) for n in node.body])
        # If rewriter inserted a block, we need to terminate it with a semicolon
        if len(node.body):
            last = node.body[-1]
            if getattr(last, "make_block", False):
                body += ";"

        if (
            self.use_catch_test_cases
            and is_void_function(node)
            and node.name.startswith("test")
        ):
            return generate_catch_test_case(node, body)

        is_python_main = getattr(node, "python_main", False)
        typenames, args = self.visit(node.args)

        args_list = []
        if len(args) and hasattr(node, "self_type"):
            # implicit this
            del typenames[0]
            del args[0]

        typedecls = []
        index = 0
        for i in range(len(args)):
            typename = typenames[i]
            arg = args[i]
            if typename == "T":
                typename = "T{0}".format(index)
                typedecls.append(typename)
                index += 1
            if is_python_main and arg in ["argv"]:
                typename = "char **"
            args_list.append(f"{typename} {arg}")

        template = "inline "
        if len(typedecls) > 0:
            typedecls_str = ", ".join([f"typename {t}" for t in typedecls])
            template = f"template <{typedecls_str}>"

        if is_python_main:
            body = (
                "pycpp::sys::argv = std::vector<std::string>(argv, argv + argc);\n"
                + body
            )
            template = ""

        if not is_void_function(node):
            if node.returns:
                typename = self._typename_from_annotation(node, attr="returns")
                return_type = f"{typename}"
            else:
                return_type = "auto"
        else:
            return_type = "void"

        args = ", ".join(args_list)
        funcdef = f"{template}{return_type} {node.name}({args}) {{"

        return funcdef + "\n" + body + "}\n"

    def visit_Attribute(self, node) -> str:
        attr = node.attr
        value_id = self.visit(node.value)

        if is_list(node.value):
            if node.attr == "append":
                attr = "push_back"

        if value_id in {"string"}:
            return f"std::{value_id}::{attr}"

        if is_enum(value_id, node.scopes):
            return f"{value_id}::{attr}"

        if is_class_or_module(value_id, node.scopes):
            return f"{value_id}.{attr}"

        if is_self_arg(value_id, node.scopes):
            return f"this->{attr}"

        ret = f"{value_id}.{attr}"
        if ret in self._attr_dispatch_table:
            ret = self._attr_dispatch_table[ret]
            return ret(self, node, value_id, attr)

        return ret

    def visit_ClassDef(self, node) -> str:
        extractor = DeclarationExtractor(CppTranspiler())
        extractor.visit(node)
        declarations = node.declarations = extractor.get_declarations()
        node.class_assignments = extractor.class_assignments

        ret = super().visit_ClassDef(node)
        if ret is not None:
            return ret

        decorators = [get_id(d) for d in node.decorator_list]
        decorators = [
            class_for_typename(t, None, self._imported_names) for t in decorators
        ]
        for d in decorators:
            if d in CLASS_DISPATCH_TABLE:
                ret = CLASS_DISPATCH_TABLE[d](self, node)
                if ret is not None:
                    return ret

        buf = [f"class {node.name} {{"]
        buf += ["public:"]
        fields = []
        index = 0
        for declaration, typename in declarations.items():
            if typename == None:
                typename = "ST{0}".format(index)
                index += 1
            fields.append(f"{typename} {declaration}")

        for b in node.body:
            if isinstance(b, ast.FunctionDef):
                b.self_type = node.name

        buf += [";\n".join(fields + [""])]
        body = [self.visit(b) for b in node.body]
        if node.is_dataclass:
            field_names = [arg for arg in declarations.keys()]
            args = ", ".join(fields)
            assignments = "; ".join(
                [f"this->{field} = {field}" for field in field_names]
            )
            constructor = f"{node.name}({args}) {{{assignments};}}"
            body = [constructor] + body
        buf += body
        buf += ["};"]
        return "\n".join(buf) + "\n"

    def _visit_enum(self, node, typename: str, fields: List[Tuple]) -> str:
        fields_list = []

        for field, value in fields:
            fields_list += [f"{field} = {value},"]
        fields_str = "".join(fields_list)
        return f"enum {node.name} : {typename} {{\n{fields_str}}};\n"

    def visit_IntEnum(self, node) -> str:
        fields = []
        for i, (member, var) in enumerate(node.class_assignments.items()):
            var = self.visit(var)
            if var == _AUTO_INVOKED:
                var = i
            fields.append((member, var))
        return self._visit_enum(node, "int", fields)

    def visit_IntFlag(self, node) -> str:
        fields = []
        for i, (member, var) in enumerate(node.class_assignments.items()):
            var = self.visit(var)
            if var == _AUTO_INVOKED:
                var = i
            fields.append((member, var))
        return self._visit_enum(node, "int", fields)

    def visit_StrEnum(self, node) -> str:
        fields = []
        definitions = []
        for i, (member, node_var) in enumerate(node.class_assignments.items()):
            var = self.visit(node_var)
            raw_string = node_var.raw_string
            if var == _AUTO_INVOKED:
                var = f'"{member}"'
            fields.append(f"static const {node.name} {member};")
            definitions.append(
                f"const {node.name} {node.name}::{member}= {raw_string};"
            )
        fields = "\n".join([f for f in fields])
        definitions = "\n".join([d for d in definitions])
        lint_exception = (
            "  // NOLINT(runtime/explicit)" if not self._no_prologue else ""
        )
        return textwrap.dedent(
            f"""\
            class {node.name} : public std::string {{
            public:
              {node.name}(const char* s) : std::string(s) {{}}{lint_exception}
              {fields}
            }};

            {definitions}

            """
        )

    def visit_Call(self, node) -> str:
        fname = self.visit(node.func)
        vargs = []

        if node.args:
            vargs += [self.visit(a) for a in node.args]
        if node.keywords:
            vargs += [self.visit(kw.value) for kw in node.keywords]

        ret = self._dispatch(node, fname, vargs)
        if ret is not None:
            return ret

        if any(i is None for i in vargs):
            raise AstNotImplementedError(f"Call {fname} ({vargs}) not supported", node)

        args = ", ".join(vargs)
        return f"{fname}({args})"

    def visit_For(self, node) -> str:
        target = self.visit(node.target)
        it = self.visit(node.iter)
        buf = []
        buf.append("for(auto {0} : {1}) {{".format(target, it))
        buf.extend([self.visit(c) for c in node.body])
        buf.append("}")
        return "\n".join(buf)

    def visit_arg(self, node):
        node_id = get_id(node)
        if node_id == "self":
            return (None, "self")
        node_id, _ = self._check_keyword(node_id)
        typename = "T"
        if node.annotation:
            typename = self._typename_from_annotation(node)
            # TODO: Generalize this to other types
            if "std::map" in typename:
                self._headers.append("#include <map>")
        return (typename, node_id)

    def visit_Lambda(self, node) -> str:
        _, args = self.visit(node.args)
        args_string = ", ".join([f"auto {a}" for a in args])
        body = self.visit(node.body)
        return f"[]({args_string}) {{ return {body}; }}"

    def visit_Str(self, node) -> str:
        """Use a C++ 14 string literal instead of raw string"""
        node.raw_string = super().visit_Str(node)
        return f"std::string{{{node.raw_string}}}"

    def visit_Bytes(self, node) -> str:
        byte_literal = super().visit_Bytes(node)
        n = len(node.s)
        return f"((std::array<unsigned char, {n}>){byte_literal})"

    def visit_Name(self, node) -> str:
        if node.id == "None":
            return "nullptr"
        else:
            return super().visit_Name(node)

    def visit_Constant(self, node) -> str:
        if node.value is True:
            return "true"
        elif node.value is False:
            return "false"
        elif isinstance(node.value, str):
            return self.visit_Str(node)
        else:
            return super().visit_Constant(node)

    def visit_Expr(self, node) -> str:
        s = super().visit_Expr(node)
        if getattr(node, "unused", False):
            s = "(void) " + s
        return s

    def _make_block(self, node):
        buf = []
        buf.append("({")
        buf.extend([self.visit(child) for child in node.body])
        if not buf[-1].endswith(";"):
            buf.append(";")
        buf.append("})")
        node.make_block = True
        return "\n".join(buf)

    def visit_If(self, node) -> str:
        body_vars = set([get_id(v) for v in node.scopes[-1].body_vars])
        orelse_vars = set([get_id(v) for v in node.scopes[-1].orelse_vars])
        node.common_vars = body_vars.intersection(orelse_vars)

        var_definitions = []
        for cv in node.common_vars:
            definition = node.scopes.find(cv)
            var_type = self._typename_from_annotation(definition)
            if var_type == self._default_type:
                var_type = decltype(definition)
            var_definitions.append("{0} {1};\n".format(var_type, cv))

        return "".join(var_definitions) + super().visit_If(node)

    def visit_UnaryOp(self, node) -> str:
        if isinstance(node.op, ast.USub):
            if isinstance(node.operand, (ast.Call, ast.Num)):
                # Shortcut if parenthesis are not needed
                return "-{0}".format(self.visit(node.operand))
            else:
                return "-({0})".format(self.visit(node.operand))
        else:
            return super().visit_UnaryOp(node)

    def visit_BinOp(self, node) -> str:
        if (
            isinstance(node.left, ast.List)
            and isinstance(node.op, ast.Mult)
            and isinstance(node.right, ast.Num)
        ):
            return "std::vector ({0},{1})".format(
                self.visit(node.right), self.visit(node.left.elts[0])
            )
        else:
            return super().visit_BinOp(node)

    def _import(self, name: str) -> str:
        name = MODULE_DISPATCH_TABLE[name] if name in MODULE_DISPATCH_TABLE else name
        if "<" in name:
            return f"#include {name}"
        return f'#include "{name}.h"'

    def _import_from(self, module_name: str, names: List[str]) -> str:
        if len(names) == 1:
            # TODO: make this more generic so it works for len(names) > 1
            name = names[0]
            lookup = f"{module_name}.{name}"
            if lookup in MODULE_DISPATCH_TABLE:
                cpp_module_name, _ = MODULE_DISPATCH_TABLE[lookup]
                cpp_module_name = cpp_module_name.replace(".", "::")
                return f'#include "{cpp_module_name}.h"'
        module_name = module_name.replace(".", "::")
        return f'#include "{module_name}.h"'

    def _get_element_type(self, node):
        _ = self._typename_from_annotation(node)
        if hasattr(node, "container_type"):
            _, element_type = node.container_type
            return element_type
        else:
            return self._default_type

    def visit_List(self, node) -> str:
        self._usings.add("<vector>")
        elements = [self.visit(e) for e in node.elts]
        elements_str = ", ".join(elements)
        element_type = self._get_element_type(node)
        if element_type == self._default_type:
            typename = decltype(node)
            # Workaround for cases where we couldn't figure out type
            if "auto" in typename:
                return f"{{{elements_str}}}"
            return f"{typename}{{{elements_str}}}"
        return f"{{{elements_str}}}"

    def visit_Set(self, node) -> str:
        self._usings.add("<set>")
        elements = [self.visit(e) for e in node.elts]
        elements_str = ", ".join(elements)
        element_type = self._get_element_type(node)
        if element_type == self._default_type:
            typename = decltype(node)
            return f"{typename}{{{elements_str}}}"
        return f"std::set<{element_type}>{{{elements_str}}}"

    def visit_Dict(self, node) -> str:
        self._usings.add("<map>")
        keys = [self.visit(k) for k in node.keys]
        values = [self.visit(k) for k in node.values]
        kv_pairs = ", ".join([f"{{ {k}, {v} }}" for k, v in zip(keys, values)])
        _ = self._typename_from_annotation(node)
        if hasattr(node, "container_type"):
            container_type, element_type = node.container_type
            key_typename, value_typename = element_type
            if key_typename == self._default_type:
                key_typename = "int"
        else:
            typename = decltype(node)
            return f"{typename}{{{kv_pairs}}}"
        return f"std::map<{key_typename}, {value_typename}>{{{kv_pairs}}}"

    def visit_Subscript(self, node) -> str:
        value = self.visit(node.value)
        if isinstance(node.slice, ast.Ellipsis):
            raise AstNotImplementedError("Ellipsis not supported", node)

        slice_value = self._slice_value(node)
        index = self.visit(slice_value)
        if hasattr(node, "is_annotation"):
            if value in self.CONTAINER_TYPES:
                value = self.CONTAINER_TYPES[value]
            return "{0}<{1}>".format(value, index)
        return f"{value}[{index}]"

    def visit_Tuple(self, node) -> str:
        self._headers.append("#include <tuple>")
        elts = [self.visit(e) for e in node.elts]
        return "std::make_tuple({0})".format(", ".join(elts))

    def visit_Try(self, node, finallybody=None) -> str:
        self._usings.add("<iostream>")
        buf = ["try {"]
        buf += [self.visit(n) for n in node.body]
        buf.append("} catch (const std::exception& e) {")

        buf += [self.visit(h) for h in node.handlers]

        if finallybody:
            buf.append("try { // finally")
            buf += [self.visit(b) for b in finallybody]
            buf.append("} throw e;")

        buf.append("}")
        buf.append(
            "catch (const std::overflow_error& e) "
            '{ std::cout << "OVERFLOW ERROR" << std::endl; }'
        )
        buf.append(
            "catch (const std::runtime_error& e) "
            '{ std::cout << "RUNTIME ERROR" << std::endl; }'
        )
        buf.append("catch (...) " '{ std::cout << "UNKNOWN ERROR" << std::endl; 0}')

        return "\n".join(buf)

    def visit_ExceptHandler(self, node) -> str:
        return "ExceptHandler /*unimplemented()*/"

    def visit_Assert(self, node) -> str:
        if not self.use_catch_test_cases:
            self._usings.add("<cassert>")
            return "assert({0});".format(self.visit(node.test))
        return "REQUIRE({0});".format(self.visit(node.test))

    def _visit_AssignOne(self, node, target) -> str:
        if isinstance(target, ast.Tuple):
            elts = [self.visit(e) for e in target.elts]
            value = self.visit(node.value)
            return "std::tie({0}) = {1};".format(", ".join(elts), value)

        if isinstance(node.scopes[-1], ast.If) and isinstance(target, ast.Name):
            outer_if = node.scopes[-1]
            if target.id in outer_if.common_vars:
                value = self.visit(node.value)
                return "{0} = {1};".format(target.id, value)

        if isinstance(target, ast.Subscript):
            target = self.visit(target)
            value = self.visit(node.value)
            return "{0} = {1};".format(target, value)

        definition = node.scopes.parent_scopes.find(get_id(target))
        if definition is None:
            definition = node.scopes.find(get_id(target))
        if isinstance(target, ast.Name) and defined_before(definition, node):
            target = self.visit(target)
            value = self.visit(node.value)
            return "{0} = {1};".format(target, value)

        if isinstance(node.value, ast.List):
            element_type = self._get_element_type(node.value)
            if element_type == self._default_type:
                typename = decltype(node)
            else:
                typename = f"std::vector<{element_type}>"
        else:
            typename = self._typename_from_annotation(target)
        target = self.visit(target)
        value = self.visit(node.value)
        lint_exception = "  // NOLINT(runtime/string)" if not self._no_prologue else ""
        if typename == "std::string" and is_global(node):
            return f"{typename} {target} = {value};{lint_exception}"

        return f"{typename} {target} = {value};"

    def visit_AnnAssign(self, node) -> str:
        target, type_str, val = super().visit_AnnAssign(node)
        return f"{type_str} {target} = {val};"

    def visit_Print(self, node) -> str:
        buf = []
        for n in node.values:
            value = self.visit(n)
            if isinstance(n, ast.List) or isinstance(n, ast.Tuple):
                buf.append(
                    "std::cout << {0} << std::endl;".format(
                        " << ".join([self.visit(el) for el in n.elts])
                    )
                )
            else:
                buf.append("std::cout << {0} << std::endl;".format(value))
        return "\n".join(buf)

    def visit_GeneratorExp(self, node) -> str:
        return self.visit_unsupported_body(node, "generator exp", [])

    def visit_Raise(self, node) -> str:
        return self.visit_unsupported_body(node, "raise", [])

    def visit_Starred(self, node) -> str:
        return self.visit_unsupported_body(node, "starred", [])
