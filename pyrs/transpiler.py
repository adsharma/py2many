import ast
from py2many.rewriters import camel_case
import textwrap

from .clike import CLikeTranspiler
from .inference import get_inferred_rust_type, map_type
from .plugins import (
    ATTR_DISPATCH_TABLE,
    CLASS_DISPATCH_TABLE,
    FUNC_DISPATCH_TABLE,
    FUNC_USINGS_MAP,
    MODULE_DISPATCH_TABLE,
    DISPATCH_MAP,
    SMALL_DISPATCH_MAP,
    SMALL_USINGS_MAP,
)

from py2many.analysis import (
    FunctionTransformer,
    get_id,
    is_global,
    is_mutable,
    is_void_function,
)
from py2many.clike import class_for_typename
from py2many.declaration_extractor import DeclarationExtractor
from py2many.exceptions import AstClassUsedBeforeDeclaration
from py2many.inference import is_reference
from py2many.tracer import is_list, defined_before, is_class_or_module

from pathlib import Path
from typing import List, Tuple, Union


class RustLoopIndexRewriter(ast.NodeTransformer):
    def visit_For(self, node):
        if hasattr(node.iter, "id"):
            definition = node.scopes.find(node.iter.id)
            if is_reference(definition):
                node.target.needs_dereference = True
        return node


class RustNoneCompareRewriter(ast.NodeTransformer):
    def visit_Compare(self, node):
        right = self.visit(node.comparators[0])
        if isinstance(right, ast.Constant) and right.value is None:
            node.left = ast.Call(
                func=ast.Name(id="Some", ctx=ast.Load()),
                args=[node.left],
                lineno=node.left.lineno,
                keywords=[],
            )
        return node


class RustStringJoinRewriter(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            if hasattr(node.func.value, "value") and isinstance(
                node.func.value.value, str
            ):
                new_call = ast.Call(
                    ast.Attribute(value=node.args[0], attr=node.func.attr, ctx="Load"),
                    args=[node.func.value],
                    keywords=[],
                )
                new_call.lineno = node.lineno
                new_call.col_offset = node.col_offset
                new_call.scopes = node.scopes
                new_call.func.scopes = node.scopes
                ast.fix_missing_locations(new_call)
                return new_call
        return node


class RustTranspiler(CLikeTranspiler):
    NAME = "rust"

    CONTAINER_TYPE_MAP = {
        "List": "Vec",
        "Dict": "HashMap",
        "Set": "HashSet",
        "Optional": "Option",
        "Result": "Result",
    }

    def __init__(self, extension: bool = False, no_prologue: bool = False):
        super().__init__()
        self._container_type_map = self.CONTAINER_TYPE_MAP
        self._default_type = "_"
        self._extension = extension
        self._rust_ignored_module_set = {"argparse_dataclass"}
        self._no_prologue = no_prologue
        self._dispatch_map = DISPATCH_MAP
        self._small_dispatch_map = SMALL_DISPATCH_MAP
        self._small_usings_map = SMALL_USINGS_MAP
        self._func_dispatch_table = FUNC_DISPATCH_TABLE
        self._func_usings_map = FUNC_USINGS_MAP
        self._attr_dispatch_table = ATTR_DISPATCH_TABLE
        self._allows = set()

    def usings(self):
        if self._extension:
            self._usings.add("pyo3::prelude::*")
            self._usings.add("pyo3::wrap_pyfunction")
        usings = sorted(list(set(self._usings)))
        deps = sorted(
            set(mod.split("::")[0] for mod in usings if not mod.startswith("std:"))
        )
        externs = [f"extern crate {dep};" for dep in deps]
        deps_str = "\n//! ".join([f'{dep} = "*"' for dep in deps])
        externs = "\n".join(externs)
        uses = "\n".join(
            f"use {mod};"
            for mod in usings
            if mod not in ("strum", "lazy_static") and mod not in deps
        )
        # This should not contain lints which mask possibly erroneous semantics, e.g. float_cmp
        # Those, and lints triggered by explicitly bad test logic, belong in
        # .github/workflows/clippy.yml
        lint_ignores = (
            textwrap.dedent(
                """
        #![allow(clippy::collapsible_else_if)]
        #![allow(clippy::double_parens)]  // https://github.com/adsharma/py2many/issues/17
        #![allow(clippy::map_identity)]
        #![allow(clippy::needless_return)]
        #![allow(clippy::print_literal)]
        #![allow(clippy::ptr_arg)]
        #![allow(clippy::redundant_static_lifetimes)]  // https://github.com/adsharma/py2many/issues/266
        #![allow(clippy::unnecessary_cast)]
        #![allow(clippy::upper_case_acronyms)]
        #![allow(clippy::useless_vec)]
        #![allow(non_camel_case_types)]
        #![allow(non_snake_case)]
        #![allow(non_upper_case_globals)]
        #![allow(unused_imports)]
        #![allow(unused_mut)]
        #![allow(unused_parens)]
        """
            )
            if not self._no_prologue
            else ""
        )
        lint_ignores += "\n".join(f"#![allow({allow})]" for allow in self._allows)
        cargo_toml = (
            f"""\
        //! ```cargo
        //! [package]
        //! edition = "2018"
        //! [dependencies]
        //! {deps_str}
        //! ```
        """
            if not self._no_prologue
            else ""
        )
        return f"{cargo_toml}\n{lint_ignores}\n\n{externs}\n{uses}\n"

    def features(self):
        if self._features:
            features = ", ".join(sorted(list(set(self._features))))
            return f"#![feature({features})]"

    def extension_module(self, tree) -> str:
        if self._extension:
            funcs = []
            tx = FunctionTransformer()
            tx.visit(tree)
            for f in tree.defined_functions:
                fname = f.name
                if fname.startswith("_"):
                    continue
                funcs.append(f"m.add_function(wrap_pyfunction!({fname}, m)?)?;")
            funcs_str = "\n".join(funcs)
            module_name = Path(tree.__file__).stem
            return textwrap.dedent(
                f"""\
            #[pymodule]
            fn {module_name}(_py: Python, m: &PyModule) -> PyResult<()> {{
                {funcs_str}

                Ok(())
            }}
            """
            )
        return ""

    def visit_Expr(self, node) -> str:
        if hasattr(node, "unused"):
            self._allows.add("clippy::no_effect")
        return super().visit_Expr(node)

    def visit_FunctionDef(self, node, async_prefix="") -> str:
        body = "\n".join([self.visit(n) for n in node.body])
        typenames, args = self.visit(node.args)

        args_list = []
        if args and args[0] == "self":
            del typenames[0]
            del args[0]
            args_list.append("&self")

        is_python_main = getattr(node, "python_main", False)
        if is_python_main:
            self._usings.add("anyhow::Result")

        typedecls = []
        index = 0
        for i in range(len(args)):
            typename = typenames[i]
            arg = args[i]

            if typename == "T":
                typename = "T{0}".format(index)
                typedecls.append(typename)
                index += 1
            args_list.append("{0}: {1}".format(arg, typename))

        return_type = "" if not is_python_main else "-> Result<()>"
        if node.returns:
            typename = self._typename_from_annotation(node, attr="returns")
            if getattr(node.returns, "rust_needs_reference", False):
                typename = f"&{typename}"
            if getattr(node, "rust_pyresult_type", False):
                if node.no_return:
                    typename = None
                else:
                    typename = self._generic_typename_from_type_node(node.returns)
                typename = map_type(typename, extension=True, return_type=True)
            if typename != "_":
                return_type = f"-> {typename}"
        else:
            if not is_void_function(node):
                return_type = "-> RT"
                typedecls.append("RT")

        template = ""
        if len(typedecls) > 0:
            template = "<{0}>".format(", ".join(typedecls))

        extension = "#[pyfunction]\n" if self.extension else ""
        args_list = ", ".join(args_list)
        funcdef = f"{extension}pub {async_prefix}fn {node.name}{template}({args_list}) {return_type}"
        return_success = (
            "Ok(())" if is_python_main else ""
        )  # TODO: generalize this to functions that return Result<T, E>
        return f"{funcdef} {{\n{body}\n {return_success}}}\n"

    def visit_arg(self, node):
        id = get_id(node)
        if id == "self":
            return (None, "self")
        typename = "T"
        if node.annotation:
            if not self._extension:
                typename = self._typename_from_annotation(node)
            else:
                typename = self._generic_typename_from_annotation(node)
                typename = map_type(typename, extension=True)
            mut = "mut " if is_mutable(node.scopes, id) else ""
            # TODO: Should we make this if not primitive instead of checking
            # for container types? That way we cover user defined structs too.
            if hasattr(node, "container_type"):
                # Python passes by reference by default. Rust needs explicit borrowing
                typename = f"&{mut}{typename}"
        return (typename, id)

    def visit_Return(self, node) -> str:
        fndef = None
        for scope in node.scopes:
            if isinstance(scope, ast.FunctionDef):
                fndef = scope
                break
        if node.value:
            ret = self.visit(node.value)
            if fndef:
                if getattr(fndef, "rust_pyresult_type", False):
                    # TODO: Design a more robust solution for this
                    # For now, PyResult and references don't mix
                    if ret.startswith("&"):
                        ret = ret[1:]
                    ret = f"Ok({ret})"
                return_type = self._typename_from_annotation(fndef, attr="returns")
                value_type = get_inferred_rust_type(node.value)
                if is_reference(node.value) and not getattr(
                    fndef.returns, "rust_needs_reference", True
                ):
                    # TODO: Handle other container types
                    ret = f"{ret}.to_vec()"
                if return_type != value_type and value_type is not None:
                    return f"return {ret} as {return_type};"
            return f"return {ret};"
        if fndef:
            if getattr(fndef, "rust_pyresult_type", False):
                return "return Ok(())"
        return "return;"

    def visit_Lambda(self, node) -> str:
        _, args = self.visit(node.args)
        args_string = ", ".join(args)
        body = self.visit(node.body)
        return "|{0}| {1}".format(args_string, body)

    def visit_Attribute(self, node) -> str:
        attr = node.attr
        if attr in self._keywords:
            attr = attr + "_"

        value_id = self.visit(node.value)

        if value_id == "sys":
            if attr == "argv":
                self._usings.add("std::env")
                return "env::args().map(|s| &*Box::leak(s.into_boxed_str())).collect()"

        if is_list(node.value):
            if node.attr == "append":
                attr = "push"
        if not value_id:
            value_id = ""

        if is_class_or_module(value_id, node.scopes):
            return "{0}::{1}".format(value_id, attr)

        ret = f"{value_id}.{attr}"
        if ret in self._attr_dispatch_table:
            ret = self._attr_dispatch_table[ret]
            return ret(self, node, value_id, attr)
        return ret

    def _func_for_lookup(self, fname) -> Union[str, object]:
        fname_for_lookup = fname.replace("::", ".")
        return super()._func_for_lookup(fname_for_lookup)

    def _func_name_split(self, fname: str) -> Tuple[str, str]:
        # string based fallback
        splits = fname.rsplit("::", maxsplit=1)
        if len(splits) == 2:
            return tuple(splits)
        else:
            return ("", splits[0])

    def _visit_struct_literal(self, node, fname: str, fndef: ast.ClassDef) -> str:
        vargs = []  # visited args
        if not hasattr(fndef, "declarations"):
            raise AstClassUsedBeforeDeclaration(fndef, node)
        if node.args:
            for arg, decl in zip(node.args, fndef.declarations.keys()):
                arg = self.visit(arg)
                if decl in self._keywords:
                    decl += "_"
                vargs += [f"{decl}: {arg}"]
        if node.keywords:
            for kw in node.keywords:
                value = self.visit(kw.value)
                if kw.arg in self._keywords:
                    kw.arg += "_"
                vargs += [f"{kw.arg}: {value}"]
        args = ", ".join(vargs)
        return f"{fname}{{{args}}}"

    def visit_Call(self, node) -> str:
        fname = self.visit(node.func)
        fndef = node.scopes.find(fname)

        if isinstance(fndef, ast.ClassDef):
            return self._visit_struct_literal(node, fname, fndef)

        vargs = []  # visited args
        if node.args:
            vargs += [self.visit(a) for a in node.args]
        if node.keywords:
            vargs += [self.visit(kw.value) for kw in node.keywords]

        ret = self._dispatch(node, fname, vargs)
        node_result_type = getattr(node, "result_type", False)
        node_func_result_type = getattr(node.func, "result_type", False)
        if ret is not None:
            if ret.startswith("std::process::exit("):
                node_result_type = False
            unwrap = "?" if node_result_type or node_func_result_type else ""
            return f"{ret}{unwrap}"

        # Check if some args need to be passed by reference
        ref_args = []
        if fndef and hasattr(fndef, "args"):
            for varg, fnarg, node_arg in zip(vargs, fndef.args.args, node.args):
                if is_reference(fnarg) and not is_reference(node_arg):
                    ref_args.append(f"&{varg}")
                else:
                    ref_args.append(varg)
        else:
            ref_args = vargs

        args = ", ".join(ref_args)
        unwrap = "?" if node_result_type or node_func_result_type else ""
        return f"{fname}({args}){unwrap}"

    def visit_For(self, node) -> str:
        target = self.visit(node.target)
        it = self.visit(node.iter)
        buf = []
        buf.append("for {0} in {1} {{".format(target, it))
        buf.extend([self.visit(c) for c in node.body])
        buf.append("}")
        return "\n".join(buf)

    def visit_Str(self, node) -> str:
        return "" + super().visit_Str(node) + ""

    def visit_Bytes(self, node) -> str:
        bytes_str = node.s
        bytes_str = bytes_str.replace(b'"', b'\\"')
        return 'b"' + bytes_str.decode("ascii", "backslashreplace") + '"'

    def visit_Compare(self, node) -> str:
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])

        if hasattr(node.comparators[0], "annotation"):
            self._generic_typename_from_annotation(node.comparators[0])
            value_type = getattr(
                node.comparators[0].annotation, "generic_container_type", None
            )
            if value_type and value_type[0] == "Dict":
                right += ".keys()"

        if not right.endswith(".keys()") and not right.endswith(".values()"):
            right += ".iter()"

        if isinstance(node.ops[0], ast.In):
            return "{0}.any(|&x| x == {1})".format(right, left)  # is it too much?
        elif isinstance(node.ops[0], ast.NotIn):
            return "{0}.all(|&x| x != {1})".format(right, left)  # is it even more?

        return super().visit_Compare(node)

    def visit_Name(self, node) -> str:
        if node.id == "None":
            return "None"
        else:
            ret = super().visit_Name(node)
            definition = node.scopes.find(node.id)
            if (
                definition
                and definition != node
                and getattr(definition, "needs_dereference", False)
            ):
                return f"*{ret}"
            return ret

    def visit_NameConstant(self, node) -> str:
        if node.value is True:
            return "true"
        elif node.value is False:
            return "false"
        elif node.value is None:
            return "None"
        else:
            return super().visit_NameConstant(node)

    def visit_If(self, node) -> str:
        body_vars = set([get_id(v) for v in node.scopes[-1].body_vars])
        orelse_vars = set([get_id(v) for v in node.scopes[-1].orelse_vars])
        node.common_vars = body_vars.intersection(orelse_vars)

        # TODO find out if this can be useful
        var_definitions = []
        # for cv in node.common_vars:
        #     definition = node.scopes.find(cv)
        #     var_type = decltype(definition)
        #     var_definitions.append("{0} {1};\n".format(var_type, cv))
        ret = "".join(var_definitions) + super().visit_If(node, use_parens=False)
        # Sometimes if True: ... gets compiled into an expression, needing a semicolon
        make_block = (
            isinstance(node.test, ast.Constant)
            and node.test.value == True
            and node.orelse == []
        )
        if make_block:
            return f"{ret};"
        return ret

    def visit_While(self, node) -> str:
        test = self.visit(node.test)
        if test == "true":
            body = [self.visit(n) for n in node.body]
            return "loop {{\n{}\n}}\n".format("\n".join(body))
        return super().visit_While(node, use_parens=False)

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
            elt, n = self.visit(node.left.elts[0]), self.visit(node.right)
            return f"vec![{elt};{n}]"
        else:
            return super().visit_BinOp(node)

    def visit_sealed_class(self, node) -> str:
        variants = []
        accessors = []
        tag_checkers = []
        camel_node_name = camel_case(node.name)
        for member, var in node.class_assignments.items():
            member_id = get_id(member)
            camel_member_id = camel_case(member_id)
            lower_member_id = member_id.lower()
            typename = node.declarations_with_defaults.get(member_id, None)
            if typename == None:
                variants.append(f"{member_id},")
            else:
                typename, _ = typename
                variants.append(f"{member_id}({typename}),")
            tag_check = textwrap.dedent(
                f"""
                    fn is_{lower_member_id}(&self) -> bool {{
                        matches!(*self, {camel_node_name}::{camel_member_id}(_))
                    }}"""
            )
            tag_checkers.append(tag_check)
            accessor = textwrap.dedent(
                f"""
                    fn {lower_member_id}(&self) -> Option<&{typename}>{{
                        if let {camel_node_name}::{camel_member_id}(val) = self {{
                            Some(val)
                        }} else {{
                            None
                        }}
                    }}"""
            )
            accessors.append(accessor)
        body_str = "\n".join(variants)
        impl_str = "\n".join(accessors) + "\n" + "\n".join(tag_checkers)
        return f"enum {camel_node_name} {{ {body_str} }}\n\n impl {camel_node_name} {{ {impl_str} }}\n\n"

    def visit_ClassDef(self, node) -> str:
        extractor = DeclarationExtractor(RustTranspiler())
        extractor.visit(node)
        node.declarations = declarations = extractor.get_declarations()
        node.declarations_with_defaults = extractor.get_declarations_with_defaults()
        node.class_assignments = extractor.class_assignments
        ret = super().visit_ClassDef(node)
        if ret is not None:
            return ret

        decorators = [get_id(d) for d in node.decorator_list]
        if "sealed" in decorators:
            # TODO: handle cases where sealed is stacked with other decorators
            return self.visit_sealed_class(node)

        decorators = [
            class_for_typename(t, None, self._imported_names) for t in decorators
        ]
        for d in decorators:
            if d in CLASS_DISPATCH_TABLE:
                ret = CLASS_DISPATCH_TABLE[d](self, node)
                if ret is not None:
                    return ret

        fields = []
        index = 0
        for declaration, typename in declarations.items():
            if typename == None:
                typename = "ST{0}".format(index)
                index += 1
            fields.append(f"pub {declaration}: {typename},")

        extension = "#[pyclass]\n" if self.extension else ""
        struct_def = "pub struct {0} {{\n{1}\n}}\n\n".format(
            node.name, "\n".join(fields)
        )
        impl_extension = "#[pymethods]\n" if self.extension else ""
        impl_def = f"{impl_extension}impl {node.name} {{\n"
        buf = [self.visit(b) for b in node.body]
        buf_str = "\n".join(buf)
        return f"{extension}{struct_def}{impl_def}{buf_str} \n}}"

    def visit_IntEnum(self, node) -> str:
        fields = []
        for member, var in node.class_assignments.items():
            var = self.visit(var)
            if var == "auto()":
                fields.append(f"{member},")
            else:
                fields.append(f"{member} = {var},")
        fields = "\n".join(fields)
        return f"#[derive(Clone, Eq, Hash, PartialEq)]\npub enum {node.name} {{\n{fields}\n}}\n\n"

    def visit_StrEnum(self, node) -> str:
        self._usings.add("strum")
        self._usings.add("strum_macros::{Display, EnumString, EnumVariantNames}")

        fields = []
        for member, var in node.class_assignments.items():
            var = self.visit(var)
            if var == "auto()":
                fields.append(f"{member},")
            else:
                fields.append(f"#[strum(serialize = {var})]{member},")
        fields = "\n".join(fields)

        return f"#[derive(Clone, Debug, Display, EnumString, EnumVariantNames, Eq, Hash, PartialEq)]\npub enum {node.name} {{\n{fields}\n}}\n\n"

    def visit_IntFlag(self, node) -> str:
        self._usings.add("flagset::flags")
        self._usings.add("std::os::raw::c_int")
        fields = []
        for member, var in node.class_assignments.items():
            var = self.visit(var)
            if var == "auto()":
                fields.append(f"{member},")
            else:
                fields.append(f"{member} = {var},")
        fields = "\n".join(["    " * 2 + f for f in fields])
        return (
            f"flags! {{\n    pub enum {node.name}: c_int {{\n{fields}\n    }}\n}}\n\n"
        )

    def _import(self, name: str) -> str:
        if name not in self._rust_ignored_module_set:
            self._usings.add(name)
        return ""

    def _import_from(self, module_name: str, names: List[str]) -> str:
        if module_name in self._rust_ignored_module_set:
            return ""
        self._usings.add(module_name)
        if len(names) == 1:
            # TODO: make this more generic so it works for len(names) > 1
            name = names[0]
            lookup = f"{module_name}.{name}"
            if lookup in MODULE_DISPATCH_TABLE:
                rust_use = MODULE_DISPATCH_TABLE[lookup]
                return f"use {rust_use};"
        module_name = module_name.replace(".", "::")
        names = ", ".join(names)
        return f"use {module_name}::{{{names}}};"

    def visit_List(self, node) -> str:
        self._usings.add("std::collections")
        if len(node.elts) > 0:
            elements = [self.visit(e) for e in node.elts]
            return "vec![{0}]".format(", ".join(elements))

        else:
            return "vec![]"

    def visit_Dict(self, node) -> str:
        self._usings.add("std::collections::HashMap")
        if len(node.keys) > 0:
            self._usings.add("std::collections::HashMap")
            kv_string = []
            for i in range(len(node.keys)):
                key = self.visit(node.keys[i])
                value = self.visit(node.values[i])
                kv_string.append("({0}, {1})".format(key, value))
            initialization = "[{0}].iter().cloned().collect::<HashMap<_,_>>()"
            return initialization.format(", ".join(kv_string))
        else:
            return "HashMap::new()"

    def _cast(self, name: str, to) -> str:
        return f"{name} as {to}"

    def visit_Subscript(self, node) -> str:
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if hasattr(node, "is_annotation"):
            if value in self.CONTAINER_TYPE_MAP:
                self._usings.add("std::collections")
                value = self.CONTAINER_TYPE_MAP[value]
            if value == "Tuple":
                return "({0})".format(index)
            return "{0}<{1}>".format(value, index)
        # TODO: optimize this. We need to compute value_type once per definition
        self._generic_typename_from_annotation(node.value)
        if hasattr(node.value, "annotation"):
            value_type = getattr(node.value.annotation, "generic_container_type", None)
            is_list = value_type is not None and value_type[0] == "List"
            if is_list:
                index_typename = get_inferred_rust_type(self._slice_value(node))
                if index_typename != "u64" or index_typename != "usize":
                    index = self._cast(index, "usize")
            is_dict = value_type is not None and value_type[0] == "Dict"
            if is_dict:
                value_type = getattr(node.value.annotation, "container_type", None)
                index_typename = get_inferred_rust_type(self._slice_value(node))
                if index_typename == value_type[1][0]:
                    index = "&" + index
        return "{0}[{1}]".format(value, index)

    def visit_Index(self, node) -> str:
        return self.visit(node.value)

    def visit_Slice(self, node) -> str:
        lower = ""
        if node.lower:
            lower = self.visit(node.lower)
        upper = ""
        if node.upper:
            upper = self.visit(node.upper)

        return "{0}..{1}".format(lower, upper)

    def visit_Tuple(self, node) -> str:
        elts = [self.visit(e) for e in node.elts]
        elts = ", ".join(elts)
        if hasattr(node, "is_annotation"):
            return elts
        return "({0})".format(elts)

    def visit_Try(self, node, finallybody=None) -> str:
        # buf = self.visit_unsupported_body(node, "try_dummy", node.body)
        buf = ["if true {"]
        buf += [self.visit(n) for n in node.body]

        for handler in node.handlers:
            buf += self.visit(handler)
        # buf.append("\n".join(excepts));

        if finallybody:
            buf += [self.visit(n) for n in finallybody]

        buf += ["}"]
        return "\n".join(buf)

    def visit_ExceptHandler(self, node):
        exception_type = ""
        if node.type:
            exception_type = self.visit(node.type)
        name = "// except!({0})".format(exception_type)
        body = self.visit_unsupported_body(node, name, node.body)
        body = [self.comment(b) for b in body]
        return body

    def visit_Assert(self, node) -> str:
        return "assert!({0});".format(self.visit(node.test))

    def _compute_kw(self, node, target) -> str:
        kw = "let"
        mut = is_mutable(node.scopes, get_id(target))
        if is_global(node) or getattr(node, "class_assignment", False):
            # Note that static are not really supported, as modifying them requires adding
            # "unsafe" blocks, which pyrs does not do.
            kw = "pub static" if mut else "pub const"
        elif mut:
            kw = "let mut"
        return kw

    def visit_AnnAssign(self, node) -> str:
        target, type_str, val = super().visit_AnnAssign(node)
        kw = self._compute_kw(node, node.target)
        return f"{kw} {target}: {type_str} = {val};"

    def _needs_cast(self, left, right) -> bool:
        if not hasattr(left, "annotation") or not hasattr(right, "annotation"):
            return False
        left_type = self._typename_from_annotation(left)
        # populate right.rust_annotation
        get_inferred_rust_type(right)
        right_type = self._typename_from_annotation(right)
        return left_type != right_type and left_type != self._default_type

    def _assign_cast(
        self, value_str: str, cast_to: str, python_annotation, rust_annotation
    ) -> str:
        # python/rust annotations provided to customize the cast if necessary
        return f"{value_str} as {cast_to}"

    def visit_AugAssign(self, node) -> str:
        target = node.target
        target_str = self.visit(node.target)
        op = self.visit(node.op)
        value = self.visit(node.value)

        needs_cast = self._needs_cast(target, node.value)
        if needs_cast:
            target_type = self._typename_from_annotation(target)
            value = self._assign_cast(
                value, target_type, target.annotation, node.value.rust_annotation
            )
        return f"{target_str} {op}= {value};"

    def _visit_AssignOne(self, node, target) -> str:
        kw = self._compute_kw(node, target)

        if isinstance(node.scopes[-1], ast.If):
            outer_if = node.scopes[-1]
            target_id = self.visit(target)
            if target_id in outer_if.common_vars:
                value = self.visit(node.value)
                return "{0} = {1};".format(target_id, value)

        if isinstance(target, ast.Subscript) or isinstance(target, ast.Attribute):
            target = self.visit(target)
            value = self.visit(node.value)
            if value == None:
                value = "None"
            return "{0} = {1};".format(target, value)

        definition = node.scopes.parent_scopes.find(get_id(target))
        if definition is None:
            definition = node.scopes.find(get_id(target))
        if isinstance(target, ast.Name) and defined_before(definition, node):
            needs_cast = self._needs_cast(target, node.value)
            target_str = self.visit(target)
            value = self.visit(node.value)
            if needs_cast:
                target_type = self._typename_from_annotation(target)
                value = self._assign_cast(
                    value, target_type, target.annotation, node.value.rust_annotation
                )
            return f"{target_str} = {value};"
        elif isinstance(node.value, ast.List) and kw.startswith("pub "):
            count = len(node.value.elts)
            target = self.visit(target)
            value = self.visit(node.value)
            # populate node.value.container_type
            self._typename_from_annotation(node.value)
            # Use arrays instead of Vec as globals must have fixed size
            if value.startswith("vec!"):
                value = value.replace("vec!", "&")
            element_type = self._default_type
            if hasattr(node.value, "container_type"):
                container_type, element_type = node.value.container_type
            return f"{kw} {target}: &[{element_type}; {count}] = {value};"
        elif isinstance(node.value, ast.Set) and kw.startswith("pub "):
            target = self.visit(target)
            value = self.visit(node.value)
            typename = self._typename_from_annotation(node.value)

            self._usings.add("lazy_static::lazy_static")
            if "str" in typename:
                typename = typename.replace("str", "'static str")
            return f"lazy_static! {{ pub static ref {target}: {typename} = {value}; }}"
        elif isinstance(node.value, ast.Dict) and kw.startswith("pub "):
            target = self.visit(target)
            value = self.visit(node.value)
            typename = self._typename_from_annotation(node.value)

            self._usings.add("lazy_static::lazy_static")
            if hasattr(node.value, "container_type"):
                container_type, element_type = node.value.container_type
                key_typename, value_typename = element_type
                if key_typename == "&str":
                    key_typename = "&'static str"
                if value_typename == "&str":
                    value_typename = "&'static str"
                typename = f"{key_typename}, {value_typename}"

            return f"lazy_static! {{ pub static ref {target}: HashMap<{typename}> = {value}; }}"
        else:
            typename = self._typename_from_annotation(target)
            needs_cast = self._needs_cast(target, node.value)
            target_str = self.visit(target)
            value = self.visit(node.value)
            if needs_cast:
                value = self._assign_cast(
                    value, typename, target.annotation, node.value.annotation
                )
            if hasattr(node.value, "container_type"):
                mut = "mut " if is_mutable(node.scopes, target_str) else ""
                typename = f"&{mut}{typename}"
                value = f"&{mut}{value}"
            optional_typename = (
                f": {typename}" if typename != self._default_type else ""
            )
            return f"{kw} {target_str}{optional_typename} = {value};"

    def visit_Delete(self, node) -> str:
        target = node.targets[0]
        target_str = self.visit(target)
        return f"drop({target_str});"

    def visit_Raise(self, node) -> str:
        if node.exc is not None:
            return "raise!({0}); //unsupported".format(self.visit(node.exc))
        # This handles the case where `raise` is used without
        # specifying the exception.
        return "raise!(); //unsupported"

    def visit_Await(self, node) -> str:
        value = self.visit(node.value)
        return f"{value}.await"

    def visit_AsyncFunctionDef(self, node) -> str:
        return self.visit_FunctionDef(node, async_prefix="async ")

    def visit_Yield(self, node) -> str:
        self._features.add("generators")
        self._features.add("generator_trait")
        self._usings.add("std::ops::Generator")
        self._usings.add("std::ops::GeneratorState")
        if node.value is not None:
            value = self.visit(node.value)
            return f"yield {value};"
        else:
            return "yield None;"

    def visit_Print(self, node) -> str:
        buf = []
        for n in node.values:
            value = self.visit(n)
            buf.append('println!("{{:?}}",{0});'.format(value))
        return "\n".join(buf)

    def visit_GeneratorExp(self, node) -> str:
        elt = self.visit(node.elt)
        generator = node.generators[0]
        target = self.visit(generator.target)
        iter = self.visit(generator.iter)

        is_range = (
            ("range" in get_id(generator.iter.func))
            if isinstance(generator.iter, ast.Call) and get_id(generator.iter.func)
            else False
        )
        # HACK for dictionary iterators to work
        if not (iter.endswith("keys()") or iter.endswith("values()")) and not is_range:
            iter += ".iter()"

        map_str = ".map(|{0}| {1})".format(target, elt)
        filter_str = ""
        if generator.ifs:
            filter_str = ".cloned().filter(|&{0}| {1})".format(
                target, self.visit(generator.ifs[0])
            )

        return "{0}{1}{2}.collect::<Vec<_>>()".format(iter, filter_str, map_str)

    def visit_ListComp(self, node) -> str:
        return self.visit_GeneratorExp(node)  # right now they are the same

    def visit_Global(self, node) -> str:
        return "//global {0}".format(", ".join(node.names))

    def visit_Starred(self, node) -> str:
        return "starred!({0})/*unsupported*/".format(self.visit(node.value))

    def visit_Set(self, node) -> str:
        self._usings.add("std::collections::HashSet")
        elts = []
        for i in range(len(node.elts)):
            elt = self.visit(node.elts[i])
            elts.append(elt)

        if elts:
            initialization = "[{0}].iter().cloned().collect::<HashSet<_>>()"
            return initialization.format(", ".join(elts))
        else:
            return "HashSet::new()"

    def visit_IfExp(self, node) -> str:
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        test = self.visit(node.test)
        return f"if {test} {{ {body} }} else {{ {orelse} }}"
