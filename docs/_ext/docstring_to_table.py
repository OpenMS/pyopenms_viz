from docutils.parsers.rst import Directive
from docutils import nodes


class DocstringToTableDirective(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "docstring": str,
        "title": str,
    }

    def run(self):
        import importlib
        import inspect
        import docstring_parser

        docstring_path = self.options.get("docstring")
        table_title = self.options.get("title")
        if not docstring_path:
            error = self.state_machine.reporter.error(
                "No :docstring: option provided to docstring_to_table directive.",
                line=self.lineno,
            )
            return [error]

        # Split module and object
        mod_name, _, obj_path = docstring_path.partition(".")
        if not obj_path:
            error = self.state_machine.reporter.error(
                f"Invalid docstring path: {docstring_path}", line=self.lineno
            )
            return [error]

        # Import module and get object
        try:
            mod = importlib.import_module(mod_name)
            obj = mod
            for attr in docstring_path.split(".")[1:]:
                obj = getattr(obj, attr)
        except Exception as e:
            error = self.state_machine.reporter.error(
                f"Could not import object '{docstring_path}': {e}", line=self.lineno
            )
            return [error]

        # Get docstring
        docstring = inspect.getdoc(obj)
        if not docstring:
            error = self.state_machine.reporter.error(
                f"No docstring found for '{docstring_path}'", line=self.lineno
            )
            return [error]

        # Parse docstring using docstring_parser
        parsed = docstring_parser.parse(docstring)
        params = []
        for param in parsed.params:
            name = param.arg_name or ""
            default = param.default or ""
            typ = param.type_name or ""
            desc = param.description or ""
            # Mark required parameters (no default) with '*'
            if not default:
                name = f"{name}*"
            params.append((name, typ, desc, default))

        # Build table
        table = nodes.table()
        if table_title:
            title_node = nodes.title(text=table_title)
            table += title_node
        tgroup = nodes.tgroup(cols=4)
        table += tgroup
        for width in [1, 1, 3, 1]:
            tgroup += nodes.colspec(colwidth=width)
        thead = nodes.thead()
        tgroup += thead
        header_row = nodes.row()
        for h in ["Parameter", "Type", "Description", "Default"]:
            entry = nodes.entry()
            entry += nodes.paragraph(text=h)
            header_row += entry
        thead += header_row
        tbody = nodes.tbody()
        tgroup += tbody
        for param in params:
            row = nodes.row()
            for cell in param:
                entry = nodes.entry()
                entry += nodes.paragraph(text=cell)
                row += entry
            tbody += row
        return [table]


def setup(app):
    app.add_directive("docstring_to_table", DocstringToTableDirective)
