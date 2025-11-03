from docutils.parsers.rst import Directive, directives
from docutils import nodes
import importlib
import inspect
import docstring_parser


DEFAULT_DOCSTRING = """
Default configuration for pyopenms_viz

Attributes:
    x (str): The column name for the X-axis data. Required.
    y (str): The column name for the Y-axis data. Required.
    by (str): The column name for the grouping variable.
    canvas (Any): Canvas for the plot. For Bokeh, this is a bokeh.plotting.Figure object. For Matplotlib, this is an Axes object, and for Plotly, this is a plotly.graph_objects.Figure object. If none, axis will be created Defaults to None.
    show_plot (bool): Whether to display the plot. Defaults to True.
"""


class DocstringToTableDirective(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "docstring": str,
        "title": str,
        "parent_depth": int,  # Number of parent classes to include
        "default_docstring": directives.flag,  # Flag, no argument required
    }

    def run(self):
        docstring_path = self.options.get("docstring")
        table_title = self.options.get("title")
        # If parent_depth is not specified, only include the base class (depth=0)
        parent_depth = self.options.get("parent_depth")
        if parent_depth is None:
            parent_depth = 0
        else:
            parent_depth = int(parent_depth)
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

        # Collect docstrings from parent classes up to parent_depth
        docstrings = []
        current_obj = obj
        for i in range(parent_depth + 1):
            docstring = inspect.getdoc(current_obj)
            if docstring:
                docstrings.append(docstring)
            bases = getattr(current_obj, "__bases__", ())
            if bases and i < parent_depth:
                current_obj = bases[0]
            else:
                break

        # Parse all collected docstrings
        params = []
        param_names = []
        # If :default_docstring: is present (flag), prepend its params
        if "default_docstring" in self.options:
            default_parsed = docstring_parser.parse(DEFAULT_DOCSTRING)
            for param in default_parsed.params:
                name = param.arg_name or ""
                default = param.default or ""
                typ = param.type_name or ""
                desc = param.description or ""
                if not default:
                    name = f"{name}*"
                params.append((name, typ, desc, default))
        for docstring in reversed(docstrings):  # Start from base class
            parsed = docstring_parser.parse(docstring)
            for param in parsed.params:
                name = param.arg_name or ""
                default = param.default or ""
                typ = param.type_name or ""
                desc = param.description or ""
                # Mark required parameters (no default) with '*'
                if not default:
                    name_out = f"{name}*"
                else:
                    name_out = name
                # Only keep the most "child" definition of each parameter
                if name in param_names:
                    # Find and remove the old parameter from the list
                    # It could be with or without a star
                    for i, (p_name, _, _, _) in enumerate(params):
                        if p_name.strip("*") == name:
                            params.pop(i)
                            break
                params.append((name_out, typ, desc, default))
                if name not in param_names:
                    param_names.append(name)

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
