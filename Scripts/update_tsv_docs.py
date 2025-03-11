#!/usr/bin/env python3
"""
TSV updater with full inheritance handling and type normalization.
"""
import re
import ast
import logging
from dataclasses import is_dataclass
from typing import Dict, List, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Configuration
CONFIG_PATH = Path("pyopenms_viz/_config.py")
DOCS_DIR = Path("docs/Parameters")
CLASS_TO_TSV_MAP = {
    "BasePlotConfig": "basePlot.tsv",
    "ChromatogramConfig": "chromatogramPlot.tsv",
    "MobilogramConfig": "mobilogramPlot.tsv",
    "PeakMapConfig": "peakMapPlot.tsv",
    "SpectrumConfig": "spectrumPlot.tsv",
}

def normalize_type(type_str: str) -> str:
    """Simplify complex type annotations for consistency."""
    type_str = re.sub(r"\bUnion\[([\w\s,]+)\]", r"\1", type_str)
    type_str = re.sub(r"\bLiteral\[[^\]]+\]", "str", type_str)
    type_str = re.sub(r"\bOptional\[([\w]+)\]", r"\1 | None", type_str)
    return type_str.strip()

def parse_docstring(doc: str) -> Dict[str, str]:
    """Extracts descriptions from class docstrings (Sphinx-style)."""
    attrs = {}
    current_param = None

    for line in doc.split("\n"):
        line = line.strip()

        if line.lower().startswith((":param", ":attr")):
            match = re.match(r":(param|attr)\s+(\w+):\s*(.*)", line)
            if match:
                current_param = match.group(2)
                attrs[current_param] = match.group(3).strip()
        elif current_param and line:
            attrs[current_param] += f" {line}"

    return attrs

def get_all_attributes(class_name, class_definitions, inheritance_map):
    """Recursively retrieves all attributes from a given class, including inherited attributes."""
    attributes = {}
    
    if class_name not in class_definitions:
        logging.error(f"Class {class_name} not found in class definitions!")
        return attributes

    node = class_definitions[class_name]
    logging.debug(f"Extracting attributes for {class_name}")

    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign):
            name = stmt.target.id if isinstance(stmt.target, ast.Name) else None
            if name:
                type_annotation = normalize_type(ast.unparse(stmt.annotation))
                default = ast.unparse(stmt.value) if stmt.value else "None"
                attributes[name] = (default, type_annotation)
                logging.debug(f"Found attribute: {name} = {default}, type: {type_annotation}")

    # Recursively retrieve attributes from parent classes
    for parent in inheritance_map.get(class_name, []):
        parent_attrs = get_all_attributes(parent, class_definitions, inheritance_map)
        for attr, values in parent_attrs.items():
            if attr not in attributes:
                attributes[attr] = values
                logging.debug(f"Inherited attribute from {parent}: {attr} = {values}")

    return attributes

def update_tsv_files():
    """Updates TSV files dynamically, ensuring inheritance is handled properly."""
    try:
        logging.info("Reading _config.py...")
        config_content = CONFIG_PATH.read_text()
        parsed = ast.parse(config_content)

        # Extract class definitions and inheritance relationships
        inheritance_map = {}
        class_definitions = {}

        for node in parsed.body:
            if isinstance(node, ast.ClassDef):
                parent_classes = [base.id for base in node.bases if isinstance(base, ast.Name)]
                inheritance_map[node.name] = parent_classes
                class_definitions[node.name] = node
        
        logging.debug(f"Extracted Classes & Inheritance Map: {inheritance_map}")

        for node in parsed.body:
            if isinstance(node, ast.ClassDef):
                if any(d.id == "dataclass" for d in node.decorator_list if isinstance(d, ast.Name)):
                    if node.name in CLASS_TO_TSV_MAP:
                        logging.info(f"Processing class: {node.name}")

                        # Retrieve all attributes including inherited ones
                        all_attributes = get_all_attributes(node.name, class_definitions, inheritance_map)
                        logging.debug(f"Attributes for {node.name}: {all_attributes}")

                        # Extract docstrings for descriptions
                        doc = ast.get_docstring(node) or ""
                        param_docs = parse_docstring(doc)
                        logging.debug(f"Parsed docstrings for {node.name}: {param_docs}")

                        tsv_path = DOCS_DIR / CLASS_TO_TSV_MAP[node.name]
                        logging.debug(f"Mapping {node.name} to TSV file: {tsv_path}")

                        # Read existing TSV content
                        existing_lines = []
                        if tsv_path.exists():
                            existing_lines = tsv_path.read_text().strip().split("\n")
                        
                        # Preserve existing descriptions
                        header = "Parameter\tDefault\tType\tDescription\n"
                        lines = []
                        existing_params = {line.split("\t")[0]: line for line in existing_lines[1:]}  # Skip header

                        for name, (default, type_) in all_attributes.items():
                            description = param_docs.get(name, "")
                            if name in existing_params:
                                # Preserve existing description
                                existing_line = existing_params[name]
                                lines.append(existing_line)
                            else:
                                lines.append(f"{name}\t{default}\t{type_}\t{description}")

                        # Write only if content has changed
                        new_content = header + "\n".join(lines)
                        if new_content != header + "\n".join(existing_lines):
                            logging.info(f"Writing to {tsv_path}")
                            temp_path = tsv_path.with_suffix(".tmp")
                            temp_path.write_text(new_content)
                            temp_path.replace(tsv_path)
                            logging.info(f"Updated {tsv_path} with inherited attributes")
                        else:
                            logging.info(f"No changes detected for {tsv_path}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")

if __name__ == "__main__":
    update_tsv_files()