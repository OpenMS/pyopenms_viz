#!/usr/bin/env python3
"""
TSV updater with full inheritance handling, type normalization, and Git hook integration.
Automatically updates TSV files when `_config.py` changes.
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Paths
CONFIG_PATH = Path("pyopenms_viz/_config.py")
DOCS_DIR = Path("docs/Parameters")

# Mapping dataclasses to TSV file names
CLASS_TO_TSV_MAP = {
    "BasePlotConfig": "basePlot.tsv",
    "ChromatogramConfig": "chromatogramPlot.tsv",
    "MobilogramConfig": "mobilogramPlot.tsv",
    "PeakMapConfig": "peakmapPlot.tsv",
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
        match = re.match(r":(param|attr)\s+(\w+):\s*(.*)", line)
        if match:
            current_param = match.group(2)
            attrs[current_param] = match.group(3).strip()
        elif current_param and line:
            attrs[current_param] += f" {line}"
    
    return attrs

def get_all_attributes(class_name, class_definitions, inheritance_map):
    """Recursively retrieves all attributes from a given class, including inherited attributes.
       Ignores external classes like `ABC` that do not have a definition in _config.py.
    """
    attributes = {}

    if class_name in ["ABC", "ABCMeta"]:  # Ignore abstract base classes
        logging.info(f"⚠️ Skipping abstract base class: {class_name}")
        return attributes  # Return empty to avoid breaking inheritance

    if class_name not in class_definitions:
        logging.warning(f"❌ Class {class_name} not found in class definitions!")
        return attributes

    node = class_definitions[class_name]
    logging.info(f"🔍 Extracting attributes for {class_name}")

    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign):
            name = stmt.target.id if isinstance(stmt.target, ast.Name) else None
            if name:
                type_annotation = normalize_type(ast.unparse(stmt.annotation))
                default = ast.unparse(stmt.value) if stmt.value else "None"
                attributes[name] = (default, type_annotation)

    # Recursively retrieve attributes from parent classes
    for parent in inheritance_map.get(class_name, []):
        if parent in ["ABC", "ABCMeta"]:  # Ignore ABC base class
            logging.info(f"⚠️ Skipping inherited abstract class: {parent}")
            continue  # Skip processing this parent
        parent_attrs = get_all_attributes(parent, class_definitions, inheritance_map)
        for attr, values in parent_attrs.items():
            if attr not in attributes:
                attributes[attr] = values

    return attributes


def update_tsv_files():
    """Updates TSV files dynamically, ensuring inheritance is handled properly."""
    try:
        logging.info("🔍 Reading _config.py...")
        config_content = CONFIG_PATH.read_text()
        parsed = ast.parse(config_content)

        # Extract class definitions and inheritance relationships
        inheritance_map = {}
        class_definitions = {}

        for node in parsed.body:
            if isinstance(node, ast.ClassDef):
                if node.name in ["ABC", "ABCMeta"]:  # Ignore abstract base classes
                    continue
                parent_classes = [base.id for base in node.bases if isinstance(base, ast.Name)]
                inheritance_map[node.name] = parent_classes
                class_definitions[node.name] = node

        logging.info(f"📌 Extracted Classes & Inheritance Map: {inheritance_map}")

        updated_files = []
        for class_name, tsv_filename in CLASS_TO_TSV_MAP.items():
            if class_name in class_definitions:
                logging.info(f"🔍 Processing class: {class_name}")

                # Retrieve all attributes including inherited ones
                all_attributes = get_all_attributes(class_name, class_definitions, inheritance_map)

                # Extract docstrings for descriptions
                doc = ast.get_docstring(class_definitions[class_name]) or ""
                param_docs = parse_docstring(doc)

                tsv_path = DOCS_DIR / tsv_filename
                logging.info(f"📂 Mapping {class_name} to TSV file: {tsv_path}")

                # Read existing TSV content
                existing_lines = tsv_path.read_text().strip().split("\n") if tsv_path.exists() else []
                existing_params = {line.split("\t")[0]: line for line in existing_lines[1:]}  # Skip header

                # Prepare new TSV content
                header = "Parameter\tDefault\tType\tDescription\n"
                lines = []
                for name, (default, type_) in all_attributes.items():
                    description = param_docs.get(name, "")
                    if name in existing_params:
                        lines.append(existing_params[name])  # Preserve existing description
                    else:
                        lines.append(f"{name}\t{default}\t{type_}\t{description}")

                new_content = header + "\n".join(lines)

                # Write only if content has changed
                if new_content != header + "\n".join(existing_lines):
                    logging.info(f"✅ Writing to {tsv_path}")
                    with open(tsv_path, "w") as f:
                        f.write(new_content)
                    updated_files.append(tsv_path)

        if updated_files:
            logging.info(f"✅ Updated {len(updated_files)} TSV files: {', '.join(map(str, updated_files))}")

    except FileNotFoundError as e:
        logging.error(f"❌ File not found: {e.filename}")
    except Exception as e:
        logging.error(f"❌ Critical error: {str(e)}")

if __name__ == "__main__":
    update_tsv_files()
