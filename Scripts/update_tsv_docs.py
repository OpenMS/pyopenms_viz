#!/usr/bin/env python3
"""
 TSV updater with error handling and type normalization
"""
import os
import re
import ast
import logging
from dataclasses import is_dataclass
from typing import Dict, List, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configuration - Keep your original paths
CONFIG_PATH = Path('pyopenms_viz/_config.py')
DOCS_DIR = Path('docs/Parameters')
CLASS_TO_TSV_MAP = {
    'BasePlotConfig': 'basePlot.tsv',
    'ChromatogramConfig': 'chromatogramPlot.tsv',
    'MobilogramConfig': 'mobilogramPlot.tsv',
    'PeakMapConfig': 'peakMapPlot.tsv',
    'SpectrumConfig': 'spectrumPlot.tsv'
}

def normalize_type(type_str: str) -> str:
    """Simplify complex type annotations"""
    type_str = re.sub(r"\bUnion\[([\w\s,]+)\]", r"\1", type_str)
    type_str = re.sub(r"\bLiteral\[[^\]]+\]", "str", type_str)
    type_str = re.sub(r"\bOptional\[([\w]+)\]", r"\1 | None", type_str)
    return type_str.strip()

def parse_docstring(doc: str) -> Dict[str, str]:
    """Improved docstring parser using attribute section detection"""
    attrs = {}
    current_param = None
    
    for line in doc.split('\n'):
        line = line.strip()
        
        if line.lower().startswith((':param', ':attr')):
            # Direct Sphinx-style parsing
            match = re.match(r":(param|attr)\s+(\w+):\s*(.*)", line)
            if match:
                current_param = match.group(2)
                attrs[current_param] = match.group(3).strip()
        elif current_param and line:
            attrs[current_param] += f" {line}"
    
    return attrs

def process_dataclass(node: ast.ClassDef) -> List[Tuple[str, str, str, str]]:
    """Process dataclass with error handling"""
    try:
        doc = ast.get_docstring(node) or ""
        param_docs = parse_docstring(doc)
        attrs = []
        
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                name = stmt.target.id if isinstance(stmt.target, ast.Name) else None
                if name:
                    type_annotation = normalize_type(ast.unparse(stmt.annotation))
                    default = ast.unparse(stmt.value) if stmt.value else ""
                    
                    # Handle default factories
                    if "default_factory" in default:
                        default = re.search(r"default_factory=(\w+)", default).group(1) + "()"
                    
                    attrs.append((
                        name,
                        default,
                        type_annotation,
                        param_docs.get(name, "")
                    ))
        return attrs
    except Exception as e:
        logging.error(f"Error processing {node.name}: {str(e)}")
        return []

def update_tsv_files():
    """Main update routine with validation"""
    try:
        config_content = CONFIG_PATH.read_text()
        parsed = ast.parse(config_content)
        
        for node in parsed.body:
            if isinstance(node, ast.ClassDef):
                if any(d.id == 'dataclass' for d in node.decorator_list if isinstance(d, ast.Name)):
                    if node.name in CLASS_TO_TSV_MAP:
                        attrs = process_dataclass(node)
                        tsv_path = DOCS_DIR / CLASS_TO_TSV_MAP[node.name]
                        
                        # Generate TSV content
                        header = "Parameter\tDefault\tType\tDescription\n"
                        lines = [
                            f"{name}\t{default}\t{type_}\t{desc.replace('\t', ' ')}"
                            for name, default, type_, desc in attrs
                        ]
                        
                        # Write with atomic replacement
                        temp_path = tsv_path.with_suffix('.tmp')
                        temp_path.write_text(header + '\n'.join(lines))
                        temp_path.replace(tsv_path)
                        logging.info(f"Updated {tsv_path}")
                        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")

if __name__ == "__main__":
    update_tsv_files()