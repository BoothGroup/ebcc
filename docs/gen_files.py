"""
Generate the code reference pages.
"""

from pathlib import Path

import mkdocs_gen_files


for path in sorted(Path("ebcc").rglob("*.py")):
    module_path = path.relative_to("ebcc").with_suffix("")
    doc_path = path.relative_to("ebcc").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = ["ebcc", *module_path.parts]

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if (
        not len(parts)
        or parts[-1] == "__main__"
        or any(module in parts for module in ("codegen", "backend"))
    ):
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write("::: " + identifier + "\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)
