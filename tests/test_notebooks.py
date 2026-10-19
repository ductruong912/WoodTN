import ast
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_and_contain_no_saved_execution_output(self):
        notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual([path.name for path in notebooks], ["efficientnet-v2-s.ipynb"])
        for path in notebooks:
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        self.assertEqual(cell.outputs, [])
                        self.assertIsNone(cell.execution_count)
                        ast.parse(cell.source, filename=path.name)


if __name__ == "__main__":
    unittest.main()
