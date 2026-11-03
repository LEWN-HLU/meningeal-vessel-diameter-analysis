"""Data-free packaging and API checks; no artificial recordings are supplied."""
import ast
import re
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class PackageTests(unittest.TestCase):
    def test_source_compiles(self):
        for path in ROOT.rglob('*.py'):
            if '__pycache__' not in path.parts:
                ast.parse(path.read_text(encoding='utf-8'),filename=str(path))

    def test_no_bundled_recordings_or_outputs(self):
        forbidden={'.csv','.tsv','.xlsx','.xls','.mp4','.docx','.pdf','.pkl','.parquet','.zip','.bak'}
        for path in ROOT.rglob('*'):
            if path.is_file(): self.assertNotIn(path.suffix.lower(),forbidden,str(path))

    def test_no_private_paths(self):
        for path in ROOT.rglob('*'):
            if path.is_file() and path.suffix in {'.py','.md','.json','.txt'}:
                if path.resolve()==Path(__file__).resolve(): continue
                text=path.read_text(encoding='utf-8')
                patterns=[
                    r'(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]',
                    r'(?m)(?<!\\)\\\\[A-Za-z0-9_.-]+\\',
                    r'/(?:Users|home)/[^/\s]+/',
                ]
                for pattern in patterns:
                    self.assertIsNone(re.search(pattern,text),str(path))

    def test_imports_and_defaults(self):
        from publication import Settings, COHORTS
        from core import pipeline, mixed, plots, dose_helpers
        self.assertEqual(Settings().baseline_sec,890)
        self.assertEqual(Settings().response_sec,590)
        self.assertEqual(len(COHORTS),5)

    def test_statistical_methods_are_available(self):
        import statistical_methods as methods
        for name in methods.__all__:
            self.assertTrue(callable(getattr(methods,name)),name)

if __name__=='__main__': unittest.main()
