"""Relocation must preserve stimulus identity and refuse escaping indexes."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from mstd.project_paths import resolve_stimulus


class ProjectPathsTests(unittest.TestCase):
    def test_moved_input_resolves_by_mapping_and_verified_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'input/stimulus.dat'
            target.parent.mkdir()
            target.write_bytes(b'original stimulus')
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            mapping = root / 'docs/migrations/20261107_layout/paths.csv'
            mapping.parent.mkdir(parents=True)
            with mapping.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['old_path', 'new_path'])
                writer.writeheader()
                writer.writerow(dict(old_path='outputs/old.dat', new_path='input/stimulus.dat'))
            self.assertEqual(resolve_stimulus('outputs/old.dat', digest, root), target)
            with self.assertRaises(FileNotFoundError):
                resolve_stimulus('outputs/old.dat', '0' * 64, root)

    def test_server_path_uses_hash_index_and_rechecks_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'input').mkdir()
            target = root / 'input/stimulus.dat'
            target.write_bytes(b'server stimulus')
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            with (root / 'input/registry.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['path', 'sha256'])
                writer.writeheader()
                writer.writerow(dict(path='input/stimulus.dat', sha256=digest))
            self.assertEqual(resolve_stimulus('/server/missing.dat', digest, root), target)
            target.write_bytes(b'changed')
            with self.assertRaises(FileNotFoundError):
                resolve_stimulus('/server/missing.dat', digest, root)

    def test_index_cannot_redirect_outside_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp)
            root = outer / 'project'
            (root / 'input').mkdir(parents=True)
            outside = outer / 'outside.dat'
            outside.write_bytes(b'outside')
            digest = hashlib.sha256(b'outside').hexdigest()
            with (root / 'input/registry.csv').open('w', newline='', encoding='utf-8') as stream:
                stream.write('path,sha256\n../outside.dat,' + digest + '\n')
            with self.assertRaises(FileNotFoundError):
                resolve_stimulus('missing.dat', digest, root)
