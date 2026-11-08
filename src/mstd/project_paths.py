"""Resolve archived stimulus locations without rewriting historical evidence."""
import csv
from pathlib import Path

from .calibration import file_sha256


def resolve_stimulus(path, expected_sha256=None, project_root=None):
    """Prefer an existing path, then exact migration paths, then indexed hash.

    All redirected candidates must stay within the project. Hash fallback only
    uses the audited stimulus index, never a basename search across the disk.
    Callers retain their native shape, timing and SHA256 validation.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    supplied = Path(path)
    if supplied.is_file():
        return supplied
    if not supplied.is_absolute() and (root / supplied).is_file():
        return root / supplied
    normalized = str(path).replace('\\', '/')
    relative = normalized
    prefix = root.as_posix() + '/'
    if relative.lower().startswith(prefix.lower()):
        relative = relative[len(prefix):]
    mapping = root / 'docs/migrations/20261107_layout/paths.csv'
    if mapping.is_file():
        with mapping.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['old_path'] != relative:
                    continue
                candidate = (root / row['new_path']).resolve()
                if candidate.is_relative_to(root) and candidate.is_file():
                    if expected_sha256 is None or file_sha256(candidate) == expected_sha256:
                        return candidate
    index = root / 'input/registry.csv'
    if expected_sha256 and index.is_file():
        with index.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['sha256'] != expected_sha256:
                    continue
                candidate = (root / row['path']).resolve()
                if (candidate.is_relative_to(root) and candidate.is_file()
                        and file_sha256(candidate) == expected_sha256):
                    return candidate
    raise FileNotFoundError(f'Stimulus unavailable: {path}; supply --stimulus with the original matching DAT')
