"""Verify frozen source bytes for historical studies after repository edits.

Training and evaluation runners still require their current sources to match.
This module only supports audits of completed evidence: every changed source
must be recovered from a verified snapshot with the exact recorded hash.
"""
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts or path.as_posix() != relative:
        raise ValueError('Source provenance requires a normalized relative path: ' + str(relative))
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError('Source provenance path escapes its recorded directory: ' + relative)
    return resolved


def merge_source_hashes(*mappings):
    """Require one unambiguous recorded hash for each implementation file."""
    merged = {}
    for mapping in mappings:
        for name, value in mapping.items():
            source_path(Path.cwd(), name)
            if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
                raise ValueError('Invalid recorded source hash: ' + name)
            if name in merged and merged[name] != value:
                raise ValueError('Frozen protocols disagree on source hash: ' + name)
            merged[name] = value
    return merged


def verify_snapshot(snapshot, expected_sources, protocol_path):
    """Check the complete declared snapshot, including its root protocol."""
    snapshot, protocol_path = Path(snapshot), Path(protocol_path)
    manifest = read(snapshot / 'manifest.json')
    expected_root_hash = digest(protocol_path)
    if (manifest['root_protocol_sha256'] != expected_root_hash
            or digest(snapshot / 'root_protocol.json') != expected_root_hash):
        raise ValueError('Executed-source snapshot refers to a different root protocol')
    if set(manifest['sources']) != set(expected_sources):
        raise ValueError('Executed-source snapshot has missing or extra frozen sources')
    for name, expected in expected_sources.items():
        entry = manifest['sources'][name]
        if (entry['snapshot_relative_path'] != name
                or entry['snapshot_sha256'] != expected
                or entry['frozen_protocol_sha256'] != expected):
            raise ValueError('Executed-source snapshot manifest differs from frozen provenance: ' + name)
        path = source_path(snapshot, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Executed-source snapshot bytes are missing or changed: ' + name)
        if 'bytes' in entry and path.stat().st_size != entry['bytes']:
            raise ValueError('Executed-source snapshot file size differs: ' + name)
    return manifest


def audit_frozen_sources(root, study, expected_sources, snapshot_sources, *, supplementary_study=None):
    """Verify live files or exact archived bytes without changing saved evidence.

    A partial historical snapshot must cover precisely ``snapshot_sources``. Any
    changed file it does not contain needs an explicitly named supplementary study
    whose complete source snapshot is bound to its own frozen root protocol.
    """
    root, study = Path(root), Path(study)
    expected_sources = merge_source_hashes(expected_sources)
    snapshot_sources = merge_source_hashes(snapshot_sources)
    if any(expected_sources.get(name) != value for name, value in snapshot_sources.items()):
        raise ValueError('Snapshot source expectations differ from the study protocols')
    changed = sorted(name for name, value in expected_sources.items()
                     if not source_path(root, name).is_file() or digest(source_path(root, name)) != value)
    snapshot = study / 'provenance/executed_training_source'
    available = (snapshot / 'manifest.json').is_file()
    verified = {}
    if available:
        verify_snapshot(snapshot, snapshot_sources, study / 'protocol.json')
        verified.update({name: str(snapshot / name) for name in snapshot_sources})
    missing = [name for name in changed if name not in verified]
    supplementary = None
    if missing and supplementary_study is not None:
        other = Path(supplementary_study)
        other_snapshot = other / 'provenance/executed_training_source'
        if (other_snapshot / 'manifest.json').is_file():
            other_protocol = read(other / 'protocol.json')
            other_sources = merge_source_hashes(other_protocol['training_source_sha256'])
            verify_snapshot(other_snapshot, other_sources, other / 'protocol.json')
            supplied = [name for name in missing if other_sources.get(name) == expected_sources[name]]
            verified.update({name: str(other_snapshot / name) for name in supplied})
            supplementary = {'study': str(other), 'snapshot_verified': True,
                             'root_protocol_sha256': digest(other / 'protocol.json'),
                             'source_paths_used': supplied}
    missing = [name for name in changed if name not in verified]
    if missing:
        raise ValueError('Frozen sources differ from the current repository and no verified snapshot supplies their exact bytes: '
                         + ', '.join(missing))
    return {'current_sources_verified': not changed,
            'current_source_differences': changed,
            'snapshot_available': available, 'snapshot_verified': available,
            'historical_frozen_sources_verified': True,
            'snapshot_sources_used': {name: verified[name] for name in changed},
            'supplementary_snapshot': supplementary,
            'scope': ('All current frozen source hashes match.' if not changed else
                      'Current sources differ from the recorded experiment. Exact original bytes for every changed file were verified in the listed source snapshots; this audit does not rerun training or evaluation.')}
