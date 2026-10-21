"""Historical audits must verify original source bytes after repository edits."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_provenance_checks', ROOT / 'experiments/source_provenance.py')
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class SourceProvenanceTests(unittest.TestCase):
    def fixture(self, directory, *, partial=False):
        root = Path(directory) / 'live'
        root.mkdir()
        files = {'src/model.py': 'model = 1\n', 'experiments/run.py': 'run = 1\n'}
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        hashes = {name: provenance.digest(root / name) for name in files}
        study = Path(directory) / 'study'
        snapshot_hashes = {'src/model.py': hashes['src/model.py']} if partial else hashes
        self.snapshot(study, root, snapshot_hashes)
        return root, study, hashes, snapshot_hashes

    def snapshot(self, study, root, hashes):
        write(study / 'protocol.json', {'training_source_sha256': hashes})
        snapshot = study / 'provenance/executed_training_source'
        snapshot.mkdir(parents=True)
        (snapshot / 'root_protocol.json').write_bytes((study / 'protocol.json').read_bytes())
        entries = {}
        for name, value in hashes.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((root / name).read_bytes())
            entries[name] = {'snapshot_relative_path': name, 'snapshot_sha256': value,
                             'frozen_protocol_sha256': value, 'bytes': path.stat().st_size}
        write(snapshot / 'manifest.json', {'root_protocol_sha256': provenance.digest(study / 'protocol.json'),
                                          'sources': entries})
        return snapshot

    def test_current_sources_and_unchanged_snapshot_are_both_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root, study, hashes, snapshot_hashes = self.fixture(directory)
            status = provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes)
            self.assertTrue(status['current_sources_verified'])
            self.assertTrue(status['snapshot_verified'])
            self.assertEqual(status['snapshot_sources_used'], {})

    def test_changed_source_uses_original_snapshot_and_reports_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root, study, hashes, snapshot_hashes = self.fixture(directory)
            (root / 'src/model.py').write_text('model = 2\n')
            status = provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes)
            self.assertFalse(status['current_sources_verified'])
            self.assertTrue(status['historical_frozen_sources_verified'])
            self.assertEqual(status['current_source_differences'], ['src/model.py'])
            self.assertEqual(provenance.digest(status['snapshot_sources_used']['src/model.py']), hashes['src/model.py'])

    def test_changed_source_without_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, study, hashes, snapshot_hashes = self.fixture(directory)
            (study / 'provenance/executed_training_source/manifest.json').unlink()
            self.assertTrue(provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes)['current_sources_verified'])
            (root / 'src/model.py').write_text('model = 2\n')
            with self.assertRaisesRegex(ValueError, 'no verified snapshot'):
                provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes)

    def test_present_snapshot_rejects_tampering_even_if_live_sources_match(self):
        for mutation in ('bytes', 'missing_file', 'source_set', 'recorded_hash', 'path', 'root_protocol'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root, study, hashes, snapshot_hashes = self.fixture(directory)
                snapshot = study / 'provenance/executed_training_source'
                manifest = provenance.read(snapshot / 'manifest.json')
                if mutation == 'bytes':
                    (snapshot / 'src/model.py').write_text('model = 9\n')
                elif mutation == 'missing_file':
                    (snapshot / 'src/model.py').unlink()
                elif mutation == 'source_set':
                    del manifest['sources']['src/model.py']
                elif mutation == 'recorded_hash':
                    manifest['sources']['src/model.py']['snapshot_sha256'] = '0'*64
                elif mutation == 'path':
                    manifest['sources']['src/model.py']['snapshot_relative_path'] = '../model.py'
                else:
                    (snapshot / 'root_protocol.json').write_text('{}')
                write(snapshot / 'manifest.json', manifest)
                with self.assertRaises(ValueError):
                    provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes)

    def test_explicit_supplement_must_match_required_hash_and_verify_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root, study, hashes, snapshot_hashes = self.fixture(directory, partial=True)
            other = Path(directory) / 'other'
            snapshot = self.snapshot(other, root, hashes)
            (root / 'experiments/run.py').write_text('run = 2\n')
            with self.assertRaisesRegex(ValueError, 'no verified snapshot'):
                provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes)
            status = provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes, supplementary_study=other)
            self.assertEqual(status['supplementary_snapshot']['source_paths_used'], ['experiments/run.py'])
            self.assertFalse(status['current_sources_verified'])
            # Even an unrelated file in the supplementary manifest must verify.
            (snapshot / 'src/model.py').write_text('model = 8\n')
            with self.assertRaisesRegex(ValueError, 'bytes are missing or changed'):
                provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes, supplementary_study=other)

    def test_valid_supplement_with_different_frozen_bytes_cannot_fill_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root, study, hashes, snapshot_hashes = self.fixture(directory, partial=True)
            (root / 'experiments/run.py').write_text('run = 2\n')
            other = Path(directory) / 'other'
            different = dict(hashes, **{'experiments/run.py': provenance.digest(root / 'experiments/run.py')})
            self.snapshot(other, root, different)
            with self.assertRaisesRegex(ValueError, 'no verified snapshot'):
                provenance.audit_frozen_sources(root, study, hashes, snapshot_hashes, supplementary_study=other)

    def test_conflicting_hashes_and_unsafe_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            provenance.merge_source_hashes({'file.py': '0'*64}, {'file.py': '1'*64})
        for name in ('../file.py', '/tmp/file.py', 'src/../file.py'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'relative path'):
                provenance.merge_source_hashes({name: '0'*64})


if __name__ == '__main__':
    unittest.main()
