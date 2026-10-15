"""The camera battery workload must discard every temporary clip and frame."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deploy.soak_camera_inference import run_one


class CameraBatteryWorkloadTest(unittest.TestCase):
    def test_records_classifies_and_discards(self):
        class Recorder:
            def record(self, dest, seconds):
                dest.write_bytes(b"video")
                return SimpleNamespace(path=dest, size_bytes=5)

        class Classifier:
            def classify(self, clip, work, event_id):
                self_test.assertTrue(clip.exists())
                (work / f"{event_id}_frame.jpg").write_bytes(b"frame")
                return SimpleNamespace(
                    frames_scored=12,
                    decision=SimpleNamespace(label="desk", confidence=0.5, is_unknown=True),
                )

        self_test = self
        with tempfile.TemporaryDirectory() as name:
            scratch = Path(name)
            result = run_one(Recorder(), Classifier(), "event", 0.01, scratch)
            self.assertEqual(result["frames_scored"], 12)
            self.assertEqual(result["clip_bytes"], 5)
            self.assertEqual(list(scratch.iterdir()), [])

    def test_discards_clip_when_inference_fails(self):
        class Recorder:
            def record(self, dest, seconds):
                dest.write_bytes(b"video")
                return SimpleNamespace(path=dest, size_bytes=5)

        class Classifier:
            def classify(self, clip, work, event_id):
                raise RuntimeError("classifier failed")

        with tempfile.TemporaryDirectory() as name:
            scratch = Path(name)
            with self.assertRaisesRegex(RuntimeError, "classifier failed"):
                run_one(Recorder(), Classifier(), "event", 0.01, scratch)
            self.assertEqual(list(scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
