import tempfile
import unittest
from pathlib import Path

import torch

from fishcls.utils import get_device, load_json, save_json, seed_everything


class SeedTest(unittest.TestCase):
    def test_same_seed_gives_same_numbers(self):
        seed_everything(0)
        first = torch.rand(3)
        seed_everything(0)
        second = torch.rand(3)
        self.assertTrue(torch.equal(first, second))


class JsonTest(unittest.TestCase):
    def test_roundtrip_creates_parent_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "data.json"
            save_json({"classes": ["Shrimp", "Sea Bass"]}, path)
            self.assertEqual(load_json(path), {"classes": ["Shrimp", "Sea Bass"]})


class DeviceTest(unittest.TestCase):
    def test_device_runs_tensors(self):
        self.assertEqual(torch.ones(2, device=get_device()).sum().item(), 2)


if __name__ == "__main__":
    unittest.main()
