import json
import os
import sys
import tempfile
import types
import unittest

# Lightweight gradio stub so helper tests can run without full GUI deps.
if "gradio" not in sys.modules:
    gradio_stub = types.SimpleNamespace(
        Blocks=object,
        update=lambda **kwargs: kwargs,
    )
    sys.modules["gradio"] = gradio_stub

import gui


class TestGuiHelpers(unittest.TestCase):
    def test_safe_resolve_rejects_escape(self):
        with self.assertRaises(ValueError):
            gui._safe_resolve("../outside")

    def test_filter_attack_data(self):
        data = [
            {"id": 1, "behavior": "alpha", "target": "x", "category": "cat-a"},
            {"id": 2, "behavior": "beta", "target": "y", "category": "cat-b"},
            {"id": 3, "behavior": "alpha beta", "target": "z", "category": "cat-a"},
        ]
        filtered = gui.filter_attack_data(data, category="cat-a", start_idx=0, end_idx=5, keyword="alpha")
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["id"], 1)
        self.assertEqual(filtered[1]["id"], 3)

    def test_summary_for_entries(self):
        entries = [
            {"Success": True, "process": [{"iteration": 2, "current_loss": 1.0}]},
            {"Success": False, "process": [{"iteration": 4, "current_loss": 3.0}]},
        ]
        summary = gui._summary_for_entries("run.jsonl", entries)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["successes"], 1)
        self.assertAlmostEqual(summary["success_rate"], 50.0)
        self.assertAlmostEqual(summary["avg_queries"], 3.0)
        self.assertAlmostEqual(summary["avg_final_loss"], 2.0)

    def test_infer_resume_index(self):
        with tempfile.TemporaryDirectory(dir=gui._ROOT_DIR) as tmpdir:
            fpath = os.path.join(tmpdir, "part.jsonl")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(json.dumps({"bidx": 3, "Success": True}) + "\n")
                f.write(json.dumps({"bidx": 5, "Success": False}) + "\n")

            rel = os.path.relpath(tmpdir, gui._ROOT_DIR)
            resume_idx = gui._infer_resume_index(rel)
            self.assertEqual(resume_idx, 6)

    def test_build_attack_cmd_source(self):
        cmd = gui._build_attack_cmd(gui._GUI_RUN_CONFIG_PATH)
        self.assertEqual(cmd[0], gui.sys.executable)
        self.assertTrue(cmd[1].endswith("attack.py"))
        self.assertEqual(cmd[2], "--config_path")
        self.assertEqual(cmd[3], gui._GUI_RUN_CONFIG_PATH)

    def test_build_attack_cmd_frozen_prefers_attack_exe(self):
        original_frozen = getattr(gui.sys, "frozen", None)
        original_executable = gui.sys.executable
        original_exists = gui.os.path.exists
        try:
            gui.sys.frozen = True
            gui.sys.executable = r"C:\dist\gui.exe"
            gui.os.path.exists = lambda p: p == r"C:\dist\attack.exe"
            cmd = gui._build_attack_cmd(gui._GUI_RUN_CONFIG_PATH)
            self.assertEqual(cmd[0], r"C:\dist\attack.exe")
        finally:
            if original_frozen is None and hasattr(gui.sys, "frozen"):
                delattr(gui.sys, "frozen")
            else:
                gui.sys.frozen = original_frozen
            gui.sys.executable = original_executable
            gui.os.path.exists = original_exists

    def test_build_attack_cmd_frozen_requires_attack_exe(self):
        original_frozen = getattr(gui.sys, "frozen", None)
        original_executable = gui.sys.executable
        original_exists = gui.os.path.exists
        try:
            gui.sys.frozen = True
            gui.sys.executable = r"C:\dist\gui.exe"
            gui.os.path.exists = lambda p: False
            with self.assertRaises(ValueError):
                gui._build_attack_cmd(gui._GUI_RUN_CONFIG_PATH)
        finally:
            if original_frozen is None and hasattr(gui.sys, "frozen"):
                delattr(gui.sys, "frozen")
            else:
                gui.sys.frozen = original_frozen
            gui.sys.executable = original_executable
            gui.os.path.exists = original_exists


if __name__ == "__main__":
    unittest.main()
