"""Monitoring must not infer job termination or submit work after network failures."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "monitor", Path(__file__).resolve().parents[1] / "scripts/local/watch_mof.py"
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class RemoteMonitorTests(unittest.TestCase):
    def test_explicit_settings_and_keepalive(self):
        result = subprocess.CompletedProcess([], 0, json.dumps({"state": "unknown"}), "")
        with patch.object(monitor.subprocess, "run", return_value=result) as run:
            self.assertEqual(monitor.remote_status("user@host", "/project", "run-1")["state"], "unknown")
        argv = run.call_args.args[0]
        self.assertIn("ServerAliveInterval=10", argv)
        self.assertNotIn("sbatch", argv)
        self.assertNotIn("scancel", argv)
        self.assertEqual(run.call_args.kwargs["timeout"], 45)

    def test_invalid_remote_or_run_is_rejected_before_connecting(self):
        with patch.object(monitor.subprocess, "run") as run:
            for host, root, run_id in [("-unsafe", "/project", "ok"),
                                       ("host", "relative", "ok"),
                                       ("host", "/project", "../escape")]:
                with self.assertRaises(ValueError):
                    monitor.remote_status(host, root, run_id)
            run.assert_not_called()

    def test_timeout_propagates_without_submitting_or_collecting(self):
        with patch.object(monitor.subprocess, "run", side_effect=subprocess.TimeoutExpired("ssh", 45)), \
                patch.object(monitor, "collect") as collect:
            with self.assertRaises(subprocess.TimeoutExpired):
                monitor.remote_status("host", "/project", "run-1")
            collect.assert_not_called()
