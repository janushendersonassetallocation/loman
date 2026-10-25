"""Exercise native provisioning with fake commands, without changing the host."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires POSIX bash")
class TestGraphvizInstaller(unittest.TestCase):
    """Keep package-manager selection and retries independent of the host PATH."""

    def setUp(self):
        """Create an isolated executable search path."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bin = Path(self.directory.name)
        self.script = Path(__file__).resolve().parents[1] / "scripts/install-graphviz.sh"
        self.command("uname", "printf 'Linux\\n'")
        self.command("id", "printf '%s\\n' \"${TEST_UID:-1000}\"")
        self.command("head", 'while IFS= read -r line; do printf "%s\\n" "$line"; break; done')
        self.command("cat", 'while IFS= read -r line; do printf "%s\\n" "$line"; done')
        self.command("sleep", 'printf "sleep\\n" >> "$PATH/calls"')
        self.command("sudo", 'printf "sudo\\n" >> "$PATH/calls"; exec "$@"')

    def command(self, name, body):
        """Write an executable stub using only shell builtins."""
        target = self.bin / name
        target.write_text("#!/bin/bash\n" + body + "\n")
        target.chmod(0o755)

    def run_installer(self, **extra):
        """Run the actual installer, keeping real package managers unreachable."""
        return subprocess.run(
            [shutil.which("bash"), str(self.script)],
            env={"PATH": str(self.bin), "GRAPHVIZ_INSTALL_BACKOFF": "0", **extra},
            capture_output=True,
            text=True,
            check=False,
        )

    def calls(self):
        """Read recorded invocations."""
        target = self.bin / "calls"
        return target.read_text().splitlines() if target.exists() else []

    def pacman(self, success=True):
        """Record arguments and optionally make dot available."""
        body = 'printf "pacman %s\\n" "$*" >> "$PATH/calls"\n'
        if success:
            body += 'printf "#!/bin/bash\\nprintf Graphviz\\\\n\\n" > "$PATH/dot"\n'
            body += f'{shutil.which("chmod")} +x "$PATH/dot"'
        else:
            body += "exit 1"
        self.command("pacman", body)

    def test_pacman_root(self):
        """Root can install directly without refreshing indexes alone."""
        self.pacman()
        result = self.run_installer(TEST_UID="0")
        assert result.returncode == 0, result.stderr
        assert self.calls() == ["pacman -S --needed --noconfirm graphviz"]
        assert "installed:" in result.stdout

    def test_nonroot_prints_instructions_without_installing(self):
        """Never invoke sudo, package managers, or backoff as a normal user."""
        for manager in ("apt-get", "dnf", "pacman", None):
            with self.subTest(manager=manager):
                if manager:
                    self.command(manager, 'printf "unexpected install\\n" >> "$PATH/calls"')
                result = self.run_installer()
                assert result.returncode != 0
                assert self.calls() == []
                assert "install it manually" in result.stderr
                assert "sudo apt-get install graphviz" in result.stderr
                assert "sudo dnf install graphviz" in result.stderr
                assert "sudo pacman -S --needed graphviz" in result.stderr
                assert "attempt" not in result.stdout
                if manager:
                    (self.bin / manager).unlink()

    def test_existing_dot_skips_install(self):
        """An existing Graphviz installation needs no package-manager call."""
        self.pacman()
        self.command("dot", "printf 'Graphviz\\n'")
        result = self.run_installer()
        assert result.returncode == 0, result.stderr
        assert self.calls() == []
        assert "already installed:" in result.stdout

    def test_unsupported_linux_fails_without_backoff(self):
        """Retrying cannot supply a missing package manager."""
        result = self.run_installer(TEST_UID="0")
        assert result.returncode != 0
        assert self.calls() == []
        assert result.stderr.count("no supported Linux package manager") == 1
        assert "sudo pacman -S --needed graphviz" in result.stderr

    def test_pacman_failure_retries_and_requires_dot(self):
        """Transient installation failures still retry and ultimately fail."""
        self.pacman(success=False)
        result = self.run_installer(TEST_UID="0", GRAPHVIZ_INSTALL_ATTEMPTS="2")
        assert result.returncode != 0
        assert self.calls() == [
            "pacman -S --needed --noconfirm graphviz",
            "sleep",
            "pacman -S --needed --noconfirm graphviz",
        ]

    def test_success_without_dot_still_fails(self):
        """A package-manager success is insufficient without the executable."""
        self.command("pacman", "exit 0")
        result = self.run_installer(TEST_UID="0", GRAPHVIZ_INSTALL_ATTEMPTS="1")
        assert result.returncode != 0
        assert "dot" in result.stderr
