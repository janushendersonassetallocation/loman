"""Tests for Git LFS template structure and files.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Verifies that LFS-related files and configurations are present.
"""


class TestLFSTemplateStructure:
    """Tests for LFS template file structure."""

    def test_lfs_makefile_exists(self, root):
        """LFS makefile should exist in make.d directory."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        assert lfs_makefile.exists(), "lfs.mk not found"

    def test_lfs_documentation_exists(self, root):
        """LFS documentation should exist."""
        lfs_doc = root / "docs" / "LFS.md"
        assert lfs_doc.exists(), "LFS.md documentation not found"

    def test_lfs_makefile_has_targets(self, root):
        """LFS makefile should define all expected targets."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        content = lfs_makefile.read_text()

        required_targets = [
            "lfs-install:",
            "lfs-pull:",
            "lfs-track:",
            "lfs-status:",
        ]

        for target in required_targets:
            assert target in content, f"Target {target} not found in lfs.mk"

    def test_lfs_makefile_has_phony_declarations(self, root):
        """LFS makefile should declare targets as phony."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        content = lfs_makefile.read_text()

        assert ".PHONY:" in content
        assert "lfs-install" in content
        assert "lfs-pull" in content
        assert "lfs-track" in content
        assert "lfs-status" in content

    def test_lfs_makefile_has_help_comments(self, root):
        """LFS makefile should have help comments for targets."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        content = lfs_makefile.read_text()

        # Check for ##@ section header
        assert "##@ Git LFS" in content

        # Check for target descriptions
        assert "##" in content

    def test_lfs_documentation_has_sections(self, root):
        """LFS documentation should have all expected sections."""
        lfs_doc = root / "docs" / "LFS.md"
        content = lfs_doc.read_text()

        expected_sections = [
            "# Git LFS",
            "## Overview",
            "## Available Make Targets",
            "## Typical Workflow",
            "## CI/CD Integration",
            "## Troubleshooting",
        ]

        for section in expected_sections:
            assert section in content, f"Section '{section}' not found in LFS.md"

    def test_lfs_documentation_describes_all_targets(self, root):
        """LFS documentation should describe all make targets."""
        lfs_doc = root / "docs" / "LFS.md"
        content = lfs_doc.read_text()

        targets = [
            "lfs-install",
            "lfs-pull",
            "lfs-track",
            "lfs-status",
        ]

        for target in targets:
            assert target in content, f"Target {target} not documented in LFS.md"

    def test_lfs_makefile_cross_platform_support(self, root):
        """LFS makefile should support multiple platforms."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        content = lfs_makefile.read_text()

        # Check for OS detection
        assert "uname -s" in content
        assert "Darwin" in content
        assert "Linux" in content

        # Check for architecture detection (macOS)
        assert "uname -m" in content
        assert "arm64" in content
        assert "amd64" in content

    def test_lfs_makefile_error_handling(self, root):
        """LFS makefile should include error handling."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        content = lfs_makefile.read_text()

        # Check for error messages
        assert "ERROR" in content
        assert "exit 1" in content

    def test_lfs_makefile_uses_color_variables(self, root):
        """LFS makefile should use standard color variables."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        content = lfs_makefile.read_text()

        # Check for color variable usage
        assert "BLUE" in content
        assert "RED" in content
        assert "RESET" in content
