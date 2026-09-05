# Release Checklist

Releases are largely automated through the rhiza release workflow. The local
`make` targets handle version bumping and tagging; the `(RHIZA) RELEASE`
workflow on GitHub Actions takes over once the tag is pushed and handles
building, SBOM generation, PyPI publishing (via OIDC Trusted Publishing), and
creating the GitHub release.

## Pre-release checks

- Confirm `main` is green on the `(RHIZA) CI` workflow.
- Update [`CHANGELOG.md`](../../CHANGELOG.md).

## Cut the release

Use `make publish` to bump the version, tag, and push in one step:

```bash
make publish
```

This invokes `rhiza-tools release --with-bump`, which:

1. Bumps the version in `pyproject.toml` and refreshes `uv.lock`.
2. Commits the bump.
3. Creates a `vX.Y.Z` tag and pushes it to the remote.

If you prefer the two phases separately:

```bash
make bump      # bump version + update uv.lock + commit
make release   # tag and push
```

Add `DRY_RUN=1` to either target to preview without applying.

## Automated workflow (triggered by tag push)

Once the `v*` tag is pushed, the `(RHIZA) RELEASE` workflow runs and:

- Validates the tag.
- Builds the sdist + wheel.
- Generates a CycloneDX SBOM with attestations.
- Drafts a GitHub release with artifacts.
- Publishes to PyPI via OIDC Trusted Publishing.
- Finalises (publishes) the GitHub release.

In parallel, `(RHIZA) BOOK` rebuilds and deploys the documentation site to
GitHub Pages.

## Post-release

Check status with:

```bash
make release-status
```

This shows the latest workflow run and the published release.
