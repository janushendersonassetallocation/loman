# GitHub Actions Configuration

This document describes the secrets used by the Rhiza-provided GitHub Actions workflows
(`.github/workflows/rhiza_*.yml`) and how to configure them.

## PAT_TOKEN

Some workflows may need to push changes to files under `.github/workflows/`. The
automatic `github.token` **cannot** do that — GitHub rejects such pushes unless
the token carries the `workflow` scope. If you need it, create a Personal Access
Token (PAT) with the `workflow` scope and store it as a repository secret named
`PAT_TOKEN`.

If `PAT_TOKEN` is not configured, workflows fall back to `github.token`.

### Creating the token

**Fine-grained PAT** (recommended):

1. Go to **Settings → Developer settings → Fine-grained tokens → Generate new token**
   (<https://github.com/settings/personal-access-tokens/new>).
2. Restrict **Repository access** to the repository (or repositories) using Rhiza.
3. Under **Repository permissions**, grant:
   - **Contents**: Read and write
   - **Workflows**: Read and write
   - **Pull requests**: Read and write (needed for the scheduled sync-PR mode)
4. Generate the token and copy it.

**Classic PAT** (alternative):

1. Go to **Settings → Developer settings → Tokens (classic) → Generate new token**.
2. Select the `repo` and `workflow` scopes.
3. Generate the token and copy it.

### Storing the secret

In the repository that consumes Rhiza:

1. Go to **Settings → Secrets and variables → Actions → New repository secret**.
2. Name: `PAT_TOKEN`
3. Value: the token created above.

Or with the GitHub CLI:

```bash
gh secret set PAT_TOKEN
```

A PAT expires; when sync pushes start failing with a `refusing to allow ... workflow` error,
regenerate the token and update the secret.

## Release workflow secrets (optional)

The release workflow (`.github/workflows/rhiza_release.yml`) supports additional secrets, all
optional depending on which release features you use:

| Secret | Purpose |
| --- | --- |
| `PYPI_TOKEN` | Publish the built package to PyPI. Not needed when using trusted publishing (OIDC). |
| `RELEASE_PUBLISH_CREDENTIALS` | Optional credentials input for a repository-owned custom publisher. Not needed when the extension uses OIDC. |
| `GH_PAT` | Git authentication for installing private dependencies during the release build. |
| `UV_EXTRA_INDEX_URL` | Extra package index URL (with credentials) for private dependencies. |

`GITHUB_TOKEN` is provided automatically by GitHub Actions and needs no configuration.

## Release publisher selection

Set the Actions **variable** `RELEASE_PUBLISHER` to choose package deployment:

| Value | Behaviour |
| --- | --- |
| Unset or `pypi` | Existing PyPI Trusted Publishing. `PYPI_REPOSITORY_URL` (variable) and `PYPI_TOKEN` (secret) still support PyPI-compatible custom endpoints. |
| `custom` | Run your repository-owned composite action at `.github/actions/release-publish/action.yml` (or `action.yaml`). No fallback to public PyPI. |
| `none` | Skip package publishing, retaining the other release outputs. |

No bundle syncs the custom action path: it remains repository-owned.
Commit the action before tagging the release;
it runs from that release revision with the `release` environment, `contents: read` and
`id-token: write`. It receives `artifacts-dir` (absolute path to the downloaded build
artifacts), `tag`, `version` (tag without `v`) and the optional `credentials` input.
It owns provider login, tool setup and upload, and must fail on unsuccessful publication.
Use the existing distributions rather than rebuilding them; the directory can also
contain provenance files.

For OIDC-capable feeds, obtain and immediately mask short-lived credentials in this
action. Otherwise, supply the optional `RELEASE_PUBLISH_CREDENTIALS` secret in the
format your action expects. Only that explicit secret is forwarded to the extension.
An optional `artifact-url` output supplies a public HTTPS artifact link for release notes;
omit it for private destinations and never include credentials or signed access tokens.

The private-package classifier still suppresses built-in PyPI uploads; `custom`
explicitly permits private distributions. Invalid modes, missing custom actions and
missing expected distributions fail closed. Failed required uploads prevent release
finalization. Conda recipes are generated only after successful public PyPI publication
with no `PYPI_REPOSITORY_URL` override.

This is a GitHub publishing extension, not private dependency-consumption setup or a
GitLab feature. See the
[publisher contract](https://jebel-quant.github.io/rhiza/reference/ARCHITECTURE/#repository-owned-publishing-action)
for details.

## Private dependency secrets in other workflows

`GH_PAT` and `UV_EXTRA_INDEX_URL` are read by the CI, benchmark, CodeQL, marimo, book,
weekly and docker workflows too. The stubs that call those reusable workflows forward each secret by
name rather than with `secrets: inherit`, because GitHub only honours `inherit` when the
caller sits in the same organisation or enterprise as `jebel-quant/rhiza` — from any other
organisation the secrets simply never arrived, and private dependencies failed to install
with `could not read Password for 'https://***@github.com'` (#1689). A secret that is not
defined is forwarded empty and the workflow falls back to `github.token`, so nothing is
required for a project with no private dependencies. Pull requests from forks never receive
secrets at all, so a fork PR that needs a private dependency fails at install; that is
GitHub's rule rather than a rhiza setting.

The docker workflow is the one place the runner's git configuration cannot reach, because
`uv sync` runs inside the image build. It passes both secrets to `docker buildx build` as
BuildKit secrets instead, which exist only for that one instruction and are written into no
layer; a build argument would be readable with `docker history` (#1691). See
`docs/development/DOCKER.md` for building such an image locally.

## Private dependency setup (opt-in)

Set `RHIZA_SETUP_ENABLED=true` and commit a consumer-owned
`.github/actions/rhiza-setup/action.yml` (or `action.yaml`). The action receives
`purpose` and the optional `RHIZA_SETUP_CREDENTIALS` secret as `credentials`.
Keep provider settings inside that action; no repository variables are forwarded.
Setup exports through `GITHUB_ENV` survive an unset legacy `UV_EXTRA_INDEX_URL`
secret. Publication is independent: built-in PyPI remains the default.

For Dependabot PRs, also define `RHIZA_SETUP_CREDENTIALS` as a Dependabot secret
when setup requires it; Actions secrets are not available to those runs. The same
applies to `BOOK_SETUP_TOKEN` for deprecated book callers.

In the shipped workflows, OIDC-based setup works only in the release `build` job,
which already has `id-token: write` for attestations. CI, book, marimo, benchmark
and weekly setup support token-based authentication only.

Default setup jobs explicitly retain read-only permissions; the book build never
inherits Pages deployment permissions. A caller-only `id-token: write` grant is
insufficient for OIDC setup: opt in with a consumer-owned workflow copy granting
OIDC only to the dependency jobs, plus the matching caller grant. See the setup
contract below for the procedure. Excluded wrappers no longer receive automatic
workflow-version updates; bump their references and maintain workflow copies manually.
Pages deployment and release signing retain their existing separate permissions.
Fork PRs and `pull_request_target` fail when setup is enabled, before the action runs.

For the CodeArtifact example, IAM/OIDC configuration, workflow coverage, book-hook
migration and deferred container/AWS validation work, see the
[setup contract](https://jebel-quant.github.io/rhiza/reference/ARCHITECTURE/#consumer-owned-dependency-setup).
Rename the former book action to `rhiza-setup`, its `token` input to `credentials`,
and `BOOK_SETUP_TOKEN` to `RHIZA_SETUP_CREDENTIALS`; remove `with.book-setup` and set
`RHIZA_SETUP_ENABLED=true` instead. The old input and secret remain deprecated aliases
until the next major release. `book-setup: true` enables setup for the book job and
emits a warning. The generic action takes precedence; if absent, the old
`rhiza-book-setup` action runs with its `token` input. `RHIZA_SETUP_CREDENTIALS`
takes precedence over `BOOK_SETUP_TOKEN`. Only one action runs.
