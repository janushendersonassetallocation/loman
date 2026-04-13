# Dependency Version Rationale

This document explains the version constraints chosen for each dependency in the Rhiza project.

## Overview

Rhiza uses carefully selected version constraints to balance stability, features, and future compatibility. This document provides the rationale for each constraint.

## Philosophy

Our dependency management follows these principles:

1. **Stability First**: Use stable versions that have been battle-tested in production
2. **Security**: Stay current enough to receive security updates
3. **Breaking Changes**: Avoid major version bumps that could introduce breaking changes
4. **Ecosystem Compatibility**: Ensure dependencies work well together
5. **Automated Updates**: Rely on Renovate to keep dependencies current within constraints

## Python Version

**Constraint**: `>=3.11`

**Rationale**:
- Python 3.11 introduces significant performance improvements (10-60% faster than 3.10)
- Provides modern syntax features like exception groups and `typing` improvements
- Maintains compatibility with most modern development environments
- Balances adoption (avoiding too new) with capability (avoiding too old)
- All dependencies support Python 3.11+

## Development Dependencies

### marimo

**Constraint**: `>=0.18.0,<1.0`

**Rationale**:
- **Lower Bound (0.18.0)**: This version introduced critical features used in our notebooks:
  - Improved chat streaming support with async generators
  - Frontend tools filtering based on chat mode
  - File explorer improvements with hidden file toggle
  - Progress display enhancements for long-running tasks
- **Upper Bound (<1.0)**: Prevents automatic upgrade to 1.0 which may introduce:
  - Breaking API changes
  - Changes to notebook execution model
  - Incompatible feature modifications
- **Use Case**: Reactive notebooks in `docs/notebooks/` for documentation and examples
- **Update Strategy**: Renovate will auto-merge minor/patch updates within the 0.x series

### numpy

**Constraint**: `>=2.4.0,<3.0`

**Rationale**:
- **Lower Bound (2.4.0)**: 
  - Includes all NumPy 2.0 improvements (10-20% performance boost)
  - Supports Python 3.11-3.14
  - Contains bug fixes and stability improvements over 2.0
  - Memory leak fixes and performance optimizations
- **Upper Bound (<3.0)**: NumPy 3.0 will likely introduce:
  - Breaking API changes
  - New type promotion rules
  - C API/ABI breakage requiring rebuild of dependent packages
  - Similar magnitude of changes as 1.x → 2.0 transition
- **Use Case**: Numerical computing in Marimo notebooks for data analysis examples
- **Update Strategy**: Stay within 2.x series; test thoroughly before moving to 3.0

**Note on NumPy 2.0**: The 2.0 release introduced significant breaking changes:
- Python API cleanup (removed deprecated functions, relocated types)
- New type promotion rules (NEP 50)
- Integer type standardization (always `int64` on 64-bit platforms)
- C API/ABI breakage requiring binary rebuilds

These changes are now stable in 2.4.x, but we pin below 3.0 to avoid the next major transition.

### plotly

**Constraint**: `>=6.5.0,<7.0`

**Rationale**:
- **Lower Bound (6.5.0)**: 
  - Mature, stable version with extensive charting capabilities
  - Full support for interactive visualizations in Marimo notebooks
  - Compatible with Narwhals for cross-framework data handling
- **Upper Bound (<7.0)**: Major version bump may introduce:
  - Breaking changes to chart configuration APIs
  - Changes to default styling or behavior
  - Updated JavaScript dependencies in the frontend
- **Use Case**: Interactive data visualization in Marimo notebooks
- **Update Strategy**: Auto-merge minor/patch updates within 6.x

### pandas

**Constraint**: `>=3,<3.1`

**Rationale**:
- **Lower Bound (3.0)**: 
  - Default string dtype (more efficient, consistent handling)
  - Copy-on-Write semantics (resolves view vs. copy confusion)
  - Removes deprecated 2.x functionality (cleaner API)
  - Requires Python 3.11+ and NumPy 1.26.0+ (aligned with our requirements)
- **Upper Bound (<3.1)**: Conservative constraint because:
  - 3.0 is a recent major release (January 2026)
  - Need time to validate stability in our notebooks
  - 3.1 may introduce new features requiring code changes
- **Use Case**: DataFrame manipulation in Marimo notebook examples
- **Update Strategy**: Will relax to `>=3,<4.0` after 3.0 proves stable in our use cases

**Note on pandas 3.0**: This major release introduced:
- String dtype as default (breaking for code checking for `object` dtype)
- Copy-on-Write enforced (eliminates `SettingWithCopyWarning`)
- Datetime resolution inference (no longer defaults to nanoseconds)
- `pd.col` syntax for declarative column operations
- Removed all 2.x deprecations

### pyyaml

**Constraint**: `>=6.0,<7.0`

**Rationale**:
- **Lower Bound (6.0)**: 
  - Stable, mature YAML parser
  - Security fixes for safe loading
  - Python 3.6+ support (well beyond our 3.11 requirement)
- **Upper Bound (<7.0)**: Major version bump may introduce:
  - Breaking changes to parsing behavior
  - Changes to safe loading defaults
  - Updated error handling
- **Use Case**: YAML parsing in validation scripts (`.rhiza/` tooling)
- **Update Strategy**: Auto-merge minor/patch updates within 6.x

## Production Dependencies

**None**: Rhiza is a template system and has zero runtime dependencies. All dependencies above are in the `[dependency-groups.dev]` section and are only needed for development, testing, and documentation.

## Dependency Management Process

### Adding New Dependencies

When adding dependencies:

1. **Research**: Check release notes for recent major versions
2. **Constrain**: Use `>=X.Y,<(X+1).0` for major version pinning
3. **Document**: Add inline comment explaining the purpose
4. **Test**: Verify with `make test` and `make deptry`
5. **Update This Doc**: Add entry to this file explaining the constraint rationale

### Updating Dependencies

Dependencies are automatically updated by Renovate:

- **Schedule**: Nightly checks for updates
- **Auto-merge**: Minor and patch updates within version constraints
- **Manual Review**: Major version bumps require human approval
- **Testing**: All updates trigger full CI pipeline

### Reviewing Version Constraints

Version constraints should be reviewed:

- **Quarterly**: Check if constraints are still appropriate
- **Before Major Releases**: Ensure dependencies are current
- **When Issues Arise**: If a specific version causes problems
- **On Security Advisories**: Immediately if a vulnerability is discovered

## Security Considerations

### Renovate Integration

- Automated dependency updates run nightly
- Security patches are auto-merged when safe
- Full test suite runs on every update
- Dependency vulnerability scanning via GitHub Security

### Advisory Monitoring

Dependencies are monitored for:

- CVE (Common Vulnerabilities and Exposures) reports
- Security advisories from package maintainers
- Deprecation notices
- End-of-life announcements

## References

- **uv Documentation**: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **Renovate Configuration**: See `renovate.json` in repository root
- **Dependency Workflows**: See `.rhiza/docs/WORKFLOWS.md`
- **Package Homepages**:
  - marimo: [https://marimo.io/](https://marimo.io/)
  - numpy: [https://numpy.org/](https://numpy.org/)
  - plotly: [https://plotly.com/python/](https://plotly.com/python/)
  - pandas: [https://pandas.pydata.org/](https://pandas.pydata.org/)
  - pyyaml: [https://pyyaml.org/](https://pyyaml.org/)

## Migration Guides

When major version updates are considered:

- **NumPy 2.x → 3.x**: Follow [NumPy Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- **pandas 2.x → 3.x**: Follow [pandas 3.0 Migration Guide](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v3.0.0.html#migration-guide)
- **marimo 0.x → 1.x**: Follow marimo's changelog and migration notes when available

## FAQ

**Q: Why do we use `<3.0` instead of `<4.0` for numpy?**  
A: NumPy 3.0 will likely introduce breaking changes similar to the 2.0 release. We prefer explicit testing before major version bumps.

**Q: Why is pandas constrained to `<3.1` instead of `<4.0`?**  
A: pandas 3.0 is very recent (January 2026). We're being conservative until we validate stability. We'll relax to `<4.0` after confirming 3.0 works well.

**Q: Can I add a new dependency?**  
A: Yes, but ensure it's necessary and follows the principles above. Use `uv add` to add it and document the rationale in this file.

**Q: What if I need a newer version for a specific feature?**  
A: Update the lower bound and document the reason. Run full tests and get PR approval.

**Q: How do I check for outdated dependencies?**  
A: Run `uv lock --upgrade` to see available updates. Renovate also creates PRs automatically.

---

*Last Updated: 2026-02-15*
