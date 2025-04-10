# Release Checklist

- Check CHANGELOG is up-to-date
- Check [Travis CI](https://travis-ci.org/janusassetallocation/loman) builds are passing
- Check [Read The Docs documentation builds](https://readthedocs.org/projects/loman/) are passing
- Update version string in
  - pyproject.toml
  - docs/conf.py
- Commit updated versions and tag
- Build the tar.gz and wheel: `python -m build`
- Upload the tar.gz and wheel: `twine upload dist\loman-x.y.z*`
- Email the community
