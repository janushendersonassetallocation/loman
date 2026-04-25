# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | ✓         |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report vulnerabilities by emailing the maintainers at the address listed on the
[PyPI project page](https://pypi.org/project/loman/) or by using
[GitHub's private vulnerability reporting](https://github.com/janushendersonassetallocation/loman/security/advisories/new).

Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- Affected versions

You can expect an acknowledgement within **5 business days** and a resolution
timeline within **90 days** of the initial report.

## Security Considerations

Loman uses [dill](https://pypi.org/project/dill/) for serialization
(`write_dill` / `read_dill`). Deserializing data from untrusted sources with
`dill.load` can execute arbitrary code. Only load `.dill` files from sources
you trust.
