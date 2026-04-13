# Security Policy

## Supported Versions

Only the latest release of Loman receives security fixes.

| Version | Supported |
| ------- | --------- |
| latest  | yes       |
| older   | no        |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report vulnerabilities privately via GitHub's built-in mechanism:
**Security > Advisories > Report a vulnerability**
(available at `https://github.com/janushendersonassetallocation/loman/security/advisories/new`)

Alternatively, email the maintainer directly at **edparcell@gmail.com**.

Include as much of the following as possible:

- Description of the vulnerability and its potential impact
- Steps to reproduce or a minimal proof-of-concept
- Affected versions
- Any suggested mitigations (optional)

## Response Process

1. You will receive an acknowledgement within **5 business days**.
2. We will investigate and aim to release a fix within **90 days** of confirmation, depending on severity.
3. A CVE will be requested where appropriate.
4. You will be credited in the release notes unless you prefer to remain anonymous.

## Scope

Loman is a pure-Python computation graph library. The most relevant security considerations are:

- **Deserialization**: loading a computation from an untrusted pickle or dill file can execute arbitrary code. Only deserialize computations from sources you trust.
- **Dependency vulnerabilities**: please report known-vulnerable transitive dependencies.

Out of scope: vulnerabilities in Python itself, the operating system, or third-party libraries not bundled with Loman.
