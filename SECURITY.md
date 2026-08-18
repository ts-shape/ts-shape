# Security Policy

## Supported Versions

ts-shape releases from `main` on every merge, and fixes land in the next
release. Only the **latest release on PyPI** receives security fixes; there are
no long-term support branches. Pin a version for reproducibility, but upgrade to
pick up a fix rather than expecting a backport.

| Version | Supported |
|---------|-----------|
| Latest release on [PyPI](https://pypi.org/project/ts-shape/) | ✅ |
| Any earlier version | ❌ — upgrade to the latest |

## Reporting a Vulnerability

**Do not open a public issue for a security problem.**

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/ts-shape/ts-shape/security/advisories/new)
on this repository. If that is unavailable to you, email the maintainer at the
address published in the package metadata (`pyproject.toml`) with `ts-shape
security` in the subject line.

Please include:

- the affected version and how it was installed,
- a minimal reproduction (a script plus the data shape it needs),
- the impact you believe it has, and
- any mitigation you have already found.

### What to expect

| Stage | Target |
|-------|--------|
| Acknowledgement of your report | within 5 working days |
| Initial assessment and severity | within 10 working days |
| Fix released, or a plan with dates | within 30 days for high severity |

You will be credited in the release notes unless you ask not to be. Please give
us the chance to release a fix before disclosing publicly.

## Scope

ts-shape is a **library**: it shapes DataFrames and reads from data sources you
configure. Reports that are in scope include, for example:

- code execution or file access through the lambda-rule expression evaluator
  (`ts_shape.eventlog.lambda_rules`), which is designed to be sandboxed;
- a loader following a configured path or URL somewhere it should not
  (path traversal, SSRF-style behaviour);
- credentials or connection strings leaking into logs, warnings, or exception
  messages;
- deserialisation of untrusted files (Parquet, JSON metadata) leading to code
  execution.

Out of scope:

- vulnerabilities in dependencies without a ts-shape-specific exploit path —
  report those upstream; our CI runs `pip-audit` and picks up fixed versions
  through Renovate;
- anything that requires the attacker to already control the Python process, the
  code being run, or the machine;
- results that are wrong rather than unsafe. Those are bugs — please open a
  normal issue, they are taken seriously too.

## Handling Credentials

ts-shape loaders accept credentials for object storage and databases. Pass them
through environment variables or your platform's secret manager; never commit
them, and never paste them into an issue, a discussion, or a reproduction
script. If you believe a credential was exposed through ts-shape's own output,
treat it as a security report and use the private channel above.
