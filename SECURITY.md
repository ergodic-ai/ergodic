# Security policy

## Reporting a vulnerability

Please report security vulnerabilities privately, by email to
**andre@ergodic.ai**, with "ergodic security" in the subject line. Don't open a public
issue, pull request or discussion for a suspected vulnerability.

Include what you can of:

- the affected version (`python -c "import ergodic; print(ergodic.__version__)"`),
- a description of the issue and its impact,
- steps or a minimal script that reproduces it.

We will acknowledge your report and agree a disclosure date with you once we
understand the issue. We credit reporters in the release notes
unless you ask us not to.

## Supported versions

Security fixes go into the latest release on PyPI. ergodic is pre-1.0, so older
minor versions don't receive backports; upgrade to the latest release to get a fix.

## Scope

ergodic is a local library: its core makes no network calls and stores nothing. The
reports that matter most are code paths that execute or deserialize untrusted input
unsafely, and dependency or packaging problems in the published distributions.
Vulnerabilities in third-party dependencies are best reported to those projects
directly; tell us too if ergodic's use of them makes the problem worse.
