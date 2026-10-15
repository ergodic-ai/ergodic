# Releasing ergodic

This is the runbook for publishing ergodic, written for the 0.2.0 joint release with
`welt-client` (DEC-078, API-146/API-147). Steps marked **[owner go]** change something
outside this repository and are hard to undo. Run each one only after the owner has
approved that specific step on release day.

Placeholders: `$OWNER_LOGIN` is the GitHub login of the person who approves PyPI
deployments.

## How publishing works

`.github/workflows/release.yml` publishes with PyPI Trusted Publishing; there are no
API tokens anywhere.

- Pushing a tag `vX.Y.Z` runs the full CI matrix (`make check` on Python 3.11 to 3.14),
  builds the sdist and wheel with uv, runs `twine check --strict`, checks that the tag
  equals both the built version and `ergodic.__version__`, installs the wheel and the
  sdist into clean virtual environments and imports them, then publishes to PyPI from
  the `pypi` environment.
- Running the workflow by hand with `testpypi` ticked does the same checks and
  publishes to TestPyPI from the `testpypi` environment. A manual run never publishes
  to PyPI.

## 0. Preconditions (no outward changes)

1. API-140 to API-145 are merged to `dev`, reviewed, and `dev` is merged to `main`.
2. `src/ergodic/__init__.py` has `__version__ = "0.2.0"` and `CHANGELOG.md` has a
   `## [0.2.0]` section.
3. CI is green on `main` (`gh run list -R ergodic-ai/ergodic --branch main --workflow ci.yml`).
4. Re-run the history scan on everything GitHub will serve and confirm it is clean.
   That includes the pull-request refs (`refs/pull/*/head`), which a normal clone or
   `git fetch` leaves out but which become public with the repository. Scan from a
   mirror:

   ```bash
   git clone --mirror https://github.com/ergodic-ai/ergodic.git ergodic-mirror.git
   git -C ergodic-mirror.git fetch origin '+refs/pull/*:refs/pull/*'   # explicit; a mirror normally has them
   git -C ergodic-mirror.git for-each-ref --format='%(refname)' | sed 's|/[^/]*$||' | sort | uniq -c
   gitleaks git --log-opts="--all" --redact ergodic-mirror.git

   # trufflehog wants a non-bare repository: copy every head, tag and PR ref in as branches.
   git init -q -b scan-unused scan-tree
   git -C scan-tree fetch -q ../ergodic-mirror.git '+refs/heads/*:refs/heads/*' \
     '+refs/tags/*:refs/tags/*' '+refs/pull/*/head:refs/heads/pull/*'
   git -C scan-tree checkout -q main   # trufflehog needs an index
   trufflehog git "file://$PWD/scan-tree" --no-verification --no-update \
     --results=verified,unknown,unverified
   ```

   Expected: gitleaks reports no leaks; trufflehog reports only the known "Box" false
   positive inside the Illustrator data of the dark logo SVG.

5. Owner decisions recorded: which stale remote branches to delete before going
   public (`codex/latent-productization`, `forecast-s5`, `scm-mvp`, `docs-redesign`,
   `metalearners-phase2`), and whether the Illustrator data in the logo SVGs
   (`public/`) stays (see the API-146 scan report; `deploy/` and `docs/assets/` went
   with the MkDocs site in 0.2.0, though history keeps them). Deleting a branch does **not** hide its
   content: the pull-request refs keep those commits reachable on GitHub (`pull/2` is
   `scm-mvp`, `pull/5` is `metalearners-phase2`; `pull/1` and `pull/3` hold 17
   pre-squash commits no branch reaches), and only GitHub support can remove PR refs.
   The API-146 scan and its independent review scanned all of them: clean.

## 1. Make the repository public **[owner go]**

```bash
gh repo edit ergodic-ai/ergodic --visibility public --accept-visibility-change-consequences
gh api repos/ergodic-ai/ergodic --jq .visibility        # expect "public"
```

Then, while you are in the settings (all optional, all reversible):

```bash
# Private vulnerability reporting, which SECURITY.md can point to later.
gh api -X PUT repos/ergodic-ai/ergodic/private-vulnerability-reporting
# Secret scanning and push protection (free for public repositories).
gh api -X PATCH repos/ergodic-ai/ergodic \
  -f 'security_and_analysis[secret_scanning][status]=enabled' \
  -f 'security_and_analysis[secret_scanning_push_protection][status]=enabled'
```

Branch protection for `main` and `dev` that requires the four `make check (Python …)`
jobs is recommended; set it under Settings → Branches once CI has run once publicly.

Going public can't really be undone: clones, forks and caches keep what they saw.

## 2. Create the deployment environments **[owner go]**

Create them before the first tag. A workflow that names a missing environment
creates it with no protection.

```bash
REVIEWER_ID=$(gh api users/$OWNER_LOGIN --jq .id)
for env in pypi testpypi; do
  gh api -X PUT repos/ergodic-ai/ergodic/environments/$env --input - <<EOF
{"reviewers":[{"type":"User","id":$REVIEWER_ID}],
 "deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}}
EOF
done
# pypi deploys only from v* tags; testpypi from main (manual dry runs).
gh api -X POST repos/ergodic-ai/ergodic/environments/pypi/deployment-branch-policies -f name='v*' -f type=tag
gh api -X POST repos/ergodic-ai/ergodic/environments/testpypi/deployment-branch-policies -f name=main -f type=branch
```

## 3. Configure the trusted publishers **[owner go]**

Both need a web login; neither has a CLI.

**PyPI** (needs a PyPI account that is an Owner of the `ergodic` project, with 2FA):
open <https://pypi.org/manage/project/ergodic/settings/publishing/>, choose
"GitHub", and enter exactly:

| Field | Value |
| --- | --- |
| Owner | `ergodic-ai` |
| Repository name | `ergodic` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

**TestPyPI** (a separate account at test.pypi.org): the `ergodic` project does not
exist on TestPyPI (checked 2026-10-14), so add a *pending* publisher at
<https://test.pypi.org/manage/account/publishing/> with project name `ergodic`, the same
owner, repository and workflow, and environment `testpypi`. The first upload creates
the project.

## 4. TestPyPI dry run **[owner go]**

`workflow_dispatch` needs `release.yml` on the default branch.

```bash
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gh workflow run release.yml -R ergodic-ai/ergodic --ref main -f testpypi=true
# Wait for this run to register, so the lookup can't pick up an older one.
until RUN=$(gh run list -R ergodic-ai/ergodic --workflow release.yml --event workflow_dispatch \
    --json databaseId,createdAt --jq "[.[] | select(.createdAt >= \"$since\")][0].databaseId // empty") \
    && [ -n "$RUN" ]; do sleep 5; done
gh run watch -R ergodic-ai/ergodic "$RUN"
# Approve the testpypi deployment when asked (Actions tab, "Review deployments").

uv venv /tmp/ergodic-testpypi
uv pip install --python /tmp/ergodic-testpypi \
  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match ergodic==0.2.0
/tmp/ergodic-testpypi/bin/python -c "import ergodic; print(ergodic.__version__)"
```

Record the run URL as the API-146 dry-run evidence. TestPyPI accepts a given version
once; a second dry run of the same version fails at upload, which is expected.

The wheel's metadata is version 2.5 (current hatchling). The dry run is also the
check that the publish action and PyPI accept it.

## 5. Prepare GitHub Pages before the tag **[owner go]**

The docs workflow (`.github/workflows/docs.yml`, from the docs builder) deploys to the
`github-pages` environment. Set Pages and that environment up **before** any `v*` tag
is pushed. Otherwise the docs deployment either fails (Pages not enabled, or the
environment only allows the default branch: "Tag v0.2.0 is not allowed to deploy to
github-pages due to environment protection rules") or goes live with no separate go.

1. **Verify the domain for the organization** (stops anyone else's Pages site
   claiming it): GitHub → ergodic-ai organization → Settings → Pages → Add a domain →
   `ergodic.dev`. GitHub shows a TXT record named
   `_github-pages-challenge-ergodic-ai.ergodic.dev`; add it in Cloudflare (see the DNS
   notes below), then click Verify. This TXT record changes nothing that is served.
2. **Enable Pages with the Actions source:**
   `gh api -X POST repos/ergodic-ai/ergodic/pages -f build_type=workflow`
   Until the custom domain is set in step 8, the site would serve at
   `https://ergodic-ai.github.io/ergodic/`.
3. **Gate the `github-pages` environment and allow release tags.** Same calls as
   step 2 (`$REVIEWER_ID` from there):

   ```bash
   gh api -X PUT repos/ergodic-ai/ergodic/environments/github-pages --input - <<EOF
   {"reviewers":[{"type":"User","id":$REVIEWER_ID}],
    "deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}}
   EOF
   gh api -X POST repos/ergodic-ai/ergodic/environments/github-pages/deployment-branch-policies -f name='v*' -f type=tag
   gh api repos/ergodic-ai/ergodic/environments/github-pages/deployment-branch-policies --jq '.branch_policies[] | .type + " " + .name'
   ```

   Pages may already have added a `main` policy; that's fine. The reviewer makes
   the docs go-live its own approval.

**Docs trigger.** `docs.yml` publishes on `release: published`: the GitHub release
created at the end of step 7, after the PyPI upload is verified, so the docs never
describe a version that isn't on PyPI. Pushing the tag doesn't deploy the docs. The
manual fallback is `gh workflow run docs.yml -R ergodic-ai/ergodic -f release_tag=v0.2.0`.
Either way, the `github-pages` reviewer gate above holds the deployment until step 8.

## 6. Delete the inherited empty 1.0.0 **[owner go]**

PyPI lists a file-less 1.0.0 ("A simple printer of nested lists") as the latest
release. Delete it just before tagging 0.2.0, so the project page shows 0.2.0 as
latest from the moment it appears.

1. Log in at <https://pypi.org> as an Owner of `ergodic`.
2. Open <https://pypi.org/manage/project/ergodic/releases/>.
3. On the `1.0.0` row choose Options → Delete, type `1.0.0` to confirm.
4. Check: `curl -s https://pypi.org/pypi/ergodic/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"`
   prints `0.1.0`.

Deletion is permanent. Never reuse 1.0.0 for a real release.

## 7. Tag and publish ergodic 0.2.0 **[owner go]**

Order for the joint release (API-147): repository public (step 1), Pages prepared
(step 5), **welt-client 0.13.0 to PyPI**, then relock ergodic (`uv lock` now resolves
`welt-client>=0.13` for the `welt` extra, which CI installs), then **ergodic 0.2.0 to
PyPI**, then both doc sites (step 8 here), then the console agent docs.
welt-client goes first because ergodic's lock and CI need it: until it is on PyPI,
`uv sync --locked` can't resolve the `welt` extra. Its `welt-client[ergodic]` extra
(`ergodic>=0.2,<0.3`) can't resolve until ergodic 0.2.0 is out, so publish ergodic
straight after.

```bash
git switch main && git pull --ff-only
uv run python -c "import ergodic; print(ergodic.__version__)"   # expect 0.2.0
git tag -a v0.2.0 -m "ergodic 0.2.0"
git push origin v0.2.0
until RUN=$(gh run list -R ergodic-ai/ergodic --workflow release.yml --event push --branch v0.2.0 \
    --json databaseId --jq '.[0].databaseId // empty') && [ -n "$RUN" ]; do sleep 5; done
gh run watch -R ergodic-ai/ergodic "$RUN"
# Approve the pypi deployment when the check and build jobs are green.
# The tag push does not deploy the docs; that starts with the GitHub release below.
```

Verify from PyPI in a clean environment:

```bash
uv venv /tmp/ergodic-pypi && uv pip install --python /tmp/ergodic-pypi "ergodic==0.2.0"
/tmp/ergodic-pypi/bin/python -c "import ergodic; print(ergodic.__version__)"
```

Create the GitHub release from the changelog section when step 8 (docs live) has its
go, because publishing the release starts the docs deployment:

```bash
gh release create v0.2.0 -R ergodic-ai/ergodic --verify-tag --title "ergodic 0.2.0" \
  --notes-file <(awk '/^## \[0.2.0\]/{f=1;next} /^## \[/{f=0} f' CHANGELOG.md)
```

## 8. Docs live at ergodic.dev **[owner go]**

### What ergodic.dev looks like today (checked 2026-10-14, read-only)

| Name | Records | Serves |
| --- | --- | --- |
| `ergodic.dev` (apex) | No A, AAAA or CNAME. TXT: Resend and Google site verification. MX: `10 inbound-smtp.eu-west-1.amazonaws.com`. No CAA. | Does not resolve (`curl: Could not resolve host`). |
| `www.ergodic.dev` | Proxied through Cloudflare (104.21.86.169, 172.67.222.101) | HTTP 525 on https and 520 on http: Cloudflare can't reach a working origin. |
| `welt.ergodic.dev` | CNAME `ghs.googlehosted.com` | The Welt docs (Google-hosted mapping). Leave alone. |
| `docs.ergodic.dev` | none | |

The zone is on **Cloudflare** (`NS fish.ns.cloudflare.com`, `west.ns.cloudflare.com`),
so records are changed in the Cloudflare dashboard of the account that owns
`ergodic.dev`. It is not in Google Cloud DNS (the Cloud DNS API is not enabled on GCP
project `ergodic`). Keep the MX and TXT records: the MX receives mail via Amazon SES
and the TXT records verify Resend and Google.

### Steps

1. **Start the deployment.** Create the GitHub release (end of step 7). That runs
   `docs.yml` (`release: published`); approve its `github-pages` deployment in the
   Actions tab. If the run doesn't start or has to be repeated, use
   `gh workflow run docs.yml -R ergodic-ai/ergodic -f release_tag=v0.2.0`.
2. **Set the custom domain:**
   `gh api -X PUT repos/ergodic-ai/ergodic/pages -f cname=ergodic.dev`
   (an Actions-deployed site ignores a `CNAME` file; the setting is what counts).
3. **Add the apex records in Cloudflare, proxy off ("DNS only", grey cloud)** so
   GitHub can issue the certificate:

   | Type | Name | Content |
   | --- | --- | --- |
   | A | `@` | `185.199.108.153` |
   | A | `@` | `185.199.109.153` |
   | A | `@` | `185.199.110.153` |
   | A | `@` | `185.199.111.153` |
   | AAAA | `@` | `2606:50c0:8000::153` |
   | AAAA | `@` | `2606:50c0:8001::153` |
   | AAAA | `@` | `2606:50c0:8002::153` |
   | AAAA | `@` | `2606:50c0:8003::153` |
   | CNAME | `www` | `ergodic-ai.github.io` (replaces the current broken proxied record, DNS only) |

   If the owner prefers a subdomain instead of the apex, add only
   `CNAME docs → ergodic-ai.github.io` (DNS only) and use `docs.ergodic.dev` in step 2.
4. **Wait for the certificate, then enforce HTTPS:**
   `gh api repos/ergodic-ai/ergodic/pages --jq '.https_certificate.state'` until
   `approved`, then `gh api -X PUT repos/ergodic-ai/ergodic/pages -F https_enforced=true`.
5. **Check:**

   ```bash
   dig +short ergodic.dev A     # the four 185.199.x.153 addresses
   dig +short ergodic.dev AAAA
   curl -sI https://ergodic.dev | head -3   # HTTP/2 200 from GitHub
   ```

6. The MkDocs site's source is gone in 0.2.0. Once the new site is live, the old
   site still served on Cloud Run (`ergodic-docs`, project `ergodic`) retires
   (DEC-078). Deleting that service is a separate owner go.

## 9. Rollback

- **A broken release on PyPI:** yank it, don't delete it. Open
  <https://pypi.org/manage/project/ergodic/release/0.2.0/>, Options → Yank, and give a
  reason. Yanked versions stay installable by exact pin (`==0.2.0`) but resolvers skip
  them otherwise. Fix forward with 0.2.1 through the same workflow. PyPI never accepts
  a re-upload of a filename that existed, so deleting a release burns its version.
- **welt-client depends on it:** `welt-client[ergodic]` accepts `>=0.2,<0.3`, so a
  0.2.1 fix reaches its users without a welt-client release. Yanking 0.2.0 with no 0.2.1
  available breaks `welt-client[ergodic]` installs; ship 0.2.1 first or together.
- **The tag was pushed but nothing was uploaded** (a check or build failed): fix on
  `main`, then move the tag: `git tag -d v0.2.0 && git push origin :refs/tags/v0.2.0`,
  re-tag and push. If anything reached PyPI, never move the tag; bump the version.
- **DNS or Pages misbehaves:** delete the A/AAAA/CNAME records added in step 8 (MX
  and TXT untouched), or unpublish with
  `gh api -X DELETE repos/ergodic-ai/ergodic/pages`.
- **Visibility:** `gh repo edit ergodic-ai/ergodic --visibility private
  --accept-visibility-change-consequences` hides the repository again, but anything
  already cloned or forked stays out there. Treat going public as permanent.

## Later releases

Bump `__version__` in `src/ergodic/__init__.py`, move the `[Unreleased]` changelog
entries under the new version, merge to `main`, then run step 7 and approve the docs deployment (step 8.1). Use
step 9 if something goes wrong. Steps 1 to 6 and the rest of step 8 are one-time.
