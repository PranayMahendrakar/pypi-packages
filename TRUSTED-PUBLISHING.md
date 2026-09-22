# Release authentication

Releases use PyPI trusted publishing. GitHub Actions proves its identity to PyPI with a
short-lived OpenID Connect token, and PyPI grants upload rights for that one package.
No API token is stored in this repository, in GitHub secrets, or on any machine.

## What you register on PyPI

For a package that does **not exist on PyPI yet**, add a *pending* publisher at
<https://pypi.org/manage/account/publishing/>. For one that already exists, add the
publisher under that project's own publishing settings instead.

Every package uses the same four values except the project name:

| Field            | Value                  |
|------------------|------------------------|
| PyPI Project Name| the package directory name, e.g. `dataset-health` |
| Owner            | `PranayMahendrakar`    |
| Repository name  | `pypi-packages`        |
| Workflow name    | `publish.yml`          |
| Environment name | `pypi`                 |

The environment is optional in PyPI's form but worth setting. It means someone who can
commit to this repository still cannot publish unless they can also run that
environment, so write access and release access are separate things.

A pending publisher does **not** reserve the name. Until the first publish, anyone else
can still create that project.

## The GitHub side

Create an environment called `pypi` under Settings, Environments. Add a required
reviewer if you want a human to approve each release.

## Releasing

Actions, "Publish one package to PyPI", Run workflow, then give the package directory
name. Tick `dry_run` to build and check without uploading.

The workflow runs the package's own test suite against an installed copy, not against
the source tree beside it, so a packaging mistake that would break a real user is caught
before anything is uploaded.

## One package at a time, on purpose

There is no matrix over every package. PyPI limits how quickly one account may create
new projects, and thirty simultaneous uploads is the exact pattern that limit exists to
stop. Releasing one at a time also keeps a bad build from affecting everything at once.


## One publisher authorises one project, not your account

This catches people out. A trusted publisher grants upload rights for a single project
name. Registering one does not cover the others, so 27 remaining names means 27
registrations. They are quick, and every field except the project name is identical, but
there is no way to do them in bulk: PyPI has no API for registering publishers.

## If that is too much clicking

`.github/workflows/publish-queue.yml` also accepts an API token, and uses it
automatically when one is present. One setup step covers every package:

1. On PyPI, Account settings, API tokens, add a token scoped to the entire account.
   It has to be account-wide, because the projects do not exist yet and a
   project-scoped token cannot create one.
2. In this repository: Settings, Secrets and variables, Actions, New repository secret.
   Name it `PYPI_API_TOKEN` and paste the token.

The workflow then uses the token and skips the trusted-publishing step.

### Which to choose

Trusted publishing is genuinely safer. Nothing is stored, and the credential GitHub
mints lives for minutes rather than forever. A token is a long-lived account-wide
credential, and although GitHub encrypts secrets at rest and never exposes them to
workflows from forked pull requests, anyone who can push to `main` here can run a
workflow that uses it.

A reasonable middle path: use the token to get the remaining packages published, then
delete it and register ordinary publishers once each project exists. At that point the
registration moves to each project's own settings page and the account-wide token is no
longer needed at all.
