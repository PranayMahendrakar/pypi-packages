# Trusted publishing setup

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
