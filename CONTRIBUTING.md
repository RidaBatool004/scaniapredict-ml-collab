# Contributing Guide

## Project Overview

This repository uses Git, GitHub Pull Requests, DVC, `uv`, pre-commit, automated testing, and GitHub Actions to maintain a reproducible machine learning workflow.

All contributors should follow the branch, commit, review, DVC, testing, and reproducibility rules below.

---

## 1. Branching Strategy

This project uses three permanent branches:

* `main` — production and release-ready code
* `staging` — release candidate and final reproducibility validation
* `dev` — integration branch for completed features, experiments, and data changes

Short-lived branches use the following naming conventions:

* `feat/<name>` — new features or functionality
* `data/<name>` — dataset or DVC changes
* `exp/<member>-<idea>` — machine learning experiments
* `fix/<name>` — bug fixes

The normal branch flow is:

```text
feature/data branch → dev → staging → main
```

Permanent branches are protected. Changes should normally enter them through pull requests.

### Experiment branches

Experiment branches must not be merged directly into `dev`, `staging`, or `main`.

Experiments should be run using DVC experiments. When a selected experiment is identified, its configuration is applied to an appropriate feature branch and then promoted through the normal PR workflow.

Abandoned experiments should remain documented in GitHub when they are required as evidence for the project.

---

## 2. Commit Message Convention

This project follows Conventional Commits:

```text
<type>: <short description>
```

Allowed commit types include:

* `feat` — new functionality
* `fix` — bug fixes
* `data` — dataset or DVC changes
* `exp` — experiment changes
* `docs` — documentation
* `test` — tests
* `ci` — CI/CD changes
* `refactor` — code restructuring
* `chore` — maintenance

Examples:

```text
feat: add Scania preprocessing pipeline
data: update train validation split
exp: compare logistic regression parameters
test: add preprocessing tests
ci: add pull request checks
docs: update contribution guidelines
```

Commit messages should describe one logical change.

---

## 3. Pull Requests and Reviews

Changes to protected branches must be submitted through pull requests.

Each pull request should clearly describe:

* What changed
* Why the change was made
* Relevant tests or experiment results
* Any limitations or known issues

Before requesting review, the author should verify:

```text
pytest
pre-commit
dvc status
dvc remote status
```

when applicable.

At least one teammate should review the pull request before merging.

Reviewers should check:

* Correctness
* Reproducibility
* Data leakage
* Fixed seeds
* Parameter configuration
* DVC synchronization
* Tests
* CI results
* Hardcoded paths
* Documentation

When a required issue is found, the reviewer should request changes rather than approving immediately.

New commits pushed after review should be reviewed again when they materially change the submitted work.

---

## 4. Merge Strategy

We use **squash merging** for pull requests into protected branches.

This keeps the protected branch history concise and represents each pull request as one logical change.

When synchronising a short-lived branch with its target branch, use a local rebase when appropriate.

Therefore:

```text
PR merge strategy: Squash
Local branch synchronization: Rebase
```

Protected branches must not be force-pushed.

---

## 5. DVC Data Workflow

Dataset files must not be committed directly to Git.

For dataset changes:

1. Add or modify the dataset.
2. Track the dataset with DVC.
3. Verify the DVC pointer.
4. Run `dvc push`.
5. Confirm the DVC remote is synchronized.
6. Commit the DVC pointer and related metadata.
7. Open a pull request into `dev`.

Example:

```powershell
uv run dvc add data/raw/<dataset>.csv
uv run dvc push
git add data/raw/<dataset>.csv.dvc
git commit -m "data: update dataset"
git push
```

### Important rule

**Always run `dvc push` before pushing the corresponding Git commit.**

A Git PR must not be considered ready for review if its required DVC artifacts are unavailable from the configured remote.

---

## 6. Reproducible Model Configuration

Model hyperparameters, data split parameters, and random seeds must be stored in `params.yaml`.

Do not hardcode experiment-specific values inside the training code.

The final pipeline must record:

* data split
* random seed
* model parameters
* preprocessing parameters
* evaluation metrics
* Git commit SHA

The evaluation stage writes the final metrics to `metrics.json`.

---

## 7. Experiment Workflow

Each team member should conduct experiments on their own `exp/<member>-<idea>` branch.

Experiments should be tracked with DVC:

```powershell
uv run dvc exp run
uv run dvc exp show
```

A selected experiment should be applied rather than manually re-entered where possible:

```powershell
uv run dvc exp apply <experiment-name>
```

The selected configuration must then be validated with the normal test and DVC workflow before being promoted.

Experiment results should be compared using clearly stated evaluation criteria rather than selecting a model without a documented basis.

---

## 8. Testing and Pre-Commit

Before opening a pull request, contributors should run:

```powershell
uv run pytest tests/
uv run pre-commit run --all-files
```

All tests and pre-commit hooks should pass before requesting review.

The project uses pre-commit hooks for:

* Ruff linting
* Ruff formatting
* notebook output stripping
* large-file detection
* secret detection

Large datasets must be managed using DVC rather than committed to Git.

---

## 9. GitHub Actions CI

All pull requests targeting:

* `dev`
* `staging`
* `main`

run the GitHub Actions CI workflow.

The CI workflow checks:

1. Ruff linting
2. Ruff formatting
3. Unit tests
4. Dataset schema and value checks
5. Smoke training on a small committed dataset sample

CML is used to report smoke-training metrics in the pull request when configured.

Required CI checks must pass before merging into protected branches.

### Deliberately failing CI test

A deliberately broken test was used during project validation to verify that branch protection actually blocks a pull request when a required CI check fails.

Therefore, contributors must not bypass a failing CI check without fixing the underlying issue.

---

## 10. Data and Test Synchronization

When project parameters are intentionally changed, related tests must be updated in the same logical change.

For example, changing the train/validation split from 80/20 to 70/30 required the parameter test to be updated from:

```python
assert params["data"]["test_size"] == 0.2
```

to:

```python
assert params["data"]["test_size"] == 0.3
```

Tests should validate the current intended configuration rather than stale historical values.

---

## 11. Branch Synchronization

Before starting a new feature, data, experiment, or fix branch:

```powershell
git switch dev
git pull origin dev
git switch -c <branch-name>
```

When a long-running branch needs the latest target-branch changes, synchronize it carefully using the team's preferred rebase workflow.

Always check:

```powershell
git status
git diff
```

before committing.

---

## 12. Handling Merge Conflicts

When a real merge conflict occurs:

1. Identify the conflicting files.
2. Understand both sides of the change.
3. Resolve the conflict intentionally.
4. Run tests.
5. Run pre-commit.
6. Run DVC validation when relevant.
7. Commit the resolved result.
8. Push the branch and request review again.

For conflicts involving `params.yaml` or `dvc.lock`, do not resolve by choosing one side blindly.

The final parameter file and DVC lock state must describe one consistent reproducible pipeline.

---

## 13. DVC Lock and Remote Issues

If DVC reports that local pipeline metadata is stale or inconsistent:

```powershell
uv run dvc status
uv run dvc status --cloud
```

If required artifacts are missing locally, use:

```powershell
uv run dvc pull
```

Do not commit an unexplained `dvc.lock` change.

If the DVC remote experiences connection or timeout problems, verify the remote configuration and retry before pushing the Git changes.

---

## 14. Release Workflow

The release flow is:

```text
dev
 ↓
release: v1.0 PR
 ↓
staging
 ↓
fresh-clone reproducibility test
 ↓
main
 ↓
model-v1.0
```

The release PR should summarize:

* included PRs
* final model configuration
* final metrics
* data version
* reproducibility information

The staging reproduction must be performed by a team member who did not train the final model.

The required commands are:

```bash
git clone <repo>
cd <repo>
git checkout staging
uv sync
dvc pull
dvc repro
```

The resulting `metrics.json` must match the reported release metrics exactly.

Only after successful reproduction should the release be promoted from `staging` to `main`.

The production release tag is:

```text
model-v1.0
```

---

## 15. Documentation Requirements

The final repository must contain a `REPORT.md` documenting:

* team members and roles
* dataset and source
* starter-code source
* reproducibility information
* experiment comparison
* selected model
* required PR links
* CI evidence
* release evidence
* retrospective
* individual contributions

Documentation changes should be committed using Conventional Commits, for example:

```text
docs: finalize project report
docs: update contribution guidelines
```

---

## 16. Lessons Incorporated From Project Issues

The following rules were added or strengthened after issues encountered during development:

### DVC artifacts must be pushed before Git changes

A model-tuning PR initially lacked required DVC artifacts. The team now explicitly verifies:

```powershell
uv run dvc status
uv run dvc status --cloud
```

and runs `dvc push` before pushing Git changes.

### Tests must track intentional parameter changes

The 70/30 split change initially caused a test failure because the test still expected the previous 80/20 value. Parameter changes must therefore be accompanied by corresponding test updates.

### CI action versions must be validated

The initial CI workflow referenced an invalid `setup-uv` action version and caused all CI jobs to fail. Action versions should be checked against the actual available release before merging CI configuration.

### Required checks must be tested, not just configured

A deliberately failing test was used to verify that branch protection actually prevents merging when a required CI check fails.

### Reproducibility must be verified from a clean environment

The final release must be reproduced by a member who did not train the model using a fresh clone, DVC pull, and DVC reproduction.

---

## 17. Contributor Checklist

Before opening a PR:

```text
[ ] Correct branch naming
[ ] Conventional Commit
[ ] No hardcoded absolute paths
[ ] Tests pass
[ ] Pre-commit passes
[ ] Parameters stored in params.yaml
[ ] Seeds configured
[ ] No data leakage
[ ] DVC status checked
[ ] DVC push completed if required
[ ] PR description completed
[ ] Relevant metrics included
[ ] Reviewer requested
```

Before merging:

```text
[ ] Required reviewer approval
[ ] Requested changes resolved
[ ] Required CI checks pass
[ ] Conversations resolved
[ ] DVC remote synchronized
[ ] No unresolved conflicts
```

---

## 18. Contact and Repository

Repository:

https://github.com/RidaBatool004/scaniapredict-ml-collab

All contributors are expected to follow this guide to maintain a reproducible, reviewable, and collaborative ML workflow.