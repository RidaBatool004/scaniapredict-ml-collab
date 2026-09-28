# Contributing Guide

## Branching Strategy

This project uses three permanent branches:

- `main` — production/release-ready code
- `staging` — release candidate and final validation
- `dev` — integration branch for completed features and data changes

Short-lived branches follow these naming conventions:

- `feat/<name>` — new features or functionality
- `data/<name>` — dataset or DVC changes
- `exp/<member>-<idea>` — machine learning experiments
- `fix/<name>` — bug fixes

The normal branch flow is:

`feature/data branch → dev → staging → main`

Experiment branches are not merged directly into `dev`, `staging`, or `main`. When an experiment produces the selected result, its changes are applied to an appropriate feature branch and then merged through the normal workflow.

The permanent branches are protected and changes should normally be made through pull requests.

## Commit Message Convention

This project follows the Conventional Commits style:

`<type>: <short description>`

Allowed commit types include:

- `feat` — new functionality
- `fix` — bug fixes
- `data` — dataset or DVC changes
- `exp` — experiment changes
- `docs` — documentation
- `test` — tests
- `ci` — CI/CD changes
- `refactor` — code restructuring
- `chore` — maintenance

Examples:

- `feat: add Scania preprocessing pipeline`
- `data: add initial dataset with DVC`
- `exp: compare random forest and logistic regression`
- `test: add preprocessing tests`
- `docs: update contribution guidelines`
- `ci: add pull request checks`

## Squash vs Rebase Decision

We use **squash merging** when merging pull requests into protected branches. This keeps the protected branch history concise and makes each merged pull request represent one logical change.

We use **rebase locally** when updating a short-lived branch with the latest changes from its target branch.

Therefore:

- PR merge strategy: **Squash**
- Local branch synchronization: **Rebase**
- Protected branches must not be force-pushed.

## Pull Requests and Reviews

Changes to protected branches should be submitted through pull requests.

Each pull request should clearly describe:

- What was changed
- Why the change was made
- Relevant tests or experiment results
- Any issues or limitations

Teammates should review pull requests and requested changes should be addressed before merging.

## DVC Data Workflow

Dataset files are not committed directly to Git.

For dataset changes:

1. Add or modify the dataset.
2. Track the dataset with DVC.
3. Run `dvc push` before pushing the Git commit.
4. Commit the DVC pointer and related metadata.
5. Open a pull request into `dev`.

This keeps large dataset files outside the Git repository while keeping their versions reproducible.