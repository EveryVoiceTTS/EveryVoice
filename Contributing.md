# Contributing to EveryVoice TTS Toolkit 💬

Feel free to dive in! [Open an issue](https://github.com/roedoejet/EveryVoice/issues/new) or submit PRs.

Issues regarding the submodules should also be submitted on the main
[EveryVoice repo](https://github.com/roedoejet/EveryVoice), with the tag
`[FastSpeech2]`, `[HiFiGAN]`, `[DeepForceAligner]`, or `[wav2vec2aligner]`
at the beginning of the issue title, as appropriate.

To submit PRs for the submodules, please submit a PR with the code changes to the
submodule repo, accompanied with a PR to this repo to update the submodules reference.

This repo follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

This repo uses automated tools to standardize the formatting of code, text files and
commits.
 - [Pre-commit hooks](#pre-commit-hooks) validate and automatically apply code
   formatting rules.
 - [gitlint](#gitlint) is used as a commit message hook to validate that
   commit messages follow the convention.

Submitting PRs in a repo with four submodules can be complicated.
See [Pull Request Recommendations](#pull-request-recommendations) for hints on making this easier.

## The TL;DR for pre-commit and gitlint hooks

Run these commands in each of your sandboxes to enable our pre-commit hooks and gitlint:

```sh
pip install -r requirements.dev.txt
pre-commit install
gitlint install-hook
git submodule foreach 'pre-commit install'
git submodule foreach 'gitlint install-hook'
```

## Pre-commit hooks

The ReadAlong Studio team has agreed to systematically use a number of pre-commit hooks to
normalize formatting of code. You need to install and enable pre-commit to have these used
automatically when you do your own commits.

Pre-commit hooks enabled:
- check-yaml validates YAML files
- end-of-file-fixer makes sure each text file ends with exactly one newline character
- trailing-whitespace removes superfluous whitespace at the end of lines in text files
- [Flake8](https://flake8.pycqa.org/) enforces good Python style rules; more info about
  using Flake8 in pre-commit hooks at:
  [Lj Miranda flake8 blog post](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
- [isort](https://pycqa.github.io/isort/) orders python imports in a standard way
- [Black](https://github.com/psf/black), the Uncompromising Code Formatter, reformats all
  Python code according to very strict rules we've agreed to follow; more info about Black
  formatting rules in
  [The Black code style](https://black.readthedocs.io/en/stable/the_black_code_style.html)
- [mypy](http://mypy-lang.org/) runs type checking for any statically-typed Python code in
  the repo

### Enabling pre-commit hooks

All the pre-commit hooks are executed using a tool called
[pre-commit](https://pre-commit.com/). Once you enable pre-commit, it will run all the
hooks each time you try to commit anything in this repo.

We've listed all the developper dependencies for the project in
[requirements.dev.txt](requirements.dev.txt) to make them easy to install:

```sh
pip install -r requirements.dev.txt
pre-commit install
```

Note that you have to run the second command in every sandbox you create, so please
don't forget to do so when you clone a new sandbox!

## gitlint

The team has also agreed to use [Conventional Commits](https://www.conventionalcommits.org/).
Install and enable [gitlint](https://jorisroovers.com/gitlint/) to have your
commit messages scanned automatically.

Convential commits look like this:

    type(optional-scope): subject (i.e., short description)

    optional body, which is free form

    optional footer

Valid types: (these are the default, which we're using as is for now)
 - build: commits for the build system
 - chore: maintain the repo, not the code itself
 - ci: commits for the continuous integration system
 - docs: adding and changing documentation
 - feat: adding a new feature
 - fix: fixing something
 - perf: improving performance
 - refactor: refactor code
 - revert: undo a previous change
 - style: working only on code or documentation style
 - test: commits for testing code

Types for partial work: you can use `pfeat`, `pfix`, `pdoc`, `ptest` or `prefactor` as the
commit type for partial work that won't be shown in the release logs. Make sure you have
the appropriate regular type commit later in the history documenting the work for the
release log, and consider squashing the partial work commits when that makes sense.

Valid scopes: the scope is optional and usually refers to which module is being changed.
 - TBD - for now not validated, should be just one word ideally

Valid subject: short, free form, what the commit is about in less than 50 or 60 characters
(not strictly enforced, but it's best to keep it short)

Optional body: this is where you put all the verbose details you want about the commit, or
nothing at all if the subject already says it all. Must be separated by a blank line from
the subject. Explain what the changes are, why you're doing them, etc, as necessary.

Optional footer: separated from the body (or subject if body is empty) by a blank line,
lists reference (e.g.: "Closes #12" "Ref #24") or warns of breaking changes (e.g.,
"BREAKING CHANGE: explanation").

These rules are inspired by these commit formatting guides:
 - [Conventional Commits](https://www.conventionalcommits.org/)
 - [Bluejava commit guide](https://github.com/bluejava/git-commit-guide)
 - [develar's commit message format](https://gist.github.com/develar/273e2eb938792cf5f86451fbac2bcd51)
 - [AngularJS Git Commit Message Conventions](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y).

### Enabling the commit linter

You can run gitlint on each commit message that you write by enabling the
commit-msg hook in Git.

Run this command in your sandbox to install and enable the commit-msg hook:

```sh
pip install -r requirements/requirements.dev.txt
gitlint install-hook
```

- Now, next time you make a change and commit it, your commit log will be checked:
  - `git commit -m'non-compliant commit log text'` outputs an error
  - `git commit -m'fix(g2p): fixing a bug in g2p integration'` works

### Initializing submodules too

The EveryVoice repo uses submodules, and the gitlint and pre-commit
initialization has to be done separately in each of one them. You can cd into
each submodule directory and run the same commands shown above, but there is a
shortcut:

```sh
git submodule foreach 'pre-commit install'
git submodule foreach 'gitlint install-hook'
```

## Pull request recommendations

### TL;DR
 - Always separate submodule update commits from other top-level commits.
 - Expect to have to rebase submodules and re-create submodule update commits.
 - For approved submodule PRs, doing fast-forward merges, from the CLI only, saves work.
 - Pay extra attention to where each submodule's HEAD is before creating submodule update commits.

### Intro

This repo has four submodules, which will get checked out under subdirectories of `everyvoice/model/` when you run the recommended `git submodule update --init` initialization command. This means that a specific state of the EveryVoice code is not just represented by a commit sha1 in the top-level repo, but also the four submodule sha1's it points to.

Every time you check out a new branch/commit/tag/PR, you should also checkout the submodules at the right commit by running `git submodule update` again.  (Running `git status` after a checkout will tell you if and which submodules need updating.)

### Creating a multi-repo PR

To create a PR for changes that affect one or more submodules, create a PR in each affected submodule, and one PR in the top-level repo to update the submodule pointers and apply any relevant code changes in the top-level repo.

Important: in the top-level repo, separate commits that update the submodules from other commits. We typically give submodule update commits a message like `chore: update submodules`.  Never combine these with other changes in the top-level repo because you will inevitably have to redo the submodule update commits when the submodule PRs are rebased, merged, improved following reviewer comments, etc.

If changes between submodules and the top-level repo are tightly tied, you can use a commit history like "refactor: rename foo->bar" for the top-level changes, and "chore: udpate submodule to rename foo->bar" for the submodule updates, but don't be tempted to merge these in one commit: the former will be easy to rebase or merge as needed, but the latter will cause conflicts and will likely need to be recreated before you're done.

### Merging a multi-repo PR

Hints for maintainers and contributors with merge privileges.

Once your PR touching submodules has been approved in all the repos, you have to merge it in the submodules first and then in the top-level repo. But you can make your life easier by performing fast-forward merges on all the submodule PRs.

**Note: Doing a fast-forward merge is not possible from the GitHub UI**: "Rebase and merge" will replay the history or a PR, creating a new sha1 for each commit in the PR. "Squash and merge" will replace all the PR commits by one new commit. "Create a merge commit" is self-explanatory, but will result in a new commit.

In all cases, like each time the submodule sha1's are changed, you have to go back to the top-level PR and recreate the submodule udpate commits to point to the newly created commits.

But you can fast-forward merge your PRs from the CLI:
```
git checkout main
git merge --ff-only dev.my-approved-pr-branch
git push origin main
```

Hint: these three steps can be replaced by a single command:
```
git push origin dev.my-approved-pr-branch:main
```
this syntax means push the local reference/commit/sha1 on the left-hand-side of the colon to the remote branch named on the right-hand-side of the colon.

### Take care not to revert submodule updates

Because an EveryVoice state is really represented by five separate sha1's (one in the top-level repo and one in each submodule), it's very easy to get confused about the right place the submodules should point.

Before commit submodule updates, inspect the history in each submodules to make sure you're only commiting the changes you expect, and did not, e.g., forget to update one submodule after rebasing your top-level PR.

This command can be quite helpful in doing this inspection:
```
git submodule foreach "git log --oneline --graph --decorate HEAD origin/main | head"
```
carefully look at the output for each submodule, making sure you agree with where HEAD points.
