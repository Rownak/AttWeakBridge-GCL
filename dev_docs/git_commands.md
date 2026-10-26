# Git Commands for AttWeakBridge-GCL

Branch and tag conventions for this repository:
- `master` is the stable branch.
- Each version is developed on its own branch named `vX.Y_<topic>`, for example `v0.3_yaml_based_pipelines` and `v0.4_sweep_experiments_hyperparam`.
- Each version is tagged `vX.Y` on `master` after it is merged.
- Commit messages start with the sprint number, for example `[Sprint-2.3] ...`.

Replace the `<...>` placeholders with real values before running a command.

# 1. Start the next version branch

Create the branch from the latest `master` (or from the current version branch if it is not merged yet):

```bash
git checkout -b v0.5_<topic>
```

The previous version branch (for example `v0.4_sweep_experiments_hyperparam`) already keeps its history, so no separate archive branch is needed.

# 2. Push branches

```bash
git push -u origin v0.5_<topic>
```

# 3. Finish a version: merge into master and tag it

Run this once the version branch is complete:

```bash
git checkout master
git merge v0.4_sweep_experiments_hyperparam
git tag -a v0.4 -m "v0.4: sampler sweep infrastructure, sweep comparison plots, alpha sweep tooling"
git push origin master
git push origin v0.4
```

Only create the tag after the merge, so `vX.Y` always points to the merged state on `master`.

# 4. Split recent commits to a new branch

Use this when commits that belong to a new version were made on the current branch.

## Step 1: Create the new branch at the current commit

```bash
git checkout -b <new-branch-name>
```

Now both the old branch and the new branch point to the latest commit.

## Step 2: Find the commit to split from

List recent commits and pick the last commit that should stay on the old branch:

```bash
git log --oneline -n 10
```

## Step 3: Switch back to the original branch

```bash
git checkout <your-original-branch-name>
```

## Step 4: Reset the original branch backward

Move the original branch back to the commit found in Step 2:

```bash
git reset --hard <commit-hash-to-keep>
```

Commits after that hash are still on `<new-branch-name>`, so they are not lost.

## Step 5: Push the rewritten branch with force

A normal push fails here:

```
error: failed to push some refs to
hint: Updates were rejected because the tip of your current branch is behind
```

This happens because `git reset --hard` rewrote the history of the original branch, so the local copy is now behind the copy on GitHub (`origin`).

```bash
git push origin <your-original-branch-name> --force-with-lease
```

- `--force` tells GitHub to accept the rewritten history even though it does not match the remote.
- `--force-with-lease` is the safer form. It checks whether anyone else has pushed to that branch since your last fetch, and it refuses the push if they have, so their work is not overwritten.

Use the force push only on your own branches. Do not rewrite `master`.
