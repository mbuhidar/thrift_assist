# Git Workflow Documentation

## Table of Contents
- [Introduction](#introduction)
- [Basic Commands](#basic-commands)
- [Branching](#branching)
- [Merging](#merging)
- [Reverting Files to Previous Commits](#reverting-files-to-previous-commits)

## Introduction

This document provides a comprehensive guide on the Git workflow used in this project. It covers basic commands, branching strategies, merging processes, and how to revert files to previous commits.

## Basic Commands

```bash
# Clone a repository
git clone <repository-url>

# Check the status of your files
git status

# Add files to staging
git add <file-name>

# Commit changes
git commit -m "Commit message"

# Push changes to remote
git push origin <branch-name>

# Pull changes from remote
git pull origin <branch-name>
```

## Branching

```bash
# Create a new branch
git branch <branch-name>

# Switch to a branch
git checkout <branch-name>

# Merge a branch into the current branch
git merge <branch-name>

# Delete a branch
git branch -d <branch-name>
```

## Merging

```bash
# Merge changes from one branch to another
git checkout <target-branch>
git merge <source-branch>

# Resolve merge conflicts
# (manual intervention required)

# Commit the merge
git commit -m "Merge branch '<source-branch>' into <target-branch>"
```

## Reverting Files to Previous Commits

### Revert Single File to Last Commit
```bash
# Revert main.py to the last committed version
git checkout HEAD -- main.py

# Alternative syntax
git restore main.py
```

### Revert File to Specific Commit
```bash
# Find the commit hash first
git log --oneline main.py

# Revert to specific commit
git checkout <commit-hash> -- main.py

# Example:
git checkout abc1234 -- main.py
```

### Revert Multiple Files
```bash
# Revert multiple files at once
git checkout HEAD -- main.py requirements.txt

# Revert entire directory
git checkout HEAD -- public/

# Revert all changes in current directory
git checkout HEAD -- .
```

### Check Changes Before Reverting
```bash
# See what changed in the file
git diff main.py

# See file history
git log --oneline -p main.py

# See current status
git status
```

### Undo Different Types of Changes

#### Unstaged Changes (not yet added)
```bash
# Revert unstaged changes
git checkout -- main.py
# or
git restore main.py
```

#### Staged Changes (added but not committed)
```bash
# Unstage the file first
git reset HEAD main.py

# Then revert the changes
git checkout -- main.py
```

#### Committed Changes
```bash
# Create new commit that undoes changes
git revert <commit-hash>

# Revert last commit for entire repository
git revert HEAD
```

### Quick Reference for File Recovery

```bash
# Revert main.py to last commit (most common)
git checkout HEAD -- main.py

# Revert and see what you're reverting
git diff main.py && git checkout HEAD -- main.py

# Revert but keep a backup first
cp main.py main.py.backup && git checkout HEAD -- main.py
```