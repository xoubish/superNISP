# Overleaf connection

Project Git URL: `https://git@git.overleaf.com/6ad2883f967dc592692b8809`

The isolated checkout lives in `.overleaf/project` at the repository root.
`.overleaf/` is ignored by the main superNISP repository. Keep the manuscript
working files in `paper/` while establishing the initial comparison.

## Publish to the blank project

The user authorized populating this blank Overleaf project with the local
manuscript. Run this in an interactive terminal from the superNISP directory:

```bash
python scripts/overleaf.py publish
```

Enter your Overleaf Git authentication token at Git's password prompt. This
command prepares a ZIP, clones the blank project, copies the manuscript,
bibliographies, class/style files and figures, commits in the isolated checkout,
and pushes to Overleaf. It supplies `main.tex` as a small entry point for
`survey_enhancements.tex`, so the blank project's default main-file setting
continues to work. The research repository is not committed or pushed.

Subsequent `publish` commands update local manuscript files on Overleaf provided
there are no new remote edits to reconcile. The helper stops if collaborators
have edited Overleaf since the previous synchronization; it never force-pushes
or deletes remote files. This is an explicit publishing command, not automatic
background synchronization or an automatic two-way merge.

The prepared upload archive is `.overleaf/superNISP-paper.zip`. Regenerate it
without accessing Overleaf using `python scripts/overleaf.py bundle`.

## Connect and compare without publishing

In an interactive terminal on this machine, from the superNISP directory:

```bash
python scripts/overleaf.py connect
```

When Git asks for a password, enter an Overleaf **Git authentication token**,
created in your Overleaf Account Settings. The username is `git`, already in
the URL. Git handles authentication; the helper does not store the token.

The helper clones the project, compares root manuscript resources and `figs/`
with `paper/`, and writes `.overleaf/comparison.json`. It does not copy files
between the two trees or push changes. Compare the existing Overleaf paper
before resolving any initial differences, including files with different names.

For a later comparison against the current local checkout:

```bash
python scripts/overleaf.py status
```

Both connection attempts reached Overleaf but required authentication. The
initial upload is authorized and prepared, but has not been pushed. Supplying
the project URL alone does not authenticate Git. Future sync must reconcile
collaborators' edits before pushing local changes.

Official instructions:
[Git authentication tokens](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/git-integration-authentication-tokens).
