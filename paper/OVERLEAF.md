# Overleaf connection

## Latest manuscript sync — 2026-10-17

Published all nine approved figures and the revised manuscript to the existing
project, commit `d7dcaecb4a06d5d6a3a98b4441bd5632f8833be4`. `main.tex` and
`survey_enhancements.tex` match the local manuscript; all 49 publication files
were hash-checked. Preserved a newer Overleaf edit that only joined the title's
acknowledgement onto one line. Remote commit verified after pushing.
Evidence: `paper/diagnostics/overleaf_figures_1_9_v01.json`.
The Overleaf compiled PDF has not been inspected; the local A&A build still
requires the missing `etex.sty` package.

Project Git URL: `https://git@git.overleaf.com/6ad2883f967dc592692b8809`

The isolated checkout lives in `.overleaf/project` at the repository root.
`.overleaf/` is ignored by the main superNISP repository. Keep the manuscript
working files in `paper/` while establishing the initial comparison.

## Publish manuscript changes

The user authorized populating this blank Overleaf project with the local
manuscript. Run this in an interactive terminal from the superNISP directory:

```bash
python scripts/overleaf.py publish --authenticate
```

Enter your Overleaf Git authentication token at the helper's hidden terminal
prompt. The token is used for fetch/push during that invocation and is not
written into a file, the remote URL or command arguments. This explicit mode
also handles servers that reject a push without prompting for credentials. This
command prepares a ZIP, clones the blank project, copies the manuscript,
bibliographies, class/style files and figures, commits in the isolated checkout,
and pushes to Overleaf. It copies the full contents of the local
`paper/survey_enhancements.tex` into Overleaf's `main.tex`, so the project's
default main-file setting continues to work. The original filename is retained
as an identical copy; each publish refreshes both from the same local source.
Continue making local manuscript edits in `paper/survey_enhancements.tex`.
The research repository is not committed or pushed.

Subsequent `publish` commands update local manuscript files on Overleaf provided
there are no new remote edits to reconcile. The helper stops if collaborators
have edited Overleaf since the previous synchronization; it never force-pushes
or deletes remote files. This is an explicit publishing command, not automatic
background synchronization or an automatic two-way merge.

The prepared upload archive is `.overleaf/superNISP-paper.zip`. Regenerate it
without accessing Overleaf using `python scripts/overleaf.py bundle`.

## Save the token for future publishes

Run once in your terminal and enter the token at the hidden prompt:

```bash
python scripts/overleaf.py publish --save-token
```

After a successful publish, this saves the token using Git's credential-store
in `.overleaf/credentials`. The file is unencrypted, readable/writable only by
your account (mode 600), and excluded from Git and manuscript exports. Anyone
with access to your account or privileged filesystem access can read it.
No working desktop keyring was available on this machine.

Subsequently, use:

```bash
python scripts/overleaf.py publish
```

The helper loads the saved token automatically. `--authenticate` overrides it
for one invocation; `--save-token` replaces it after a successful publish.
Delete `.overleaf/credentials` to stop storing it locally. Revoke it in Overleaf
Account Settings if you also want to invalidate the token. This setup applies
to this helper; a plain push in the research repository does not publish the paper.

## Connect and compare without publishing

In an interactive terminal on this machine, from the superNISP directory:

```bash
python scripts/overleaf.py connect
```

When Git asks for a password, enter an Overleaf **Git authentication token**,
created in your Overleaf Account Settings. The username is `git`, already in
the URL. The helper does not persist the token. `connect --authenticate` can
also be used to enter it explicitly once for the initial clone.

The helper clones the project, compares root manuscript resources and `figs/`
with `paper/`, and writes `.overleaf/comparison.json`. It does not copy files
between the two trees or push changes. Compare the existing Overleaf paper
before resolving any initial differences, including files with different names.

For a later comparison against the current local checkout:

```bash
python scripts/overleaf.py status
```

The initial upload succeeded using `publish --authenticate`. Future sync must
reconcile collaborators' edits before pushing local changes.

Official instructions:
[Git authentication tokens](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/git-integration-authentication-tokens).
