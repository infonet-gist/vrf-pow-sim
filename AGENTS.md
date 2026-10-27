# Agent instructions

## Research repository hygiene

- Keep reusable source, tests, configurations, manuscript sources, cited figures and explicitly selected deliverables in Git. Do not move implementation libraries between repositories as part of cleanup.
- Create one-off edits, page renders, visual QA, downloaded references and intermediate builds outside the checkout. Use `RESEARCH_TEMP` when set; otherwise use an OS temporary directory with a repository/task subdirectory.
- Do not create new `tmp`, render, QA or version-copy directories inside this repository. Existing historical records and published/submitted files are retained unless a separate cleanup is requested.
- Keep only the artifacts needed to reproduce reported results; put large/raw outputs outside Git and document their location and generation command. Preserve reusable experiment scripts and existing evidence.
- Do not globally ignore PDF, PPTX, PNG, CSV, JSON, notebooks or logs: some are source, fixtures or selected deliverables. Use narrow generated-output paths.
- Before committing, inspect `git status` and the staged diff. Stage explicit paths; never include unrelated manuscript changes, local credentials or unpublished drafts in a housekeeping commit.
- Keep existing research-specific instructions and user edits. Use Git history for intermediate versions; preserve separately named submission/shared snapshots when required by the project.
