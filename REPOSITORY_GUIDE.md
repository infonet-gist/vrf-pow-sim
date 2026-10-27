# Repository layout and generated files

Repositories use a stable research slug and role: `<research>-docs` for manuscripts, slides and research documents; `<research>-code` for implementation and tests; `<research>-sim` for a standalone simulator. Local checkout names match GitHub names.

Keep document and implementation repositories separate, under the same local research directory. A documentation repository may retain small experiment scripts used to generate paper figures; splitting those without updating reproducibility instructions is not part of housekeeping.

Temporary workspace: use `RESEARCH_TEMP/<repository>/<task>/`, or an OS temporary directory. Rendered pages, review screenshots, one-off editing scripts and intermediate builds go there. They are not research deliverables.

Track sources, dependencies, test fixtures, figure-generation scripts and selected final/submitted artifacts. Raw datasets, downloaded reading libraries and large experiment outputs belong in external storage with a recorded source/location. Existing reference collections and evidence remain available; this cleanup does not rewrite Git history.

`.gitignore` prevents future accidental staging. It does not remove previously committed files. Review `git diff --cached --stat` and `git diff --cached` before pushing, and stage explicit paths.
