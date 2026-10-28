# Fresh server inputs implementation plan

**Goal:** v1.8 uses server-side fresh preparation for all scopes, with no migration or SCP requirement; last1 remains the only current stage.

**Architecture:** Remove migration entrypoints. Own an initially empty INPUT_ROOT with a version/budget preparation marker and root lock; resume only that preparation, verify registered bundles without rebuilding. Bind generated metadata to the fresh-input policy and reject older inputs at training/preflight.

**Tasks:**
- Update authoritative design first and synchronize the repository snapshot, retaining dated history.
- Modify prepare.py, data.py, protocol.py and input_bundle.py; remove migrate_inputs.py and its ACP route. Keep training, noise and LR budgets unchanged.
- Update current scripts and operator docs to inputs-v1.8/results/nanogpt-v1.8 with only inputs → smoke last1 → formal last1.
- Replace migration tests with empty-root, preparation restart, immutable registration, rejection of old/incompatible inputs and blocked concurrent preparation tests; retain optimizer/resume regressions.
- Run CPU suite, Bash syntax and design-sync checks; commit, push main and check CI. Do not generate the real corpus or start GPU training locally.
