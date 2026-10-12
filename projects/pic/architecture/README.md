# EPDE architecture sources

LikeC4 sources describe the current framework, benchmark layer, notebook and local Streamlit workflow. The labels are English so the diagrams can be reused in presentations.

- `specification.c4`: element types and relation styles.
- `model-epde.c4`: interface, structures, preprocessing, caches, optimizers, operators and solvers.
- `model-pic.c4`: dataset, command-line, notebook, app and external benchmark layer.
- `views.c4`: package views and workflow, epoch, candidate-fit and benchmark sequences.

From this directory, run `likec4 validate .` to check the sources or `likec4 start` to explore them. The Streamlit page “How EPDE works” embeds compact per-stage views (`stage*` in `views.c4`); clicking one opens the full model browser. The embedded script is generated from these sources and must be regenerated after editing them:

```bash
likec4 gen webcomponent -o ../app/static/likec4-views.js .
```

Validated on 8 October 2026: four source files. Framework diagrams describe optional solvers as capabilities; their presence does not imply a discovered PDE was solved in a particular benchmark record.
