"""RGCN retrain pipeline (rgcn-retrain branch).

Runnable, path-parameterized replacements for the two cluster notebooks
(build_graph.ipynb, train_gnn.ipynb), fixing the three released defects:
  1. meteorological drivers were never attached to the graph (all-zero inputs);
  2. static watershed features were attached but never fed to the model;
  3. normalization leaked validation data (per-node stats over 1980-2020).

See RGCN_RETRAIN_PLAN.md for the full rationale.
"""
