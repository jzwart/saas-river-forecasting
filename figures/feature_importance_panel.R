# Combined feature-importance figure across models.
#
# Merges the four per-model feature-importance plots from the manuscript
# (Figs. 3-6: Logistic Regression, XGBoost, LSTM HOBO-only, LSTM all-sites)
# into a single multi-panel figure for easy cross-model comparison.
#
# ============================================================================
# IMPORTANT: the importance values in this script are MANUALLY TRANSCRIBED from
# the notebooks -- they are hard-coded here, NOT read from any data file. They
# were copied verbatim from the cell output of each model's notebook:
#     lr/lr.ipynb, xgb/xgb.ipynb,
#     lstm/lstm_hobo_sites.ipynb, lstm/lstm_all_sites.ipynb
#
# Because they are a static snapshot, they will silently go STALE if any of
# those notebooks or the underlying models are re-run (e.g. after retraining,
# a new data pull, a changed random seed, or a feature-set edit). If you re-run
# any notebook, you MUST re-copy its top-10 feature importances into the
# `importance` tribble below (and update the display names / metric labels if
# the feature set changed) so this figure stays in sync with the models.
# ============================================================================
#
# Following those notebooks, importance is the absolute value of the raw metric,
# max-scaled to 0-1, showing the top 10 features. Underlying metric differs by
# model (see panel captions) but all are 0-1 max-scaled, so bar lengths are
# comparable across panels.

library(tidyverse)

# --- Feature -> category color scheme ---------------------------------------
# Colorblind-safe categorical palette. This 5-hue set clears every hard gate of
# the data-viz validator on the all-pairs test against a white surface (worst
# CVD ΔE 13.0, target >= 8; worst normal-vision ΔE 16.3, hard floor >= 15), so
# any two categories remain distinguishable under protan/deutan CVD regardless
# of which bars land adjacent. Hues echo the notebooks' original scheme for
# continuity (green/blue/violet). The two lower-contrast hues (yellow, magenta)
# are assigned to Degrees and Lagged Target, whose bars are the longest in the
# panels where they appear; every bar is also directly labeled (relief rule).
category_colors <- c(
  "Lagged Target" = "#e87ba4", # magenta (was red)
  "Degrees"       = "#eda100", # yellow  (was orange)
  "Drivers"       = "#2a78d6", # blue
  "Static"        = "#008300", # green
  "Other Obs"     = "#4a3aa7"  # violet  (was purple)
)

# --- Raw importances (top 10 per model, as printed in the notebooks) --------
# `raw` is the model's native metric; sign is dropped via abs() below.
# NOTE: these are hand-copied from the notebook outputs. Re-run a notebook or
# retrain a model -> update the corresponding rows here (see header note).
importance <- tribble(
  ~model, ~feature, ~raw, ~category,
  # (a) Logistic Regression -- |standardized coefficient|
  "LR", "out_degree",     4.140090, "Degrees",
  "LR", "lagged_target",  3.454425, "Lagged Target",
  "LR", "Slope",         -2.998770, "Static",
  "LR", "aspect_se_pct",  2.593444, "Static",
  "LR", "aspect_ne_pct", -2.449138, "Static",
  "LR", "sph",            2.433181, "Drivers",
  "LR", "rhmin",          2.068056, "Drivers",
  "LR", "rhmax",         -1.743166, "Drivers",
  "LR", "elev_min_cm",   -1.539184, "Static",
  "LR", "curv_mean",      1.477246, "Static",
  # (b) XGBoost -- gain-based importance
  "XGB", "lagged_target", 0.816903, "Lagged Target",
  "XGB", "out_degree",    0.040024, "Degrees",
  "XGB", "elev_min_cm",   0.036470, "Static",
  "XGB", "aspect_ne_pct", 0.010387, "Static",
  "XGB", "vp",            0.008559, "Drivers",
  "XGB", "slp_mean_pct",  0.008066, "Static",
  "XGB", "rhmax",         0.007656, "Drivers",
  "XGB", "rhmin",         0.007633, "Drivers",
  "XGB", "aspect_se_pct", 0.007348, "Static",
  "XGB", "LengthKM",      0.005559, "Static",
  # (c) LSTM (HOBO only) -- permutation dF1
  "LSTM_hobo", "lagged_target",  0.044626, "Lagged Target",
  "LSTM_hobo", "slp_median_pct", 0.016543, "Static",
  "LSTM_hobo", "aspect_sw_pct",  0.016543, "Static",
  "LSTM_hobo", "elev_mean_cm",   0.014636, "Static",
  "LSTM_hobo", "out_degree",     0.014076, "Degrees",
  "LSTM_hobo", "curv_median",    0.013083, "Static",
  "LSTM_hobo", "sph",            0.009862, "Drivers",
  "LSTM_hobo", "Slope",          0.009569, "Static",
  "LSTM_hobo", "curv_mean",      0.008205, "Static",
  "LSTM_hobo", "srad",           0.008205, "Drivers",
  # (d) LSTM (all sites) -- permutation dF1
  "LSTM_all", "lagged_target",   0.046933, "Lagged Target",
  "LSTM_all", "srad",            0.013011, "Drivers",
  "LSTM_all", "etgrass",         0.006464, "Drivers",
  "LSTM_all", "etalfalfa",       0.005381, "Drivers",
  "LSTM_all", "vp",              0.005381, "Drivers",
  "LSTM_all", "rhmax",           0.004300, "Drivers",
  "LSTM_all", "sph",             0.004300, "Drivers",
  "LSTM_all", "aspect_ne_pct",   0.003222, "Static",
  "LSTM_all", "prcp",            0.003222, "Drivers",
  "LSTM_all", "MaxDepth_Censor", 0.002146, "Other Obs"
)

# Short, descriptive display names for the raw feature keys (definitions from
# the manuscript figure captions). Kept terse so they fit inside the panels.
feature_labels <- c(
  lagged_target   = "Lagged wet/dry (t-3)",
  out_degree      = "Downstream connections",
  Slope           = "Reach slope (%)",
  slp_mean_pct    = "Mean slope (%)",
  slp_median_pct  = "Median slope (%)",
  aspect_ne_pct   = "NE aspect",
  aspect_se_pct   = "SE aspect",
  aspect_sw_pct   = "SW aspect",
  elev_min_cm     = "Min. elevation",
  elev_mean_cm    = "Mean elevation",
  curv_mean       = "Mean curvature",
  curv_median     = "Median curvature",
  LengthKM        = "Reach length (km)",
  sph             = "Specific humidity",
  rhmin           = "Min. rel. humidity",
  rhmax           = "Max. rel. humidity",
  vp              = "Vapor press. deficit",
  srad            = "Solar radiation",
  prcp            = "Precipitation",
  etgrass         = "Ref. ET (grass)",
  etalfalfa       = "Ref. ET (alfalfa)",
  MaxDepth_Censor = "Detectable depth (0/1)"
)

# Panel labels and their display order.
model_labels <- c(
  LR        = "(a) Logistic Regression\n| standardized coefficient |",
  XGB       = "(b) XGBoost\ngain-based importance",
  LSTM_hobo = "(c) LSTM (HOBO only)\npermutation ΔF1",
  LSTM_all  = "(d) LSTM (all sites)\npermutation ΔF1"
)

# Reproduce the notebooks' transform: abs value, max-scale to 0-1 per model.
plot_df <- importance |>
  group_by(model) |>
  mutate(scaled = abs(raw) / max(abs(raw))) |>
  ungroup() |>
  mutate(
    feature  = unname(feature_labels[feature]),
    model    = factor(model, levels = names(model_labels), labels = model_labels),
    category = factor(category, levels = names(category_colors))
  )

# Per-panel ordering (same feature can rank differently across models), so we
# order within each facet and strip the disambiguating suffix from labels.
reorder_within <- function(x, by, within, sep = "___") {
  stats::reorder(paste(x, within, sep = sep), by)
}
scale_y_reordered <- function(..., sep = "___") {
  scale_y_discrete(labels = function(x) gsub(paste0(sep, ".+$"), "", x), ...)
}

plot <- ggplot(plot_df, aes(
  x = scaled,
  y = reorder_within(feature, scaled, model),
  fill = category
)) +
  geom_col(width = 0.72) +
  facet_wrap(~model, scales = "free_y", ncol = 2) +
  scale_y_reordered() +
  scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1), expand = expansion(mult = c(0, 0.02))) +
  scale_fill_manual(values = category_colors, name = "Feature type", drop = FALSE) +
  labs(
    x = "Feature importance (0-1 max-scaled)",
    y = NULL
  ) +
  guides(fill = guide_legend(nrow = 1)) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    strip.text = element_text(face = "bold", size = 10, hjust = 0, lineheight = 1.05),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "grey92"),
    panel.spacing.x = unit(1.4, "lines"),
    panel.spacing.y = unit(1.0, "lines"),
    axis.text.y = element_text(size = 9),
    axis.text = element_text(color = "grey30"),
    plot.margin = margin(12, 16, 12, 12)
  )

ggsave(
  filename = "figures/feature_importance_panel.png",
  plot = plot,
  width = 11,
  height = 7.5,
  dpi = 300,
  bg = "white"
)
