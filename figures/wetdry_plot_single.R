library(tidyverse)

example_site <- 55000900029021
example_year <- 2018

# Confusion-cell colours (after talking-headwaters plot_flow_timeseries_sample):
# saturated = correct call, pale = error; blues sit on the wet (=1) rail,
# oranges on the dry (=0) rail. Each observed day is coloured by whether the
# 3-day-ahead prediction hit or missed once P(wet) is thresholded at 0.5.
cell_levels <- c(
  "Obs wet, pred wet (hit)",
  "Obs wet, pred dry (miss)",
  "Obs dry, pred wet (false wet)",
  "Obs dry, pred dry (hit)"
)
cell_cols <- c(
  "Obs wet, pred wet (hit)"       = "#1b4965", # dark blue
  "Obs wet, pred dry (miss)"      = "#9ecae1", # light blue
  "Obs dry, pred wet (false wet)" = "#f6b48a", # light orange
  "Obs dry, pred dry (hit)"       = "#c8471a"  # dark orange
)

data <- read_csv("data/huggingface/train_val_predictions_day3.csv") |>
  filter(
    !is.na(true_wetdry),
    site_id == example_site,
    date >= as.Date(paste0(example_year, "-01-01")),
    date <= as.Date(paste0(example_year, "-12-31"))
  ) |>
  select(date, site_id, true_wetdry, pred_wetdry_prob, pred_wetdry_label) |>
  arrange(date) |>
  mutate(
    obs_wet  = true_wetdry == 1,
    pred_wet = pred_wetdry_prob >= 0.5,
    cell = factor(case_when(
       obs_wet &  pred_wet ~ "Obs wet, pred wet (hit)",
       obs_wet & !pred_wet ~ "Obs wet, pred dry (miss)",
      !obs_wet &  pred_wet ~ "Obs dry, pred wet (false wet)",
      !obs_wet & !pred_wet ~ "Obs dry, pred dry (hit)"
    ), levels = cell_levels)
  )

plot <- ggplot(data, aes(x = date)) +
  geom_line(aes(y = pred_wetdry_prob), color = "grey40", linewidth = 0.4, alpha = 0.9) +
  geom_point(
    aes(y = true_wetdry, color = cell),
    size = 1.6, alpha = 0.85
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey70", linewidth = 0.3) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = c(0, 0.5, 1),
    labels = c("0\n(Dry)", "0.5", "1\n(Wet)")
  ) +
  scale_x_date(date_labels = "%b", date_breaks = "1 month") +
  scale_color_manual(
    values = cell_cols, drop = FALSE,
    name = NULL
  ) +
  guides(
    color = guide_legend(nrow = 2, override.aes = list(size = 3, alpha = 1))
  ) +
  labs(
    x = NULL,
    y = "P(wet)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "grey92"),
    axis.text = element_text(color = "grey40"),
    plot.margin = margin(12, 16, 12, 12)
  )

ggsave(
  filename = "figures/wetdry_plot_single.png",
  plot = plot,
  width = 12,
  height = 5,
  dpi = 300,
  bg = "white"
)
