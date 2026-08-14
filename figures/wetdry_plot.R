library(tidyverse)

data <- read_csv("data/huggingface/train_val_predictions_day3.csv") |>
  filter(!is.na(true_wetdry)) |>
  select(date, site_id, true_wetdry, pred_wetdry_prob, pred_wetdry_label) |>
  mutate(
    observed = factor(true_wetdry, levels = c(0, 1), labels = c("Dry", "Wet"))
  )

plot <- ggplot(data, aes(x = date)) +
  geom_line(aes(y = pred_wetdry_prob), color = "grey40", linewidth = 0.3, alpha = 0.9) +
  geom_point(
    aes(y = true_wetdry, color = observed),
    size = 0.6, alpha = 0.7
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey70", linewidth = 0.3) +
  facet_wrap(~site_id, ncol = 1, strip.position = "right") +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = c(0, 1),
    labels = c("Dry", "Wet")
  ) +
  scale_x_date(date_labels = "%Y", date_breaks = "5 years") +
  scale_color_manual(
    values = c(Dry = "#e07a5f", Wet = "#1b4965"),
    name = "Observed status"
  ) +
  labs(
    title = "Predicted probability of stream water presence",
    subtitle = "3-day-ahead RGCN forecast (line) vs. observed wet/dry status (points), by site",
    x = NULL,
    y = "P(wet)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(color = "grey35", margin = margin(b = 10)),
    strip.text.y.right = element_text(angle = 0, hjust = 0, face = "bold", size = 9),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "grey92"),
    panel.spacing = unit(0.6, "lines"),
    axis.text = element_text(color = "grey40"),
    plot.margin = margin(12, 16, 12, 12)
  )

ggsave(
  filename = "figures/wetdry_plot.png",
  plot = plot,
  width = 12,
  height = 18,
  dpi = 300,
  bg = "white"
)
