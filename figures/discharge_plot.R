library(tidyverse)

data <- read_csv("data/huggingface/train_val_predictions_day3.csv") |>
  filter(!is.na(true_discharge_cms), date > as.Date("2012-01-01"), date < as.Date("2014-01-01")) |>
  select(date, site_id, true_discharge_cms, pred_discharge_cms) |>
  mutate(pred_discharge_cms = ifelse(pred_discharge_cms < 0, 0.0001, pred_discharge_cms)) |>
  pivot_longer(
    cols = c(true_discharge_cms, pred_discharge_cms),
    names_to = "series",
    values_to = "discharge_cms"
  ) |>
  mutate(
    series = recode(series,
      true_discharge_cms = "Observed",
      pred_discharge_cms = "Predicted"
    )
  )

plot <- ggplot(data, aes(x = date, y = discharge_cms, color = series)) +
  geom_line(linewidth = 0.4, alpha = 0.9) +
  facet_wrap(~site_id, scales = "free_y", ncol = 1, strip.position = "right") +
  scale_y_log10(labels = scales::label_number()) +
  scale_x_date(date_labels = "%b %Y", date_breaks = "1 month") +
  scale_color_manual(
    values = c(Observed = "#1b4965", Predicted = "#e07a5f"),
    name = NULL
  ) +
  labs(
    title = "Observed vs. predicted daily discharge",
    subtitle = "3-day-ahead RGCN forecast by site",
    x = NULL,
    y = expression("Discharge (m"^3 * " s"^-1 * ", log scale)")
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    legend.key.width = unit(1.5, "lines"),
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
  filename = "figures/discharge_plot.png",
  plot = plot,
  width = 12,
  height = 12,
  dpi = 300,
  bg = "white"
)
