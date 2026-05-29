library(tidyverse)

network <- read_csv("data/huggingface/nhd_id_stream_order_permanence.csv")

stream_order_summary <- network |>
  group_by(StreamOrde) |>
  summarise(
    n_discharge = sum(n_discharge, na.rm = TRUE),
    n_water_presence = sum(n_water_presence, na.rm = TRUE),
    n_sites = n(),
    n_sites_with_data = sum(has_data),
    .groups = "drop"
  ) |>
  mutate(stream_type = ifelse(StreamOrde > 2, "tailwater", "headwater")) |>
  group_by(stream_type) |>
  summarise(
    n_discharge = sum(n_discharge, na.rm = TRUE),
    n_water_presence = sum(n_water_presence, na.rm = TRUE),
    n_sites = sum(n_sites),
    n_sites_with_data = sum(n_sites_with_data),
    pct_sites_with_data = n_sites_with_data / n_sites * 100,
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = n_discharge:pct_sites_with_data,
    names_to = "variable",
    values_to = "value"
  ) |>
  mutate(
    variable = factor(
      variable,
      levels = c(
        "n_discharge",
        "n_water_presence",
        "n_sites",
        "pct_sites_with_data",
        "n_sites_with_data"
      )
    )
  )

plot_out <- ggplot(filter(
  stream_order_summary,
  !variable %in% c("n_sites_with_data", "pct_sites_with_data")
)) +
  geom_bar(
    aes(x = variable, y = value, fill = stream_type),
    stat = "identity",
    position = "fill"
  ) +
  scale_x_discrete(
    labels = c(
      "n_discharge" = "Discharge\n Observations",
      "n_water_presence" = "Water Presence\n Observations",
      "n_sites" = "Stream\nSegments"
    )
  ) +
  scale_fill_manual(
    values = c("headwater" = "#708cb7ff", "tailwater" = "#140694ff"),
    labels = c("headwater" = "Headwater", "tailwater" = "Tailwater")
  ) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = NULL, y = "Fraction", fill = "Stream Type") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    axis.text = element_text(size = 12)
  )

ggsave(
  "figures/data_distribution_by_stream_type.png",
  plot = plot_out,
  width = 5,
  height = 5,
  units = "in",
  dpi = 300
)
