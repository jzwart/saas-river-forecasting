library(tidyverse)
library(knitr)

network <- read_csv(
  "data/huggingface/nhd_id_stream_order_permanence.csv",
  show_col_types = FALSE
) |>
  mutate(
    n_water_presence = ifelse(n_water_presence < 10, 0, n_water_presence),
    has_data = coalesce(n_discharge, 0) > 0 | coalesce(n_water_presence, 0) > 0
  )

stream_order_table <- network |>
  mutate(
    stream_order_group = ifelse(StreamOrde >= 4, "4+", as.character(StreamOrde))
  ) |>
  group_by(stream_order_group) |>
  summarise(
    n_segments = n(),
    n_segments_with_data = sum(has_data),
    n_discharge_obs = sum(n_discharge, na.rm = TRUE),
    n_water_presence_obs = sum(n_water_presence, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    pct_segments_with_data = round(n_segments_with_data / n_segments * 100, 1),
    total_obs = n_discharge_obs + n_water_presence_obs
  ) |>
  select(
    `Stream Order` = stream_order_group,
    `Total Segments` = n_segments,
    `Segments with Data` = n_segments_with_data,
    `% with Data` = pct_segments_with_data,
    `Discharge Obs` = n_discharge_obs,
    `Water Presence Obs` = n_water_presence_obs,
    `Total Obs` = total_obs
  )

kable(stream_order_table, format = "html")
