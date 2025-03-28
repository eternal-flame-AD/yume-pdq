library(tidyverse)
library(scales)
library(cowplot)
library(patchwork)

streams <- tribble(
    ~machine, ~nt, ~throughput,
    "7950x-ddr5", 1, 48412.1,
    "7950x-ddr5", 2, 58996.8,
    "7950x-ddr5", 4, 69170.1,
    "7950x-ddr5", 8, 67738.8,
    "7950x-ddr5", 16, 50890.2,
    "7950x-ddr5", 32, 45506.8,
    "xeongold-ddr4", 1, 10301.3,
    "xeongold-ddr4", 2, 19377.1,
    "xeongold-ddr4", 4, 35227.8,
    "xeongold-ddr4", 8, 55905.4,
    "xeongold-ddr4", 16, 90553.0,
    "xeongold-ddr4", 32, 173363.1,
) 

df <- read_csv("bench_parallel.csv") |>
    mutate(njobs = ifelse(is.na(nt), 1, 48)) |>
    mutate(pps = njobs / time * 1e6) |>
    rowwise() |>
    mutate(throughput = eval(parse(text = throughput)))




st.ingest <- df |>
    filter(is.na(nt)) |>
    bind_rows(mutate(streams, kernel = "stream benchmark") %>% filter(nt == 1)) |>
    ggplot(aes(y = kernel, x = throughput, fill = repr)) +
    scale_x_continuous(labels = label_scientific(digits = 2)) +
    geom_col(position = "dodge") +
    facet_wrap(~machine, scales = "free") +
    labs(x = "Kernel", y = "Data Ingestion Rate (MB/s)", title = "Single-thread data ingestion rate") +
    theme_cowplot() +
    theme(legend.position = "top")
    
mt.ingest <- df |>
    filter(!is.na(nt))  |>
    ggplot(aes(x = nt, y = throughput, color = kernel)) +
    scale_x_continuous(trans = "log2") +
    geom_line() +
    geom_line(data = mutate(streams, kernel = "STREAM benchmark") %>% expand_grid(repr = c("u8", "f32")), aes(x = nt, y = throughput), linetype = "dashed", inherit.aes = FALSE, color = "black") +
    scale_y_continuous(labels = label_scientific(digits = 2)) +
    facet_wrap(~machine + repr) +
    labs(x = "Number of threads", 
        y = "Data Ingestion Rate (MB/s)", title = "Multi-thread data ingestion rate",
        caption = 
            "Dashed lines are the results from the STREAM benchmark.\n Multithreading is done by naively using the `rayon` crate with no hw locality optimizations."
        ) +
    theme_cowplot() +
    theme(legend.position = "top")

st.hash <- df |>
    filter(is.na(nt)) |>
    ggplot(aes(y = kernel, x = pps, fill = repr)) +
    scale_x_continuous(labels = label_scientific(digits = 2)) +
    geom_col(position = "dodge") +
    facet_wrap(~machine, scales = "free") +
    labs(y = "Kernel", x = "Hashes per second", title = "Single-thread hash rate") +
    theme_cowplot() +
    theme(legend.position = "top")

mt.hash <- df |>
    filter(!is.na(nt)) |>
    ggplot(aes(x = nt, y = pps, color = kernel)) +
    scale_x_continuous(trans = "log2") +
    scale_y_continuous(labels = label_scientific(digits = 2)) +
    geom_line() +
    facet_wrap(~machine + repr, scales = "free") +
    labs(x = "Number of threads", y = "Hashes per second", title = "Multi-thread hash rate",
        caption = 
            "Multithreading is done by naively using the `rayon` crate with no hw locality optimizations."
        ) +
    theme_cowplot() +
    theme(legend.position = "top")

layout <- "
AAABBDDDD
CCCBBDDDD
"

final <- wrap_plots(
    A = st.ingest,
    B = mt.ingest,
    C = st.hash,
    D = mt.hash,
    design = layout
) + plot_annotation(title = "Yume-PDQ Parallel Benchmarks")

ggsave("bench_parallel.png", final, width = 25, height = 10, bg = "white")
