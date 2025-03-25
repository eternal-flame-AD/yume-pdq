library(tidyverse)
library(cowplot)
library(scales)
data <- read_csv("bench.csv") %>%
    mutate(class = case_when(
        class == "hash" ~ "Overall",
        class == "hash_4threads" ~ "Overall (4 threads)",
        class == "hash_8threads" ~ "Overall (8 threads)",
        class == "dct2d" ~ "DCT transform",
        class == "jarosz_compress" ~ "Compress",
        class == "quantize" ~ "Quantize",
        TRUE ~ class
    )) %>%
    mutate(case = case_when(
        case == "reference" ~ "Reference (Precise)",
        case == "scalar" ~ "Scalar (auto-vectorized)",
        case == "avx2" ~ "AVX2 (f32x8)",
        case == "avx512" ~ "AVX512 (f32x16)",
        TRUE ~ case
    ))

data.overall <- data %>%
    filter(str_detect(class, "Overall")) %>%
    filter(!str_detect(class, "flip"))

data.sub <- data %>%
    filter(!str_detect(class, "Overall")) %>%
    filter(!str_detect(class, "flip"))

data.flip <- data %>%
    filter(str_detect(class, "flip"))

data.sub %>%
    mutate(class = factor(class), case = factor(case)) %>%
    mutate(class = fct_relevel(class, "Compress", "DCT transform", "Quantize")) %>%
    mutate(case = fct_relevel(case, "Reference (Precise)", "Scalar (auto-vectorized)", "AVX2 (f32x8)", "AVX512 (f32x16)")) %>%
    filter(class != "sum_of_gradients") %>%
    ggplot(aes(x = case, y = thrptm, fill = case, ymin = thrptl, ymax = thrpth)) +
    geom_bar(stat = "identity") +
    geom_errorbar(width = 0.2) +
    scale_y_continuous(expand = expansion(mult = 0.03, add = 0.02), limits = c(0, NA), labels = label_bytes(units = "auto_binary")) +
    facet_wrap(~class, scales = "free_y") +
    theme_minimal() +
    theme(axis.text.x = element_blank()) + 
    labs(
        x = "Case",
        y = "Throughput (per second)",
        caption = "AMD Ryzen 9 7950x, 2x32G RAM@6000MT/s",
        title = "Sub-operations"
    )

ggsave("sub.png", width = 10, height = 6)
ggsave("sub.jpg", width = 10, height = 6)

data.overall %>%
    mutate(class = factor(class), case = factor(case)) %>%
    mutate(class = fct_relevel(class, "Overall", "Overall (4 threads)", "Overall (8 threads)")) %>%
    mutate(case = fct_relevel(case, "Reference (Precise)", "Scalar (auto-vectorized)", "AVX2 (f32x8)", "AVX512 (f32x16)")) %>%
    filter(class != "sum_of_gradients") %>%
    ggplot(aes(x = case, y = thrptm, fill = case, ymin = thrptl, ymax = thrpth)) +
    geom_bar(stat = "identity") +
    geom_errorbar(width = 0.2) +
    scale_y_continuous(expand = expansion(mult = 0.03, add = 0.02), limits = c(0, NA), labels = label_bytes(units = "auto_binary")) +
    facet_wrap(~class) +
    theme_minimal() +
    theme(axis.text.x = element_blank()) +  
    labs(
        x = "Case",
        y = "Throughput (per second)",
        caption = "AMD Ryzen 9 7950x, 2x32G RAM@6000MT/s",
        title = "Overall Ingestion Bandwidth"
    )

ggsave("overall.png", width = 10, height = 6)
ggsave("overall.jpg", width = 10, height = 6)

data.overall %>%
    mutate(class = factor(class), case = factor(case)) %>%
    mutate(class = fct_relevel(class, "Overall", "Overall (4 threads)", "Overall (8 threads)")) %>%
    # multi-threaded benchmarks are done in 48 blocks of images
    mutate(multiplier = case_when(
        class == "Overall" ~ 1,
        class == "Overall (4 threads)" ~ 48,
        class == "Overall (8 threads)" ~ 48,
        TRUE ~ 1
    )) %>%
    mutate(case = fct_relevel(case, "Reference (Precise)", "Scalar (auto-vectorized)", "AVX2 (f32x8)", "AVX512 (f32x16)")) %>%
    filter(class != "sum_of_gradients") %>%
    ggplot(aes(x = case, y = 1/timem * multiplier, fill = case, ymin = 1/timel * multiplier, ymax = 1/timeh * multiplier)) +
    geom_bar(stat = "identity") +
    geom_errorbar(width = 0.2) +
    scale_y_continuous(expand = expansion(mult = 0.03, add = 0.02), limits = c(0, NA)) +
    facet_wrap(~class) +
    theme_minimal() +
    theme(axis.text.x = element_blank()) +  
    labs(
        x = "Case",
        y = "Operations per second",
        caption = "AMD Ryzen 9 7950x, 2x32G RAM@6000MT/s",
        title = "Overall Speed"
    )

ggsave("overall_ops.png", width = 10, height = 6)
ggsave("overall_ops.jpg", width = 10, height = 6)

data.flip %>%
    mutate(class = case_when(
        class == "hash_flip8" ~ "All 8 dihedrals",
        class == "hash_diagflip" ~ "Diagonal flip",
        TRUE ~ class
    )) %>%
    mutate(class = factor(class), case = factor(case)) %>%
    mutate(class = fct_relevel(class, "All 8 dihedrals", "Diagonal flip")) %>%
    mutate(case = fct_relevel(case, "Scalar (auto-vectorized)", "AVX2 (f32x8)", "AVX512 (f32x16)")) %>%
    ggplot(aes(x = case, y = 1/timem, fill = case, ymin = 1/timel, ymax = 1/timeh)) +
    geom_bar(stat = "identity") +
    geom_errorbar(width = 0.2) +
    scale_y_continuous(expand = expansion(mult = 0.03, add = 0.02), limits = c(0, NA)) +
    facet_wrap(~class) +
    theme_minimal() +
    theme(axis.text.x = element_blank()) +  
    labs(
        x = "Case",
        y = "Operations per second",
        caption = "AMD Ryzen 9 7950x, 2x32G RAM@6000MT/s",
        title = "Hash Flipping"
    )

ggsave("flip.png", width = 10, height = 6)
ggsave("flip.jpg", width = 10, height = 6)
