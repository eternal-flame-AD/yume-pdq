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
    filter(str_detect(class, "Overall"))

data.sub <- data %>%
    filter(!str_detect(class, "Overall"))

data.sub %>%
    mutate(class = factor(class), case = factor(case)) %>%
    mutate(class = fct_relevel(class, "Compress", "DCT transform", "Quantize")) %>%
    mutate(case = fct_relevel(case, "Reference (Precise)", "Scalar (auto-vectorized)", "AVX2 (f32x8)", "AVX512 (f32x16)")) %>%
    select(-starts_with("time")) %>%
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
        y = "Throughput per Second",
        caption = "AMD Ryzen 9 7950x, 2x32G RAM@6000MT/s"
    )

ggsave("sub.png", width = 10, height = 5)
ggsave("sub.jpg", width = 10, height = 5)

data.overall %>%
    mutate(class = factor(class), case = factor(case)) %>%
    mutate(class = fct_relevel(class, "Overall", "Overall (4 threads)", "Overall (8 threads)")) %>%
    mutate(case = fct_relevel(case, "Reference (Precise)", "Scalar (auto-vectorized)", "AVX2 (f32x8)", "AVX512 (f32x16)")) %>%
    select(-starts_with("time")) %>%
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
        y = "Throughput per Second",
        caption = "AMD Ryzen 9 7950x, 2x32G RAM@6000MT/s"
    )

ggsave("overall.png", width = 10, height = 5)
ggsave("overall.jpg", width = 10, height = 5)
