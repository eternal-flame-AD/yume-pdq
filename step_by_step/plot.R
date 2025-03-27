library(ggplot2)
library(patchwork)
library(pixmap)
library(grid)
library(gridExtra)

# Function to read and prepare an image for plotting
prepare.image <- \(path, rotate = TRUE) {
  img <- read.pnm(path)
  img.array <- array(
    c(img@red, img@green, img@blue),
    dim = c(img@size[1], img@size[2], 3)
  )
  if (!rotate) {
    return(img.array)
  }
  rotated <- array(
    dim = c(img@size[2], img@size[1], 3)  # swap dimensions
  )
  for (i in 1:3) {
    rotated[,,i] <- t(apply(img.array[,,i], 2, rev))
  }
  return(rotated)
}

    
# Create a legend data frame
legend.data <- data.frame(
  color = c(
    rgb(0/255, 130/255, 200/255),   # BLUE
    rgb(245/255, 130/255, 48/255),  # ORANGE
    rgb(255/255, 255/255, 255/255),  # WHITE
    rgb(170/255, 110/255, 40/255),  # BROWN
    rgb(255/255, 225/255, 25/255),  # YELLOW
    rgb(128/255, 128/255, 128/255)# GRAY
  ),
  description = factor(
    c(
      "Correctly below median",
      "Should be below median",
      "On median",
      "Should be above median",
      "Correctly above median",
      "Refused to identify"
    ),
    levels = c(
      "Correctly below median",
      "Should be below median",
      "On median",
      "Should be above median",
      "Correctly above median",
      "Refused to identify"
    )
  )
)
get.legend <- ggplot(legend.data, aes(y = description, fill = color)) +
  geom_tile(aes(x = 1), width = 2, height = 0.8) +  # Make tiles wider and shorter
  scale_fill_identity() +
  theme_minimal() +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_text(size = 24),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    plot.margin = margin(5, 20, 5, 20),
    plot.title = element_text(size = 32, hjust = 0.5)
  ) +
  labs(title = "Hand-vectorized iterative thresholding")


# Create a plot for a single image
plot.iteration <- \(img.array, title = NULL, marginxy = 0.2) {
  df <- expand.grid(
    x = seq_len(dim(img.array)[2]),
    y = seq_len(dim(img.array)[1])
  )
  df$r <- as.vector(img.array[,,1])
  df$g <- as.vector(img.array[,,2])
  df$b <- as.vector(img.array[,,3])
  df$col <- rgb(df$r, df$g, df$b)
  
  p <- ggplot(df, aes(x = x, y = y)) +
    geom_raster(aes(fill = col)) +
    scale_fill_identity() +
    coord_fixed() +
    theme_void()
  
  if (!is.null(title)) {
    p <- p + labs(title = title) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 32),
        plot.margin = unit(c(0.2, marginxy, 0.2, marginxy), "cm")
      )
  }
  return(p)
}

compression.input <- plot.iteration(prepare.image("compress/avx2/input.ppm"), title = "Input Image (512x512 Luma)")
compression.output <- plot.iteration(prepare.image("compress/avx2/output.ppm"), title = "Tent Filter Compression Using Vectorized FMA Lookup Tables (127x127 Luma, padded to 128x128)")

# Create input and output plots
quantize.input <- plot.iteration(prepare.image("quantize/avx2/input.ppm"), title = expression(Hand-vectorized ~ DCTII ~ (D %*% A %*% D^t)))
quantize.output <- plot.iteration(prepare.image("quantize/avx2/output.ppm"), title = "Bit-Packed Hash Output")

# Create iteration plots
iter_plots <- list()
for (i in 0:7) {
  img <- prepare.image(sprintf("quantize/avx2/iter_%d.ppm", i))
  iter_plots[[i + 1]] <- plot.iteration(img, sprintf("Iteration %d", i), marginxy = 1.5)
}

print(iter_plots)

# Modified layout to include legend at the bottom
layout <- "
AAAAAA
AAAAAA
AAAAAA
AAAAAA
BBBBBB
BBBBBB
CCCCCC
CCCCCC
EEDDDD
EEDDDD
FFFFFF
FFFFFF
"

final <- wrap_plots(
  A = compression.input,
  B = compression.output,
  C = quantize.input,
  D = wrap_plots(iter_plots, ncol = 4),
  E = get.legend,
  F = quantize.output,
  design = layout
) +
  plot_annotation(
    title = "Yume-PDQ Pipeline Overview",
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 32, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 32),
    )
  )

# Adjust dimensions to accommodate legend
ggsave("../pipeline_overview.png", final, 
       width = 24, height = 22, dpi = 300)