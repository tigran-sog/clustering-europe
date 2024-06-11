library(tidyverse)
library(ggrepel)
library(showtext)
showtext_auto()
font_add("Titillium Web", "C:/WINDOWS/FONTS/TITILLIUMWEB-REGULAR.TTF")
font_add("Titillium Web SemiBold", "C:/WINDOWS/FONTS/TITILLIUMWEB-SEMIBOLD.TTF")

celia <- read_csv('C:/Users/user/Documents/GitHub/clustering-europe/data/european clusters.csv')

k_colors <- c("#7fc97f",
              "#beaed4",
              "#fdc086",
              "#ffff99")
k_names <- c(0:3)


ggplot(celia,
       aes(x = HDI, y = `Inequality-loss`, label = ShortName, col = factor(k4))) +
  geom_point(alpha = 0.5, size = 16) +
  #scale_color_manual(values = setNames(k_colors,k_names)) +
  geom_text_repel(col = '#222222',
                  hjust = -0.15, vjust = 0.05,
                  show.legend = FALSE,
                  family = 'Titillium Web',
                  alpha = 0.9,
                  force = 10,
                  point.padding = 2,
                  size = 16) +
  scale_x_continuous(limits = c(0.72, 1), expand = c(0, 0.01)) +
  scale_y_continuous(limits = c(4.5, 15), expand = c(0, 0)) +
  labs(x = 'Human Development Index',
       y = 'Inequality (loss %)',
       title = 'Clustering of European countries by\nhuman development and inequality (k=4)') +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = '#ebebeb', size = 1), # Adjust gridline thickness here
    panel.grid.minor = element_line(color = '#ebebeb', size = 0), # Adjust minor gridline thickness here
    text = element_text(family = 'Titillium Web'),
    plot.caption = element_text(size = 20),
    plot.title = element_text(size = 70, color = '#1D5B79',
                              family = 'Titillium Web SemiBold',
                              margin = margin(t = 30, b = 20)), # Adjust title size
    plot.subtitle = element_text(size = 35, color = '#222222',
                                 margin = margin(b = 20)), # Adjust subtitle size
    axis.title.x = element_text(size = 45, margin = margin(t = 20, b = 20)), # Adjust X axis title size and margin
    axis.title.y = element_text(size = 45, margin = margin(r = 20, l = 20)), # Adjust Y axis title size and margin
    axis.text = element_text(size = 45, vjust = .5),
    axis.line = element_line(linewidth = .5, color = '#1D5B79'),  # adjust line size as needed
    legend.title = element_text(size = 35), # adjust size as needed
    legend.text = element_text(size = 35)  # adjust size as needed
  ) +
  guides(size = 'none',
         col = 'none') -> k4_plot
ggsave("C:/Users/user/Documents/GitHub/clustering-europe/viz/k4 plot.pdf", k4_plot, width = 29, height = 29,
       bg = '#f5f5f5')
ggsave("C:/Users/user/Documents/GitHub/clustering-europe/viz/k4 plot.png", k4_plot, width = 29, height = 29, dpi = 96,
       bg = '#f5f5f5')

