
library(ggplot2)

data <- read.csv("results/features.csv")

plot <- ggplot(data, aes(x=value)) + geom_density(aes(group=correct, colour=correct, fill=correct), alpha=0.3) + facet_grid(sent ~ name) + scale_y_sqrt()

ggsave("results/features.pdf", plot)