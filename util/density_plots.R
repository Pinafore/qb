
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
granularity <- args[1]

data <- read.csv(sprintf("results/%s.features_cont.csv", granularity))

bw<-function(b, x) { b/bw.nrd0(x) }

plot <- ggplot(data, aes(x=value)) + geom_density(aes(group=correct, colour=correct, fill=correct), alpha=0.3, kernel='o', adjust=bw(0.005, data$value)) + facet_grid(sent ~ name) + scale_y_sqrt()

ggsave(sprintf("results/%s.features_cont.pdf", granularity), plot, width=10, height=20)

data <- read.csv(sprintf("results/%s.features_disc.csv", granularity))
plot <- ggplot(data, aes(x=frequency,y=ig,label=name)) + facet_grid(feature ~ sent) + geom_text()
ggsave(sprintf("results/%s.features_disc.pdf", granularity), plot, width=10, height=20)
