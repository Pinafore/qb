library(ggplot2)


args <- commandArgs(trailingOnly = TRUE)

file <- args[1]
# file <- "/Users/jbg/temp/report.csv"
results <- read.csv(file)

correct <- ggplot(results, aes(sent, val))
correct <- correct + geom_area(aes(color = type, fill= type), position = 'stack') + facet_grid(weight ~ vw)

outfile <- args[2]
ggsave(outfile, correct, width=11, height=7)
