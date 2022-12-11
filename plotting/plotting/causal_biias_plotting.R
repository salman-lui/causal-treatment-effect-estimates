## Plotting 


library(readr)

## loading data

lr <- read.csv(file = '/Users/salman/Downloads/causal-final-project-main/lin_result.csv')

bart <- read.csv(file = '/Users/salman/Downloads/causal-final-project-main/bart_result.csv')

rf_lr <- read.csv(file = '/Users/salman/Downloads/causal-final-project-main/std_biases_lr_rf.csv')

rf_doublyrobust <- read.csv(file = '/Users/salman/Downloads/causal-final-project-main/std_biases_dr_rf.csv')

gbm_lr <- read.csv(file = '/Users/salman/Downloads/causal-final-project-main/std_biases_lr_gbm.csv')

gbm_doublyrobust <- read.csv(file = '/Users/salman/Downloads/causal-final-project-main/std_biases_dr_gbm.csv')


## eight design dataframe

design_111 <- data.frame(lr$X111, bart$X111, rf_lr$X111, rf_doublyrobust$X111, gbm_lr$X111, gbm_doublyrobust$X111)


design_211 <- data.frame(lr$X211, bart$X211, rf_lr$X211, rf_doublyrobust$X211, gbm_lr$X211, gbm_doublyrobust$X211)


design_121 <- data.frame(lr$X121, bart$X121, rf_lr$X121, rf_doublyrobust$X121, gbm_lr$X121, gbm_doublyrobust$X121)


design_221 <- data.frame(lr$X221, bart$X221, rf_lr$X221, rf_doublyrobust$X221, gbm_lr$X221, gbm_doublyrobust$X221)


design_112 <- data.frame(lr$X112, bart$X112, rf_lr$X112, rf_doublyrobust$X112, gbm_lr$X112, gbm_doublyrobust$X112)


design_212 <- data.frame(lr$X212, bart$X212, rf_lr$X212, rf_doublyrobust$X212, gbm_lr$X212, gbm_doublyrobust$X212)


design_122 <- data.frame(lr$X122, bart$X122, rf_lr$X122, rf_doublyrobust$X122, gbm_lr$X122, gbm_doublyrobust$X122)


design_222 <- data.frame(lr$X222, bart$X222, rf_lr$X222, rf_doublyrobust$X222, gbm_lr$X222, gbm_doublyrobust$X222)


## Plottting 
library(ggplot2)
plot_design <- function(design, title){
  new_design <- stack(design)
  levels(new_design$ind) <- c("LR", "BART", "RF-LR", "RF-DR", "GBM-LR", "GBM-DR")
  ggplot(new_design, aes(ind, values)) +  # Boxplot with updated labels
  geom_boxplot()+ ggtitle(title) +
  xlab("") + ylab("standardized bias")
}


## Design - 1 -111

plot_design(design = design_111, title = "Non-ignorable, non-interactive, negative alignment")

## Design - 2 - 211

plot_design(design = design_211, title = "Ignorable, non-interactive, negative alignment")


## Design - 3 - 121

plot_design(design = design_121, title = "Non-ignorable, interactive, negative alignment")


## Design - 4 - 221

plot_design(design = design_221, title = "Ignorable, interactive, negative alignment")


## Design - 5 - 112

plot_design(design = design_112, title = "Non-ignorable, non-interactive, positive alignment")


## Design - 6

plot_design(design = design_212, title = "Ignorable, non-interactive, positive alignment")


## Design - 7

plot_design(design = design_122, title = "Non-ignorable, interactive, positive alignment")


## Design - 8

plot_design(design = design_222, title = "Ignorable, interactive, positive alignment")
