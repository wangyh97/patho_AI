install.packages('magrittr')
install.packages('tidyverse')
library(magrittr)
library(tidyverse)
dev.off()

clin_TMB <- read.csv('~/uro_biomarker/patho_AI/config/merged_clinical_data_with_TMB.csv')
str(clin_TMB, list.len=ncol(clin_TMB))

sorted <- subset(clin_TMB,clin_TMB$treatment_type=='Pharmaceutical Therapy, NOS' & clin_TMB$days_to_death!="'--")
sorted_include_radio <- subset(clin_TMB,clin_TMB$days_to_death!="'--")


final <- select(sorted,ID,vital_status,TMB,days_to_death)
final_with_radio <- select(sorted_include_radio,ID,vital_status,TMB,days_to_death)


write.table(final,file = './config/TMB&D2D.txt',sep = '\t',,row.names=F,quote = F)
write.table(final_with_radio,file =  './config/TMB&D2D_with_radio.txt',sep = '\t',,row.names=F,quote = F)




dev.off()
