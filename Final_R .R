################################# PACKAGES IMPORT

### set working environment
# add packages to the list as needed
pkgs <- list("glmnet", "reticulate", "stringr", "rstudioapi", "data.table", "parallel", "minpack.lm", "doParallel",
             "foreach", "pROC", "gplots", "pwr", "dplyr", "caret", "sm", "ggplot2", "scales", "reshape2", "Hmisc",
             "bayesAB", "gridExtra", "plotly", "flux", "RColorBrewer", "plm", "xts", "pdp", "vip", "ranger", "vioplot",
             "randomForest")

# install packages in list
##lapply(pkgs, install.packages, character.only = T)

# load packages in list
lapply(pkgs, require, character.only = T)

################################# DATA IMPORT

library(readr)
data <- read_csv("Desktop/capstone/d14_d30_data_v3.csv")
#View(data)


################################# CLV Prediction

#covert data type
data$d30_spend <- as.numeric(data$d30_spend)
data$d14_spend <- as.numeric(data$d14_spend)
data$d7_spend <- as.numeric(data$d7_spend)
data$d3_spend <- as.numeric(data$d3_spend)
data$d1_spend <- as.numeric(data$d1_spend)
data$count_p_1 <- as.numeric(data$count_p_1)
data$max_p_2 <- as.numeric(data$max_p_2)
data$p_4 <- as.numeric(data$p_4)
data$p_5 <- as.numeric(data$p_5)
data$sum_p_6 <- as.numeric(data$sum_p_6)
data$count_p_7 <- as.numeric(data$count_p_7)
data$count_p_8 <- as.numeric(data$count_p_8)
data$count_p_9 <- as.numeric(data$count_p_9)
data$count_p_10 <- as.numeric(data$count_p_10)
data$count_p_11 <- as.numeric(data$count_p_11)
data$count_p_12 <- as.numeric(data$count_p_12)
data$count_p_13 <- as.numeric(data$count_p_13)
data$len_p_14 <- as.numeric(data$len_p_14)
data$count_p_15 <- as.numeric(data$count_p_15)
data$count_p_19 <- as.numeric(data$count_p_19)
data$count_p_20 <- as.numeric(data$count_p_20)

#virtulization
library(VIM)

#================================= campaign
#delete the symbol can not be converted
data$campaign <- gsub('[c_]', '', data$campaign)
##counting the frequency
campaign_counts <- table(data$campaign)
#View(data)
data$campaign <- as.numeric(data$campaign)
##hist
hist(data$campaign)
data$campaign <- as.factor(data$campaign)
##order bar chart
library(forcats)
ggplot(data, aes(fct_infreq(campaign))) +
  geom_bar()

#define the split benchmark
campaign_number <- data%>%
  count(campaign)%>%
  arrange(-n)%>%
  slice_max(n,n=27)

data$campaign <- as.numeric(data$campaign)

#================================= media_source
#delete the symbol can not be converted
data$media_source <- gsub('[ms_]', '', data$media_source)
#View(data)
#define the split benchmark
data$media_source <- as.numeric(data$media_source)

media_source <- data%>%
  count(media_source)%>%
  arrange(-n)%>%
  slice_max(n,n=17)

#================================= dummy variable 
#data$u_3_1 <- ifelse(data$u_3 == "Non-organic", 1, 0)
#data$u_3_2 <- ifelse(data$u_3 == "Organic", 1, 0)
#data$u_3_3 <- ifelse(is.null(data$u_3), 1, 0)
#data <- subset(data, select = -c(u_3_1, u_3_2, u_3_3) )

#install.packages("fastDummies")
library(fastDummies)
library(tidyr)

##u_2
data <- dummy_cols(data, select_columns = "u_2")

##u_3
data <- dummy_cols(data, select_columns = "u_3")

##u_4
data <- dummy_cols(data, select_columns = "u_4")

##m_3
data <- dummy_cols(data, select_columns = "m_3")

##m_4
data <- dummy_cols(data, select_columns = "m_4")

##max_p_2
data <- dummy_cols(data, select_columns = "max_p_2")

##p_3
data <- dummy_cols(data, select_columns = "p_3")

##campaign
data$c_6 <- ifelse(data$campaign==6,1,0)
data$c_10 <- ifelse(data$campaign==10,1,0)
data$c_0 <- ifelse(data$campaign==0,1,0)
data$c_19 <- ifelse(data$campaign==19,1,0)
data$c_5 <- ifelse(data$campaign==5,1,0)
data$c_7 <- ifelse(data$campaign==7,1,0)
data$c_4 <- ifelse(data$campaign==4,1,0)
data$c_20 <- ifelse(data$campaign==20,1,0)
data$c_18 <- ifelse(data$campaign==18,1,0)
data$c_24 <- ifelse(data$campaign==24,1,0)
data$c_37 <- ifelse(data$campaign==37,1,0)
data$c_17 <- ifelse(data$campaign==17,1,0)
data$c_9 <- ifelse(data$campaign==9,1,0)
data$c_30 <- ifelse(data$campaign==30,1,0)
data$c_15 <- ifelse(data$campaign==15,1,0)
data$c_45 <- ifelse(data$campaign==45,1,0)
data$c_25 <- ifelse(data$campaign==25,1,0)
data$c_2 <- ifelse(data$campaign==2,1,0)
data$c_65 <- ifelse(data$campaign==65,1,0)
data$c_1 <- ifelse(data$campaign==1,1,0)
data$c_12 <- ifelse(data$campaign==12,1,0)
data$c_31 <- ifelse(data$campaign==31,1,0)
data$c_28 <- ifelse(data$campaign==28,1,0)
data$c_29 <- ifelse(data$campaign==29,1,0)
data$c_67 <- ifelse(data$campaign==67,1,0)
data$c_21 <- ifelse(data$campaign==21,1,0)
data$c_71 <- ifelse(data$campaign==71,1,0)
#campaign_list <- c(6, 10, 0, 19, 5, 7, 4, 20, 18, 24, 37, 17, 9, 30, 15, 45, 25, 2, 65, 1, 12, 31, 28, 29, 67, 21, 27)
#data$other <- ifelse(data$campaign != campaign_list, 1, 0)

#media_source
data$ms_1 <- ifelse(data$media_source==1,1,0)
data$ms_6 <- ifelse(data$media_source==6,1,0)
data$ms_0 <- ifelse(data$media_source==0,1,0)
data$ms_8 <- ifelse(data$media_source==8,1,0)
data$ms_3 <- ifelse(data$media_source==3,1,0)
data$ms_7 <- ifelse(data$media_source==7,1,0)
data$ms_2 <- ifelse(data$media_source==2,1,0)
data$ms_12 <- ifelse(data$media_source==12,1,0)
data$ms_15 <- ifelse(data$media_source==15,1,0)
data$ms_16 <- ifelse(data$media_source==16,1,0)
data$ms_11 <- ifelse(data$media_source==11,1,0)
data$ms_5 <- ifelse(data$media_source==5,1,0)
data$ms_20 <- ifelse(data$media_source==20,1,0)
data$ms_26 <- ifelse(data$media_source==26,1,0)
data$ms_4 <- ifelse(data$media_source==4,1,0)
data$ms_25 <- ifelse(data$media_source==25,1,0)
data$ms_10 <- ifelse(data$media_source==10,1,0)

#View(data)

#Delete some variables that cannot be in the model
cor_vars <- data[ , -grep("first_login|m_1|m_2|u_1", colnames(data))]
cor_vars <- subset(cor_vars, select = -...1)

#Deletes variables that have been converted with dummy
#cor_vars <- cor_vars[ , -grep("u_2|u_3|u_4|m_3|m_4|max_p_2|p_3|campaign|media_source", colnames(cor_vars))]
cor_vars <- subset(cor_vars, select = -u_2)
cor_vars <- subset(cor_vars, select = -u_3)
cor_vars <- subset(cor_vars, select = -u_4)
cor_vars <- subset(cor_vars, select = -m_3)
cor_vars <- subset(cor_vars, select = -m_4)
cor_vars <- subset(cor_vars, select = -max_p_2)
cor_vars <- subset(cor_vars, select = -p_3)
cor_vars <- subset(cor_vars, select = -campaign)
cor_vars <- subset(cor_vars, select = -media_source)


#Dealing with missing value here
cor_vars[is.na(cor_vars) == TRUE] <- 0
sum(is.na(cor_vars))

#================================= corralation ship
#sub1
library(corrplot)
##subset: col chose
cor_vars_sub1 <- cor_vars[ , grep("d1_spend|d3_spend|d7_spend|d14_spend|d30_spend", colnames(cor_vars))]
##calculate cor
res_cor_sub1 <- cor(cor_vars_sub1)
## pic
sub_1_pic <- corrplot(corr = res_cor_sub1, order = "original",type="upper",tl.pos = "tp")
sub_1_pic <- corrplot(corr = res_cor_sub1, add=TRUE, type="lower", method="number",order="original",diag=FALSE,tl.pos="n", cl.pos="n")

#sub2
##subset: col chose
cor_vars_sub2 <- cor_vars[5:25]
##calculate cor
res_cor_sub2 <- round(cor(cor_vars_sub2),2)
## pic
sub_2_pic <- corrplot(corr = res_cor_sub2, order="original", type="upper", tl.pos="tp")
sub_2_pic <- corrplot(corr = res_cor_sub2, add=TRUE, type="lower", method="number",order="original", col="black",diag=FALSE,tl.pos="n", cl.pos="n")

#sub3
##subset: col chose
cor_vars_sub3 <- cor_vars[55:66]
##calculate cor
res_cor_sub3 <- round(cor(cor_vars_sub3),2)
## pic
sub_3_pic <- corrplot(corr = res_cor_sub3, order="original", type="upper", tl.pos="tp")
sub_3_pic <- corrplot(corr = res_cor_sub3, add=TRUE, type="lower", method="number",order="original", col="black",diag=FALSE,tl.pos="n", cl.pos="n")

#sub4
##subset: col chose
cor_vars_sub4 <- cor_vars[65:91]
##calculate cor
res_cor_sub4 <- round(cor(cor_vars_sub4),2)
## pic
sub_4_pic <- corrplot(corr = res_cor_sub4, order="original", type="upper", tl.pos="tp")
sub_4_pic <- corrplot(corr = res_cor_sub4, add=TRUE, type="lower", method="number",order="original", col="black",diag=FALSE,tl.pos="n", cl.pos="n")

# helper function to get upper triangle of the correlation matrix

cormat <- round(cor(cor_vars),2)
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = TRUE)

# label levels of factors to make heatmap nice to read
#levels(melted_cormat$Var1)[levels(melted_cormat$Var1)=="years_since_first_spend"] <- "Years active on platform"
#levels(melted_cormat$Var2)[levels(melted_cormat$Var2)=="years_since_first_spend"] <- "Years active on platform"

or_heat <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "red", high = "blue", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  theme(axis.text.y = element_text(size = 12))+
  coord_fixed()

#================================= output of dataset
# select x vars
x1_vars <- cor_vars[ , -grep("d30_spend", colnames(cor_vars))]
x2_vars <- cor_vars[ , -grep("d1_spend|d3_spend|d7_spend|d14_spend|d30_spend", colnames(cor_vars))]

#set saving path
setwd("Desktop/capstone")
#save the document
write.csv(x = x1_vars, file = "x1_vars")
write.csv(x = x2_vars, file = "x2_vars")
write.csv(x = cor_vars, file = "cor_vars")

