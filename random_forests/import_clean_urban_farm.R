#Importing main data soures into R
#Remove columns will all NAs 

#Function to check for NAs
tvs_nas<-function(tvs){
  for(i in 1:ncol(tvs)){
    print(paste("Column number",i))
    print(sum(is.na(tvs[i])))
  }
}

#Begin imports
library(readxl)
#Import Primary data
#IM_SPVPTDATA-20190925
tvs <- read_excel("C:/Users/tvs2.xlsx", 
                   col_types = c("text", "text", "text","text", "text", "text", "text", "text", "text", 
                                 "numeric", "numeric", "numeric", "numeric", 
                                 "text", 
                                 "skip", 
                                 "numeric", "numeric", "numeric", "numeric", "numeric",
                                 "text", "text", 
                                 "numeric", 
                                 "date", "text", 
                                 "numeric", 
                                 "text", 
                                 "numeric", "numeric", 
                                 "text", 
                                 "skip",
                                 "text", "text", "text", "text", 
                                 "date", 
                                 "text", 
                                 "numeric","numeric", 
                                 "text"))
  #Check for NAs
  tvs_nas(tvs)
  #Remove columns with all NAs
  #Columns 13, 20, 23, 34, 35, 38
  tvs1 <- tvs[-c( 13, 20, 23, 34, 35, 38)]
  tvs_nas(tvs1)
 
  

#Import Habbitat data
hab <- read_excel("C:/Users/tvs3.xlsx", 
                   col_types = c("text", "text", "text", 
                                 "text", "numeric", "text", "text", "text", 
                                 "numeric", "numeric", "numeric", 
                                 "text", "numeric", "numeric", "numeric", 
                                 "numeric", "numeric", "numeric", 
                                 "skip", "numeric", "text", "numeric", 
                                 "text", "numeric", "numeric", "numeric", 
                                 "numeric", "numeric", "text", "numeric", 
                                 "text", "text", "text", "text", "date", 
                                 "numeric", "text", "text", "text", 
                                 "numeric", "text", "text", "numeric"))
  #Check for NAs
  tvs_nas(hab)
  #Remove Columns
  #27, 34, 35
  hab1 <- hab[-c(27, 34, 35)]
  tvs_nas(hab1)
 
  
#Import data for subplots
#IM_SPLOTDATA.xlsx (tvs4) is the same as Habitats file
  

