library(readxl)
library("parallel")
#library(raster)
#library(rgdal)
library(sf)
#library(rgeos)
library(sp)
#library(stringr)
library(dplyr)

#import forest data
df<- read_excel("C:/Plot_properties-20201012.xls")
species <- read_excel("C:/Plants_in_each_plot-20201012.xlsx")


#rename plot id
df<- rename(df,"PLOT_ID" ="PLOT ID")

#make plot id for species data
species$PLOT_ID<-gsub('.{4}$', '', species$SUBPLOT_ID)
#merge data
Merged<-right_join(species,df, by = c("PLOT_ID"))


#Find coords
xy <- cbind(Merged$x_97,Merged$y_97)
colnames(xy) <- c('Longitude', 'Latitude')
coords <- coordinates(xy)

#current coord system 
TWD97 <- CRS('+proj=tmerc +lat_0=0 +lon_0=121 +k=0.9999 +x_0=250000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ')
c97 <- SpatialPoints(coords, TWD97)

#convert to wgs84

WGS84 <- CRS('+proj=longlat +ellps=WGS84')
c84 <- spTransform(c97, WGS84)
rm(c97,xy,coords,TWD97,WGS84)
ctran<-as.data.frame(c84)
Merged$Longitude<-ctran$Longitude
Merged$Latitude<-ctran$Latitude
rm(ctran,df,c84,species)

#pepare data
forest<-data.frame(Merged[c(3,17,18)])
colnames(forest)<-(c("Name","Longitude","Latitude"))

#get forest list
Forest_List<-as.data.frame(table(forest$Name))
Forest_List<-(Forest_List[order(-Forest_List$Freq),])
ForList<-Forest_List[which(Forest_List$Freq >= 10),]

#get invasive status
ni<-read_excel("C:/TaiwanSpecies20191213.xlsx")
s_names<-ni[which(ni$kingdom=="Plantae"),]
invf<-s_names[which(s_names$`alien_status (0: native, 1&2&3: introduced)`==1),]#####
rm(s_names,ni)
library(stringr)
#t<-str_split_fixed(tt[[1]][1], " ", 3)
#tt<-invf[which(invf$genus=="Litsea" & invf$species=="acuminata"),]

#split names for comparision
genus<-list()
species<-list()
for(i in 1:nrow(ForList)){
  nme<-str_split_fixed(ForList[[1]][i], " ", 3)
  genus[[i]]<-nme[c(1)]
  species[[i]]<-nme[c(2)]
}

For_Names<-data.frame(unlist(genus),unlist(species))
colnames(For_Names)<-c("genus","species")

#find the invasives
f_invasives<-list()
for(j in 1:nrow(For_Names)){
  f_invasives[[j]]<-invf[which(invf$genus==as.character(For_Names$genus[j]) 
                               & invf$species==as.character(For_Names$species[j])),]
  
}
# row.names(f_invasives[[289]])
# class(row.names(f_invasives[[21]]))
#attributes(f_invasives[[1]])

f_invasive_n<-list()
for(k in 1:length(f_invasives)){
  if(length(row.names(f_invasives[[k]]))!=0){
    f_invasive_n[[k]]<-f_invasives[[k]]
  }else{
    f_invasive_n[[k]]<-NA
  }
}

#find invasive index
fi_index<-data.frame(is.na(f_invasive_n)==FALSE)
colnames(fi_index)<-c("TF")
inv_index<-which(fi_index$TF=="TRUE")

#make into dataframe
for(m in 1:nrow(fi_index)){
  if(fi_index$TF[m]=="FALSE"){
    fi_index$TF[m]<-0
  }
}

ForList$status<-fi_index

InvListF<-ForList[which(ForList$status==1),]
NatListF<-ForList[which(ForList$status==0),]
rm(ForList)

