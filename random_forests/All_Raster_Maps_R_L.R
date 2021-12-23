library(sf)
library(raster)
library(ggplot2) # tidyverse data visualization package
require("graphics")
library(ggpubr)
library(parallel)
detectCores()
library('randomForest') #need this to see models
# load the model
IV_M <- readRDS("C:/Users/Invasive_Models_lowland.rds")
N_M <- readRDS("C:/Users/Natives_Models_lowland.rds")
F_M <- readRDS("C:/Users/forest_models.rds")

IV_M[[1]]$Title<-"Bidens alba (L.) DC. var. radiata (Sch. Bip.)"
gc()
memory.limit(size=10000000000000)

spacial_nat<-list()
for(i in 1:10){
  spdf <- as(N_M[[i]]$SpatialPredictions , "SpatialPixelsDataFrame")
  spacial_nat[[i]]<- as.data.frame(spdf) 
}
spacial_inv<-list()
for(i in 1:10){
  spdf <- as(IV_M[[i]]$SpatialPredictions , "SpatialPixelsDataFrame")
  spacial_inv[[i]]<- as.data.frame(spdf) 
}

for_10<-list()
for(i in 1:10){
  spdf <- as(F_M[[i]]$SpatialPredictions , "SpatialPixelsDataFrame")
  for_10[[i]]<- as.data.frame(spdf) 
}

gc()
memory.limit(size=10000000000000)

plot_nat<-function(species){
p1<-ggplot() +
  geom_raster(data = spacial_nat[[species]] , aes(x = x, y = y, fill = layer)) +
  coord_equal() +
  geom_path() +
  scale_fill_gradientn(colours = rev(terrain.colors(20)),name="Predicted") +
  labs(title = N_M[[species]]$Title) + xlab("") + ylab("") +
  theme_classic() +
  theme(plot.title = element_text(size = 6, face = "italic")) 
return(p1)
}

gc()
memory.limit(size=10000000000000)

plot_inv<-function(species){
  p1<-ggplot() +
    geom_raster(data = spacial_inv[[species]] , aes(x = x, y = y, fill = layer)) +
    coord_equal() +
    geom_path() +
    scale_fill_gradientn(colours = rev(terrain.colors(20)),name="Predicted") +
    labs(title = IV_M[[species]]$Title) + xlab("") + ylab("") +
    theme_classic() +
    theme(plot.title = element_text(size = 6, face = "italic")) 
  return(p1)
}

plot_for<-function(species){
  p1<-ggplot() +
    geom_raster(data = for_10[[species]] , aes(x = x, y = y, fill = layer)) +
    coord_equal() +
    geom_path() +
    scale_fill_gradientn(colours = rev(terrain.colors(20)),name="Predicted") +
    labs(title = F_M[[species]]$Title) + xlab("") + ylab("") +
    theme_classic() +
    theme(plot.title = element_text(size = 6, face = "italic")) 
  return(p1)
}

gc()
memory.limit(size=10000000000000)

#Native plots
ggarrange(plot_nat(1),plot_nat(2),plot_nat(3),plot_nat(4),
          plot_nat(5),plot_nat(6),plot_nat(7),plot_nat(8),
          plot_nat(9),plot_nat(10),
  nrow=2,ncol=5,common.legend = TRUE, legend = "bottom")

gc()
memory.limit(size=10000000000000)

#Invasive plots

ggarrange(plot_inv(1),plot_inv(2),plot_inv(3),plot_inv(4),
          plot_inv(5),plot_inv(6),plot_inv(7),plot_inv(8),
          plot_inv(9),plot_inv(10),
          nrow=2,ncol=5,common.legend = TRUE, legend = "bottom")
gc()
memory.limit(size=10000000000000)

#forest plots
ggarrange(plot_for(1),plot_for(2),plot_for(3),plot_for(4),
          plot_for(5),plot_for(6),plot_for(7),plot_for(8),
          plot_for(9),plot_for(10),
          nrow=2,ncol=5,common.legend = TRUE, legend = "bottom")



