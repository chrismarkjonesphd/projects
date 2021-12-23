#Clear memory 
rm(f_df,Invasives,Invasives_List,Natives,Natives_List)
rm(f_invasives,f_invasive_n,fi_index,For_Names,Forest_List,genus,species,Merged,nme,invf)
rm(inv_index,i,j,k,m)

gc()
memory.limit(size=10000000000000)
library(parallel)
detectCores()

require(rgeos)
ptm <- proc.time()

#Get road data
#####################################
library(raster)
all_rasters <-  stack("C:/ALL_density_map_final.grd")
names(all_rasters[[1]])<-"Elevation"
names(all_rasters[[2]])<-"RoadE1"
names(all_rasters[[3]])<-"RoadU1"
names(all_rasters[[4]])<-"RoadW1"
names(all_rasters[[5]])<-"RoadU2"
names(all_rasters[[6]])<-"RoadW2"
names(all_rasters[[7]])<-"RoadU3"
names(all_rasters[[8]])<-"RoadW3"
names(all_rasters[[9]])<-"RoadW4"
names(all_rasters[[10]])<-"RoadAL"
names(all_rasters[[11]])<-"RoadHU"
names(all_rasters[[12]])<-"RoadHW"
names(all_rasters[[13]])<-"RoadOR"
names(all_rasters[[14]])<-"RoadOT"
names(all_rasters[[15]])<-"RoadRD"
names(all_rasters[[16]])<-"RoadRE"
names(all_rasters[[17]])<-"RiversLakes"
names(all_rasters[[18]])<-"Bridges"
names(all_rasters[[19]])<-"Tunnels"
names(all_rasters[[20]])<-"Railways"
names(all_rasters[[21]])<-"SpeeRail"
names(all_rasters[[22]])<-"Soil0"
names(all_rasters[[23]])<-"Soil8"
names(all_rasters[[24]])<-"Soil9"
names(all_rasters[[25]])<-"Soil10"
names(all_rasters[[26]])<-"Soil11"
names(all_rasters[[27]])<-"Soil13"
names(all_rasters[[28]])<-"Soil14"
names(all_rasters[[29]])<-"Soil15"
names(all_rasters[[30]])<-"Soil16"
names(all_rasters[[31]])<-"Soil18"
names(all_rasters[[32]])<-"Soil19"
names(all_rasters[[33]])<-"Soil21"
names(all_rasters[[34]])<-"Soil22"
names(all_rasters[[35]])<-"Soil23"
names(all_rasters[[36]])<-"Soil24"
names(all_rasters[[37]])<-"Soil25"
names(all_rasters[[38]])<-"Soil26"
names(all_rasters[[39]])<-"Soil27"
names(all_rasters[[40]])<-"Soil28"
names(all_rasters[[41]])<-"Soil29"
names(all_rasters[[42]])<-"Soil30"
names(all_rasters[[43]])<-"Soil33"
names(all_rasters[[44]])<-"Soil35"
names(all_rasters[[45]])<-"Soil38"
names(all_rasters[[46]])<-"Soil39"
names(all_rasters[[47]])<-"Soil40"
names(all_rasters[[48]])<-"Soil41"
names(all_rasters[[49]])<-"Soil42"
names(all_rasters[[50]])<-"Soil43"
names(all_rasters[[51]])<-"Soil44"
names(all_rasters[[52]])<-"Soil45"
names(all_rasters[[53]])<-"Soil46"
names(all_rasters[[54]])<-"Soil47"
names(all_rasters[[55]])<-"Soil48"
names(all_rasters[[56]])<-"Soil49"
names(all_rasters[[57]])<-"Soil54"
names(all_rasters[[58]])<-"Soil55"
names(all_rasters[[59]])<-"Soil59"
names(all_rasters[[60]])<-"Soil60"
names(all_rasters[[61]])<-"Soil61"
names(all_rasters[[62]])<-"Soil62"
names(all_rasters[[63]])<-"Soil63"
names(all_rasters[[64]])<-"Soil64"
names(all_rasters[[65]])<-"Soil65"
names(all_rasters[[66]])<-"Soil66"
names(all_rasters[[67]])<-"Soil67"
names(all_rasters[[68]])<-"Soil68"
names(all_rasters[[69]])<-"Soil69"
names(all_rasters[[70]])<-"Soil71"
names(all_rasters[[71]])<-"Soil75"
names(all_rasters[[72]])<-"Soil76"
names(all_rasters[[73]])<-"Soil78"
names(all_rasters[[74]])<-"Soil79"
names(all_rasters[[75]])<-"Soil80"
names(all_rasters[[76]])<-"Soil81"
names(all_rasters[[77]])<-"Soil82"
names(all_rasters[[78]])<-"Soil83"
names(all_rasters[[79]])<-"Soil84"
names(all_rasters[[80]])<-"Soil3637"
names(all_rasters[[81]])<-"Soil5051"
names(all_rasters[[82]])<-"Soil5253"
names(all_rasters[[83]])<-"Soil6734"
names(all_rasters[[84]])<-"planform"
names(all_rasters[[85]])<-"NEAR_DIST"
names(all_rasters[[86]])<-"percip_jan"
names(all_rasters[[87]])<-"percip_feb"
names(all_rasters[[88]])<-"percip_mar"
names(all_rasters[[89]])<-"percip_apr"
names(all_rasters[[90]])<-"percip_may"
names(all_rasters[[91]])<-"percip_jun"
names(all_rasters[[92]])<-"percip_jul"
names(all_rasters[[93]])<-"percip_aug"
names(all_rasters[[94]])<-"percip_sep"
names(all_rasters[[95]])<-"percip_oct"
names(all_rasters[[96]])<-"percip_nov"
names(all_rasters[[97]])<-"percip_dec"
names(all_rasters[[98]])<-"MAP"
names(all_rasters[[99]])<-"MSP"
names(all_rasters[[100]])<-"WPR"
names(all_rasters[[101]])<-"Tmin_jan"
names(all_rasters[[102]])<-"Tmin_feb"
names(all_rasters[[103]])<-"Tmin_mar"
names(all_rasters[[104]])<-"Tmin_apr"
names(all_rasters[[105]])<-"Tmin_may"
names(all_rasters[[106]])<-"Tmin_jun"
names(all_rasters[[107]])<-"Tmin_jul"
names(all_rasters[[108]])<-"Tmin_aug"
names(all_rasters[[109]])<-"Tmin_sep"
names(all_rasters[[110]])<-"Tmin_oct"
names(all_rasters[[111]])<-"Tmin_nov"
names(all_rasters[[112]])<-"Tmin_dec"
names(all_rasters[[113]])<-"Tave_jan"
names(all_rasters[[114]])<-"Tave_feb"
names(all_rasters[[115]])<-"Tave_mar"
names(all_rasters[[116]])<-"Tave_apr"
names(all_rasters[[117]])<-"Tave_may"
names(all_rasters[[118]])<-"Tave_jun"
names(all_rasters[[119]])<-"Tave_jul"
names(all_rasters[[120]])<-"Tave_aug"
names(all_rasters[[121]])<-"Tave_sep"
names(all_rasters[[122]])<-"Tave_oct"
names(all_rasters[[123]])<-"Tave_nov"
names(all_rasters[[124]])<-"Tave_dec"
names(all_rasters[[125]])<-"MAT"
names(all_rasters[[126]])<-"Tmax_jan"
names(all_rasters[[127]])<-"Tmax_feb"
names(all_rasters[[128]])<-"Tmax_mar"
names(all_rasters[[129]])<-"Tmax_apr"
names(all_rasters[[130]])<-"Tmax_may"
names(all_rasters[[131]])<-"Tmax_jun"
names(all_rasters[[132]])<-"Tmax_jul"
names(all_rasters[[133]])<-"Tmax_aug"
names(all_rasters[[134]])<-"Tmax_sep"
names(all_rasters[[135]])<-"Tmax_oct"
names(all_rasters[[136]])<-"Tmax_nov"
names(all_rasters[[137]])<-"Tmax_dec"
names(all_rasters[[138]])<-"TD"
names(all_rasters[[139]])<-"AHM"
names(all_rasters[[140]])<-"SHM"
names(all_rasters[[141]])<-"WI"
names(all_rasters[[142]])<-"aspect_adj"

#Need these librarys
#####################################
library(dplyr)
library(sp) #For spatial data
library(maptools) #Makes raster maps
library(raster) #To import world climate
library(dismo) #To find extent of points
library(rpart) #For CART
library(randomForest) #For random forest
library(caret) #To save models to file


#Function to create training sets
r_train<-function(all_rasters, train_set){
  #Extract rasters with training and test data
  #####################################
  r_train <- extract(all_rasters, train_set[,2:3]) ##World climate data at training data
  #Need to get the extent all of the points from the data
  e <- extent(SpatialPoints(all_data[, 2:3])) 
  #####################################
  
  #Create points for unobsered areas 
  #####################################
  #number of zeros in the training set
  #taking a random sample from the roads where there are no observations
  
  #Make a paramter 1 and 0 to represent where the collected data is and isn't
  set.seed(0)
  raster_train_set <- sampleRandom(all_rasters,nrow(train_set) , ext=e) 
  #####################################
  
  #Now produce training set
  #####################################
  training_set <- data.frame(rbind(cbind(param=1, r_train), cbind(param=0,raster_train_set))) #Now stick them together
  training_set<-training_set[order(-training_set$param),]
  training_set<-na.omit(training_set) #remove all the nas because random forests hate them
  gc()
  memory.limit(size=10000000000000)
  #####################################
  return(training_set)
}

#Very important, build models for each species
modeler<-function(all_rasters, all_data){
  #Grow a random forest
  #####################################
  param <- as.factor(training_set[, 'param']) #dependent variable as a factor 1 or 0 for classification
  set.seed(23)
  model1 <- randomForest(training_set[, 2:ncol(training_set)], param) #first spot are all the predictors, second in the dependent
  
  #Find an optimal tuning parameter
  set.seed(91)
  tuned <- tuneRF(training_set[, 2:ncol(training_set)], training_set[, 'param']) #Find the best tunning parameter for mtry
  M_try <- tuned[which.min(tuned[,2]), 1] #finds the lowest oobe
  rm(model1,tuned)
  
  #Re-fit random forest
  set.seed(23)
  model2 <- randomForest(training_set[, 2:ncol(training_set)], training_set[, 'param'], mtry=M_try, ntree = 500)
  #####################################
  return(model2)
}

#Finds predictions for each species
predictor<-function(all_rasters, model,e){
  #Spatial prediction
  spatial_pred <- predict(all_rasters, model, ext=e)
  return(spatial_pred)}

#Test models 
r_test<-function(all_rasters, test_set){
  #Extract rasters with training and test data
  r_test <- extract(all_rasters, test_set[,2:3])
  #Need to get the extent all of the points from the data
  e <- extent(SpatialPoints(all_data[, 2:3])) 
  #####################################
  set.seed(1)
  raster_test_set <- sampleRandom(all_rasters, nrow(test_set), ext=e) 
  #####################################
  
  #Now produce training set
  #####################################
  testing_set <- as.data.frame(rbind(cbind(param=1, r_test), cbind(param=0, raster_test_set))) #Now stick them together
  testing_set<-testing_set[order(-testing_set$param),]
  testing_set<-na.omit(testing_set)
  gc()
  memory.limit(size=10000000000000)
  return(testing_set)
}

#Evaluate model
eval<-function(testing_set,model){
  #eval
  eval <- evaluate(testing_set[testing_set$pa==1, ],
                   testing_set[testing_set$pa==0, ], model)
  
  gc()
  return(eval)
}

#This is for low land data
Invasive_Models_F <-list()
for (i in 1:nrow(InvListF)){
  
  title<-as.character(InvListF$Var1[[i]])
  print("Model:")
  print(i)
  print("Species:")
  print(title)
  all_data<-forest[which(forest$Name == title),]
  
  # Split into training and testing sets
  #####################################
  set.seed(33)
  test_set<-sample_frac(all_data,0.3) #Testing
  r_s <- as.numeric(rownames(test_set)) #Testing index
  train_set <- all_data[-r_s,] #Training
  rm(r_s)
  #####################################
  training_set<-r_train(all_rasters,train_set)
  model<-modeler(all_rasters,training_set)
  e <- extent(SpatialPoints(lowland[, 2:3])) #GPS coords
  spatial_pred<-predictor(all_rasters,model,e)
  testing_set<-r_test(all_rasters,test_set)
  evaluation<-eval(testing_set,model)
  preds<-predict(model, newdata=testing_set)
  
  
  Invasive_Models_F[[i]]<-list("Title" = title,"Model"= model, "SpatialPredictions" = spatial_pred, 
                             "ModelEvaluation" = evaluation, "Predictions" = preds, "SpatialExtent" = e)
}

# save the model to disk
saveRDS(Invasive_Models_F, "C:/Invasive_Models_Forest.rds")


Native_Models_F <-list()
for (i in 1:nrow(NatListF)){
  
  title<-as.character(NatListF$Var1[[i]])
  print("Model:")
  print(i)
  print("Species:")
  print(title)
  all_data<-forest[which(forest$Name == title),]
  
  # Split into training and testing sets
  #####################################
  set.seed(33)
  test_set<-sample_frac(all_data,0.3) #Testing
  r_s <- as.numeric(rownames(test_set)) #Testing index
  train_set <- all_data[-r_s,] #Training
  rm(r_s)
  #####################################
  training_set<-r_train(all_rasters,train_set)
  model<-modeler(all_rasters,training_set)
  e <- extent(SpatialPoints(lowland[, 2:3])) #GPS coords
  spatial_pred<-predictor(all_rasters,model,e)
  testing_set<-r_test(all_rasters,test_set)
  evaluation<-eval(testing_set,model)
  preds<-predict(model, newdata=testing_set)
  
  
  Native_Models_F[[i]]<-list("Title" = title,"Model"= model, "SpatialPredictions" = spatial_pred, 
                               "ModelEvaluation" = evaluation, "Predictions" = preds, "SpatialExtent" = e)
}

# save the model to disk
saveRDS(Native_Models_F, "C:/Native_Models_Forest.rds")






#This is for lowland data
Invasive_Models <-list()
for (i in 1:nrow(InvList)){
  
  title<-as.character(InvList$Var1[[i]])
  print("Model:")
  print(i)
  print("Species:")
  print(title)
  all_data<-lowland[which(lowland$Name == title),]
  
  # Split into training and testing sets
  #####################################
  set.seed(33)
  test_set<-sample_frac(all_data,0.3) #Testing
  r_s <- as.numeric(rownames(test_set)) #Testing index
  train_set <- all_data[-r_s,] #Training
  rm(r_s)
  #####################################
  training_set<-r_train(all_rasters,train_set)
  model<-modeler(all_rasters,training_set)
  e <- extent(SpatialPoints(lowland[, 2:3])) #GPS coords
  spatial_pred<-predictor(all_rasters,model,e)
  testing_set<-r_test(all_rasters,test_set)
  evaluation<-eval(testing_set,model)
  preds<-predict(model, newdata=testing_set)
  
  
  Invasive_Models[[i]]<-list("Title" = title,"Model"= model, "SpatialPredictions" = spatial_pred, 
                             "ModelEvaluation" = evaluation, "Predictions" = preds, "SpatialExtent" = e)
}

# save the model to disk
saveRDS(Invasive_Models, "C:/Invasive_Models_lowland.rds")

Natives_Models <-list()
for (i in 1:nrow(NatList)){
  title<-as.character(NatList$Var1[[i]])
  print("Model:")
  print(i)
  print("Species:")
  print(title)
  all_data<-lowland[which(lowland$Name == title),]
  
  # Split into training and testing sets
  #####################################
  set.seed(33)
  test_set<-sample_frac(all_data,0.3) #Testing
  r_s <- as.numeric(rownames(test_set)) #Testing index
  train_set <- all_data[-r_s,] #Training
  rm(r_s)
  #####################################
  training_set<-r_train(all_rasters,train_set)
  model<-modeler(all_rasters,training_set)
  e <- extent(SpatialPoints(lowland[, 2:3])) #GPS coords
  spatial_pred<-predictor(all_rasters,model,e)
  testing_set<-r_test(all_rasters,test_set)
  evaluation<-eval(testing_set,model)
  preds<-predict(model, newdata=testing_set)
  
  
  Natives_Models[[i]]<-list("Title" = title,"Model"= model, "SpatialPredictions" = spatial_pred, 
                            "ModelEvaluation" = evaluation, "Predictions" = preds, "SpatialExtent" = e)
}

saveRDS(Natives_Models, "C:/Natives_Models_lowland.rds")

proc.time() - ptm

