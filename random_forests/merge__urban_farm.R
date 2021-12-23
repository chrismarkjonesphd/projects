#WHEN SAVING, USE SAVE WITH ENCODING UTF-8 !!!!!!!!!!!!!!!!

#This will be used for any other manipulations to the data frames 
library(dplyr)
#This will remove all NAs from the specified column
#NOTE: This will return a data frame where where the Entire Row is removed
remove_nas<-function(data,column){
  column<-as.numeric(column)
  temp<-data[!is.na(data[column]),]
}

#Merge tvs1 and hab1

#Rename Habitat varaibles
  hab_rename<- rename(hab1,"Serial_Number" ="Serial number",
                "Plot_ID" = "PLOT_ID (Plot number)",
                "Subplot_ID" = "SUB_ID (Subplot number)",
                "Habitat_Type" = "HAB_TYPE (Habitat type)",
                "Subplot_Type"="SUB_TYPE (Type of subplot)",
                "Survary_Date"= "SUBPDATE (Survey date)" ,
                "Investigator_ID"= "EXAMINE",
                "Data_Recoder_ID"= "INTR",
                "Altitude_handheld"= "ALTT (m) (Altitude (measured with hand-held GPS))",
                "Subplot_x_coor_UTMCS08"= "TM2X",
                "Subplot_y_coor_UTMCS08"= "TM2Y",
                "Data_Examiner_ID"= "CONFIRM",
                "Subplot_x_coor_UTMCS78"= "TWD67X",
                "Subplot_y_coor_UTMCS78"= "TWD67Y",
                "Longitude_08"= "DD97X (Longitude)",
                "Latitude_08"= "DD97Y (Latitude)",
                "Longitude_78"= "DDX",
                "Latitude_78"= "DDY",
                "Uploader_ID"= "C_PER",
                "Date_Uploaded"= "C_DATE",
                "Altitude"= "TM2Z",
                "Grid_Number"= "GRIDNO",
                "Aspect"= "TWASPECT (aspect)",
                "Slope"= "TWSLOPE (slope)",
                "Sky_Un-obscured_Percentage"= "TWSVF (Percent sky unobscured by topography; 986= 98.6%)",
                "ALTTZ"= "ALTTZ" ,
                "Survay_Year"= "SURV_YEAR",
                "Data_Error"= "ERROR_L",
                "Data_Error_2"= "VERROR_L",
                "PERROR_L"= "PERROR_L",
                "Coastal_Areas"= "SEA_MARK (Plots in the coastal areas; 0, no;1, yes)",
                "Urban_Areas"= "CITY_MARK(Plots in the urban areas; 0, no;1, yes)",
                "Recreation_Areas"= "AMUM_MARK(Plots in the recreation forests; 0, no;1, yes)",
                "Low_inland_Areas"= "INLA_MARK(Plots in the lowland inland areas; 0, no;1, yes)",
                "COLROW"= "COLROW",
                "onMain_offIsland"= "MAIN_OFF  (0, main island; 1, off island)",
                "Plot_Type"= "PLOT_TYPE",
                "Plot_Mark"= "PROT_MARK",
                "TMZ"= "TMZ")
  #Remove variables that are of no use at this point
  Habitat <- hab_rename[-c(26, 30, 35)]
  
  #Rename Vegitation variables 
  tvs_rename<- rename(tvs1,"Serial_Number" ="Serial Number",
                "Plot_ID" = "PLOT_ID",
                "Subplot_ID" = "SUB_ID (subplot ID)",
                "Habitat_Type" = "HAB_TYPE (habitat type)",
                "Site_ID" = "SITE_ID",
                "Species_Code_Number" = "V_ID",
                "Growth_Form_Herb_Wood" = "V_TYPE (growth form type)",
                "Family" = "S1 (Family)",
                "Genus" = "S2 (Genus)",
                "Type" = "TYPE",
                "Growth_Form" = "H1 (growth form)",
                "Deciduousness" = "H2 (Deciduousness)",
                "Scientific_Name" = "SNAME (scientific name)",
                "Percent_Cover" = "COV(%)(cover)",
                "Bud_zeroNo_oneYes" = "BRON(bud; 0, no; 1, yes)",
                "Flower_zeroNo_oneYes" = "FLON(flower; 0, no; 1, yes)",
                "Fruit_zeroNo_oneYes" = "FRON(fruit; 0, no; 1, yes)",
                "Seed_zeroNo_oneYes" = "SDON(seed; 0, no; 1, yes)",
                "DBH_CHK" = "DBH_CHK",
                "Uploader_ID" = "C_PER",
                "Survay_Year" = "SURV_YEAR",
                "Data_Error" = "ERROR_L",
                "Synonym" = "OR_CNAME",
                "Subplot_Type" = "SUB_TYPE",
                "Floristic_Region" = "FR",
                "LF_Growth_Form" = "LF(growth form)",
                "zeroAs_Native_oneAs_Introduced" = "0, native; 1, introduced",
                "World_Weed" = "W (Listed in the world weeds or not)",
                "Usage" = "U",
                "Origin" = "ORIGIN",
                "onMain_offIsland" = "MAIN_OFF",
                "TMZ" = "TMZ")
  #Remove variables that are of no use at this point
  Vegitation <- tvs_rename[-c(1,11, 19, 23)]
  
  #MERGE BOTH TOGETHER
  Merged<-right_join(Habitat,Vegitation, by = c("Plot_ID", "Subplot_ID", "Habitat_Type","Survay_Year","Uploader_ID", 
                                                "Data_Error", "onMain_offIsland","Subplot_Type","TMZ"))
  
  #Remove redundant objects
  rm(tvs,tvs1,tvs_rename,
     hab,hab_rename,hab1,
     Habitat,Vegitation)
  
  
  #Set Variables to correct format
  #Check data types
  sapply(Merged, class)
  
  Merged$Coastal_Areas[which(is.na(Merged$Coastal_Areas))] <- as.character("-1")
  Merged$Low_inland_Areas[which(is.na(Merged$Low_inland_Areas))] <- as.character("-1")
  Merged$Plot_Mark[which(is.na(Merged$Plot_Mark))] <- as.character("-1")
  Merged$Species_Code_Number[which(is.na(Merged$Species_Code_Number))] <- as.character("-1")
  Merged$Urban_Areas[which(is.na(Merged$Urban_Areas))] <- as.character("-1")
  Merged$Recreation_Areas[which(is.na(Merged$Recreation_Areas))] <- as.character("-1")
  Merged$onMain_offIsland[which(is.na(Merged$onMain_offIsland))] <- as.numeric("-1")
  Merged$Plot_Type[which(is.na(Merged$Plot_Type))] <- as.character("-1")
  Merged$TMZ[which(is.na(Merged$TMZ))] <- as.numeric("-1")
  Merged$Deciduousness[which(is.na(Merged$Deciduousness))] <- as.character("-1")
  Merged$World_Weed[which(is.na(Merged$World_Weed))] <- as.character("Not_Listed")
  Merged$Usage[which(is.na(Merged$Usage))] <- as.character("None_or_Unknown")
  Merged$Origin[which(is.na(Merged$Origin))] <- as.character("Taiwan_or_Unknown")
  
  
  #summary(model_df$zeroAs_Native_oneAs_Introduced) (code run before conversion, then pasted below)
  #Convert "未知"(Wèizhi) to the english translation "Unknown"
  #Convert "栽培"(Zaipéi) to the english translation "Cultivation"
  Merged$zeroAs_Native_oneAs_Introduced[which(Merged$zeroAs_Native_oneAs_Introduced == ("未知"))] <- as.character("Unknown")
  Merged$zeroAs_Native_oneAs_Introduced[which(Merged$zeroAs_Native_oneAs_Introduced == ("栽培"))] <- as.character("Cultivation")
  model_df<-transform(Merged, 
                      #Serial_Number as numeric
                      Serial_Number=as.numeric(Serial_Number),
                      #Subplot_ID as numeric
                      Subplot_ID=as.numeric(Subplot_ID),
                      #Investigator_ID as factor 
                      Investigator_ID=as.factor(Investigator_ID),
                      #Subplot_x_coor_UTMCS08 keep numeric 
                      #Subplot_x_coor_UTMCS78 keep numeric
                      #Latitude_08 keep numeric
                      #Uploader_ID as factor 
                      Uploader_ID=as.factor(Uploader_ID),
                      #Grid_Number as numeric
                      Grid_Number=as.numeric(Grid_Number),
                      #Sky_Unsobscured_Perecntage keep as numeric 
                      #Data_Error_2 as factor 
                      Data_Error_2=as.factor(Data_Error_2),
                      #Recreation_Areas as factor 
                      Recreation_Areas=as.factor(Recreation_Areas),
                      #Plot_Type as factor
                      Plot_Type=as.factor(Plot_Type),
                      #Site_ID as numeric
                      Site_ID=as.numeric(Site_ID),
                      #Family as factor
                      Family=as.factor(Family),
                      #Deciduousness as factor
                      Deciduousness=as.factor(Deciduousness),
                      #Bud_zeroNo_oneYes factor
                      Bud_zeroNo_oneYes=as.factor(Bud_zeroNo_oneYes),
                      #Seed_zeroNo_oneYes as factor
                      Seed_zeroNo_oneYes=as.factor(Seed_zeroNo_oneYes),
                      #zeroAs_Native_oneAs_Introduced as factor
                      zeroAs_Native_oneAs_Introduced=as.factor(zeroAs_Native_oneAs_Introduced),
                      #Origin keep character
                      Origin=as.factor(Origin),
                      #Plot_ID as numeric
                      Plot_ID=as.numeric(Plot_ID),
                      #Subplot_Type as factor
                      Subplot_Type=as.factor(Subplot_Type),
                      #Data_Recoder_ID as factor
                      Data_Recoder_ID=as.factor(Data_Recoder_ID),
                      #Subplot_y_coor_UTMCS08 keep numeric
                      #Subplot_y_coor_UTMCS78 keep numeric 
                      #Longitude_78 keep numeric 
                      #Date_Uploaded as date might be hard to convert
                      #Aspect keep numeric 
                      #Survay_Year as factor 
                      Survay_Year=as.factor(Survay_Year),
                      #Coastal_Areas as factor 
                      Coastal_Areas=as.factor(Coastal_Areas),
                      #Low_inland_Areas as factor 
                      Low_inland_Areas=as.factor(Low_inland_Areas),
                      #Plot_Mark as factor 
                      Plot_Mark=as.factor(Plot_Mark),
                      #Species_Code_Number as character 
                      Species_Code_Number=as.character(Species_Code_Number),
                      #Genus as factor 
                      Genus=as.factor(Genus),
                      #Scientific_Name keep character
                      #Flower_zeroNo_oneYes as factor
                      Flower_zeroNo_oneYes=as.factor(Flower_zeroNo_oneYes),
                      #Floristic_Region as factor 
                      Floristic_Region=as.factor(Floristic_Region),
                      #World_Weed as factor
                      World_Weed=as.factor(World_Weed),
                      #Habitat_Type as factor
                      Habitat_Type=as.factor(Habitat_Type),
                      #Survary_Date as date hard to convert
                      #Altitude_handheld keep numeric
                      #Data_Examiner_ID as factor 
                      Data_Examiner_ID=as.factor(Data_Examiner_ID),
                      #Longitude_08 keep numeric
                      #Latitude_78 keep numeric
                      #Altitude keep numeric 
                      #Slope keep numeric
                      #Data_Error as factor 
                      Data_Error=as.factor(Data_Error),
                      #Urban_Areas as factor
                      Urban_Areas=as.factor(Urban_Areas),
                      #onMain_offIsland as factor
                      onMain_offIsland=as.factor(onMain_offIsland),
                      #TMZ keep numeric
                      #Growth_Form_Herb_Wood as factor 
                      Growth_Form_Herb_Wood=as.factor(Growth_Form_Herb_Wood),
                      #Type as factor
                      Type=as.factor(Type),
                      #Percent_Cover keep numeric
                      #Fruit_zeroNo_oneYes as factor 
                      Fruit_zeroNo_oneYes=as.factor(Fruit_zeroNo_oneYes),
                      #Growth_Form as factor 
                      Growth_Form=as.factor(LF_Growth_Form)
                      #Usage keep character
  )
  df1 <- model_df[!(model_df$Latitude_08 %in% c(NA)),]
  f_df<-na.omit(df1)
  rm(df1,Merged,model_df)
  
  lowland<-data.frame(f_df[c(44,15,16,52)])
  colnames(lowland)<-(c("Name","Longitude","Latitude","zeroAs_Native_oneAs_Introduced"))
  #Invasives
  Invasives<-lowland[which(lowland$zeroAs_Native_oneAs_Introduced == "1" ),]
  Invasives_List<-as.data.frame(table(Invasives$Name))
  Invasives_List<-(Invasives_List[order(-Invasives_List$Freq),])
  InvList<-Invasives_List[which(Invasives_List$Freq >= 10),]
  
  #Natives
  Natives<-lowland[which(lowland$zeroAs_Native_oneAs_Introduced == "0" ),]
  Natives_List<-as.data.frame(table(Natives$Name))
  Natives_List<-(Natives_List[order(-Natives_List$Freq),])
  NatList<-Natives_List[which(Natives_List$Freq >= 10),]
  


  


  

  
  