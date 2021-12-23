#!/usr/bin/env python
# coding: utf-8

# # Rhus.succedanea.L. Population Prediction

# We will predict the population density of Rhus.succedanea.L., which is an invasive species in Taiwan. To predict population density, we will count how many were observed at each sampling location. Then using the k-nearest neighbors algorithm we will find the distance and count of each neighboring Rhus.succedanea.L. observation, as well as the distance to nearest roads.

# In[1]:


#For mapping and getting k-lag data
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint , LineString
from shapely.ops import nearest_points
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import folium
from folium import plugins
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
import branca
import branca.colormap as cm


# In[2]:


import random
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfg = tfp.glm
import tensorflow.keras.backend as kb
from keras.callbacks import EarlyStopping


# In[3]:


def create_geo(df, x='Longitude', y='Latitude'):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs={'init':'EPSG:4326'}) #creates a geometry 


# In[4]:


def get_nearest(n_points, candidates, k_neighbors=1):
   
    # Nearest neighbors between points
    tree = BallTree(candidates, leaf_size=15, metric='haversine') #balltree algorthim for finding points
    distances, indices = tree.query(n_points, k=k_neighbors) #gets point and distance

    #Transpose and get indices
    distances = distances.transpose() 
    indices = indices.transpose()
    closest = indices[0]
    closest_dist = distances[0]

    return (closest, closest_dist) 


# In[5]:


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):

    left_geom_col = left_gdf.geometry.name #Main points of interest
    right_geom_col = right_gdf.geometry.name #Points nearby 

    right = right_gdf.copy().reset_index(drop=True) #get right index sequentialy

    #Need to get all points and convert to radians
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    closest, dist = get_nearest(n_points=left_radians, candidates=right_radians) # points closest
    closest_points = right.loc[closest] #returns nearest points

    closest_points = closest_points.reset_index(drop=True) #get left index sequentialy

    #get distance
    if return_dist:
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius #radians to meters

    return closest_points


# In[6]:


#Read in data
fc = pd.read_csv('C:/Users/cjon377/Desktop/Research/tranfer_learning/Forest_data/Forest_Counts.csv',encoding = "ISO-8859-1")
fc


# In[7]:


fc=fc.iloc[:,1:] # Remove index values from file

df = pd.DataFrame(fc['Long_Lat'].str.split(',', expand=True)) #Split Long_Lat into two collumns
df=df.rename(columns={0: "Longitude", 1: "Latitude"}) 
df_g=create_geo(df) #Create a geometry shape for long and lat

fc['Longitude']=df_g['Longitude']
fc['Latitude']=df_g['Latitude']
fc['geometry']=df_g['geometry']


# In[8]:


df_g #Our locations


# In[9]:


x_dist=np.array(fc['geometry'].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list()) #Get distances
tree = BallTree(x_dist, leaf_size=10, metric='haversine')  
dist, ind = tree.query(x_dist, k=10) #knn


# In[10]:


print(ind[:,1])
print(dist[:,1:4]* 6371000)


# In[11]:


lag_1=fc.iloc[ind[:,1]] #first lag data
lag_1


# In[12]:


#Look at the distribtion of the first k=10 neighbor distances 
plt.figure(figsize=(15, 5))
plt.hist(dist[:,1:6]* 6371000,bins=50)
plt.plot()
#notice the distribution


# In[13]:


#If we and to use all the laged points (change length for all)
k_lags=[]
for k in range(1,3):
    
    k_lag=fc.loc[ind[:,k]]
    k_lag=k_lag.iloc[:,1:2546]
    k_lag=np.array(k_lag,dtype='float64')
    k_lags.append(k_lag)
    #np.append(k_lag, k_lag, 1).shape
    
all_lags=np.concatenate(k_lags,1)


# In[14]:


#First lag of Rhus.succedanea.L.
lag=np.array(lag_1.iloc[:,1997],dtype='float64')
lag
X=np.array(dist[:,1]* 6371000)
X2=np.array(dist[:,2]* 6371000)
y=np.array(fc.iloc[:,1997],dtype='float64')

#Look at distance and count
plt.scatter(X,y,alpha=.3)
plt.show()

#Look at lagged count on count
plt.scatter(lag,y,alpha=.3)
plt.show()


# In[15]:


#Get all roads shapefile
roads=gpd.read_file("C:/Users/cjon377/Desktop/Research/tranfer_learning/Factors/Road_longlat.shp")
roads=roads['geometry']
roads


# In[16]:


#Need to pull each point out of linestrings
def coord_lister(geom):
    coords = list(geom.coords)
    return (coords)

coordinates = roads.geometry.apply(coord_lister)


# In[17]:


#Need to flatten the list of arrays and put into data frame
coord=[c for c in coordinates]
c_df = []
for k in range(len(coord)):
    for j in range(len(coord[k])):
        c_df.append(coord[k][j])
    
c_df=pd.DataFrame(c_df) 


# In[18]:


c_df.rename(columns={0: 'Longitude', 1: 'Latitude'}, inplace=True)
c_df #cleaned data


# In[19]:


c_df=create_geo(c_df) #Convert into a new geometry


# In[20]:


#Finally we find the nearest point to each road
closest_roads = nearest_neighbor(df_g,c_df, return_dist=True)
closest_roads


# In[21]:


#Look at counts and distance to roads
plt.hist(closest_roads['distance'])
plt.show()


# In[22]:


#Look at distance and count
plt.scatter(closest_roads['distance'],y,alpha=.3)
plt.show()


# In[23]:


closest_roads = closest_roads.rename(columns={'geometry': 'closest_roads_geom'})
closest_roads 

#Join with plant point locations
df_g=df_g.join(closest_roads,  how='left', lsuffix=['Longitude','Latitude'], rsuffix = ['Longitude','Latitude']) 


# In[24]:


df_g


# In[25]:


# Now we can create a linestring to show graphically how close each point is to each road
df_g['line'] = df_g.apply(lambda row: LineString([row['geometry'], row['closest_roads_geom']]), axis=1)
line_gdf = df_g[['closest_roads_geom', 'line']].set_geometry('line') #creates line
line_gdf.crs = crs={"init":"epsg:4326"} #make sure the coordinate reference systems still matches


# In[26]:


#Create map
mp1 = folium.Map([24.62179, 121.391991], tiles='CartoDb dark_matter')

locs = zip(fc.Latitude, fc.Longitude)
locsr = zip(closest_roads.Latitude, closest_roads.Longitude)

for location in locs:
    folium.CircleMarker(location=location, 
        color='green',   radius=4).add_to(mp1)

    
for location in locsr:
    folium.CircleMarker(location=location, 
              color='red', radius=2).add_to(mp1)

folium.GeoJson(data=line_gdf['line']).add_to(mp1)

mp1.save('map1.html')

mp1


# In[27]:


#Split data for training, validation, and testing
X_train, X_test, y_train, y_test = train_test_split(np.stack((stats.zscore(X),stats.zscore(lag),stats.zscore(closest_roads.iloc[:,3])),axis=1), fc.iloc[:,1997], test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=1)


# In[28]:


#If we want to include enviromental variables
#Split for training and testing
#The size of the training and testing set is spatial dependent, meaning the amount of rows increases/decreases 
#the area and will effect the total pop

'''
env=fc.iloc[:,2488:2546]
data=np.stack((X,X2,closest_roads.iloc[:,3],
               env['X1'],env['X2'], env['X3'], env['X4'],env['X5'], env['X6'], env['X7'],env['X8'],env['X9'],env['X10'],
               env['X11'],env['X12'], env['X13'],env[ 'X14'],env[ 'X15'], env['X16'], env['X17'],env['X18'], env['X19'],
               env['X20'],env['X21'],env['X22'],env['X23'],env['X24'],env['X25'],env['X26'],env['X27'],env['X28'], 
               env['X29'],env['X30'],env['X31'],env['X32'], env['X33'],env['X34'],env['X35'],env['X36'],env['X37'],
               env['X38'],env['X39'],env['X40'],env['X41'], env['X42'], env['X43'],env['X44'],env['X45'],env['X46'],
               env['X47'],env['X48'], env['X49'], env['X50'], env['X51'], env['X52'], env['X53'], env['X54'], env['X55'],
               env['X56'],env['X57'], env['X58']),axis=1)

'''


# # Neural Network Model

# Part 1: We will construct a family of models. Each model will use k-fold cross validation for training. We will stop training when the model does not improve. Each model will construct a probability distribution of counts which are zero inflated. We will use a probabilistic deep learning framework, where the output layer can either be a Negative Binomial or Zero-inflated Poisson model. We will use a Zero-inflated Poisson for now (Negative Binomial coming at a later date). We will then take each sum of the counts from each model output and average them for each model. It is shown that the average prediction is extremely close to the sum of the counts of testing set.   

# In[29]:


def zero_inf(out): 
    rate = tf.squeeze(tf.math.exp(out[:,0:1])) #gets lambda and flattends tensor
    s = tf.math.sigmoid(out[:,1:2]) #zero-infaltion with sigmoid (0,1)
    probs = tf.concat([1-s, s], axis=1) #C get probs of 0 and poisson 
    #get mixture of models
    return tfd.Mixture(
          cat=tfd.Categorical(probs=probs),
          components=[
          tfd.Deterministic(loc=tf.zeros_like(rate)), #zero-inflation model
          tfd.Poisson(rate=rate, force_probs_to_zero_outside_support=True), #poisson model
        ])

def zero_inflated(n_inputs,n_outputs,layer1_size):

    inputs = tf.keras.layers.Input(shape=(n_inputs,)) 
    x = tf.keras.layers.Dense(layer1_size, kernel_initializer='Zeros', activation="relu")(inputs) 
    x = tf.keras.layers.Dense(1000, kernel_initializer='Zeros', activation="relu")(x) 
    x = tf.keras.layers.Dense(1000,  kernel_initializer='Zeros',activation="relu")(x) 

    outputs = tf.keras.layers.Dense(2)(x)
    zi = tfp.layers.DistributionLambda(zero_inf)(outputs) #output to zip model
    model_zi= tf.keras.Model(inputs=inputs, outputs=zi)
    model_zi.compile(loss='Poisson', optimizer='adam',
                    metrics=[tf.keras.metrics.AUC(from_logits=True)]) #using poisson as a loss function
    
    return model_zi


# In[30]:


# k-fold cross-validation
def evaluate_model(X, y,layer1_size,X_val,y_val,x_shape, y_shape):
    results = list()
    n_inputs, n_outputs = x_shape, y_shape
    cv = RepeatedKFold(n_splits=2, n_repeats=20, random_state=1) #cross validate
    models = []
    index = 0
    for train_ix, test_ix in cv.split(X):
      
        X_train, X_test = X[train_ix], X[test_ix] 
        y_train, y_test = y[train_ix], y[test_ix]
       
        model = zero_inflated(n_inputs =  n_inputs, n_outputs =n_outputs ,layer1_size = layer1_size) #get model
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
        model.fit(x=X_train, y=y_train, epochs=10, verbose=1, validation_data=(X_val, y_val),callbacks=[es]) #model fit
   
        index +=1
        print('Model:',index)

        models.append(model) #get family of models
    return models


# In[31]:


model = zero_inflated(n_inputs =  61, n_outputs =1 ,layer1_size = 30)
model.summary()


# In[32]:


models=evaluate_model(X=X_train, y=np.array(y_train,dtype='float64'),layer1_size=30,
                      X_val=X_val,y_val=np.array(y_val,dtype='float64'), x_shape=3, y_shape=1)


# In[33]:


#Look at learning rates of each model
index = 0
for model in range(len(models)):
    index +=1
    print('Model:',index)

    loss_train = models[model].history.history['loss']
    loss_val = models[model].history.history['val_loss']
    epochs = range(0,len(models[model].history.history['loss']))
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# In[34]:


#Model AUC will always be different will you evaluate it.
#So we evaluate each model multple times and get a distriution
AUC_all = []

for i in range(0,30):
    AUC = []
    for model in range(len(models)):
        AUC.append(models[model].evaluate(X_val, y_val, verbose=0)[1])
    AUC_all.append(AUC)


# In[35]:


AUC_means=pd.DataFrame(AUC_all).T.mean(axis=1)
plt.hist(AUC_means)
plt.show()
print(np.mean(AUC_means))


# In[36]:


#get predictions from the family of models
preds = []
for ms in range(len(models)):
    preds.append(models[ms].predict(X_test))


# In[37]:


#Notice they come from the same distribtution 
plt.hist(preds)
plt.show


# In[38]:


#Here we take a bootstraped distribution for the true points
#This is our theoretical distribtution 
samp = []
for r in range(1000):
    np.random.seed(r)
    samp.append(sum(np.random.choice(y_test, y_test.shape[0],replace=True)))
plt.hist(samp)
print(np.mean(samp)) #mean of the theoretical distribution 

#Here is what was observed 
sum(y_test)
#notice the average of the theoretical distribtuion is nearly the same as what was observed


# In[39]:


#We can also see the validation set theoretical distribtution 

samp = []
for r in range(1000):
    np.random.seed(r)
    samp.append(sum(np.random.choice(y_val, y_val.shape[0],replace=True)))
plt.hist(samp)
print(np.mean(samp)) #mean of the theoretical distribtuion

#Here is what was observed 
sum(y_val)


# In[40]:


#Here is the distribution of the testing set
plt.hist(y_train,bins=40)
plt.hist


# In[41]:


#Here we explude values above 1.5 since it has a very low probability of being observed
#We take the mean of each distribution
pops=[]
for p in range(len(preds)):
    pops.append(np.sum(preds[p][preds[p]<=1.5]))


# In[42]:


#Now we get the mean of the means
print(pops)
plt.hist(pops)
print(np.mean(pops))


# Part 2: Putting the models together we find each location has a different probability distribution for counts. We will map the maximum value of each distribution back to the GPS coordinate to visualize the most we would expect to find in a certain area. (The maximum value has a very low probability of being observed, while observing a 0 count is highly probable.) We then place a marker for each location where Rhus.succedanea.L. was found and plot a circle around each point color coded to the true count. Grey points represent training and validation data. 

# In[43]:


#get predicted values from each model and take the average
p_=pd.DataFrame(preds)
plt.hist(p_)
plt.show()
p_


# In[44]:


#Distribtution of max values
p_max=p_.T.max(axis=1)
p_max=np.array(p_max)
plt.hist(p_max)
plt.show()


# In[45]:


#Get locations of points with counts above 0
y_test_locs=pd.concat([fc.iloc[:,2546:2548], y_test[y_test>0]], axis=1)
y_test_locs.dropna(inplace=True)
y_test_locs


# In[46]:


#Get all GPS points of testing set 
p_data=pd.concat([fc.iloc[:,2546:2548], y_test], axis=1)
p_data.dropna(inplace=True)
p_data


# In[47]:


#get color map for predictions
colormap = cm.LinearColormap(colors=['gold','green','blue'], index=[0,np.max(p_max)/2,np.max(p_max)],vmin=0,vmax=np.max(p_max))
colormap


# In[48]:


#color map of true values
colormap2 = cm.LinearColormap(colors=['gold','green','blue','purple'], index=[0,np.max(p_max)/2,np.max(p_max),np.max(y_test)],vmin=0,vmax=np.max(y_test))
colormap2


# In[49]:



mp2 = folium.Map([24.62179, 121.391991], tiles='CartoDb dark_matter')

locs = zip(fc.Latitude, fc.Longitude)

p_locs_max = zip(p_data.Latitude, p_data.Longitude , p_max)



y_t_locs =zip(y_test_locs.Latitude,y_test_locs.Longitude,y_test_locs['Rhus.succedanea.L.'])

for location in locs:
    folium.CircleMarker(location=location, 
        color='grey',   radius=1).add_to(mp2)

    
for location in y_t_locs:
    #icon=folium.Icon(color=colormap(location[2]))
    folium.Marker(location=location[0:2]).add_to(mp2)
    folium.CircleMarker(location=location[0:2],color=colormap2(location[2]), radius=5).add_to(mp2)
    
for location in p_locs_max:
    folium.CircleMarker(location=location[0:2],color=colormap(location[2]), radius=2).add_to(mp2)
    
    

    
mp2.save('map2.html')

mp2

