#!/usr/bin/env python
# coding: utf-8

# In[84]:


pip install labelme tensorflow tensorflow-gpu opencv-python matplotlib albumentations 


# In[2]:


import os
import time
import uuid
import cv2


# In[3]:


uuid.uuid1() #unique id to be given to images using this


# In[4]:


IMAGES_PATH=os.path.join('data','images')
number_images=30 #raw data,and labels would be folder where data would be unpartitioned and then used to create folders for train test and validate data


# In[5]:


cap=cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret,frame=cap.read()
    imgname=os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname,frame)
    cv2.imshow('frame',frame)
    time.sleep(0.5) #this will give us time to move around cause we also have to feed the model with data where our faace is not visible
                         
    if cv2.waitKey(1) & 0xFF == ord('q'):
                         break
cap.release()
cv2.destroyAllWindows()
# collected 239 different images
                         


# In[6]:


pip install labelme


# In[18]:


get_ipython().system('labelme')


# In[35]:



import tensorflow as tf #use to build pipeline and then build our deep learning model it takes sometime to load
import cv2
import json #as our labels are in json format
import numpy as np #help in preprocessing
import matplotlib.pyplot as plt #this was running with matplotlib.pyplot and not with from matplotlib import pyplot


# In[36]:


#to ensure not to get too many errors while running tensorflow we use the below code to limit GPU Memory Growth
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


# In[38]:


#to test if our gpu is available or not mine was not available for time being
tf.config.list_physical_devices('GPU')


# In[40]:


#loading our images in tensorflow data pipeline
images=tf.data.Dataset.list_files('data\\images\\*.jpg',shuffle=False) #'*' means we look for everything that has .jpg extension that is looking for all our images


# In[41]:


#checking if tensorflow is picking up our images
images.as_numpy_iterator().next()
# this shows the exact path to the images


# In[42]:


#now to load images one by one from the folder
def load_image(x): #x is full file path
    byte_img=tf.io.read_file(x) #byte encoded image would be obtained
    img=tf.io.decode_jpeg(byte_img) #this will then be decoded and returned
    return img


# In[43]:


images=images.map(load_image) #fucntion is applied on each image in the image file path


# In[44]:


images.as_numpy_iterator().next() #converted the images into numpy array


# In[45]:


type(images) #this shows this a tensorflow data pipeline


# In[172]:


#to view data
image_generator = images.batch(4).as_numpy_iterator() #combining 4 images as one instead of doing one by one in order to increase the speed


# In[61]:


plot_images=image_generator.next() #each time we run this pipeline we get a new batch of four images


# In[63]:


fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx,image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()


# In[67]:


#partitioning unaugmented data
239*0.7 #for training


# In[68]:


239*0.15 #for val and test each 15 percent
#moving all those images manually 


# In[71]:


#moving the matching labels
for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data',folder,'images')):
        filename=file.split('.')[0]+'.json'
        existing_filepath=os.path.join('data','labels',filename)
        if os.path.exists(existing_filepath):
            new_filepath=os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath,new_filepath)


# In[92]:


import albumentations as alb
# error was coming that the module cant found but I solved by shifting the location of those modules to anaconda3/lib/site-packages


# In[93]:


#defining augmentation pipeline
augmentor=alb.Compose([alb.RandomCrop(width=450,height=450),
                       alb.HorizontalFlip(p=0.5),
                       alb.RandomGamma(p=0.2),
                       alb.RandomBrightnessContrast(p=0.2),
                       alb.RGBShift(p=0.2),
                      alb.VerticalFlip(p=0.5),
                      ],bbox_params=alb.BboxParams(format='albumentations',label_fields=['class_labels']))
                        #bbox parameters chosen here is albumentation notation that is scaling each points there are other notations also like yolo or coco or pascal voc that can help in changing the bounding box parameters


# In[103]:


#loading test image and annotation using opencv and json
img=cv2.imread(os.path.join('data','train','images','2aaf1f91-01d7-11ee-be23-94e23ca46077.jpg'))


# In[104]:


with open(os.path.join('data','train','labels','2aaf1f91-01d7-11ee-be23-94e23ca46077.json'),'r') as f:
    label=json.load(f)


# In[107]:


type(label)
#thus we can grab the values from the keys


# In[108]:


label


# In[352]:


label['shapes'][0]['points']


# In[353]:


#extracting coordinates and rescale to match image Resolution basically transforming coordinates into a single vector
coords=[0,0,0,0]
coords[0]=label['shapes'][0]['points'][0][0]
coords[1]=label['shapes'][0]['points'][0][1]
coords[2]=label['shapes'][0]['points'][1][0]
coords[3]=label['shapes'][0]['points'][1][1]


# In[354]:


coords


# In[355]:


#rescaling using the following method to match image resolution
coords=list(np.divide(coords,[640,480,640,480])) #this is albumentation format


# In[356]:


coords


# In[357]:


#Applying augmentation
augmented=augmentor(image=img,bboxes=[coords],class_labels=['face'])


# In[360]:


augmented['bboxes']


# In[120]:


augmented.keys()


# In[173]:


augmented['image']


# In[123]:


augmented['image'].shape


# In[363]:


#to draw boxes around the face
cv2.rectangle(augmented['image'],
             tuple(np.multiply(augmented['bboxes'][0][:2],[450,450]).astype(int)),       #grabbing first two values for first corner point
             tuple(np.multiply(augmented['bboxes'][0][2:],[450,450]).astype(int)),
                  (0,255,0),2)                                                           #grabbing last two values for second corner point
plt.imshow(augmented['image'])

#Also we will apply this pipeline to all images
#bounding box is drawn open cv reads color as BGR and matplotlib show in RGB


# In[138]:


#Now we will run this pipeline over all test train and val sets
for partition in ['train','test','val']:
    for image in os.listdir(os.path.join('data',partition,'images')):
        img=cv2.imread(os.path.join('data',partition,'images',image))
        
        coords=[0,0,0.00001,0.00001]
        label_path=os.path.join('data',partition,'labels',f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path,'r') as f:
                label=json.load(f)
                
            coords[0]=label['shapes'][0]['points'][0][0]
            coords[1]=label['shapes'][0]['points'][0][1]
            coords[2]=label['shapes'][0]['points'][1][0]
            coords[3]=label['shapes'][0]['points'][1][1]
            coords=list(np.divide(coords,[640,480,640,480]))
            
        try:
            for x in range(60): #we would be running each image with its 60 images variation thus total images become 239*60
                augmented=augmentor(image=img,bboxes=[coords],class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data',partition,'images',f'{image.split(".")[0]}.{x}.jpg'),augmented['image'])
                
                annotation={}
                annotation['image']=image
                
                if os.path.exists(label_path):
                    if len(augmented['bboxes'])==0:
                        
                        annotation['bbox']=[0,0,0,0]
                        annotation['class']=0
                        
                    
                    else:
                        
                        annotation['bbox']=augmented['bboxes'][0]
                        annotation['class']=1
                else:
                    annotation['bbox']=[0,0,0,0]
                    annotation['class']=0
                    
                with open(os.path.join('aug_data',partition,'labels',f'{image.split(".")[0]}.json'),'w') as f:
                    json.dump(annotation,f)
        
        except Exception as e: #to print out any error if occurs
            print(e)
                    
                
                
            


# In[364]:


coords


# In[388]:


#loading augmented images to tensorflow dataset
train_images=tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg',shuffle=False) #shuffle is false cause we are going to load label in the order as of images
train_images=train_images.map(load_image)
train_images=train_images.map(lambda x:tf.image.resize(x,(120,120))) #compressing the images more so that more efficient neural network 
train_images=train_images.map(lambda x:x/255) #also we are scaling the images down so that the value or color of images is between 0 and 1 and hence sigmoid activation fucntion can be applied


# In[389]:


test_images=tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg',shuffle=False) #shuffle is false cause we are going to load label in the order as of images
test_images=test_images.map(load_image)
test_images=test_images.map(lambda x:tf.image.resize(x,(120,120))) #compressing the images more so that more efficient neural network 
test_images=test_images.map(lambda x:x/255) #also we are scaling the images down so that the value or color of images is between 0 and 1 and hence sigmoid activation fucntion can be applied


# In[390]:


val_images=tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg',shuffle=False) #shuffle is false cause we are going to load label in the order as of images
val_images=val_images.map(load_image)
val_images=val_images.map(lambda x:tf.image.resize(x,(120,120))) #compressing the images more so that more efficient neural network 
val_images=val_images.map(lambda x:x/255) #also we are scaling the images down so that the value or color of images is between 0 and 1 and hence sigmoid activation fucntion can be applied


# In[391]:


train_images.as_numpy_iterator().next() #this is scaled down so that to sent the data to sigmoid activation function


# In[392]:


coords


# In[393]:


#Preparing label
#label load fucntion
def load_labels(label_path):
    with open(label_path.numpy(),'r',encoding="utf-8") as f: #numpy function to grab those labels
        label=json.load(f)
        
    return [label['class']],label['bbox'] #retrurn the class in form of array


# In[394]:


train_labels=tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json',shuffle=False)
train_labels=train_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))


# In[369]:


train_labels.as_numpy_iterator().next()


# In[395]:


test_labels=tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json',shuffle=False)
test_labels=test_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))


# In[396]:


val_labels=tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json',shuffle=False)
val_labels=val_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))


# In[397]:


#Combining labels and image smaples
len(train_images),len(train_labels),len(test_images),len(test_labels),len(val_images),len(val_labels)


# In[398]:


coords


# In[399]:


#creating new final dataset[images/labels]
train=tf.data.Dataset.zip((train_images,train_labels))
train=train.shuffle(5000)
train=train.batch(8) #each batch would be representing 8 images and 8 labels
train=train.prefetch(4) #eliminate bottlenecks when loading and training data in neural network


# In[532]:


test=tf.data.Dataset.zip((test_images,test_labels)) #zip generator basically combine those images with labels
test=test.shuffle(1300)
test=test.batch(8)
test=test.prefetch(4)


# In[533]:


test.as_numpy_iterator().next()


# In[401]:


val=tf.data.Dataset.zip((val_images,val_labels))
val=val.shuffle(5000)
val=val.batch(8)
val=val.prefetch(4)


# In[402]:


data_samples=train.as_numpy_iterator()


# In[403]:


res= data_samples.next()


# In[492]:


res[1][0]


# In[557]:


fig, ax=plt.subplots(ncols=4,figsize=(20,20))
for idx in range(4):
    sample_image=res[0][idx]
    sample_coords=res[1][1][idx]
    
    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2],[120,120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:],[120,120]).astype(int)),
                      (0,255,0),2)

    ax[idx].imshow(sample_image)


    
   


# In[405]:


coords


# In[406]:


#building deep learning model using functional api
#import layer and base network

from tensorflow import keras 

#from tensorflow.keras.models import Model

#from tensorflow.python.keras.models import Input

#from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Activation, Flatten #dont include keras library directly import it through tensorflow

#from tensorflow.keras.applications import VGG16

#tensorflow.keras is a legacy now


# In[407]:


#download VGG 16
#how to determine which neural network or how many units should be used for that go through research papers and all
vgg=keras.applications.VGG16(include_top=False)#we dont need top layer as VGG 16 is a classification model but we need both classification as well as regression model


# In[408]:


vgg.summary() #we will be having our parameters i.e. channels, width and height


# In[513]:


#building our neural network
#first step is to define single input or multiple inputs and  then have a single output or bunch of output
def build_model():
    input_layer=keras.layers.Input(shape=(120,120,3))

    vgg=keras.applications.VGG16(include_top=False)(input_layer) #intialising vgg as well as passing input
    #we get two prediciton head here

    #Classification model

    f1=keras.layers.GlobalMaxPooling2D()(vgg)
    class1=keras.layers.Dense(2048,activation='relu')(f1)#we are first condensing all the output we are getting from vgg so instead off having 512 layers we will be taking max to get 512 values
    class2=keras.layers.Dense(1,activation='sigmoid')(class1)

    #Regression Model or bounding box model

    f2=keras.layers.GlobalMaxPooling2D()(vgg)
    regress1=keras.layers.Dense(2048,activation='relu')(f2)
    regress2=keras.layers.Dense(4,activation='sigmoid')(regress1) #4 neurons should be there for 4 coordinates silly msitake

    facetracker=keras.models.Model(inputs=input_layer,outputs=[class2,regress2]) #these output match with those we have create training data bith class value is and four coordinate values
    return facetracker

    


# In[420]:


facetracker=build_model() #intialising the model


# In[421]:


coords


# In[422]:


facetracker.summary()


# In[423]:


X,y=train.as_numpy_iterator().next() #X would be images y would be labels, unpacking of training set


# In[424]:


X.shape #(batches of 8)


# In[425]:


y[1]


# In[428]:


coords


# In[427]:


#now model will predict
classes,coords=facetracker.predict(X)


# In[418]:


classes,coords #now we have to test how accurate is this


# In[289]:


len(train)


# In[429]:


#defining losses and optimizer
batches_per_epoch=len(train)
#epoch means how many batches it should cover in one go throught the model
#we should have learning decay rate so that the model doesnt overfit or overuse the gradient
lr_decay=(1./0.75-1)/batches_per_epoch
#learning rate decay determines how much learning should be descreased each time we go through the epoch


# In[430]:


lr_decay


# In[318]:


#optimizer used would be Adam which would help in adjusting hyperparameters and backpropogate it then

opt=keras.optimizers.legacy.Adam(learning_rate=0.0001,decay=lr_decay)


# In[431]:


#def localization loss
import tensorflow
def localization_loss(y_true,yhat):
    delta_coords=tensorflow.reduce_sum(tensorflow.square(y_true[:,:2]-yhat[:,:2]))
    
    h_true=y_true[:,3]-y_true[:,1]
    w_true=y_true[:,2]-y_true[:,0]
    
    h_pred=yhat[:,3]-yhat[:,1]
    w_pred=yhat[:,2]-yhat[:,0]
    
    delta_size=tensorflow.reduce_sum(tensorflow.square(w_true-w_pred)+tensorflow.square(h_true-h_pred))
    
    return delta_coords+delta_size


# In[432]:


classloss=keras.losses.BinaryCrossentropy()
regressloss=localization_loss


# In[433]:


localization_loss(y[1],coords)


# In[434]:


classloss(y[0],classes)


# In[474]:


#Train Neural Network
#creating custom model class
class FaceTracker(Model):
    def __init__(self,eyetracker,**kwargs):   # init__ method is used to pass initial parameters,here we are passing initial model facetracker  
        super().__init__(**kwargs)
        self.model=eyetracker
        
    def compile(self,opt,classloss,localizationloss,**kwargs): #whenever we create keras model during compiling we pass optimizer and loss function
        super().compile(**kwargs)
        self.closs=classloss            #here we are setting up the losses in a class
        self.lloss=localizationloss
        self.opt=opt
        
    def train_step(self,batch,**kwargs):  #here we train our neural network, it will take one batch of data and then train on it
        X,y=batch
        
        with tensorflow.GradientTape() as tape:  
            classes,coords=self.model(X,training=True)  #here self.moddel is our original facretack model we are activating the training model of neural layers if any one of them requires it by inference
            
            batch_classloss=self.closs(y[0],classes)
            batch_localizationloss=self.lloss(tensorflow.cast(y[1],tensorflow.float32),coords) #casting the values so that the loss function can work properly
            
            total_loss=batch_localizationloss+0.5*batch_classloss #50 percent of classloss is randomly chosen
            
            grad=tape.gradient(total_loss,self.model.trainable_variables) #this step is important when we are using a custom training step we have to calculate the gradient with respect to our loss function
            
        opt.apply_gradients(zip(grad,self.model.trainable_variables))  #here we are applying gradient descent that is we are optimizing closer towards minimizing the loss, this is basically backpropagating
        
        return{"total_loss":total_loss,"class_loss":batch_classloss,"regress_loss":batch_localizationloss} #dict is obtained done to save the progress
    
    def test_step(self,batch,**kwargs): #in test step this would be activated when we pass validation dataset and here we wont be backpropagating like above
        X,y=batch
        
        classes,coords=self.model(X,training=True)
        
        batch_classloss=self.closs(y[0],classes)
        batch_localizationloss=self.lloss(tensorflow.cast(y[1],tensorflow.float32),coords)
        total_loss=batch_localizationloss+0.5*batch_classloss
        
        
        return{"total_loss":total_loss,"class_loss":batch_classloss,"regress_loss":batch_localizationloss}
    
    def call(self,X,**kwargs): #if we use .predict we have to include this
        return self.model(X,**kwargs)
        
        
            
            


# In[475]:


model= FaceTracker(facetracker)


# In[476]:


model.compile(opt,classloss,regressloss) #model is compiled successfully


# In[477]:


#Now we will train
logdir='logs' #specifying our log dir here our tensorboard model will log out to


# In[478]:


tensorflow_callback=tensorflow.keras.callbacks.TensorBoard(log_dir=logdir) #to review our model performance


# In[479]:


hist=model.fit(train,epochs=40,validation_data=val,callbacks=[tensorflow_callback]) #here model.fit would trigger our train step above and when we pass our validation data it will trigger our test step


# In[483]:


hist.history #performance of the model


# In[484]:


#plotting the performance
fig, ax = plt.subplots(ncols=3, figsize=(20,5))
ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()
plt.show()

#problem is with the no of train label was 160 and no of train images was 9600


# In[486]:


#making prediction on our test data set now
test_data=test.as_numpy_iterator() #starting header of test_data


# In[522]:


test


# In[490]:


test_sample=test_data.next() #now taking batch of 8 images one by one and give values of coordinates


# In[536]:


test_sample


# In[519]:


yhat[0][0]


# In[491]:


yhat=facetracker.predict(test_sample[0]) #now we are predicting the values one by one


# In[587]:


fig,ax = plt.subplots(ncols=4,figsize=(20,20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]
    if yhat[0][idx] > 0.5:
        cv2.rectangle(sample_image,
                       tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                               (255,0,0), 2)
    ax[idx].imshow(sample_image)
    
plt.show()


# In[589]:


#loading and saving our model in tensorflow
from tensorflow.keras.models import load_model


# In[590]:


facetracker.save('facetracker.h5')


# In[591]:


#once saved the model can be loaded now
facetracker=load_model('facetracker.h5')


# Real Time Detection

# In[597]:


cap=cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame=cap.read()
    frame=frame[50:500,50:500,:]
    
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #since tensorflow can only read in rgb while opencv reads in bgr
    resized=tensorflow.image.resize(rgb,(120,120))
    
    yhat=facetracker.predict(np.expand_dims(resized/255,0))
    
    sample_coords=yhat[1][0]
    
    if yhat[0]>0.5:
        #"controls the main rectangle"
        cv2.rectangle(frame,tuple(np.multiply(sample_coords[:2],[450,450]).astype(int)),
                     tuple(np.multiply(sample_coords[2:],[450,450]).astype(int)),
                     (0,255,0),2)
        #controls the label rectangle
        cv2.rectangle(frame,tuple(np.add(np.multiply(sample_coords[:2],[450,450]).astype(int),
                      [0,-30])),
                     tuple(np.add(np.multiply(sample_coords[2:],[450,450]).astype(int),
                      [80,0])),
                     (0,255,0),-1)
        
        #controls the text rendered
        
        cv2.putText(frame,'Rounak Raman',tuple(np.add(np.multiply(sample_coords[:2],[450,450]).astype(int),
                                              [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        
    cv2.imshow('FaceTrack',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break
        
cap.release()
    
cv2.destroyALLWindows()
        
        
        
        
    

