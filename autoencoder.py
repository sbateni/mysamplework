##python version:3.6
##author: Ehsan Bateni
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D


from sklearn import metrics
from sklearn.cluster import KMeans

from keras.datasets import mnist






def CAE():

    input_shape=(28,28,1)
    K=[32, 64, 128, 10]


    model = Sequential()

    model.add(Conv2D(K[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(K[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(K[2], 3, strides=2, padding='valid', activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=K[3], name='latent'))
    model.add(Dense(units=K[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), K[2])))
    model.add(Conv2DTranspose(K[1], 3, strides=2, padding='valid', activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(K[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    
    return model

def ClusterUpdate(input_array,n_clusters):
     kmeans = KMeans(n_clusters=n_clusters,init='k-means++').fit(input_array)
     centroids=kmeans.cluster_centers_
     labels=kmeans.labels_
     centroids_extent=centroids[labels]
     return centroids_extent, labels

def main():
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (len(x_train),28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test),28, 28, 1))
    X=x_train
    batch_size=256
    NMI=[]
    n_clusters=10
    n_epochs = 50
    n_epochs_pretrain = 5
    
   
    
    #pretrain cnn with reconstruction loss only
    model=CAE() 
    model.compile(loss='mse', optimizer='adam') 
    model.fit(X, X, batch_size=batch_size, epochs=n_epochs_pretrain)

    #define clustering model with two losses
    output1= model.get_layer(name='latent').output
    output2= model.get_layer(name='deconv1').output
    clustering_model=Model(inputs=model.input, outputs=[output1, output2])
    clustering_model.compile(loss=['mse', 'mse'],loss_weights=[1, 1], optimizer='adam') 
    
    #train cnn with optimizing clustering and reconstruction loss
    n_batches = int(X.shape[0] / batch_size)
    
    for epoch in range(n_epochs):
            
       
       for i in range(n_batches):
           #updating clusters with the whole data 
           Z, _ =clustering_model.predict(X, verbose=0)
           centroids, y_pred=ClusterUpdate (Z,n_clusters)
           NMI.append(metrics.normalized_mutual_info_score(y_train, y_pred))  
           
           batch_start, batch_end=i * batch_size,(i + 1) * batch_size
           if i == n_batches-1:
               batch_end=X.shape[0]-1
           #updating network batch to batch
           loss = clustering_model.train_on_batch(x=X[batch_start:batch_end],
                                                     y=[centroids[batch_start:batch_end], X[batch_start:batch_end]]) 
    
       clustering_model.save_weights('model' + '/deep_clustering_model_' + str(epoch) + '.h5') 
   
    Z, _ =clustering_model.predict(X, verbose=0)
    centroids, y_pred=ClusterUpdate (Z,n_clusters)
    NMI.append(metrics.normalized_mutual_info_score(y_train, y_pred))  
    print(NMI)    

               
    
   
if __name__ == '__main__':
    main()
     




