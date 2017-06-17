from Utilities import *

def TestModel(Original,Rescaled,H,W):
   
    #region low level Net
    lowLev_layer1 = tf.nn.relu(Conv2d(Rescaled, Low_weights['W_conv1'],2) + Low_biases['b_conv1']) 
    lowLev_layer2 = tf.nn.relu(Conv2d(lowLev_layer1, Low_weights['W_conv2'], 1) + Low_biases['b_conv2']) 
    lowLev_layer3 = tf.nn.relu(Conv2d(lowLev_layer2, Low_weights['W_conv3'], 2) + Low_biases['b_conv3']) 
    lowLev_layer4 = tf.nn.relu(Conv2d(lowLev_layer3, Low_weights['W_conv4'], 1) + Low_biases['b_conv4']) 
    lowLev_layer5 = tf.nn.relu(Conv2d(lowLev_layer4, Low_weights['W_conv5'], 2) + Low_biases['b_conv5']) 
    lowLev_layer6 = tf.nn.relu(Conv2d(lowLev_layer5, Low_weights['W_conv6'], 1) + Low_biases['b_conv6']) 
    #endregion
 
    #region low level Net
    lowLev2_layer1 = tf.nn.relu(Conv2d(Original, Low_weights['W_conv1'],2) + Low_biases['b_conv1']) 
    lowLev2_layer2 = tf.nn.relu(Conv2d(lowLev2_layer1, Low_weights['W_conv2'], 1) + Low_biases['b_conv2']) 
    lowLev2_layer3 = tf.nn.relu(Conv2d(lowLev2_layer2, Low_weights['W_conv3'], 2) + Low_biases['b_conv3']) 
    lowLev2_layer4 = tf.nn.relu(Conv2d(lowLev2_layer3, Low_weights['W_conv4'], 1) + Low_biases['b_conv4']) 
    lowLev2_layer5 = tf.nn.relu(Conv2d(lowLev2_layer4, Low_weights['W_conv5'], 2) + Low_biases['b_conv5']) 
    lowLev2_layer6 = tf.nn.relu(Conv2d(lowLev2_layer5, Low_weights['W_conv6'], 1) + Low_biases['b_conv6'])  
    #endregion
 
    #region Mid level Net
    MidLev_layer1 = tf.nn.relu(Conv2d(lowLev2_layer6, Mid_weights['W_conv1'], 1) + Mid_biases['b_conv1']) 
    MidLev_layer2 = tf.nn.relu(Conv2d(MidLev_layer1, Mid_weights['W_conv2'], 1) + Mid_biases['b_conv2']) 
    #endregion
 
    #region Global level Net
    GlobalLev_layer1 = tf.nn.relu(Conv2d(lowLev_layer6, Global_weights['W_conv1'], 2) + Global_biases['b_conv1']) 
    GlobalLev_layer2 = tf.nn.relu(Conv2d(GlobalLev_layer1, Global_weights['W_conv2'], 1) + Global_biases['b_conv2']) 
    GlobalLev_layer3 = tf.nn.relu(Conv2d(GlobalLev_layer2, Global_weights['W_conv3'], 2) + Global_biases['b_conv3']) 
    GlobalLev_layer4 = tf.nn.relu(Conv2d(GlobalLev_layer3, Global_weights['W_conv4'], 1) + Global_biases['b_conv4'])  
    #endregion
 
    #region ML Net
    ML_Net = ConstructML(GlobalLev_layer4,3,[1024,512,256])
    #endregion
 
    #region Fusion Layer
    Fuse = Fusion_layer(MidLev_layer2, ML_Net,1,H,W)
    #endregion
 
    #region Colorization Net
    H = tf.cast((H / 8), tf.int32) 
    W = tf.cast((W / 8), tf.int32) 
    
    Color_layer1 = tf.nn.relu(Conv2d(Fuse, Color_weights['W_conv1'], 1) + Color_biases['b_conv1'])     
 
    Color_layer2 = tf.nn.relu(Conv2d(Color_layer1, Color_weights['W_conv2'], 1) + Color_biases['b_conv2'])     
    Color_layer2_up = tf.image.resize_nearest_neighbor(Color_layer2,[H * 2,W * 2])
 
    Color_layer3 = tf.nn.relu(Conv2d(Color_layer2_up, Color_weights['W_conv3'], 1) + Color_biases['b_conv3']) 
    Color_layer4 = tf.nn.relu(Conv2d(Color_layer3, Color_weights['W_conv4'], 1) + Color_biases['b_conv4']) 
    Color_layer4_up = tf.image.resize_nearest_neighbor(Color_layer4,[H * 4,W * 4])
 
    Color_layer5 = tf.nn.relu(Conv2d(Color_layer4_up, Color_weights['W_conv5'], 1) + Color_biases['b_conv5']) 
    Color_layer6 = tf.nn.sigmoid(Conv2d(Color_layer5, Color_weights['W_conv6'], 1) + Color_biases['b_conv6']) 
 
    Output = tf.image.resize_nearest_neighbor(Color_layer6,[H * 8,W * 8])
 
 
    #endregion
  
    return Output

def TestingImage(TestImageName,FirstRun):
    
    if(FirstRun == False):
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('Model Directory/our_model.meta')
        saver.restore(sess, 'Model Directory/our_model')

    RescaledImageBatch = []
    OriginalImageBatch = []

    OriginalImage = Image.open(TestingImgPath + TestImageName).convert('RGB').convert('L')  
    width,height = OriginalImage.size
    OriginalImage = OriginalImage.resize((int(width / 8) * 8,int(height / 8) * 8),Image.ANTIALIAS)      
    RescaledImage = OriginalImage.resize((224,224),Image.ANTIALIAS)      

    OriginalImage = np.asanyarray(OriginalImage) 
    RescaledImage = np.asanyarray(RescaledImage) 

    ImageShape = OriginalImage.shape
    OriginalImage = OriginalImage.reshape(ImageShape[0],ImageShape[1], GreyChannels)#[H,W,1]
    OriginalImageBatch.append(OriginalImage)#[#imgs,224,224,1]
    
    RescaledImage = RescaledImage.reshape(224, 224, GreyChannels)#[224,224,1]
    RescaledImageBatch.append(RescaledImage)#[#imgs,224,224,1]

    Rescaled = tf.placeholder(dtype=tf.float32,shape=[1,224,224,1])
    original = tf.placeholder(dtype=tf.float32,shape=[1,None,None,1])
    
    Prediction = TestModel(original,Rescaled,OriginalImage.shape[0],OriginalImage.shape[1]) 
    Chrominance = sess.run(Prediction,feed_dict={Rescaled:RescaledImageBatch,original:OriginalImageBatch})

    ColoredImage = np.empty((OriginalImage.shape[0],OriginalImage.shape[1],3))
    for i in range(len(OriginalImage[:,1,0])):
      for j in range(len(OriginalImage[1,:,0])):
         ColoredImage[i,j,0] = 0 + ((OriginalImage[i,j,0] - 0) * (100 - 0) / (255 - 0))  
    ColoredImage[:,:,1] = DeNormalize(Chrominance[0,:,:,0],0,1)
    ColoredImage[:,:,2] = DeNormalize(Chrominance[0,:,:,1],0,1)
    ColoredImage = color.lab2rgb(ColoredImage)
    plt.imsave(TestingImgPath + TestImageName[0:-4] + "_Colored" + TestImageName[len(TestImageName) - 4:],ColoredImage)