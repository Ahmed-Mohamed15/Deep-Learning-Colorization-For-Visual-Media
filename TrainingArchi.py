from Utilities import *

def TriainModel(Input):
    
    global Batch_size

    #region low level Net
    lowLev_layer1 = tf.nn.relu(Conv2d(Input, Low_weights['W_conv1'],2) + Low_biases['b_conv1']) 
    lowLev_layer2 = tf.nn.relu(Conv2d(lowLev_layer1, Low_weights['W_conv2'], 1) + Low_biases['b_conv2']) 
    lowLev_layer3 = tf.nn.relu(Conv2d(lowLev_layer2, Low_weights['W_conv3'], 2) + Low_biases['b_conv3']) 
    lowLev_layer4 = tf.nn.relu(Conv2d(lowLev_layer3, Low_weights['W_conv4'], 1) + Low_biases['b_conv4']) 
    lowLev_layer5 = tf.nn.relu(Conv2d(lowLev_layer4, Low_weights['W_conv5'], 2) + Low_biases['b_conv5']) 
    lowLev_layer6 = tf.nn.relu(Conv2d(lowLev_layer5, Low_weights['W_conv6'], 1) + Low_biases['b_conv6'])  
    #endregion
 
    #region Mid level Net
    MidLev_layer1 = tf.nn.relu(Conv2d(lowLev_layer6, Mid_weights['W_conv1'], 1) + Mid_biases['b_conv1']) 
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
    Fuse = Fusion_layer(MidLev_layer2, ML_Net,Batch_size,224,224)
    #endregion
 
    #region Colorization Net
    Color_layer1 = tf.nn.relu(Conv2d(Fuse, Color_weights['W_conv1'], 1) + Color_biases['b_conv1'])     
 
    Color_layer2 = tf.nn.relu(Conv2d(Color_layer1, Color_weights['W_conv2'], 1) + Color_biases['b_conv2'])     
    Color_layer2_up = tf.image.resize_nearest_neighbor(Color_layer2,[56,56])
 
    Color_layer3 = tf.nn.relu(Conv2d(Color_layer2_up, Color_weights['W_conv3'], 1) + Color_biases['b_conv3']) 
    Color_layer4 = tf.nn.relu(Conv2d(Color_layer3, Color_weights['W_conv4'], 1) + Color_biases['b_conv4']) 
    Color_layer4_up = tf.image.resize_nearest_neighbor(Color_layer4,[112,112])
 
    Color_layer5 = tf.nn.relu(Conv2d(Color_layer4_up, Color_weights['W_conv5'], 1) + Color_biases['b_conv5']) 
    Color_layer6 = tf.nn.sigmoid(Conv2d(Color_layer5, Color_weights['W_conv6'], 1) + Color_biases['b_conv6']) 
 
    Output = tf.image.resize_nearest_neighbor(Color_layer6,[224,224])
 
    #endregion
 
    return Output

def Train():

    global CurrentBatch_indx
    global EpochsNum
    global ExamplesNum
    global Batch_size

    InputImages = tf.placeholder(dtype=tf.float32,shape=[None,224,224,1])
    Ab_LabelsTensor = tf.placeholder(dtype=tf.float32,shape=[None,224,224,2])
    Prediction = TriainModel(InputImages) 
    Colorization_MSE = tf.reduce_mean((Frobenius_Norm(tf.sub(Prediction,Ab_LabelsTensor))))
    Optmizer = tf.train.AdamOptimizer().minimize(Colorization_MSE)
    sess.run(tf.global_variables_initializer())
    
    #saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph('Model Directory/our_model.meta')
    #saver.restore(sess, 'Model Directory/our_model')
    
    PrevLoss = 0
    for epoch in range(EpochsNum):
        epoch_loss = 0
        CurrentBatch_indx = 1
        for i in range(int(ExamplesNum / Batch_size)):#Over batches
           print("Batch Num ",i + 1)
           GreyImages_Batch,AbColores_values = ReadNextBatch()
           a, c = sess.run([Optmizer,Colorization_MSE],feed_dict={InputImages:GreyImages_Batch,Ab_LabelsTensor:AbColores_values})
           epoch_loss += c
        print("epoch: ",epoch + 1, ",Loss: ",epoch_loss,", Diff:",PrevLoss - epoch_loss)
        PrevLoss = epoch_loss
    
    #saver.save(sess, 'Model Directory/our_model',write_meta_graph=False)
