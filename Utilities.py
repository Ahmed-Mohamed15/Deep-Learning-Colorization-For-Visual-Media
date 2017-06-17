import tensorflow as tf
import glob
from PIL import Image
import numpy
import numpy as np
import math
import os
from skimage import io, color
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy import misc

sess = tf.InteractiveSession() 

#region Intialize Data and Layers weights

ColorImgsPath = os.path.realpath(__file__[0:-16])+"/" # replace This Path with  colored images dataset path
TestingImgPath =os.path.realpath(__file__[0:-16])+"/"

AbColores_values = None
GreyImages_Batch = [] 
ColorImages_Batch = []
Batch_size = 1
CurrentBatch_indx = 1
EpochsNum = 100
ExamplesNum = 4    # Number of all Images in Db Dir
Imgsize = 224, 224 
GreyChannels = 1
Fusion_output = None
 
Low_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,1,64], stddev=0.001),name="Low1"),
               'W_conv2':tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.001),name="Low2"),
               'W_conv3':tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.001),name="Low3"),
               'W_conv4':tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.001),name="Low4"), 
               'W_conv5':tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.001),name="Low5"),
               'W_conv6':tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.001),name="Low6")}
 
Low_biases = {'b_conv1':tf.Variable(tf.truncated_normal([64], stddev=0.001)),
              'b_conv2':tf.Variable(tf.truncated_normal([128], stddev=0.001)),
              'b_conv3':tf.Variable(tf.truncated_normal([128], stddev=0.001)),
              'b_conv4':tf.Variable(tf.truncated_normal([256], stddev=0.001)),
              'b_conv5':tf.Variable(tf.truncated_normal([256], stddev=0.001)),
              'b_conv6':tf.Variable(tf.truncated_normal([512], stddev=0.001))}
 
Mid_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
               'W_conv2':tf.Variable(tf.truncated_normal([3,3,512,256], stddev=0.001))}
 
Mid_biases = {'b_conv1':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
              'b_conv2':tf.Variable(tf.truncated_normal([256], stddev=0.001))}
 
Global_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                  'W_conv2':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                  'W_conv3':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                  'W_conv4':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001))}
 
Global_biases = {'b_conv1':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                 'b_conv2':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                 'b_conv3':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                 'b_conv4':tf.Variable(tf.truncated_normal([512], stddev=0.001))}
 
Color_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,256], stddev=0.001)),
                 'W_conv2':tf.Variable(tf.truncated_normal([3,3,256,128], stddev=0.001)),
                 'W_conv3':tf.Variable(tf.truncated_normal([3,3,128,64], stddev=0.001)),
                 'W_conv4':tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.001)),
                 'W_conv5':tf.Variable(tf.truncated_normal([3,3,64,32], stddev=0.001)),
                 'W_conv6':tf.Variable(tf.truncated_normal([3,3,32,2], stddev=0.001))}
 
Color_biases = {'b_conv1':tf.Variable(tf.truncated_normal([256], stddev=0.001)),
                    'b_conv2':tf.Variable(tf.truncated_normal([128], stddev=0.001)),
                    'b_conv3':tf.Variable(tf.truncated_normal([64], stddev=0.001)),
                    'b_conv4':tf.Variable(tf.truncated_normal([64], stddev=0.001)),
                    'b_conv5':tf.Variable(tf.truncated_normal([32], stddev=0.001)),
                    'b_conv6':tf.Variable(tf.truncated_normal([2], stddev=0.001))}    
  
MLnode_for_each_layer = [1024, 512, 256]
MLHidden_layer = []
MLHidden_layer.append({'weights': tf.Variable(tf.truncated_normal([7 * 7 * 512, MLnode_for_each_layer[0]], stddev=0.001)),
                     'biases': tf.Variable(tf.truncated_normal([MLnode_for_each_layer[0]],stddev=0.001))})
for i in range(1, 3):
    MLHidden_layer.append({'weights': tf.Variable(tf.truncated_normal([MLnode_for_each_layer[i - 1], MLnode_for_each_layer[i]], stddev=0.001)),
 
                         'biases': tf.Variable(tf.truncated_normal([MLnode_for_each_layer[i]], stddev=0.001))})
 
 #endregion

def Fusion_layer(MiddNetOutput, GlobalNetOutput,BatchSize,H,W):
    """A network that fuses the output of midd Net  and output of MLP(Global) Net
    together.
    Args:
    MiddNetOutput: Size of [?, H/8, W/8, 256].
    GlobalNetOutput: Size of [?,256]
    """ 
    H = tf.cast((H / 8), tf.int32) 
    W = tf.cast((W / 8), tf.int32) 
   
    GlobalNetOutput = tf.tile(GlobalNetOutput,[1,H * W])
    GlobalNetOutput = tf.reshape(GlobalNetOutput,[Batch_size,H,W,256])
    Fusion_output = tf.concat(3,[MiddNetOutput,GlobalNetOutput])
    return Fusion_output  

def ConstructML(input_tensor, layers_count, node_for_each_layer):
    """A fully connected Network MLP connected to Global Feature outputs
    Args:
    input_tensor : the output of global feature , shape = [ 1, 7, 7, 512]
    layer_count : number of layer in MLP actually = 3 or more
    node_for_each_layer : a list contains number of nodes in each layer [ 1024, 512, 256]
    """
    global   ML_OUTPUT 
 
    FeatureVector = tf.reshape(input_tensor,shape=[-1,7 * 7 * 512])
    layers_output = tf.add(tf.matmul(FeatureVector, MLHidden_layer[0]['weights']), MLHidden_layer[0]['biases'])
    layers_output = tf.nn.relu(layers_output)
 
    for j in range(1, layers_count):
        layers_output = tf.add(tf.matmul(layers_output, MLHidden_layer[j]['weights']), MLHidden_layer[j]['biases'])
        layers_output = tf.nn.relu(layers_output)
 
    return layers_output

def Conv2d(inp, W ,Stride):
   """Computes a 2-D convolution over a 4-D input
    Args:
    inp: 4-D tensor
    W: Kernel
    Stride: The kenrel stride over the input
    """
   return tf.nn.conv2d(inp, W, strides=[1, Stride ,Stride, 1], padding='SAME')

def ReadNextBatch(): 
    '''Reads the Next (grey,Color) Batch and computes the Color_Images_batch Chrominance (AB colorspace values)
 
    Return:
     GreyImages_Batch: List with all Greyscale images [Batch size,224,224,1]
     ColorImages_Batch: List with all Colored images [Batch size,Colored images]
    '''
  
    global GreyImages_Batch
    global ColorImages_Batch
    global CurrentBatch_indx
    global Batch_size

    GreyImages_Batch = []
    ColorImages_Batch = []
   
    for ind in range(Batch_size):
        Colored_img = Image.open(ColorImgsPath + str(CurrentBatch_indx) + '.png')
        ColorImages_Batch.append(Colored_img)
        Grey_img = Colored_img.convert('L')       
        Grey_img = np.asanyarray(Grey_img) 
        img_shape = Grey_img.shape
        img_reshaped = Grey_img.reshape(img_shape[0],img_shape[1], GreyChannels)#[224,224,1]
        GreyImages_Batch.append(img_reshaped)#[#imgs,224,224,1]
        CurrentBatch_indx = CurrentBatch_indx + 1
    Get_Batch_Chrominance() 
    return GreyImages_Batch, AbColores_values

def Get_Batch_Chrominance():
    ''''Convert every image in the batch to LAB Colorspace and normalize each value of it between [0,1]
 
    Return:
     AbColores_values array [batch_size,2224,224,2] 0-> A value, 1-> B value color
    '''
   
    global AbColores_values
    global ColorImages_Batch
    AbColores_values = np.empty((Batch_size,224,224,2),"float32")
    for indx in range(Batch_size):
        lab = color.rgb2lab(ColorImages_Batch[indx])
        Min_valueA = np.amin(lab[:,:,1])
        Max_valueA = np.amax(lab[:,:,1])
        Min_valueB = np.amin(lab[:,:,2])
        Max_valueB = np.amax(lab[:,:,2])
        AbColores_values[indx,:,:,0] = Normalize(lab[:,:,1],-128,127)
        AbColores_values[indx,:,:,1] = Normalize(lab[:,:,2],-128,127)
    return AbColores_values

def Normalize(value,MinValue,MaxValue):
    '''Normalize the input value between specific range
 
    Args:
     value = pixel value
     MinValue = Old Min value
     MaxValue = Old Max value
 
   Return:
    Normalized Value
    '''
 
    MinNormalize_val = 0
    MaxNormalize_val = 1
    value = MinNormalize_val + (((MaxNormalize_val - MinNormalize_val) * (value - MinValue)) / (MaxValue - MinValue))
    return value

def DeNormalize(value,MinValue,MaxValue):
    '''DeNormalize the input value between specific range
 
    Args:
     value = pixel value
     MinValue = Old Min value
     MaxValue = Old Max value
 
   Return:
    Normalized Value
    ''' 
    MinNormalize_val = -128
    MaxNormalize_val = 127
    value = MinNormalize_val + (((MaxNormalize_val - MinNormalize_val) * (value - MinValue)) / (MaxValue - MinValue))
    return value

def Frobenius_Norm(M):
    '''Calculate Frobenius Normalization using formula Sqrt( sum(each (values^2) in input) )
 
    Args:
     M: Input Tensor     
    '''
    return tf.reduce_sum(M ** 2) ** 0.5
