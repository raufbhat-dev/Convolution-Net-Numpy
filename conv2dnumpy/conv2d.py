import numpy as np
import importlib

from neuralnetnumpy  import layer 

Layer = importlib.reload(layer)

class Conv2D(Layer.Layer):
    def __init__(self, kernel_size, kernel_count, stride, padding, activation_func, layer_type_category, layer_type):
        super().__init__(kernel_size[0]*kernel_size[1], kernel_count,activation_func, layer_type_category, layer_type)
        self.stride = stride
        self.kernel_width = kernel_size[0]
        self.kernel_height = kernel_size[1]
        self.kernel_count = kernel_count
        self.padding = padding
        
    def __call__(self,inputs):
        if self.padding == 'same':
            height_padding = inputs.shape[0]%(self.kernel_height+self.stride)
            width_padding = inputs.shape[1]%(self.kernel_width+self.stride)
            depth_padding = inputs.shape[2]
            height_padding_all = (height_padding//2,height_padding//2+height_padding%2)
            width_padding_all = (width_padding//2,width_padding//2+width_padding%2)
            
            inputs = np.pad(inputs,(height_padding_all,width_padding_all,(depth_padding,depth_padding)),'constant',constant_values = (0,0))
        y_fin = None
        for w in self.w.reshape(self.kernel_count,self.kernel_width, self.kernel_height):
            y= None
            for row_index in range(0, inputs.shape[0],self.stride):
                y_out = np.array([])
                col_end_index = row_end_index = 0
                row_end_index = row_index + self.kernel_height
                for col_index in range(0,inputs.shape[1],self.stride):
                    col_end_index = col_index + self.kernel_width
                    if row_end_index <= inputs.shape[0] and col_end_index <= inputs.shape[1]:
                        y_mul  = np.einsum('ijk,ij->ij',inputs[row_index:row_end_index,col_index:col_end_index,],w)
                        y_out = np.append(y_out, np.sum(y_mul))
                if y is None:
                    y = y_out
                elif y_out.shape[0]!=0 :
                    y = np.vstack((y, y_out))
            y = y.reshape(y.shape[0],y.shape[1],1)
            if y_fin is None:
                y_fin = y
            else:
                y_fin = np.dstack((y_fin, y))
        self.y = y_fin
        self.y_activation = self.activation_method(self.y)
        self.activation_derivative = self.activation_method.activation_derivative
        return self.y_activation
    
