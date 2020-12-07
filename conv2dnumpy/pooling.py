import numpy as np

class Pooling:
    def __init__(self, frame_size, stride, layer_type_category, layer_type, pooling_type = 'average'):
        self.frame_width = frame_size[0]
        self.frame_height = frame_size[1]
        self.pooling_type = pooling_type
        self.stride = stride
        self.layer_type_category = layer_type_category
        self.layer_type = layer_type
        self.y = None
    
    def __call__(self, inputs):
        for row_index in range(0, inputs.shape[0],self.stride):
            y_out = np.array([])
            col_end_index = row_end_index = 0
            row_end_index = row_index + self.frame_height
            for col_index in range(0,inputs.shape[1],self.stride):
                col_end_index = col_index + self.frame_width
                if row_end_index <= inputs.shape[0] and col_end_index <= inputs.shape[1]:
                    if self.pooling_type == 'average':
                        y_reduced  = np.nansum(inputs[row_index:row_end_index,col_index:col_end_index,],axis =(0,1))/self.frame_width*self.frame_height
                    elif self.pooling_type == 'maximum':
                        y_reduced  = np.max(inputs[row_index:row_end_index,col_index:col_end_index,],axis =(0,1))
                    y_reduced = y_reduced.reshape(y_reduced.shape[0],1)
                    if y_out.shape[0] == 0:
                        y_out = y_reduced
                    else:
                        y_out = np.hstack((y_out, y_reduced))
            if y_out.shape[0]!=0:
                y_out = y_out.reshape(y_out.shape[0],y_out.shape[1],1)
                if self.y is None:
                    self.y = y_out
                else:
                    self.y = np.dstack((self.y, y_out))
        self.y = self.y.reshape(self.y.shape[2], self.y.shape[1], self.y.shape[0])
        return self.y
