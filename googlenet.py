from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD

input = Input(shape=(3, 224, 224))
conv1_7x7_s2 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1/7x7_s2',W_regularizer=l2(0.0002))(input)
googlenet = Model(input=input, output=conv1_7x7_s2)

output_path = 'C:/Users/M543015/Desktop/GitHub/deeplearning/'
with open(output_path+'googlenet_test.json', 'w') as outfile:
    outfile.write(googlenet.to_json())


from keras.layers.core import Layer
import theano.tensor as T
from keras.models import model_from_json


class LRN(Layer):

    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2 # half the local region
        input_sqr = T.sqr(x) # square the input
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c) # make an empty tensor with zero pads along channel dimension
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input
        scale = self.k # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):
    
    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        return x[:,:,1:,1:]
    
    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




path = 'C:/Users/M543015/Desktop/GitHub/googlenet/googlenet/'
model = model_from_json(open(path+'googlenet_architecture.json').read(), custom_objects={"PoolHelper": PoolHelper, "LRN": LRN})
model.load_weights(path+'googlenet_weights.h5')



# import h5py
# import sys
# import json
# import argparse
# import tensorflow as tf
# from keras.models import model_from_config
# from pathlib import Path

# import warnings
# warnings.filterwarnings('ignore')

# sys.argv = ['program', '-o', '/path/to/output/directory', 
# '-m', '/path/to/model/file.hdf5', '/path/to/model/directory']

# parser = argparse.ArgumentParser()
# parser.add_argument('-m','--models', nargs='+', type=Path, help='List of model files/directories to fix.')
# parser.add_argument('-x','--ext', default='hdf5', help='Model extension (if using directories)')
# parser.add_argument('-o','--output', type=Path, help='Output directory.')
# args = parser.parse_args()

# for p in args.models:
#     models = []
#     if p.is_dir():
#         models.extend([_ for _ in p.glob('*.{}'.format(args.ext))])
#     else:
#         if p.exists():
#             models.append(p)
#         else:
#             print('Missing file: {}'.format(p))

# args.models = models


# def fix_model_file(fp,od=Path('.')):
#     assert fp.exists()
#     if not od.is_dir():
#         od.mkdir(parents=True, exist_ok=True)
#     op = str(od / fp.name)
#     fp = str(fp)
    
#     with h5py.File(fp) as h5:
#         config = json.loads(h5.attrs.get("model_config")
#                             .decode('utf-8')
#                             .replace('input_dtype','dtype'))
#     with tf.Session('') as sess:
#         model = model_from_config(config)
#         model.load_weights(fp)
#         model.save(op)
#         del model
#     del sess
#     print(op)

# if __name__ == '__main__':
#     for model in args.models:
#         fix_model_file(model, od=args.output)