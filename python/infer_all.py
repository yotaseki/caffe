#!/usr/bin/python
# Just like the Shellhammer version except it saves the segmented file

import sys,os
import time
import numpy as np
from PIL import Image
import caffe
import glob

def main():
    files = [x.split('/')[-1] for x in glob.glob(argv[1] + "/*.jpg")]
    for f in files:
        print(f)
        predict(argv[1],f)

def predict(path,filename):
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(path+filename)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

# shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    out_8 = np.empty_like(out, dtype=np.uint8)
    np.copyto(out_8, out, casting='unsafe')
    img = Image.fromarray(out_8)
    gray = img.convert("L")
    gray.save("out/"+filename)

if __name__=='__main__':
    argv = sys.argv
    argv[1] = argv[1] + "/"
    caffe.set_mode_gpu()
    deploy = "fixed_SoccerField3D_Blur_Bleeding.fcn-8s-digits/deploy.prototxt"
    model = "fixed_SoccerField3D_Blur_Bleeding.fcn-8s-digits/snapshot_iter_10010.caffemodel"
    net = caffe.Net(
            deploy,
            model,
            caffe.TEST
            )
    main()
