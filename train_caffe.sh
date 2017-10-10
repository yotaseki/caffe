#!/bin/bash

DATE = `date +"%Y%m%d%k%M%S" |sed "s/ /0/"`
echo "START" > Logs/$DATE.log
date >> Logs/$DATE.log

./tools/caffe train -gpu 0 -solver models/whiteline10-fcn8s/solver.prototxt

echo "END" >> Logs/$DATE.log
date >> Logs/$DATE.log
cat /tmp/caffe.INFO >> Logs/$DATE.log

