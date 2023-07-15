dir="./onnx"
if [ ! -d "$dir" ];then
mkdir $dir
echo "创建onnx文件夹成功"
else
echo "onnx文件夹已经存在"
fi

#!/bin/bash
dir="./statistics"
if [ ! -d "$dir" ];then
mkdir $dir
echo "创建statistics文件夹成功"
else
echo "statistics文件夹已经存在"
fi


python train0.py
python train1.py
python train2.py
python train3.py
