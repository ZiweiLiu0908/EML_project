## To Gorden:

##### 在自己电脑上执行train.sh脚本 （时间可能会很长 不需要GPU）

```bash
source train.sh
```

在onnx文件目录下会生成onnx模型权重文件， 在statistics文件目录下会生成loss以及accuracy的变化图像（一共30epochs）。训练时，每个epoch的loss以及accuracy会分别保存在exp0.txt，exp1.txt，exp2.txt，exp3.txt之中。

##### 在树莓派上运行test.sh脚本

```bash
source test.sh
```

输出不同模型的运行时间以及在测试集上的准确率，结果同样会保存在test_result.txt之中。

## 命名规则：

如

```
model1_{2}_{}.onnx
```

此模型为model1，weight_bit_width = 2时 在所有epoch中，准确率最佳的模型。

如：

```
model2_{3}_{4}.onnx
```

此模型为model2，weight_bit_width = 3， bit_width = 4 时 在所有epoch中，准确率最佳的结果模型。

## To zbw & lb

所有模型最多有2个变量 即weight_bit_width， bit_width

本项目对 2<=weight_bit_width<=8,  2<=bit_width<=8 进行对比

当weight_bit_width = 1， loss为0无法训练 不晓得为什么

当bit_width = 1， 程序崩溃 不晓得为什么

详情可参考官方文档

Ps. 可利用exp0.txt ~ exp4.txt 以及test_result.txt 中的数据生成表格 做分析实验

