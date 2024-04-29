#!/bin/bash

# 循环复制文件
for i in {0000..0020}
do
    cp -r "$i/predicted_txts/"* "summary/$i"
done