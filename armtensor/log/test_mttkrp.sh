#!/bin/bash


for ((a=0; a<10; a++))   #循环mode次，设置行数
do
  ../build/test/test_tilespatsr -i ../data/whtdata/10_100000.tns -m $a -t 1
  ../build/test/test_tilespatsr -i ../data/whtdata/10_100000.tns -m $a -t 64
done