#!/bin/bash


for ((a=7; a<9; a++))   #循环mode次，设置行数
do
  ../build/test/test_tilespatsr_ttv -i ../data/whtdata/10_900000.tns -m $a -t 1
  ../build/test/test_tilespatsr_ttv -i ../data/whtdata/10_900000.tns -m $a -t 64
done