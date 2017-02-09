# encoding: utf-8



# numpy模組 - 基本用法:創建矩陣
import numpy as np

# 建立一個 np array
arr = np.array([[1,2,3],[4,5,6]])    #放入list就會轉換喔!
print "建立一個 np array"
print arr
print arr.shape
#印出 [[1,2,3],[4,5,6]]的 2*3 矩陣


# 建立一個 全為 0 的 2*3 矩陣，並指定為 int16格式
zero_arr = np.zeros((2,3), dtype=np.int16)
print "建立一個 全為 0 的 2*3 矩陣，並指定為 int16格式"
print zero_arr
print zero_arr.shape[0]
#印出 [[0,0,0],[0,0,0]] 的 2*3 矩陣


# 建立一個長度為12、數值從0~11、格式為 int64 的一維陣列
onedim_arr = np.arange(12, dtype=np.int64)
print "建立一個長度為12、數值從0~11、格式為 int64 的一維陣列"
print onedim_arr
print onedim_arr.shape[0]
#印出 [0,1,2,3,4,5,6,7,8,9,10,11]


# 建立一個數值從0~11、格式為int64，但是是 3*4 的矩陣
reshape_arr = np.arange(12, dtype=np.int64).reshape( (3,4) )
print "建立一個數值從0~11、格式為int64，但是是 3*4 的矩陣"
print reshape_arr
#印出 [ [0,1,2,3],[4,5,6,7],[8,9,10,11] ] 的 3*4矩陣


# 建立一個最小值1，最大值50，共分五塊相等間距的矩陣
slice_arr = np.linspace(1,50,5)
print "建立一個最小值1，最大值50，共分五塊相等間距的矩陣"
print slice_arr
#印出 [  1.    13.25  25.5   37.75  50.  ] 的 1*5 矩陣
#要轉成 5*1 的矩陣就在後面加上 .reshape((5*1)) 就可以喔！


# 建立一個隨機在0~1之間分布的 4*2 矩陣
random_arr = np.random.random( (4, 2) )
print "建立一個隨機在0~1之間分布的 4*2 矩陣"
print random_arr
#印出 [[ 0.92615029  0.12676952], [ 0.33146324  0.51952945],
#[ 0.37010443  0.23092601], [ 0.11052635  0.78243148]]
#值是隨機的，所以你印出來一定跟我不一樣喔！
