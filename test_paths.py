#!/usr/bin/env python3
# 测试文件路径
import os
import pandas as pd

print("当前工作目录:", os.getcwd())
print("检查文件是否存在:")

# 检查数据文件
train_path = 'sol_3/train.csv'
test_path = 'sol_3/test.csv'

print(f"训练数据文件: {train_path} - {'存在' if os.path.exists(train_path) else '不存在'}")
print(f"测试数据文件: {test_path} - {'存在' if os.path.exists(test_path) else '不存在'}")

if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    print(f"训练数据形状: {train_df.shape}")
    print("前5行:")
    print(train_df.head())
else:
    print("无法找到训练数据文件")

if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
    print(f"测试数据形状: {test_df.shape}")
    print("前5行:")
    print(test_df.head())
else:
    print("无法找到测试数据文件")
