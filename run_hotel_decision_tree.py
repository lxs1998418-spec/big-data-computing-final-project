#!/usr/bin/env python3
"""
运行酒店预订决策树脚本
Run Hotel Booking Decision Tree Script
"""
import os
import sys
import subprocess

# 切换到sol_3目录
os.chdir('sol_3')

# 运行决策树脚本
try:
    result = subprocess.run([sys.executable, 'hotel_booking_decision_tree.py'], 
                          capture_output=True, text=True, check=True)
    print("脚本执行成功!")
    print("输出:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("脚本执行失败!")
    print("错误输出:")
    print(e.stderr)
    print("标准输出:")
    print(e.stdout)
except Exception as e:
    print(f"执行过程中出现错误: {e}")
