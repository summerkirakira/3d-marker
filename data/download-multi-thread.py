#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def download_scan(scan_id):
    """下载单个scan"""
    print(f"开始下载 {scan_id}")
    cmd = f"python download-scannet.py --id {scan_id}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"成功下载 {scan_id}")
    except subprocess.CalledProcessError as e:
        print(f"下载 {scan_id} 失败: {e}")

def main():
    # 获取当前目录
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    # 读取scan列表
    with open("scans.txt", "r") as f:
        scan_ids = [line.strip() for line in f if line.strip()]
    
    # 使用线程池下载
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(download_scan, scan_ids)

if __name__ == "__main__":
    main()
