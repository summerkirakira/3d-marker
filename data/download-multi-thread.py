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
    cmd = f"python download-scannet.py -o /mnt/ssd0/kira-home/Dataset/ScanNet/Raw --id {scan_id}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"成功下载 {scan_id}")
    except subprocess.CalledProcessError as e:
        print(f"下载 {scan_id} 失败: {e}")

def main():
    # 获取当前目录
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    # 生成0-10的scan id
    scan_ids = [f"scene{i:04d}_00" for i in range(400, 601)]
    
    # 使用线程池下载
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(download_scan, scan_ids)

if __name__ == "__main__":
    main()
