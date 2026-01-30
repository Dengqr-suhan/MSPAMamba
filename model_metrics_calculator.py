#!/usr/bin/env python3
"""
统一模型评估工具 - 计算参数量(Param)、内存占用(Memo)和FLOPs指标
支持所有配置文件，一键计算所有模型指标
"""

import os
import sys
import torch
import torch.nn as nn
import psutil
import gc
import argparse
from pathlib import Path
from thop import profile, clever_format
import traceback
from tools.cfg import py2cfg

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage():
    """获取当前内存使用量(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def calculate_model_metrics(config_path, input_size=(1, 3, 512, 512)):
    """
    计算单个模型的指标
    Args:
        config_path: 配置文件路径
        input_size: 输入张量大小 (batch, channels, height, width)
    Returns:
        dict: 包含模型名称和各项指标的字典
    """
    try:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 加载配置
        config = py2cfg(config_path)
        
        # 记录加载配置后的内存（作为基准）
        initial_memory = get_memory_usage()
        
        model = config.net
        
        # 获取模型名称
        model_name = config_path.stem
        if hasattr(config, 'weights_name'):
            model_name = config.weights_name
        
        # 移动到GPU（如果可用）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        # 计算参数量
        params_count = count_parameters(model)
        params_mb = params_count * 4 / (1024 * 1024)  # 假设float32，每个参数4字节
        
        # 记录模型加载后内存
        model_memory = get_memory_usage()
        memory_usage = model_memory - initial_memory
        
        # 如果内存使用量为负数或过小，使用参数量估算
        if memory_usage <= 0:
            memory_usage = params_mb
        
        # 创建输入张量
        input_tensor = torch.randn(input_size).to(device)
        
        # 计算FLOPs
        try:
            flops, thop_params = profile(model, inputs=(input_tensor,), verbose=False)
            # 修复：正确使用clever_format
            flops_formatted, params_formatted = clever_format([flops, thop_params], "%.3f")
            # 将原始FLOPs值也保存，用于数值比较
            flops_raw = flops
        except Exception as e:
            print(f"Warning: Could not calculate FLOPs for {model_name}: {e}")
            flops_raw = 0
            flops_formatted = "N/A"
        
        # 清理内存
        del model, input_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'model_name': model_name,
            'config_path': str(config_path),
            'params_count': params_count,
            'params_mb': params_mb,
            'memory_mb': memory_usage,
            'flops': flops_raw,
            'flops_formatted': flops_formatted,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Error processing {config_path}: {e}")
        traceback.print_exc()
        return {
            'model_name': config_path.stem,
            'config_path': str(config_path),
            'params_count': 0,
            'params_mb': 0,
            'memory_mb': 0,
            'flops': 0,
            'flops_formatted': "Error",
            'status': f'error: {str(e)}'
        }

def find_all_configs(config_dir="config"):
    """查找所有配置文件，排除LoveDA"""
    config_path = Path(config_dir)
    configs = []
    
    # 遍历所有子目录，排除loveda
    for dataset_dir in config_path.iterdir():
        if (dataset_dir.is_dir() and 
            dataset_dir.name != "__pycache__" and 
            dataset_dir.name != "loveda"):  # 排除loveda目录
            for config_file in dataset_dir.glob("*.py"):
                if not config_file.name.startswith("__"):
                    configs.append(config_file)
    
    return sorted(configs)

def print_results_table(results):
    """打印结果表格"""
    print("\n" + "="*120)
    print("模型性能指标统计表")
    print("="*120)
    print(f"{'模型名称':<30} {'数据集':<12} {'参数量(M)':<12} {'内存(MB)':<12} {'FLOPs':<15} {'状态':<15}")
    print("-"*120)
    
    for result in results:
        dataset = result['config_path'].split('/')[-2] if '/' in result['config_path'] else 'unknown'
        params_m = result['params_count'] / 1e6 if result['params_count'] > 0 else 0
        
        print(f"{result['model_name']:<30} {dataset:<12} {params_m:<12.2f} "
              f"{result['memory_mb']:<12.1f} {result['flops_formatted']:<15} {result['status']:<15}")
    
    print("-"*120)
    
    # 统计信息
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        total_models = len(successful_results)
        avg_params = sum(r['params_count'] for r in successful_results) / total_models / 1e6
        avg_memory = sum(r['memory_mb'] for r in successful_results) / total_models
        
        print(f"统计信息: 成功评估 {total_models} 个模型")
        print(f"平均参数量: {avg_params:.2f}M")
        print(f"平均内存占用: {avg_memory:.1f}MB")
    
    print("="*120)

def save_results_csv(results, output_file="model_metrics.csv"):
    """保存结果到CSV文件"""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model Name', 'Dataset', 'Config Path', 'Params(M)', 'Memory(MB)', 'FLOPs', 'FLOPs_Raw', 'Status'])
        
        for result in results:
            dataset = result['config_path'].split('/')[-2] if '/' in result['config_path'] else 'unknown'
            params_m = result['params_count'] / 1e6 if result['params_count'] > 0 else 0
            
            writer.writerow([
                result['model_name'],
                dataset,
                result['config_path'],
                f"{params_m:.2f}",
                f"{result['memory_mb']:.1f}",
                result['flops_formatted'],
                result['flops'],  # 添加原始FLOPs数值
                result['status']
            ])
    
    print(f"\n结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='计算所有模型的性能指标')
    parser.add_argument('--config_path', type=str, help='单个配置文件路径')
    parser.add_argument('--config_dir', type=str, default='config', help='配置文件目录')
    parser.add_argument('--input_size', type=str, default='1,3,512,512', 
                       help='输入张量大小，格式: batch,channels,height,width')
    parser.add_argument('--output_csv', type=str, default='model_metrics.csv', 
                       help='输出CSV文件名')
    parser.add_argument('--dataset', type=str, choices=['vaihingen', 'potsdam'], 
                       help='只评估指定数据集的模型')
    
    args = parser.parse_args()
    
    # 解析输入大小
    input_size = tuple(map(int, args.input_size.split(',')))
    
    results = []
    
    if args.config_path:
        # 评估单个配置文件
        config_path = Path(args.config_path)
        if config_path.exists():
            result = calculate_model_metrics(config_path, input_size)
            results.append(result)
        else:
            print(f"配置文件不存在: {config_path}")
            return
    else:
        # 评估所有配置文件
        configs = find_all_configs(args.config_dir)
        
        # 过滤数据集
        if args.dataset:
            configs = [c for c in configs if args.dataset in str(c)]
        
        print(f"找到 {len(configs)} 个配置文件")
        
        for i, config_path in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] 正在评估: {config_path}")
            result = calculate_model_metrics(config_path, input_size)
            results.append(result)
    
    # 打印结果
    print_results_table(results)
    
    # 保存CSV
    save_results_csv(results, args.output_csv)

if __name__ == "__main__":
    main()