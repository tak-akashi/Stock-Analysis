#!/usr/bin/env python3
"""
週次属性情報取得スクリプト（yfinance）
cronから実行される週次タスク
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.yfinance.data_processor import TSEDataProcessor

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='週次属性情報取得スクリプト')
    parser.add_argument('--yfinance-only', action='store_true', 
                       help='yfinanceデータ取得のみ実行')
    parser.add_argument('--analysis-only', action='store_true',
                          help='統合分析のみ実行')
    return parser.parse_args()

def main():
    """週次属性情報取得処理"""
    args = parse_args()
    
    # オプション競合チェック
    if sum([args.yfinance_only, args.analysis_only]) > 1:
        print("エラー: --yfinance-only, --analysis-only は同時に指定できません")
        sys.exit(1)
    
    # デフォルトは全て実行
    run_yfinance = not (args.analysis_only)
    run_analysis = not (args.yfinance_only)

    print(f"=== 週次タスク開始 {datetime.now()} ===")
    
    try:
        if run_yfinance:
            print(f"=== yfinanceデータ取得開始 {datetime.now()} ===")
            processor = TSEDataProcessor(max_workers=4, rate_limit_delay=0.7)
            processor.run()
            print(f"=== yfinanceデータ取得完了 {datetime.now()} ===")

        if run_analysis:
            print(f"=== 統合分析処理開始 {datetime.now()} ===")
            analysis_script_path = os.path.join(project_root, 'backend', 'analysis', 'integrated_analysis2.py')
            subprocess.run([sys.executable, analysis_script_path], check=True)
            print(f"=== 統合分析処理完了 {datetime.now()} ===")
        
        print(f"=== 週次タスク完了 {datetime.now()} ===")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()