#!/usr/bin/env python3
"""
週次属性情報取得スクリプト（yfinance）
cronから実行される週次タスク
"""

import sys
import os
from datetime import datetime

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.yfinance.data_processor import TSEDataProcessor

def main():
    """週次属性情報取得処理"""
    print(f"=== yfinance週次データ取得開始 {datetime.now()} ===")
    
    try:
        # プロセッサーの初期化と実行
        processor = TSEDataProcessor(max_workers=4, rate_limit_delay=0.7)
        processor.run()
        
        print(f"=== yfinance週次データ取得完了 {datetime.now()} ===")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()