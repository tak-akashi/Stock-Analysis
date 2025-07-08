#!/usr/bin/env python3
"""
日次株価データ取得スクリプト（J-Quants API）
cronから実行される日次タスク
"""

import sys
import os
from datetime import datetime

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.jquants.data_processor import JQuantsDataProcessor

def main():
    """日次株価データ取得処理"""
    print(f"=== J-Quants日次データ取得開始 {datetime.now()} ===")
    
    try:
        # データベースパス
        db_path = os.path.join(project_root, "data", "jquants.db")
        
        # プロセッサーの初期化と実行
        processor = JQuantsDataProcessor()
        processor.update_prices_to_db(db_path)
        
        print(f"=== J-Quants日次データ取得完了 {datetime.now()} ===")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()