#!/usr/bin/env python3
"""
月次マスターデータ更新スクリプト
cronから実行される月次タスク
"""

import sys
import os
from datetime import datetime

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.master.master_db import StockMasterDB

def main():
    """月次マスターデータ更新処理"""
    print(f"=== マスターデータ月次更新開始 {datetime.now()} ===")
    
    try:
        # マスターDBの初期化と更新
        master_db = StockMasterDB()
        success = master_db.update_master_data()
        
        if success:
            # 統計情報の出力
            stats = master_db.get_statistics()
            print(f"更新完了 - 総銘柄数: {stats['total_stocks']}, アクティブ: {stats['active_stocks']}")
            print(f"=== マスターデータ月次更新完了 {datetime.now()} ===")
        else:
            print("マスターデータの更新に失敗しました")
            sys.exit(1)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()