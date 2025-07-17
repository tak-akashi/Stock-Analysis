#!/usr/bin/env python3
"""
日次株価データ取得スクリプト（J-Quants API）
cronから実行される日次タスク
"""

import sys
import os
import logging
from datetime import datetime

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 最適化版の使用
from backend.jquants.data_processor_optimized import JQuantsDataProcessorOptimized


def setup_logging():
    """ログ設定"""
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"jquants_daily_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def main():
    """日次株価データ取得処理"""
    logger = setup_logging()
    
    logger.info("=== J-Quants日次データ取得開始 ===")
    
    try:
        # データベースパス
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "jquants.db")
        
        logger.info(f"データベースパス: {db_path}")
        
        processor = JQuantsDataProcessorOptimized(
            max_concurrent_requests=3,  # 調整可能
            batch_size=100,
            request_delay=0.1
        )

        # データベースの存在確認
        db_exists = os.path.exists(db_path)
        logger.info(f"データベース存在: {'はい' if db_exists else 'いいえ'}")
        
        if not db_exists:
            logger.info("初回実行: 過去5年分のデータを取得します")
            processor.get_all_prices_for_past_5_years_to_db_optimized(db_path)
        else:
            logger.info("差分更新を実行します")
            processor.update_prices_to_db_optimized(db_path)
        
        # 統計情報を表示
        stats = processor.get_database_stats(db_path)
        if stats:
            logger.info("データベース統計:")
            logger.info(f"  レコード数: {stats.get('record_count', 'N/A')}")
            logger.info(f"  銘柄数: {stats.get('code_count', 'N/A')}")
            logger.info(f"  データ期間: {stats.get('date_range', 'N/A')}")
        
        logger.info("=== J-Quants日次データ取得完了 ===")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        logger.error("環境変数 EMAIL, PASSWORD が正しく設定されているか確認してください")
        sys.exit(1)

if __name__ == "__main__":
    main()