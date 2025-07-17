"""
Script to create database indexes for improved performance.
This should be run once to optimize database query performance.
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.utils.parallel_processor import BatchDatabaseProcessor


def setup_logging():
    """Setup logging configuration."""
    log_filename = f"create_indexes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_jquants_indexes(db_path: str):
    """Create indexes for jquants database."""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating indexes for jquants database: {db_path}")
    
    db_processor = BatchDatabaseProcessor(db_path)
    
    indexes = [
        {
            'name': 'idx_daily_quotes_code',
            'table': 'daily_quotes',
            'columns': ['Code']
        },
        {
            'name': 'idx_daily_quotes_date',
            'table': 'daily_quotes',
            'columns': ['Date']
        },
        {
            'name': 'idx_daily_quotes_code_date',
            'table': 'daily_quotes',
            'columns': ['Code', 'Date'],
            'unique': True
        },
        {
            'name': 'idx_daily_quotes_date_code',
            'table': 'daily_quotes',
            'columns': ['Date', 'Code']
        }
    ]
    
    # Check existing constraints for daily_quotes table
    logger.info("Checking constraints for daily_quotes table")
    constraints = check_existing_constraints(db_path, 'daily_quotes')
    logger.info(f"  Primary keys: {constraints['primary_keys']}")
    logger.info(f"  Existing indexes: {len(constraints['existing_indexes'])}")
    
    # Filter indexes based on existing constraints
    filtered_indexes = []
    skipped_count = 0
    
    for index_def in indexes:
        should_create, reason = should_create_index(index_def, constraints)
        if should_create:
            filtered_indexes.append(index_def)
            logger.info(f"  Will create index: {index_def['name']}")
        else:
            skipped_count += 1
            logger.info(f"  Skipping index {index_def['name']}: {reason}")
    
    try:
        if filtered_indexes:
            db_processor.create_indexes(filtered_indexes)
            logger.info(f"Successfully created {len(filtered_indexes)} indexes for jquants database")
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} indexes due to existing constraints")
        else:
            logger.warning("No indexes to create (all were skipped)")
    except Exception as e:
        logger.error(f"Error creating jquants indexes: {e}")
        raise


def check_existing_constraints(db_path: str, table_name: str) -> Dict[str, List]:
    """Check existing constraints and indexes for a table."""
    constraints = {
        'primary_keys': [],
        'unique_constraints': [],
        'existing_indexes': []
    }
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get table info for primary keys
            cursor.execute(f"PRAGMA table_info({table_name})")
            for row in cursor.fetchall():
                if row[5]:  # pk column is 1 for primary key
                    constraints['primary_keys'].append(row[1])  # column name
            
            # Get existing indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            for row in cursor.fetchall():
                index_name = row[1]
                is_unique = row[2]
                
                # Get index columns
                cursor.execute(f"PRAGMA index_info({index_name})")
                columns = [col[2] for col in cursor.fetchall()]
                
                constraints['existing_indexes'].append({
                    'name': index_name,
                    'columns': columns,
                    'unique': bool(is_unique)
                })
                
                if is_unique:
                    constraints['unique_constraints'].append(columns)
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error checking constraints for {table_name}: {e}")
    
    return constraints


def check_duplicate_data(db_path: str, table_name: str, columns: List[str]) -> int:
    """Check for duplicate data that would prevent UNIQUE index creation."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            columns_str = ', '.join(columns)
            query = f"""
                SELECT COUNT(*) as duplicates
                FROM (
                    SELECT {columns_str}, COUNT(*) as cnt
                    FROM {table_name}
                    GROUP BY {columns_str}
                    HAVING COUNT(*) > 1
                )
            """
            cursor.execute(query)
            return cursor.fetchone()[0]
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error checking duplicates for {table_name}: {e}")
        return 0


def clean_duplicate_data(db_path: str, table_name: str, columns: List[str]) -> int:
    """Remove duplicate data, keeping only the latest row for each combination."""
    logger = logging.getLogger(__name__)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get table columns to determine which ones to keep
            cursor.execute(f"PRAGMA table_info({table_name})")
            all_columns = [row[1] for row in cursor.fetchall()]
            
            columns_str = ', '.join(columns)
            all_columns_str = ', '.join(all_columns)
            
            # Create a temporary table with deduplicated data
            # Use rowid to keep the latest row in case of duplicates
            dedup_query = f"""
                CREATE TEMP TABLE {table_name}_dedup AS
                SELECT {all_columns_str}
                FROM {table_name}
                WHERE rowid IN (
                    SELECT MAX(rowid)
                    FROM {table_name}
                    GROUP BY {columns_str}
                )
            """
            cursor.execute(dedup_query)
            
            # Count original rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            original_count = cursor.fetchone()[0]
            
            # Count deduplicated rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}_dedup")
            dedup_count = cursor.fetchone()[0]
            
            if original_count > dedup_count:
                # Replace original table with deduplicated data
                cursor.execute(f"DELETE FROM {table_name}")
                cursor.execute(f"INSERT INTO {table_name} SELECT * FROM {table_name}_dedup")
                
                removed_count = original_count - dedup_count
                logger.info(f"  Removed {removed_count} duplicate rows from {table_name}")
                return removed_count
            else:
                logger.info(f"  No duplicates found in {table_name}")
                return 0
                
    except Exception as e:
        logger.error(f"Error cleaning duplicates for {table_name}: {e}")
        return 0


def should_create_index(index_def: Dict, constraints: Dict) -> Tuple[bool, str]:
    """Check if an index should be created based on existing constraints."""
    index_columns = index_def['columns']
    is_unique = index_def.get('unique', False)
    
    # Check if this exact index already exists
    for existing_idx in constraints['existing_indexes']:
        if existing_idx['columns'] == index_columns:
            return False, f"Index with same columns already exists: {existing_idx['name']}"
    
    # Check if unique index conflicts with existing unique constraints
    if is_unique:
        # Check primary key conflict
        if set(index_columns) == set(constraints['primary_keys']):
            return False, "Conflicts with PRIMARY KEY constraint"
        
        # Check other unique constraints
        for unique_cols in constraints['unique_constraints']:
            if set(index_columns) == set(unique_cols):
                return False, "Conflicts with existing UNIQUE constraint"
    
    return True, "OK"


def create_results_indexes(db_path: str):
    """Create indexes for analysis results database."""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating indexes for results database: {db_path}")
    
    db_processor = BatchDatabaseProcessor(db_path)
    
    # Check which tables exist
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
    
    logger.info(f"Found tables in results database: {existing_tables}")
    
    indexes = []
    
    # Relative strength table indexes
    if 'relative_strength' in existing_tables:
        indexes.extend([
            {
                'name': 'idx_rs_date',
                'table': 'relative_strength',
                'columns': ['Date']
            },
            {
                'name': 'idx_rs_code',
                'table': 'relative_strength',
                'columns': ['Code']
            },
            {
                'name': 'idx_rs_date_code',
                'table': 'relative_strength',
                'columns': ['Date', 'Code'],
                'unique': True
            },
            {
                'name': 'idx_rs_rsp',
                'table': 'relative_strength',
                'columns': ['RelativeStrengthPercentage']
            },
            {
                'name': 'idx_rs_rsi',
                'table': 'relative_strength',
                'columns': ['RelativeStrengthIndex']
            }
        ])
    
    # Minervini table indexes
    if 'minervini' in existing_tables:
        indexes.extend([
            {
                'name': 'idx_minervini_date',
                'table': 'minervini',
                'columns': ['Date']
            },
            {
                'name': 'idx_minervini_code',
                'table': 'minervini',
                'columns': ['Code']
            },
            {
                'name': 'idx_minervini_date_code',
                'table': 'minervini',
                'columns': ['Date', 'Code'],
                'unique': True
            }
        ])
    
    # HL ratio table indexes
    if 'hl_ratio' in existing_tables:
        indexes.extend([
            {
                'name': 'idx_hl_ratio_date',
                'table': 'hl_ratio',
                'columns': ['Date']
            },
            {
                'name': 'idx_hl_ratio_code',
                'table': 'hl_ratio',
                'columns': ['Code']
            },
            {
                'name': 'idx_hl_ratio_date_code',
                'table': 'hl_ratio',
                'columns': ['Date', 'Code'],
                'unique': True
            },
            {
                'name': 'idx_hl_ratio_hlratio',
                'table': 'hl_ratio',
                'columns': ['HlRatio']
            }
        ])
    
    # Filter indexes based on existing constraints
    filtered_indexes = []
    skipped_count = 0
    
    for table_name in existing_tables:
        if table_name in ['relative_strength', 'minervini', 'hl_ratio']:
            logger.info(f"Checking constraints for table: {table_name}")
            constraints = check_existing_constraints(db_path, table_name)
            logger.info(f"  Primary keys: {constraints['primary_keys']}")
            logger.info(f"  Existing indexes: {len(constraints['existing_indexes'])}")
            
            # Filter indexes for this table
            table_indexes = [idx for idx in indexes if idx['table'] == table_name]
            for index_def in table_indexes:
                should_create, reason = should_create_index(index_def, constraints)
                if should_create:
                    # Check for duplicate data if this is a unique index
                    if index_def.get('unique', False):
                        duplicate_count = check_duplicate_data(db_path, table_name, index_def['columns'])
                        if duplicate_count > 0:
                            logger.warning(f"  Found {duplicate_count} duplicate combinations for {index_def['name']}")
                            removed = clean_duplicate_data(db_path, table_name, index_def['columns'])
                            if removed > 0:
                                logger.info("  Cleaned duplicates, can now create unique index")
                            else:
                                logger.warning("  Could not clean duplicates, skipping unique index")
                                skipped_count += 1
                                continue
                    
                    filtered_indexes.append(index_def)
                    logger.info(f"  Will create index: {index_def['name']}")
                else:
                    skipped_count += 1
                    logger.info(f"  Skipping index {index_def['name']}: {reason}")
    
    try:
        if filtered_indexes:
            db_processor.create_indexes(filtered_indexes)
            logger.info(f"Successfully created {len(filtered_indexes)} indexes for results database")
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} indexes due to existing constraints")
        else:
            logger.warning("No indexes to create (all were skipped or no tables found)")
    except Exception as e:
        logger.error(f"Error creating results indexes: {e}")
        raise


def optimize_database_settings(db_path: str):
    """Optimize SQLite database settings for better performance."""
    logger = logging.getLogger(__name__)
    logger.info(f"Optimizing database settings for: {db_path}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Set synchronous to NORMAL for better performance
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Increase cache size (in KB)
            conn.execute("PRAGMA cache_size=10000")
            
            # Set temp store to memory
            conn.execute("PRAGMA temp_store=MEMORY")
            
            # Enable memory mapping
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            # Optimize for faster writes
            conn.execute("PRAGMA optimize")
            
            conn.commit()
            
        logger.info(f"Database settings optimized for {db_path}")
    except Exception as e:
        logger.error(f"Error optimizing database settings for {db_path}: {e}")
        raise


def analyze_database_stats(db_path: str):
    """Analyze database statistics after index creation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing database statistics for: {db_path}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Update statistics
            conn.execute("ANALYZE")
            
            # Get database size
            cursor = conn.cursor()
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Database size: {db_size / (1024*1024):.2f} MB")
            logger.info(f"Tables: {', '.join(tables)}")
            
            # Get index information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            logger.info(f"Indexes: {len(indexes)} total")
            
    except Exception as e:
        logger.error(f"Error analyzing database statistics for {db_path}: {e}")


def main():
    """Main function to create all database indexes."""
    logger = setup_logging()
    logger.info("Starting database index creation process")
    
    # Database paths
    jquants_db_path = "/Users/tak/Markets/Stocks/Stock-Analysis/data/jquants.db"
    results_db_path = "/Users/tak/Markets/Stocks/Stock-Analysis/data/analysis_results.db"
    
    try:
        # Check if databases exist
        if not os.path.exists(jquants_db_path):
            logger.error(f"jquants database not found at {jquants_db_path}")
            return False
        
        if not os.path.exists(results_db_path):
            logger.warning(f"Results database not found at {results_db_path}, will be created when needed")
        
        # Create indexes for jquants database
        logger.info("Creating indexes for jquants database...")
        create_jquants_indexes(jquants_db_path)
        optimize_database_settings(jquants_db_path)
        analyze_database_stats(jquants_db_path)
        
        # Create indexes for results database if it exists
        if os.path.exists(results_db_path):
            logger.info("Creating indexes for results database...")
            create_results_indexes(results_db_path)
            optimize_database_settings(results_db_path)
            analyze_database_stats(results_db_path)
        
        logger.info("Database index creation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during index creation process: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)