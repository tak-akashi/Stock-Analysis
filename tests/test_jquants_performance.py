"""
Test script to measure JQuants data processor performance.
"""

import os
import sys
import time
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.jquants.data_processor import JQuantsDataProcessor


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'jquants_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def test_performance(test_codes: list, days_back: int = 30):
    """
    Test performance of JQuants data processor.
    
    Args:
        test_codes: List of stock codes to test with
        days_back: Number of days back to fetch data for
    """
    logger = setup_logging()
    logger.info("Starting JQuants processor performance test")
    
    # Test dates
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = os.path.join(temp_dir, "test.db")
        
        try:
            # Test with different configurations
            configurations = [
                {
                    'name': 'Conservative',
                    'params': {'max_concurrent_requests': 2, 'batch_size': 50, 'request_delay': 0.2}
                },
                {
                    'name': 'Standard',
                    'params': {'max_concurrent_requests': 3, 'batch_size': 100, 'request_delay': 0.1}
                },
                {
                    'name': 'Aggressive',
                    'params': {'max_concurrent_requests': 5, 'batch_size': 200, 'request_delay': 0.05}
                }
            ]
            
            results = []
            
            for config in configurations:
                logger.info("=" * 60)
                logger.info(f"Testing {config['name']} configuration")
                logger.info(f"Parameters: {config['params']}")
                logger.info("=" * 60)
                
                start_time = time.time()
                processor = JQuantsDataProcessor(**config['params'])
                
                # Process all codes in batch
                import asyncio
                batch_results = asyncio.run(processor.process_codes_batch(
                    [str(code) for code in test_codes], from_date, to_date
                ))
                
                successful = sum(1 for _, df in batch_results if not df.empty)
                total_records = sum(len(df) for _, df in batch_results if not df.empty)
                elapsed_time = time.time() - start_time
                
                # Calculate metrics
                codes_per_second = len(test_codes) / elapsed_time if elapsed_time > 0 else 0
                records_per_second = total_records / elapsed_time if elapsed_time > 0 else 0
                
                result = {
                    'config': config['name'],
                    'time': elapsed_time,
                    'successful': successful,
                    'total_codes': len(test_codes),
                    'total_records': total_records,
                    'codes_per_second': codes_per_second,
                    'records_per_second': records_per_second
                }
                results.append(result)
                
                logger.info(f"Results for {config['name']}:")
                logger.info(f"  Time: {elapsed_time:.2f} seconds")
                logger.info(f"  Successful: {successful}/{len(test_codes)}")
                logger.info(f"  Total records: {total_records}")
                logger.info(f"  Rate: {codes_per_second:.2f} codes/second")
                logger.info(f"  Rate: {records_per_second:.2f} records/second")
                
                # Add delay between tests
                time.sleep(2)
            
            # Display comparison
            logger.info("=" * 60)
            logger.info("PERFORMANCE COMPARISON")
            logger.info("=" * 60)
            logger.info(f"Test configuration:")
            logger.info(f"  Stock codes: {len(test_codes)}")
            logger.info(f"  Date range: {from_date} to {to_date}")
            logger.info("")
            
            # Find best configuration
            best_config = min(results, key=lambda x: x['time'])
            
            for result in results:
                speedup_vs_best = result['time'] / best_config['time'] if best_config['time'] > 0 else 1
                logger.info(f"{result['config']}:")
                logger.info(f"  Time: {result['time']:.2f}s")
                logger.info(f"  Codes/sec: {result['codes_per_second']:.2f}")
                logger.info(f"  Records/sec: {result['records_per_second']:.2f}")
                if result['config'] == best_config['config']:
                    logger.info(f"  ⭐ FASTEST")
                else:
                    logger.info(f"  {speedup_vs_best:.2f}x slower than {best_config['config']}")
                logger.info("")
            
            # Extrapolate to full dataset
            full_dataset_size = 4000  # Approximate number of listed companies
            logger.info(f"Estimated time for full dataset ({full_dataset_size} codes):")
            for result in results:
                estimated_time = (result['time'] / len(test_codes)) * full_dataset_size
                logger.info(f"  {result['config']}: {estimated_time/60:.1f} minutes")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during performance test: {e}")
            return None


def test_batch_sizes(test_codes: list):
    """
    Test different batch sizes to find optimal configuration.
    
    Args:
        test_codes: List of stock codes to test
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing different batch sizes")
    
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    batch_sizes = [25, 50, 100, 200]
    results = []
    
    for batch_size in batch_sizes:
        logger.info(f"\nTesting batch size: {batch_size}")
        
        try:
            start_time = time.time()
            processor = JQuantsDataProcessor(
                max_concurrent_requests=3,
                batch_size=batch_size,
                request_delay=0.1
            )
            
            import asyncio
            batch_results = asyncio.run(processor.process_codes_batch(
                [str(code) for code in test_codes], from_date, to_date
            ))
            
            elapsed_time = time.time() - start_time
            successful = sum(1 for _, df in batch_results if not df.empty)
            
            results.append({
                'batch_size': batch_size,
                'time': elapsed_time,
                'successful': successful,
                'codes_per_second': len(test_codes) / elapsed_time if elapsed_time > 0 else 0
            })
            
            logger.info(f"  Time: {elapsed_time:.2f}s")
            logger.info(f"  Success rate: {successful}/{len(test_codes)}")
            
            # Add delay between tests
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error testing batch size {batch_size}: {e}")
    
    # Find optimal batch size
    if results:
        optimal = min(results, key=lambda x: x['time'])
        logger.info(f"\nOptimal batch size: {optimal['batch_size']} (completed in {optimal['time']:.2f}s)")
    
    return results


def test_error_recovery(test_codes: list):
    """
    Test error recovery and retry mechanisms.
    
    Args:
        test_codes: List of stock codes to test
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing error recovery mechanisms")
    
    # Include some invalid codes to test error handling
    test_codes_with_errors = test_codes + ["99999", "00000", "INVALID"]
    
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        processor = JQuantsDataProcessor(
            max_concurrent_requests=3,
            batch_size=100,
            request_delay=0.1
        )
        
        import asyncio
        results = asyncio.run(processor.process_codes_batch(
            [str(code) for code in test_codes_with_errors], from_date, to_date
        ))
        
        successful = sum(1 for _, df in results if not df.empty)
        failed = len(test_codes_with_errors) - successful
        
        logger.info(f"Error recovery test results:")
        logger.info(f"  Total codes: {len(test_codes_with_errors)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Recovery rate: {(successful/len(test_codes))*100:.1f}% (excluding invalid codes)")
        
        return successful >= len(test_codes)  # Should successfully process all valid codes
        
    except Exception as e:
        logger.error(f"Error during error recovery test: {e}")
        return False


def main():
    """Main test function."""
    logger = setup_logging()
    
    # Test with a subset of major stock codes
    test_codes = ["7203", "6758", "9984", "8306", "4563", "6861", "8035", "7974", 
                  "9432", "8058", "6902", "7267", "8001", "9433", "4502", "6501"]
    
    logger.info("Starting JQuants processor performance tests")
    logger.info(f"Test codes: {test_codes}")
    
    try:
        # Test 1: Performance with different configurations
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Performance Comparison")
        logger.info("="*60)
        performance_results = test_performance(test_codes, days_back=7)
        
        # Test 2: Optimal batch size
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Batch Size Optimization")
        logger.info("="*60)
        batch_results = test_batch_sizes(test_codes[:8])  # Use fewer codes for batch test
        
        # Test 3: Error recovery
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Error Recovery")
        logger.info("="*60)
        error_recovery_passed = test_error_recovery(test_codes[:8])
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("="*60)
        
        if performance_results:
            best_config = min(performance_results, key=lambda x: x['time'])
            logger.info(f"✅ Best configuration: {best_config['config']}")
            logger.info(f"   Processing rate: {best_config['codes_per_second']:.2f} codes/second")
            logger.info(f"   Estimated time for 4000 codes: {(4000/best_config['codes_per_second'])/60:.1f} minutes")
        
        if batch_results:
            optimal_batch = min(batch_results, key=lambda x: x['time'])
            logger.info(f"✅ Optimal batch size: {optimal_batch['batch_size']}")
        
        if error_recovery_passed:
            logger.info("✅ Error recovery: PASSED")
        else:
            logger.info("❌ Error recovery: FAILED")
        
        logger.info("\n🎉 Performance testing completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)