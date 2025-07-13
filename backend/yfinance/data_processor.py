"""
TSE Stock Data Processor

Fetches TSE stock data from yfinance and stores it in SQLite database.
"""

import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sqlite3

warnings.filterwarnings('ignore')

# Load .env
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path)
DATA_DIR = "data"
DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "yfinance.db"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==== TSE Data Source ====
os.makedirs(DATA_DIR, exist_ok=True)

def init_db(db_path: str):
    """データベースとテーブルを初期化する"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            ticker TEXT PRIMARY KEY,
            longName TEXT,
            sector TEXT,
            industry TEXT,
            marketCap INTEGER,
            trailingPE REAL,
            forwardPE REAL,
            dividendYield REAL,
            website TEXT,
            currentPrice REAL,
            regularMarketPrice REAL,
            currency TEXT,
            exchange TEXT,
            shortName TEXT,
            previousClose REAL,
            open REAL,
            dayLow REAL,
            dayHigh REAL,
            volume INTEGER,
            averageDailyVolume10Day INTEGER,
            averageDailyVolume3Month INTEGER,
            fiftyTwoWeekLow REAL,
            fiftyTwoWeekHigh REAL,
            fiftyDayAverage REAL,
            twoHundredDayAverage REAL,
            beta REAL,
            priceToBook REAL,
            enterpriseValue INTEGER,
            profitMargins REAL,
            grossMargins REAL,
            operatingMargins REAL,
            returnOnAssets REAL,
            returnOnEquity REAL,
            freeCashflow INTEGER,
            totalCash INTEGER,
            totalDebt INTEGER,
            earningsGrowth REAL,
            revenueGrowth REAL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        logger.info(f"Database initialized at {db_path}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def save_stock_info_to_db(info: Dict[str, Any]):
    """yfinanceのticker.infoをデータベースに保存（または更新）する"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        stock_data = (
            info.get('symbol'),
            info.get('longName'),
            info.get('sector'),
            info.get('industry'),
            info.get('marketCap'),
            info.get('trailingPE'),
            info.get('forwardPE'),
            info.get('dividendYield'),
            info.get('website'),
            info.get('currentPrice'),
            info.get('regularMarketPrice'),
            info.get('currency'),
            info.get('exchange'),
            info.get('shortName'),
            info.get('previousClose'),
            info.get('open'),
            info.get('dayLow'),
            info.get('dayHigh'),
            info.get('volume'),
            info.get('averageDailyVolume10Day'),
            info.get('averageDailyVolume3Month'),
            info.get('fiftyTwoWeekLow'),
            info.get('fiftyTwoWeekHigh'),
            info.get('fiftyDayAverage'),
            info.get('twoHundredDayAverage'),
            info.get('beta'),
            info.get('priceToBook'),
            info.get('enterpriseValue'),
            info.get('profitMargins'),
            info.get('grossMargins'),
            info.get('operatingMargins'),
            info.get('returnOnAssets'),
            info.get('returnOnEquity'),
            info.get('freeCashflow'),
            info.get('totalCash'),
            info.get('totalDebt'),
            info.get('earningsGrowth'),
            info.get('revenueGrowth'),
            datetime.datetime.now()
        )

        cursor.execute("""
        INSERT OR REPLACE INTO stocks (
            ticker, longName, sector, industry, marketCap,
            trailingPE, forwardPE, dividendYield, website,
            currentPrice, regularMarketPrice, currency, exchange, shortName,
            previousClose, open, dayLow, dayHigh, volume,
            averageDailyVolume10Day, averageDailyVolume3Month,
            fiftyTwoWeekLow, fiftyTwoWeekHigh, fiftyDayAverage, twoHundredDayAverage,
            beta, priceToBook, enterpriseValue, profitMargins, grossMargins,
            operatingMargins, returnOnAssets, returnOnEquity, freeCashflow,
            totalCash, totalDebt, earningsGrowth, revenueGrowth, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, stock_data)

        conn.commit()
        logger.debug(f"[{info.get('symbol')}] Successfully saved/updated to DB.")

    except Exception as e:
        logger.error(f"Error saving stock info for {info.get('symbol')}: {e}")
    finally:
        if conn:
            conn.close()

def download_tse_listed_stocks():
    """Downloads the TSE listed stocks excel file from the JPX website."""
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        excel_link = soup.find("a", href=lambda href: href and (".xls" in href or ".xlsx" in href))

        if excel_link:
            excel_url = urljoin(url, excel_link["href"])
            excel_response = requests.get(excel_url)
            excel_response.raise_for_status()
            file_extension = ".xlsx" if excel_url.endswith(".xlsx") else ".xls"
            file_path = os.path.join(DATA_DIR, f"tse_listed_stocks{file_extension}")
            with open(file_path, "wb") as f:
                f.write(excel_response.content)
            logger.info(f"Successfully downloaded TSE listed stocks to {file_path}")
            return file_path
        else:
            logger.error("Could not find the link to the excel file.")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading TSE listed stocks: {e}")
        return None

def fetch_and_store_tse_data(max_workers: int = 5, delay: float = 0.5) -> None:
    """Fetch all TSE stock data and store in SQLite database."""
    logger.info("Fetching all TSE data and storing in database...")
    start_time = time.time()
    
    file_path = download_tse_listed_stocks()
    if not file_path:
        return

    try:
        df = pd.read_excel(file_path, header=0)
        df = df[df['市場・商品区分'].str.contains('ETF') == False]
        df.rename(columns={
            'コード': 'symbol',
            '銘柄名': 'name',
            '33業種区分': 'sector'
        }, inplace=True)
        df['symbol'] = df['symbol'].astype(str)

    except Exception as e:
        logger.error(f"Failed to load TSE stock list: {e}")
        return

    all_stocks = []
    for _, row in df.iterrows():
        symbol = row["symbol"] + ".T"
        name = row["name"]
        sector = row["sector"]
        all_stocks.append((symbol, name, sector))

    logger.info(f"Fetching data for {len(all_stocks)} stocks...")

    def fetch_single(symbol: str, name: str, sector: str) -> None:  # name and sector are used for validation
        try:
            time.sleep(delay)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info and 'symbol' in info:
                save_stock_info_to_db(info)

        except Exception as e:
            logger.error(f"Fetch failed for {symbol}: {e}")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single, *stock): stock for stock in all_stocks}
            completed_count = 0
            total_count = len(futures)

            for future in as_completed(futures):
                future.result()
                completed_count += 1

                if completed_count % 100 == 0 or completed_count == total_count:
                    logger.info(f"Progress: {completed_count}/{total_count} stocks processed")

    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")

    load_time = time.time() - start_time
    logger.info(f"Successfully processed {len(all_stocks)} stocks in {load_time:.2f} seconds")

# ==== Main Functions ====

class TSEDataProcessor:
    def __init__(self, max_workers: int = 4, rate_limit_delay: float = 0.7):
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        
        os.makedirs(DATA_DIR, exist_ok=True)
        init_db(DB_PATH)

    def run(self) -> None:
        """Run the data fetching and storage process."""
        logger.info("Starting TSE data processing...")
        fetch_and_store_tse_data(max_workers=self.max_workers, delay=self.rate_limit_delay)
        logger.info("TSE data processing completed.")