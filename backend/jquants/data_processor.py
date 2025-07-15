import os
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import sqlite3
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

API_URL = "https://api.jquants.com"

class JQuantsDataProcessor:
    def __init__(self, refresh_token: Optional[str] = None):
        """
        Args:
            refresh_token (str, optional): J-Quants APIのリフレッシュトークン. Defaults to None.
                                           指定しない場合は .env ファイルまたは環境変数 JQUANTS_REFRESH_TOKEN が使用されます.
        """
        self._refresh_token = refresh_token or os.getenv("JQUANTS_REFRESH_TOKEN")
        if not self._refresh_token:
            raise ValueError("リフレッシュトークンが指定されていないか、.env ファイルまたは環境変数 JQUANTS_REFRESH_TOKEN が設定されていません。")
        self._id_token = self._get_id_token()

    def _get_id_token(self) -> str:
        """リフレッシュトークンを元にIDトークンを取得する"""
        res = requests.post(f"{API_URL}/v1/token/auth_refresh?refreshtoken={self._refresh_token}")
        if res.status_code == 200:
            print("idTokenの取得に成功しました。")
            return res.json()['idToken']
        else:
            raise Exception(f"idTokenの取得に失敗しました: {res.text}")

    @property
    def _headers(self) -> dict:
        """APIリクエスト用のヘッダーを返す"""
        return {'Authorization': f'Bearer {self._id_token}'}

    def get_listed_info(self) -> pd.DataFrame:
        """
        東証上場銘柄の一覧を取得します。
        
        Returns:
            pd.DataFrame: 上場銘柄一覧のDataFrame
        """
        print("上場銘柄一覧を取得しています...")
        params = {}
        res = requests.get(f"{API_URL}/v1/listed/info", params=params, headers=self._headers)
        
        if res.status_code != 200:
            raise Exception(f"銘柄一覧の取得に失敗しました: {res.text}")

        d = res.json()
        data = d["info"]
        while "pagination_key" in d:
            params["pagination_key"] = d["pagination_key"]
            res = requests.get(f"{API_URL}/v1/listed/info", params=params, headers=self._headers)
            if res.status_code != 200:
                raise Exception(f"ページネーション中の銘柄一覧取得に失敗しました: {res.text}")
            d = res.json()
            data += d["info"]
        
        df = pd.DataFrame(data)
        print(f"{len(df)}件の銘柄情報を取得しました。")
        return df

    def get_daily_quotes(self, code: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        指定された銘柄コードと期間の株価四本値を取得します。
        
        Args:
            code (str): 銘柄コード
            from_date (str): 取得開始日 (YYYY-MM-DD)
            to_date (str): 取得終了日 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 株価四本値のDataFrame
        """
        params = {
            "code": code,
            "from": from_date,
            "to": to_date,
        }
        res = requests.get(f"{API_URL}/v1/prices/daily_quotes", params=params, headers=self._headers)

        if res.status_code != 200:
            print(f"銘柄コード: {code} の株価取得に失敗しました: {res.text}")
            return pd.DataFrame()

        d = res.json()
        data = d["daily_quotes"]
        while "pagination_key" in d:
            params["pagination_key"] = d["pagination_key"]
            res = requests.get(f"{API_URL}/v1/prices/daily_quotes", params=params, headers=self._headers)
            if res.status_code != 200:
                print(f"銘柄コード: {code} のページネーション中の株価取得に失敗しました: {res.text}")
                break
            d = res.json()
            data += d["daily_quotes"]
            
        return pd.DataFrame(data)

    def _get_last_date_for_code(self, db_path: str, code: str) -> str:
        """
        指定された銘柄コードの最新データ日付を取得します。
        
        Args:
            db_path (str): SQLiteデータベースのパス
            code (str): 銘柄コード
            
        Returns:
            str: 最新データの日付 (YYYY-MM-DD形式)、データが存在しない場合は5年前の日付
        """
        try:
            with sqlite3.connect(db_path) as con:
                cursor = con.cursor()
                cursor.execute('''
                    SELECT MAX(Date) FROM daily_quotes WHERE Code = ?
                ''', (code,))
                result = cursor.fetchone()
                
                if result[0]:
                    return result[0]
                else:
                    # データが存在しない場合は5年前から開始
                    return (datetime.now() - relativedelta(years=5)).strftime('%Y-%m-%d')
        except Exception:
            # テーブルが存在しない場合も5年前から開始
            return (datetime.now() - relativedelta(years=5)).strftime('%Y-%m-%d')

    def get_all_prices_for_past_5_years_to_db(self, db_path: str):
        """
        すべての東証上場銘柄について、過去5年分の株価四本値を取得し、銘柄ごとにDBに保存します。
        
        Args:
            db_path (str): SQLiteデータベースのパス
        """
        listed_info_df = self.get_listed_info()
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - relativedelta(years=5)).strftime('%Y-%m-%d')
        
        total_codes = len(listed_info_df)
        successful_codes = 0
        failed_codes = []
        
        # データベースの初期化
        with sqlite3.connect(db_path) as con:
            # テーブルが存在しない場合に作成
            con.execute('''
                CREATE TABLE IF NOT EXISTS daily_quotes (
                    Code TEXT,
                    Date TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    Volume INTEGER,
                    TurnoverValue REAL,
                    AdjustmentFactor REAL,
                    AdjustmentOpen REAL,
                    AdjustmentHigh REAL,
                    AdjustmentLow REAL,
                    AdjustmentClose REAL,
                    AdjustmentVolume INTEGER,
                    PRIMARY KEY (Code, Date)
                )
            ''')
        
        for i, (_, row) in enumerate(listed_info_df.iterrows()):
            code = row["Code"]
            print(f"銘柄コード: {code} の株価を取得しています... ({int(i)+1}/{total_codes}) ({from_date} - {to_date})")
            
            # APIへの過度な負荷を避けるための待機処理
            time.sleep(0.5)
            try:
                quotes_df = self.get_daily_quotes(str(code), from_date, to_date)
                if not quotes_df.empty:
                    # 銘柄ごとにDBに保存
                    with sqlite3.connect(db_path) as con:
                        quotes_df.to_sql('daily_quotes', con, if_exists='append', index=False)
                    successful_codes += 1
                    print(f"  → {len(quotes_df)}件のデータを保存しました")
                else:
                    print(f"  → {code} のデータが取得できませんでした")
                    failed_codes.append(code)
            except Exception as e:
                print(f"  → エラーが発生しました: {e}")
                failed_codes.append(code)
        
        print(f"株価データの取得が完了しました。成功: {successful_codes}件、失敗: {len(failed_codes)}件")
        if failed_codes:
            print(f"失敗した銘柄コード: {failed_codes[:10]}{'...' if len(failed_codes) > 10 else ''}")

    def update_prices_to_db(self, db_path: str):
        """
        すべての東証上場銘柄について、DBに保存されている最新日付以降の株価データを取得し、
        銘柄ごとにDBに保存します（差分更新）。
        
        Args:
            db_path (str): SQLiteデータベースのパス
        """
        listed_info_df = self.get_listed_info()
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        total_codes = len(listed_info_df)
        successful_codes = 0
        failed_codes = []
        updated_codes = 0
        
        # データベースの初期化
        with sqlite3.connect(db_path) as con:
            # テーブルが存在しない場合に作成
            con.execute('''
                CREATE TABLE IF NOT EXISTS daily_quotes (
                    Code TEXT,
                    Date TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    Volume INTEGER,
                    TurnoverValue REAL,
                    AdjustmentFactor REAL,
                    AdjustmentOpen REAL,
                    AdjustmentHigh REAL,
                    AdjustmentLow REAL,
                    AdjustmentClose REAL,
                    AdjustmentVolume INTEGER,
                    PRIMARY KEY (Code, Date)
                )
            ''')
        
        for i, (_, row) in enumerate(listed_info_df.iterrows()):
            code = row["Code"]
            
            # 各銘柄の最新データ日付を取得
            last_date = self._get_last_date_for_code(db_path, str(code))
            
            # 最新データの翌日から今日までを取得対象とする
            try:
                last_datetime = datetime.strptime(last_date, '%Y-%m-%d')
                from_date = (last_datetime + relativedelta(days=1)).strftime('%Y-%m-%d')
            except ValueError:
                # 日付の解析に失敗した場合は5年前から開始
                from_date = (datetime.now() - relativedelta(years=5)).strftime('%Y-%m-%d')
            
            # 開始日が今日より後の場合はスキップ
            if from_date > to_date:
                print(f"銘柄コード: {code} は最新データです。スキップします... ({i+1}/{total_codes})")
                continue
            
            # APIへの過度な負荷を避けるための待機処理
            time.sleep(0.5)
            
            try:
                quotes_df = self.get_daily_quotes(str(code), from_date, to_date)
                if not quotes_df.empty:
                    # 銘柄ごとにDBに保存
                    with sqlite3.connect(db_path) as con:
                        quotes_df.to_sql('daily_quotes', con, if_exists='append', index=False)
                    successful_codes += 1
                    updated_codes += 1
                    print(f"銘柄コード: {code} の{len(quotes_df)}件の株価データを保存しました")
                else:
                    print(f"  → {code} のデータが取得できませんでした")
                    successful_codes += 1  # データがないだけでエラーではない
            except Exception as e:
                print(f"  → エラーが発生しました: {e}")
                failed_codes.append(code)
        
        print(f"株価データの更新が完了しました。処理成功: {successful_codes}件、更新: {updated_codes}件、失敗: {len(failed_codes)}件")
        if failed_codes:
            print(f"失敗した銘柄コード: {failed_codes[:10]}{'...' if len(failed_codes) > 10 else ''}")

def main():
    """
    メイン処理
    東証全銘柄の株価データを取得し、銘柄ごとにSQLite3データベースに保存します。
    DBが存在しない場合は過去5年分、存在する場合は差分更新を行います。
    """
    try:
        processor = JQuantsDataProcessor()
        
        # プロジェクトルートのdataディレクトリに保存
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(output_dir, exist_ok=True)
        db_path = os.path.join(output_dir, "jquants.db")
        
        # DBの存在確認
        db_exists = os.path.exists(db_path)
        
        if not db_exists:
            print("データベースが存在しません。過去5年分のデータを取得します。")
            # 初回実行時：過去5年分の株価データを取得
            processor.get_all_prices_for_past_5_years_to_db(db_path)
        else:
            print("データベースが存在します。差分更新を行います。")
            # 2回目以降：差分更新
            processor.update_prices_to_db(db_path)
        
        # 保存したデータの確認
        with sqlite3.connect(db_path) as con:
            # テーブルの存在確認
            cursor = con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_quotes'")
            if cursor.fetchone():
                # データの件数確認
                count_df = pd.read_sql('SELECT COUNT(*) as count FROM daily_quotes', con)
                print(f"データベースに保存されたレコード数: {count_df.iloc[0]['count']}件")
                
                # 銘柄数の確認
                codes_df = pd.read_sql('SELECT COUNT(DISTINCT Code) as code_count FROM daily_quotes', con)
                print(f"保存された銘柄数: {codes_df.iloc[0]['code_count']}銘柄")
                
                # 最新データの日付範囲を表示
                date_range_df = pd.read_sql('SELECT MIN(Date) as min_date, MAX(Date) as max_date FROM daily_quotes', con)
                print(f"データの期間: {date_range_df.iloc[0]['min_date']} - {date_range_df.iloc[0]['max_date']}")
                
                # データの先頭5行を表示
                retrieved_df = pd.read_sql('SELECT * FROM daily_quotes ORDER BY Date DESC LIMIT 5', con)
                print("最新データの先頭5行:")
                print(retrieved_df)
            else:
                print("データベースにテーブルが作成されませんでした。")

    except (ValueError, Exception) as e:
        print(f"エラーが発生しました: {e}")
        print(".env ファイルまたは環境変数 'JQUANTS_REFRESH_TOKEN' が正しく設定されているか確認してください。")


if __name__ == "__main__":
    main()