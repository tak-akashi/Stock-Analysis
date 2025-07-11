import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sqlite3
import tempfile

# テスト対象のモジュールをインポート
from backend.jquants.data_processor import JQuantsDataProcessor

# テスト用の固定値を設定
TEST_REFRESH_TOKEN = "test_refresh_token"
TEST_ID_TOKEN = "test_id_token"

@pytest.fixture
def mock_requests():
    """ requests.post と requests.get をモック化する fixture """
    with patch('requests.post') as mock_post, patch('requests.get') as mock_get:
        # auth_refresh のレスポンスを設定
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'idToken': TEST_ID_TOKEN}
        )

        # APIエンドポイントごとに異なるレスポンスを返すように設定
        def get_side_effect(url, params=None, headers=None):
            if "listed/info" in url:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        'info': [
                            {'Code': '1301', 'CompanyName': '極洋'},
                            {'Code': '1305', 'CompanyName': 'ダイワ上場投信－トピックス'}
                        ]
                    }
                )
            elif "prices/daily_quotes" in url:
                code = params.get("code")
                if code == '1301':
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            'daily_quotes': [
                                {'Date': '2020-07-08', 'Code': '1301', 'Open': 1000, 'High': 1100, 'Low': 900, 'Close': 1050, 'Volume': 10000},
                            ]
                        }
                    )
                elif code == '1305':
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            'daily_quotes': [
                                {'Date': '2020-07-08', 'Code': '1305', 'Open': 2000, 'High': 2100, 'Low': 1900, 'Close': 2050, 'Volume': 20000},
                            ]
                        }
                    )
            return MagicMock(status_code=404, json=lambda: {"message": "Not Found"})

        mock_get.side_effect = get_side_effect
        yield mock_post, mock_get

@pytest.fixture
def processor(mock_requests):
    """ テスト用の JQuantsDataProcessor インスタンスを作成する fixture """
    # 環境変数を設定
    os.environ["JQUANTS_REFRESH_TOKEN"] = TEST_REFRESH_TOKEN
    processor = JQuantsDataProcessor()
    # テスト終了後に環境変数を削除
    del os.environ["JQUANTS_REFRESH_TOKEN"]
    return processor


def test_init_success(processor):
    """ JQuantsDataProcessor の初期化が成功することをテストする """
    assert processor._refresh_token == TEST_REFRESH_TOKEN
    assert processor._id_token == TEST_ID_TOKEN

def test_init_no_token():
    """ トークンがない場合に ValueError が発生することをテストする """
    if "JQUANTS_REFRESH_TOKEN" in os.environ:
        del os.environ["JQUANTS_REFRESH_TOKEN"]
    with pytest.raises(ValueError):
        JQuantsDataProcessor()

def test_get_listed_info(processor, mock_requests):
    """ 上場銘柄一覧の取得をテストする """
    df = processor.get_listed_info()
    assert not df.empty
    assert len(df) == 2
    assert df.iloc[0]['Code'] == '1301'

def test_get_daily_quotes(processor, mock_requests):
    """ 株価四本値の取得をテストする """
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - relativedelta(years=5)).strftime('%Y-%m-%d')
    df = processor.get_daily_quotes('1301', from_date, to_date)
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]['Code'] == '1301'

@patch('time.sleep', return_value=None) # time.sleep を無効化
def test_get_all_prices_for_past_5_years(mock_sleep, processor, mock_requests):
    """ 全銘柄の過去5年分の株価取得をテストする """
    # This method saves to DB and doesn't return a dataframe
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db = os.path.join(temp_dir, "test.db")
        processor.get_all_prices_for_past_5_years_to_db(temp_db)
        # Verify the database was created
        assert os.path.exists(temp_db)

@patch('time.sleep', return_value=None)
def test_main_saves_to_db(mock_sleep, processor, mock_requests):
    """ main関数がデータベースにデータを保存することをテストする """
    with patch('backend.jquants.data_processor.JQuantsDataProcessor') as mock_processor_class:
        # JQuantsDataProcessor のインスタンスをモック化
        mock_instance = MagicMock()
        mock_processor_class.return_value = mock_instance

        # get_all_prices_for_past_5_years の戻り値を設定
        mock_df = pd.DataFrame({
            'Date': ['2020-07-08', '2020-07-08'],
            'Code': ['1301', '1305'],
            'Open': [1000, 2000],
            'High': [1100, 2100],
            'Low': [900, 1900],
            'Close': [1050, 2050],
            'Volume': [10000, 20000]
        })
        mock_instance.get_all_prices_for_past_5_years.return_value = mock_df

        # 一時的なデータベースパスを使用
        with tempfile.TemporaryDirectory() as temp_dir:
            # dataディレクトリを一時ディレクトリ内に作成
            data_dir = os.path.join(temp_dir, 'data')
            os.makedirs(data_dir)
            db_path = os.path.join(data_dir, "test.db")

            # os.path.join が常に正しいパスを返すようにモック化
            def mock_path_join(*args):
                # backend/jquants/data_processor.py からの呼び出しを想定
                if args[-1] == 'data':
                    return data_dir
                if args[-1] == 'jquants.db':
                    return db_path
                return os.path.join(*args)

            with patch('backend.jquants.data_processor.os.path.join', side_effect=mock_path_join):
                from backend.jquants.data_processor import main
                main()

                # データベースが作成されたか確認
                assert os.path.exists(db_path)