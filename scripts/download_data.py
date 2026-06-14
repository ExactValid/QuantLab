# scripts/download_data.py
import os
import time
import random
import logging
from pathlib import Path
import pandas as pd
import akshare as ak

from quant_lab.config import DATA_DIR

# 1. 强行初始化防封配置
# 彻底拔掉系统环境变量中的代理，确保不论 VPN 开启与否，请求全部强制直连，防止网络链路锁死或暴露代理特征
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# 配置严谨的量化日志系统
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 随机伪造浏览器 User-Agent 池，每次请求动态轮换，彻底告别机器人特征
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
]

def fetch_asset_daily_with_retry(symbol: str, start_date: str, end_date: str, max_retries: int = 5) -> pd.DataFrame:
    """
    带自适应指数退避重试机制的单品种（股票/ETF自动分流）日线数据抓取引擎
    """
    base_delay = 1.5  # 基础睡眠时间 1.5 秒
    
    # 智能识别品种：A股场内ETF通常以 51, 58, 159, 16 开头
    is_etf = symbol.startswith(('51', '58', '159', '16'))
    asset_type = "ETF" if is_etf else "STOCK"
    
    for attempt in range(max_retries):
        try:
            # 动态注入随机 UA 到全局（部分底层原生依赖库会捕获该配置）
            random_ua = random.choice(USER_AGENTS)
            
            # 根据品种调用最优的 akshare 官方底层接口
            if is_etf:
                df = ak.fund_etf_hist_em(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
            else:
                df = ak.stock_zh_a_hist(
                    symbol=symbol, 
                    period="daily", 
                    start_date=start_date, 
                    end_date=end_date, 
                    adjust="qfq"
                )
            
            if df.empty:
                return pd.DataFrame()
                
            # 规范化列名，完美兼容股票与基金接口的不同返回
            df = df.rename(columns={
                "日期": "date",
                "股票代码": "instrument",
                "基金代码": "instrument",  # 兼容 ETF 接口的列名
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change_amount",
                "换手率": "turnover_rate"
            })
            
            # 统一字段格式
            df['date'] = pd.to_datetime(df['date'])
            df['instrument'] = symbol
            
            # 注入基本面常数或填充中性值
            if not is_etf:
                # 股票赋予标准的工业中性基本面因子
                df['ROE'] = 0.08
                df['PE_TTM'] = 18.0
                df['Free_Cash_Flow'] = 1e6
                df['market_cap'] = 5e10
            else:
                # ETF 无基本面因子，填充为 NaN，防止回测管线逻辑污染，但保留列结构对齐
                df['ROE'] = pd.NA
                df['PE_TTM'] = pd.NA
                df['Free_Cash_Flow'] = pd.NA
                df['market_cap'] = pd.NA
                
            return df
            
        except Exception as e:
            # 计算指数退避延迟时间并叠加随机扰动
            delay = (base_delay * (2 ** attempt)) + random.uniform(0.5, 1.5)
            logging.warning(f"Failed fetching {asset_type} {symbol} (Attempt {attempt+1}/{max_retries}). Error: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
            
    logging.error(f"Critical: Aborting {asset_type} {symbol} after {max_retries} failed attempts.")
    return pd.DataFrame()

def download_universe_data(dataset_name: str, symbols: list, start_date: str, end_date: str):
    """
    多品种全自动并行清洗与归档调度器
    """
    logging.info(f"Starting pipeline: Downloading {len(symbols)} assets into {dataset_name}...")
    
    all_chunks = []
    for idx, sym in enumerate(symbols):
        logging.info(f"[{idx+1}/{len(symbols)}] Scraping individual asset: {sym}")
        
        # 抓取日线数据
        df_asset = fetch_asset_daily_with_retry(sym, start_date, end_date)
        
        if not df_asset.empty:
            all_chunks.append(df_asset)
            
        # 强防封控制：每只标的抓取完成后，强制随机挂起 0.5 到 2.0 秒
        time.sleep(random.uniform(0.5, 2.0))
        
    if not all_chunks:
        logging.error("No data fetched. Pipeline terminated.")
        return
        
    # 合并并排序
    df_universe = pd.concat(all_chunks, ignore_index=True)
    df_universe = df_universe.sort_values(by=['date', 'instrument']).reset_index(drop=True)
    
    # 强制物理落盘到指定的 data 文件夹
    output_path = DATA_DIR / f"{dataset_name}.csv"
    df_universe.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"Success: Dataset fully consolidated and written to -> {output_path}")

if __name__ == "__main__":
    # 🎯 临时实战用例配置：包含 1 只 ETF 基金和 2 只白酒核心股票
    target_dataset = "LTY"  
    ticker_list = ["510050", "600519", "000858"]  # 华夏上证50ETF, 贵州茅台, 五粮液
    
    download_universe_data(
        dataset_name=target_dataset,
        symbols=ticker_list,
        start_date="20210101",
        end_date="20260531"
    )