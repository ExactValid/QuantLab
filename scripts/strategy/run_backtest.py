# scripts/strategy/run_backtest.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from quant_lab.config import DATA_DIR, MODELS_DIR, RECORDS_DIR, TRANSACTION_COST_RATE
# 🤝 完美对接你重构好的正确变量名 OPTIMIZER_REGISTRY
from scripts.strategy.optimizer_pool import OPTIMIZER_REGISTRY 
from scripts.strategy.html_renderer import generate_human_dashboard

def parse_dataset_from_name(model_id):
    parts = model_id.split('_')
    if len(parts) < 3: return "Unknown"
    if parts[0] == "EXP":
        return parts[1] # 提取如 CZKJ 这样的数据源名称
    return "Unknown"

def execute_pipeline(model_id, dataset_name, opt_key):
    opt_config = OPTIMIZER_REGISTRY[opt_key]
    opt_name = opt_config["name"]
    opt_func = opt_config["func"]
    
    model_path = MODELS_DIR / f"{model_id}.pkl"
    if not model_path.exists(): return
        
    artifacts = joblib.load(model_path)
    model = artifacts['model_body']
    feature_cols = artifacts['features_list']
    split_date_str = artifacts['split_date']
    # 动态捕获当前模型的验证集准确率
    val_accuracy = artifacts.get('val_performance', {}).get('accuracy', 'N/A') 
    
    csv_path = DATA_DIR / f"{dataset_name}.csv"
    if not csv_path.exists(): return
        
    df_all = pd.read_csv(csv_path, parse_dates=['date']).set_index(['date', 'instrument']).sort_index()
    
    # ==================== 🛠️ 特征工程全量对齐修复区 ====================
    df_all['feat_ret_5d'] = df_all.groupby('instrument')['close'].pct_change(5)
    df_all['ma_10'] = df_all.groupby('instrument')['close'].transform(lambda x: x.rolling(10).mean())
    df_all['feat_ma_ratio'] = df_all['close'] / df_all['ma_10']
    df_all['prev_amount'] = df_all.groupby('instrument')['amount'].shift(1)
    df_all['feat_amount_change'] = np.log(df_all['amount'] / (df_all['prev_amount'] + 1e-9))
    df_all['feat_volatility'] = df_all.groupby('instrument')['close'].transform(lambda x: x.rolling(10).std())
    df_all['feat_roe'] = df_all.get('ROE', df_all.get('roe', 0.0))
    
    # 💡 补齐先前引发 KeyError 崩溃的 3 个硬核特征，增强对缺失列的安全保护
    if 'Free_Cash_Flow' in df_all.columns:
        df_all['fcf_ma5'] = df_all.groupby('instrument')['Free_Cash_Flow'].transform(lambda x: x.rolling(5).mean())
        df_all['feat_fcf_trend'] = df_all['Free_Cash_Flow'] - df_all['fcf_ma5']
    else:
        df_all['feat_fcf_trend'] = 0.0
        
    df_all['feat_pe'] = df_all.get('PE_TTM', df_all.get('pe', 15.0))
    
    # 兼容处理市值字段并完成对数转换
    mkt_col = [c for c in ['market_cap', 'mkt_cap', 'Market_Cap'] if c in df_all.columns]
    if mkt_col:
        df_all['feat_log_market_cap'] = np.log(df_all[mkt_col[0]] + 1e-9)
    else:
        df_all['feat_log_market_cap'] = np.log(1e10)
    
    # 基础收益率映射
    df_all['daily_return'] = df_all.groupby('instrument')['close'].pct_change(1)
    df_all['next_daily_return'] = df_all.groupby('instrument')['daily_return'].shift(-1)
    # =============================================================
    
    df_ml = df_all.dropna().reset_index()
    df_test = df_ml[df_ml['date'] > split_date_str].copy()
    if df_test.empty: return
        
    df_test['pred_prob'] = model.predict_proba(df_test[feature_cols])[:, 1]
    df_history = df_ml.pivot(index='date', columns='instrument', values='daily_return').fillna(0)
    
    all_raw_dates = sorted(df_all.reset_index()['date'].unique())
    test_dates = [d for d in all_raw_dates if pd.Timestamp(d) > pd.Timestamp(split_date_str)]
    
    strategy_returns = {}
    position_records = {}  
    prob_records = {}      
    turnover_records = {}  
    last_weights_dict = {}
    
    # 🌟 方案A核心：上游概率 EMA 物理平滑器
    last_ema_prob = None 
    ALPHA = 0.25  
    
    for t_date in test_dates:
        t_date_str = pd.Timestamp(t_date).strftime('%Y-%m-%d')
        day_slice = df_test[df_test['date'] == t_date]
        
        if len(day_slice) < 1:
            strategy_returns[t_date_str] = 0.0
            position_records[t_date_str] = 0.0
            prob_records[t_date_str] = 0.5
            turnover_records[t_date_str] = 0.0
            continue
            
        valid_inst = [i for i in day_slice['instrument'].values if i in df_history.columns]
        if len(valid_inst) < 1:
            strategy_returns[t_date_str] = 0.0
            position_records[t_date_str] = 0.0
            prob_records[t_date_str] = 0.5
            turnover_records[t_date_str] = 0.0
            continue
            
        day_slice_valid = day_slice[day_slice['instrument'].isin(valid_inst)].set_index('instrument').loc[valid_inst]
        raw_prob = float(np.mean(day_slice_valid['pred_prob'].values))
        
        if last_ema_prob is None:
            ema_prob = raw_prob
        else:
            ema_prob = ALPHA * raw_prob + (1.0 - ALPHA) * last_ema_prob
        last_ema_prob = ema_prob
        
        if opt_config["type"] == "Single":
            weights = np.array([opt_func(np.array([ema_prob]), None)])
            actual_pos = float(weights[0])
        else:
            cov_matrix = df_history.loc[:t_date].tail(60)[valid_inst].cov().values + np.eye(len(valid_inst)) * 1e-4
            weights = opt_func(day_slice_valid['pred_prob'].values, cov_matrix)
            prob_limit = np.max(day_slice_valid['pred_prob'].values)
            leverage = 1.0 / (1.0 + np.exp(-35 * (prob_limit - 0.52))) if prob_limit >= 0.50 else 0.0
            weights = weights * leverage
            actual_pos = float(np.sum(weights))
            
        current_weights_dict = dict(zip(valid_inst, weights))
        all_assets = set(current_weights_dict.keys()).union(set(last_weights_dict.keys()))
        turnover = sum(abs(current_weights_dict.get(a, 0.0) - last_weights_dict.get(a, 0.0)) for a in all_assets)
        
        net_return = np.sum(np.array([current_weights_dict.get(i,0.0) for i in valid_inst]) * day_slice_valid['next_daily_return'].values) - (turnover * TRANSACTION_COST_RATE)
        strategy_returns[t_date_str] = float(net_return)
        position_records[t_date_str] = actual_pos  
        prob_records[t_date_str] = ema_prob       
        turnover_records[t_date_str] = float(turnover)  
        last_weights_dict = current_weights_dict
        
    s_ret = pd.Series(strategy_returns).sort_index()
    
    cum_prod = (1 + s_ret).prod() - 1
    ann_ret = (1 + cum_prod) ** (242.0 / len(s_ret)) - 1
    sharpe = (s_ret.mean() / (s_ret.std() + 1e-9)) * np.sqrt(242)
    cum_rolled = (1 + s_ret).cumprod()
    max_dd = ((cum_rolled - cum_rolled.cummax()) / cum_rolled.cummax()).min()
    
    daily_pct = s_ret.values
    win_rate = float(np.sum(daily_pct > 0) / len(daily_pct)) if len(daily_pct) > 0 else 0.0
    pos_pct = daily_pct[daily_pct > 0]
    neg_pct = daily_pct[daily_pct < 0]
    ratio_p_l = float(np.mean(pos_pct) / abs(np.mean(neg_pct))) if len(neg_pct) > 0 else 1.0
    
    # 🌟🌟🌟 版本化资产隔离舱：将独立的 pkl 模型 ID 作为独立的 json 文件名存储，防止相互冲刷！
    record_file_path = RECORDS_DIR / dataset_name / f"{model_id}.json"
    record_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if record_file_path.exists():
        with open(record_file_path, 'r', encoding='utf-8') as f: model_meta = json.load(f)
    else:
        model_meta = {
            "metadata": {
                "model_id": model_id,
                "source_dataset": f"{dataset_name}.csv",
                "features_used": feature_cols
            },
            "val_performance": {"accuracy": val_accuracy if isinstance(val_accuracy, str) else f"{val_accuracy:.2%}"},
            "backtest_records": {}
        }
        
    model_meta["backtest_records"][opt_name] = {
        "backtest_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optimizer_params": f"流派类型: {opt_config['type']} | EMA平滑开启(alpha=0.25)",
        "total_return": f"{cum_prod:.2%}",
        "annualized_return": f"{ann_ret:.2%}",
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown": f"{max_dd:.2%}",
        "win_rate": f"{win_rate:.1%}",           
        "profit_loss_ratio": round(ratio_p_l, 2),    
        "date_series": list(cum_rolled.index),
        "equity_series": [round(x, 4) for x in cum_rolled.values],
        "drawdown_series": [round(x, 4) for x in ((cum_rolled - cum_rolled.cummax()) / cum_rolled.cummax()).values],
        "position_series": [round(position_records[d], 4) for d in cum_rolled.index],
        "prob_series": [round(prob_records[d], 4) for d in cum_rolled.index],
        "turnover_series": [round(turnover_records[d], 4) for d in cum_rolled.index] 
    }
    with open(record_file_path, 'w', encoding='utf-8') as f: 
        json.dump(model_meta, f, ensure_ascii=False, indent=4)

def run_main():
    pkl_paths = list(MODELS_DIR.glob("*.pkl"))
    if not pkl_paths: return
    print("\n" + "="*80)
    model_list = []
    for idx, path in enumerate(pkl_paths):
        model_id = path.stem
        dataset_name = parse_dataset_from_name(model_id)
        model_list.append((model_id, dataset_name))
        print(f"[{idx:<3}] [{dataset_name:<11}] {model_id[:42]:<45}")
    print("="*80)
    
    m_choice = int(input("请选择需要回测的模型索引编号: ").strip())
    selected_model_id, selected_dataset = model_list[m_choice]

    print(" [ALL_M] -> 一键轰炸全部多股资产配置优化算法\n [ALL_S] -> 一键轰炸全部单股时序择时控仓算法")
    choice = input("请输入选择的流派编号或轰炸指令: ").strip()
    
    if choice == "ALL_M":
        for k, v in OPTIMIZER_REGISTRY.items():
            if v["type"] == "Multiple": execute_pipeline(selected_model_id, selected_dataset, k)
    elif choice == "ALL_S":
        for k, v in OPTIMIZER_REGISTRY.items():
            if v["type"] == "Single": execute_pipeline(selected_model_id, selected_dataset, k)
    else:
        execute_pipeline(selected_model_id, selected_dataset, choice)
            
    generate_human_dashboard(selected_dataset)
    print(f"[Success] 完美的资产多模型隔离看板生成成功: outputs/reports/Dashboard_{selected_dataset}.html")

if __name__ == "__main__":
    run_main()