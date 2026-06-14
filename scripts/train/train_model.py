# scripts/train/train_model.py
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from quant_lab.config import DATA_DIR, MODELS_DIR, RECORDS_DIR

def train_pipeline(dataset_name, model_type):
    """全自动模型拟合与注册流水线（修复过拟合与虚假准确率）"""
    csv_path = DATA_DIR / f"{dataset_name}.csv"
    if not csv_path.exists():
        print(f"[Error] 找不到训练所需的基础数据源: {csv_path}")
        return
        
    df = pd.read_csv(csv_path, parse_dates=['date'])
    
    # ==================== 🧱 统一对齐的特征工程区 ====================
    df['feat_ret_5d'] = df.groupby('instrument')['close'].pct_change(5)
    df['ma_10'] = df.groupby('instrument')['close'].transform(lambda x: x.rolling(10).mean())
    df['feat_ma_ratio'] = df['close'] / df['ma_10']
    df['prev_amount'] = df.groupby('instrument')['amount'].shift(1)
    df['feat_amount_change'] = np.log(df['amount'] / (df_all['prev_amount'] + 1e-9) if 'df_all' in locals() else df['amount'] / (df['prev_amount'] + 1e-9))
    df['feat_volatility'] = df.groupby('instrument')['close'].transform(lambda x: x.rolling(10).std())
    df['feat_roe'] = df.get('ROE', df.get('roe', 0.0))
    
    if 'Free_Cash_Flow' in df.columns:
        df['fcf_ma5'] = df.groupby('instrument')['Free_Cash_Flow'].transform(lambda x: x.rolling(5).mean())
        df['feat_fcf_trend'] = df['Free_Cash_Flow'] - df['fcf_ma5']
    else:
        df['feat_fcf_trend'] = 0.0
        
    df['feat_pe'] = df.get('PE_TTM', df.get('pe', 15.0))
    
    mkt_col = [c for c in ['market_cap', 'mkt_cap', 'Market_Cap'] if c in df.columns]
    if mkt_col:
        df['feat_log_market_cap'] = np.log(df[mkt_col[0]] + 1e-9)
    else:
        df['feat_log_market_cap'] = np.log(1e10)
        
    # 标签：预测明天是否上涨
    df['target_label'] = (df.groupby('instrument')['close'].shift(-1) > df['close']).astype(int)
    # =============================================================
    
    df_clean = df.dropna().copy()
    feature_cols = [c for c in df_clean.columns if c.startswith('feat_')]
    
    # 🌟 真正的全时段样本切分：2024-06-01之前训练，之后作为验证集
    train_split_date = "2024-06-01"
    backtest_start_date = "2025-09-19" # 留给回测的样本边界
    
    train_mask = df_clean['date'] <= train_split_date
    val_mask = (df_clean['date'] > train_split_date) & (df_clean['date'] <= backtest_start_date)
    
    train_data = df_clean[train_mask]
    val_data = df_clean[val_mask]
    
    if train_data.empty or val_data.empty:
        print("[Error] 样本切分失败，请检查数据集时间跨度。")
        return
        
    X_train, y_train = train_data[feature_cols], train_data['target_label']
    X_val, y_val = val_data[feature_cols], val_data['target_label']
    
    # 🌟 限制树深，防止模型无脑死记硬背
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42, n_jobs=-1)
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(n_estimators=60, learning_rate=0.03, max_depth=3, random_state=42)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        print(f"[Error] 未定义该类型的机器学习模型: {model_type}")
        return
        
    # 拟合
    model.fit(X_train, y_train)
    
    # 🌟 计算真正的验证集准确率
    train_acc = model.score(X_train, y_train)
    real_val_acc = model.score(X_val, y_val)
    
    print(f"\n[🔬 拟合报告] 训练集记忆率: {train_acc:.2%} | 🌟 真实验证集准确率: {real_val_acc:.2%}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"EXP_{dataset_name}_{model_type}_{timestamp}"
    
    # 固化导出
    model_path = MODELS_DIR / f"{model_id}.pkl"
    artifacts = {
        'model_body': model,
        'features_list': feature_cols,
        'split_date': backtest_start_date, # 对接给回测的起始边界
        'source_dataset': f"{dataset_name}.csv",
        'val_performance': {"accuracy": real_val_acc} # 将真正的验证集表现固化进去
    }
    joblib.dump(artifacts, model_path)
    print(f"[Train Success] 物理资产已固化: outputs/models/{model_id}.pkl")
    
    # 注册履历
    target_cabin_dir = RECORDS_DIR / dataset_name
    target_cabin_dir.mkdir(parents=True, exist_ok=True)
    
    initial_meta = {
        "metadata": {
            "experiment_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split_date": backtest_start_date,
            "features_used": feature_cols,
            "source_dataset": f"{dataset_name}.csv"
        },
        "model_hyperparameters": model.get_params() if hasattr(model, 'get_params') else {},
        "val_performance": { "accuracy": f"{real_val_acc:.2%}" }, # 注册真实的验证成绩
        "backtest_records": {}
    }
    
    with open(target_cabin_dir / f"{model_id}.json", 'w', encoding='utf-8') as f:
        json.dump(initial_meta, f, ensure_ascii=False, indent=4)
    print(f"[Register Success] 履历已注册: records/{dataset_name}/{model_id}.json")

def run_menu():
    csv_paths = list(DATA_DIR.glob("*.csv"))
    if not csv_paths:
        print("[Error] data/ 目录下没有任何 .csv 数据源文件。")
        return
        
    print("\n" + "="*60)
    print(" QUANT LAB 机器学习模型训练发射台 (防作弊修正版)")
    print("="*60)
    
    print("[第一步] 请选择本次实验的目标数据集:")
    datasets = [p.stem for p in csv_paths]
    for idx, d_name in enumerate(datasets):
        print(f"  [{idx}] -> {d_name}")
    try:
        d_choice = int(input("请输入数据集编号: ").strip())
        selected_dataset = datasets[d_choice]
    except Exception:
        return
        
    print("\n[第二步] 请选择要拟合的机器学习核心算法:")
    model_types = ["RandomForest", "GradientBoosting", "LogisticRegression"]
    for idx, m_type in enumerate(model_types):
        print(f"  [{idx}] -> {m_type}")
    try:
        m_choice = int(input("请输入算法编号: ").strip())
        selected_model = model_types[m_choice]
    except Exception:
        return
        
    train_pipeline(selected_dataset, selected_model)

if __name__ == "__main__":
    run_menu()