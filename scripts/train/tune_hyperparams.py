# scripts/train/tune_hyperparams.py
import json
import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from quant_lab.config import DATA_DIR, RECORDS_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, df_clean, feature_cols, X_train, y_train, X_val, y_val):
    """
    贝叶斯优化核心目标函数
    trial会自动在设定的参数空间内进行智能抽样
    """
    # 1. 定义超参数的贝叶斯搜索空间
    params = {
        # 树的棵树：不能太多，防止过拟合
        'n_estimators': trial.suggest_int('n_estimators', 40, 150, step=10),
        # 学习率：越小越稳健，但需要配合更多的树
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),
        # 树的最大深度：金融高噪声数据，严禁树太深，限制在2到7之间
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        # 叶子节点最小样本数：压制过拟合的核心武器
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100, step=10),
        'random_state': 42
    }
    
    # 2. 实例化当前这组参数的模型
    model = GradientBoostingClassifier(**params)
    
    # 3. 拟合训练
    model.fit(X_train, y_train)
    
    # 4. 计算验证集准确率
    val_acc = model.score(X_val, y_val)
    
    # 5. 返回给优化器，目标是最大化这个验证集得分
    return val_acc

def run_tuning():
    print("\n" + "="*60)
    print(" QUANT LAB 贝叶斯超参数自动优化控制台")
    print("="*60)
    
    csv_paths = list(DATA_DIR.glob("*.csv"))
    if not csv_paths:
        print("[Error] 没有发现任何 .csv 数据集")
        return
        
    datasets = [p.stem for p in csv_paths]
    for idx, d in enumerate(datasets): 
        print(f"  [{idx}] -> {d}")
    d_choice = int(input("请选择要调参的数据集编号: ").strip())
    dataset_name = datasets[d_choice]
    
    # 加载数据并对齐特征工程逻辑
    df = pd.read_csv(DATA_DIR / f"{dataset_name}.csv", parse_dates=['date'])
    df['feat_ret_5d'] = df.groupby('instrument')['close'].pct_change(5)
    df['ma_10'] = df.groupby('instrument')['close'].transform(lambda x: x.rolling(10).mean())
    df['feat_ma_ratio'] = df['close'] / df['ma_10']
    df['prev_amount'] = df.groupby('instrument')['amount'].shift(1)
    df['feat_amount_change'] = np.log(df['amount'] / (df['prev_amount'] + 1e-9))
    df['feat_volatility'] = df.groupby('instrument')['close'].transform(lambda x: x.rolling(10).std())
    df['feat_roe'] = df.get('ROE', df.get('roe', 0.0))
    
    if 'Free_Cash_Flow' in df.columns:
        df['fcf_ma5'] = df.groupby('instrument')['Free_Cash_Flow'].transform(lambda x: x.rolling(5).mean())
        df['feat_fcf_trend'] = df['Free_Cash_Flow'] - df['fcf_ma5']
    else:
        df['feat_fcf_trend'] = 0.0
    df['feat_pe'] = df.get('PE_TTM', df.get('pe', 15.0))
    
    mkt_col = [c for c in ['market_cap', 'mkt_cap', 'Market_Cap'] if c in df.columns]
    df['feat_log_market_cap'] = np.log(df[mkt_col[0]] + 1e-9) if mkt_col else np.log(1e10)
    
    df['target_label'] = (df.groupby('instrument')['close'].shift(-1) > df['close']).astype(int)
    df_clean = df.dropna().copy()
    feature_cols = [c for c in df_clean.columns if c.startswith('feat_')]
    
    # 严格时序切分样本（与训练脚本完美一致）
    train_split_date = "2024-06-01"
    backtest_start_date = "2025-09-19"
    
    train_data = df_clean[df_clean['date'] <= train_split_date]
    val_data = df_clean[(df_clean['date'] > train_split_date) & (df_clean['date'] <= backtest_start_date)]
    
    X_train, y_train = train_data[feature_cols], train_data['target_label']
    X_val, y_val = val_data[feature_cols], val_data['target_label']
    
    # 创建Optuna贝叶斯研究工坊，优化方向为最大化（maximize）
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    n_trials = 40  # 设定迭代轰炸次数
    print(f"\n[Running] 正在启动 {n_trials} 轮贝叶斯演化研究，请稍候...")
    
    def callback(study, trial):
        print(f"  | ➔ 第 {trial.number+1:02d}/{n_trials} 轮 | 本轮ACC: {trial.value:.2%} | 历史最高ACC: {study.best_value:.2%}")

    study.optimize(
        lambda trial: objective(trial, df_clean, feature_cols, X_train, y_train, X_val, y_val), 
        n_trials=n_trials,
        callbacks=[callback]
    )
    
    print("\n" + "="*60)
    print(" 贝叶斯优化收敛完成！最强黄金超参数组合如下：")
    print("="*60)
    for k, v in study.best_params.items():
        print(f"   {k:<18} : {v}")
    print(f"   真验证集最高准确率: {study.best_value:.2%}")
    print("="*60)
    print("下一步：请将上方打印出来的超参数填入你的 train_model.py 中固化训练即可。")

if __name__ == "__main__":
    run_tuning()