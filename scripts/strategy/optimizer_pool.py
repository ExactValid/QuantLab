# scripts/strategy/optimizer_pool.py
import numpy as np

# ==================== 板块一：横截面多资产配置算法 (Multiple Assets) ====================

def run_gurobi_safe(alpha, cov_matrix, gurobi_solver_func=None):
    if gurobi_solver_func:
        return gurobi_solver_func(alpha, cov_matrix, lmbda=3.0, max_w=0.2)
    return np.ones(len(alpha)) / len(alpha)

def run_gurobi_aggressive(alpha, cov_matrix, gurobi_solver_func=None):
    if gurobi_solver_func:
        return gurobi_solver_func(alpha, cov_matrix, lmbda=0.1, max_w=0.4)
    return np.ones(len(alpha)) / len(alpha)

def run_top5_equal_weight(alpha, cov_matrix, gurobi_solver_func=None):
    n_assets = len(alpha)
    k = min(5, n_assets)
    top_indices = np.argsort(alpha)[-k:]
    weights = np.zeros(n_assets)
    weights[top_indices] = 1.0 / k
    return weights

def run_gurobi_unconstrained(alpha, cov_matrix, gurobi_solver_func=None):
    if gurobi_solver_func:
        return gurobi_solver_func(alpha, cov_matrix, lmbda=1.0, max_w=1.0)
    return np.ones(len(alpha)) / len(alpha)


# ==================== 板块二：纵向单股时序择时控仓算法 (Single Asset) ====================

def _sigmoid_sizer(prob, k, p0):
    """
    底层连续动态控仓 Sigmoid 核心函数
    """
    prob = np.clip(prob, 0.001, 0.999)
    weight = 1.0 / (1.0 + np.exp(-k * (prob - p0)))
    # 当预测概率低于多空分水岭 50% 时主动清仓，且过滤掉低于 5% 的无效碎股微调
    return float(weight) if prob >= 0.50 and weight >= 0.05 else 0.0

def run_sigmoid_balanced(alpha, cov_matrix=None):
    # ====================================================================
    #  降维修复：将原本过分陡峭的 k=35 暴力调降到 7.0
    # 让经过 EMA 平滑后的温和概率（如 53% ~ 56%）能丝滑滑行在 55% ~ 75% 的多级丰富持仓带
    # ====================================================================
    return _sigmoid_sizer(alpha[0], k=7.0, p0=0.52)

def run_sigmoid_aggressive(alpha, cov_matrix=None):
    # ====================================================================
    #  降维修复：将激进型爆破斜率 k=50 调降到 10.0，回归理性连续控仓
    # ====================================================================
    return _sigmoid_sizer(alpha[0], k=10.0, p0=0.50)

def run_kelly_half(alpha, cov_matrix=None):
    p = np.clip(alpha[0], 0.001, 0.999)
    # 凯利公式博弈（设定赔率 1.2）
    kelly_w = (p * 1.2 - (1.0 - p)) / 1.2
    return float(np.clip(kelly_w * 0.5, 0.0, 1.0)) if kelly_w > 0 else 0.0

def run_threshold_ladder(alpha, cov_matrix=None):
    """
    双阶梯硬切控仓。
    概率被 EMA 烫平后，该刚性阶梯刚好形成了极佳的半仓/满仓稳健风控防御网。
    """
    p = float(alpha[0])
    if p > 0.58: return 1.0
    if p > 0.53: return 0.5
    if p > 0.50: return 0.1
    return 0.0


# ==================== 🗂️ 优化器双板块中央注册池 ====================

# 正确命名为 OPTIMIZER_REGISTRY，完美对接修复后的 run_backtest.py 脚本
OPTIMIZER_REGISTRY = {
    # 多股资产配置板块
    "M0": {"name": "Gurobi_Safe_Risk3.0", "func": run_gurobi_safe, "type": "Multiple", "desc": "多股马科维茨稳健配置"},
    "M1": {"name": "Gurobi_Aggressive_Risk0.1", "func": run_gurobi_aggressive, "type": "Multiple", "desc": "多股马科维茨激进配置"},
    "M2": {"name": "Top5_Equal_Weight", "func": run_top5_equal_weight, "type": "Multiple", "desc": "多股Alpha前5等权平铺"},
    "M3": {"name": "Gurobi_Unconstrained_Risk1.0", "func": run_gurobi_unconstrained, "type": "Multiple", "desc": "多股无个股上限限制优化"},
    
    # 单股择时控仓板块
    "S0": {"name": "Sigmoid_Balanced", "func": run_sigmoid_balanced, "type": "Single", "desc": "单股均衡型 S 曲线控仓"},
    "S1": {"name": "Sigmoid_Aggressive", "func": run_sigmoid_aggressive, "type": "Single", "desc": "单股激进型爆破控仓"},
    "S2": {"name": "Kelly_Sizer_Half", "func": run_kelly_half, "type": "Single", "desc": "单股半凯利公式博弈"},
    "S3": {"name": "Threshold_Ladder", "func": run_threshold_ladder, "type": "Single", "desc": "单股置信度阶梯硬切风控"}
}