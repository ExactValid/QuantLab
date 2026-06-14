# quant_lab/config.py
from pathlib import Path

# 定位项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 核心数据源目录
DATA_DIR = BASE_DIR / "data"

# 系统日志目录
LOGS_DIR = BASE_DIR / "logs"

# 输出资产大类目录
OUTPUTS_DIR = BASE_DIR / "outputs"

# 物理资产舱：存放训练好的模型二进制文件 (.pkl)
MODELS_DIR = OUTPUTS_DIR / "models"

# 实验记录舱：按数据源二级物理分舱存放分布式实验履历 (.json)
RECORDS_DIR = OUTPUTS_DIR / "records"

# 表现层看板舱：按数据源独立存放全景 HTML 看板
REPORTS_DIR = OUTPUTS_DIR / "reports"

# 依赖修复：为可能存在的旧版脚本保留统一大账本的读写物理路径
LEDGER_PATH = OUTPUTS_DIR / "experiment_ledger.json"

# 自动化物理初始化：确保所有物理目录在导入配置时自动建立
for target_dir in [DATA_DIR, LOGS_DIR, MODELS_DIR, RECORDS_DIR, REPORTS_DIR]:
    target_dir.mkdir(parents=True, exist_ok=True)

# 基础交易摩擦费率配置（包含印花税、佣金、滑点综合度量）
TRANSACTION_COST_RATE = 0.0015