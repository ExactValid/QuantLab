# scripts/train/manage_models.py
import json
import argparse
from pathlib import Path
from quant_lab.config import MODELS_DIR, RECORDS_DIR
from scripts.strategy.html_renderer import generate_human_dashboard

def wipe_single_model(model_id, dataset_name):
    """粉碎单个实验单元的物理资产与逻辑履历"""
    pkl_path = MODELS_DIR / f"{model_id}.pkl"
    if pkl_path.exists():
        pkl_path.unlink()
        print(f" -> Removed binary asset: outputs/models/{pkl_path.name}")
        
    json_path = RECORDS_DIR / dataset_name / f"{model_id}.json"
    if json_path.exists():
        json_path.unlink()
        print(f" -> Removed distributed record: records/{dataset_name}/{model_id}.json")

def delete_model_pipeline():
    parser = argparse.ArgumentParser(description="Quant Lab Extreme Asset Management Tools")
    parser.add_argument("-m", "--model", type=str, help="秒杀单个特定模型(不经过交互确认)")
    parser.add_argument("-d", "--dataset", type=str, help="一键格式化清空某个特定数据集下的所有实验")
    args = parser.parse_args()

    # 🚀 极速响应分支一：单点秒杀
    if args.model:
        parts = args.model.split('_')
        dataset_name = "22_CSI100" if (len(parts) > 2 and parts[1] == "22" and parts[2] == "CSI100") else parts[1]
        print(f"\n[Strike Mode] Wiping model: {args.model}")
        wipe_single_model(args.model, dataset_name)
        generate_human_dashboard(dataset_name)
        print("[Success] Cleaning command pipeline executed successfully.")
        return

    # 🚀 极速响应分支二：全板块推倒重来
    if args.dataset:
        target_cabin = RECORDS_DIR / args.dataset
        print(f"\n[Danger Zone] Resetting everything under data channel: {args.dataset}")
        if target_cabin.exists():
            for json_path in list(target_cabin.glob("*.json")):
                wipe_single_model(json_path.stem, args.dataset)
        generate_human_dashboard(args.dataset)
        print(f"[Success] All histories linked to [{args.dataset}] have been neutralized.")
        return

    # 🔄 分支三：标准交互菜单（当没有传入命令行参数时自动降级启动）
    json_paths = list(RECORDS_DIR.rglob("*.json"))
    if not json_paths:
        print("Notification: No distributed experiment records found.")
        return

    print("\n" + "="*95)
    print(f"{'Index':<6}{'Dataset':<15}{'Model ID':<45}{'Val Acc':<10}{'Timestamp'}")
    print("="*95)
    
    record_list = []
    for idx, path in enumerate(json_paths):
        try:
            with open(path, 'r', encoding='utf-8') as f: 
                model_data = json.load(f)
            acc = model_data.get("val_performance", {}).get("accuracy", "N/A")
            t_time = model_data.get("metadata", {}).get("experiment_time", "N/A")
            record_list.append({"model_id": path.stem, "dataset_name": path.parent.name, "record_path": path})
            print(f"[{idx:<3}] [{path.parent.name:<11}] {path.stem[:42]:<45}{acc:<10}{t_time}")
        except Exception: 
            continue
    print("="*95)

    try:
        choice = input("Enter Index to DELETE (or 'q' to quit): ").strip()
        if choice.lower() == 'q': return
        target_item = record_list[int(choice)]
    except Exception: 
        return

    confirm = input(f"Confirm wiping {target_item['model_id']}? (y/n): ").strip().lower()
    if confirm != 'y': return

    wipe_single_model(target_item['model_id'], target_item['dataset_name'])
    generate_human_dashboard(target_item["dataset_name"])
    print(f"\n[Success] Targets safely liquidated. Board updated.")

if __name__ == "__main__":
    delete_model_pipeline()