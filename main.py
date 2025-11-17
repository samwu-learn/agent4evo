
from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from config import ReevoConfig
from utils.utils import setup_logging, _safe_load_yaml, _ensure_required_imports
from utils.llm_interface import OpenAIEmbedding
from evo_agent import EvoAgent
from pool_interface import ExperiencePool, ChromaMemoryBackend, JSONMemoryBackend
from planner_interface import PlannerInterface  # 新增：导入规划器


def run_validation(config, best_info):
    """基于最佳解重新执行验证阶段，失败时返回 None。"""
    best_dir = config.paths.best_dir
    best_dir.mkdir(parents=True, exist_ok=True)
    target_py = (
        config.project_root
        / "problems"
        / config.problem.problem_name
        / "gpt.py"
    )

    # 准备待验证代码：优先使用内存文本，其次尝试从文件读取。
    code_text = best_info.get("code")
    code_text = _ensure_required_imports(code_text)
    target_py.parent.mkdir(parents=True, exist_ok=True)

    try:
        target_py.write_text(code_text.rstrip() + "\n", encoding="utf-8")
    except OSError as exc:
        logging.warning("写入验证代码失败，跳过验证: %s", exc)
        return None

    eval_script_name = (
        "eval_black_box.py"
        if config.problem.problem_type == "black_box"
        else "eval.py"
    )
    eval_script = (
        config.project_root
        / "problems"
        / config.problem.problem_name
        / eval_script_name
    )
    if not eval_script.is_file():
        logging.warning("验证脚本不存在，跳过验证: %s", eval_script)
        return None

    val_stdout = best_dir / "validation_stdout.txt"
    env = os.environ.copy()
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{config.project_root}:{existing_py}"
        if existing_py
        else str(config.project_root)
    )

    logging.info("运行验证脚本: %s", eval_script)
    try:
        with open(val_stdout, "w", encoding="utf-8") as stdout:
            subprocess.run(
                ["python", str(eval_script), "-1", str(config.project_root), "val"],
                stdout=stdout,
                stderr=stdout,
                env=env,
                check=False,
            )
    except OSError as exc:
        logging.warning("执行验证脚本失败: %s", exc)
        return None

    logging.info("验证输出保存至: %s", val_stdout)
    return val_stdout if val_stdout.exists() else None


def build_experience_pool(config: ReevoConfig, config_path, llm):
    """根据配置初始化 ExperiencePool """

    pool_config = _safe_load_yaml(config_path)

    memory_cfg = pool_config.get("memory")
    backend_name = memory_cfg.get("backend")

    if backend_name == "chroma":
        memory_backend = ChromaMemoryBackend(**memory_cfg.get("chroma", {}))
    else:
        json_cfg = dict(memory_cfg.get("json", {}))
        default_filename = json_cfg.get("filepath", "memory.json")
        run_memory_path = (config.paths.run_dir / default_filename).resolve()
        run_memory_path.parent.mkdir(parents=True, exist_ok=True)
        json_cfg["filepath"] = str(run_memory_path)
        memory_backend = JSONMemoryBackend(**json_cfg)
        logging.info("Memory backend initialized at: %s", run_memory_path)

    embmodel_cfg = pool_config.get("emb_model", {})
    embedding_model = OpenAIEmbedding(
        api_key=embmodel_cfg["api_key"],
        base_url=embmodel_cfg["base_url"],
        model=embmodel_cfg["embedding_model"],
        **embmodel_cfg.get("request_kwargs", {}),
    )

    return ExperiencePool(memory_backend, llm, embedding_model, problem_cfg=config.problem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfg/config.yaml", help="配置文件路径")
    parser.add_argument("--pool_config", type=str, default="cfg/pool.yaml", help="经验池配置文件路径")
    args = parser.parse_args()

    config = ReevoConfig(args.config)
    config.load()

    log_file = setup_logging(config)
    config.summary()
    config.snapshot_config()

    # 初始化LLM
    generator_llm = config.create_llm_client("generator_llm")
    reflector_llm = config.create_llm_client("reflector_llm")
    planner_llm = config.create_llm_client("planner_llm")

    # 初始化经验池
    pool = build_experience_pool(config, args.pool_config, reflector_llm)
    
    # ===== 初始化规划智能体 =====
    use_planner = True
    if use_planner:
        logging.info("启用规划智能体")
        planner = PlannerInterface(
            llm=planner_llm,
            problem_cfg=config.problem,
            prompt_dir=Path("prompts/common")
        )
        logging.info("规划智能体初始化完成")
    else:
        planner = None
        logging.info("未启用规划智能体，使用传统固定算子策略")
    
    # 创建进化智能体（传入规划器）
    agent = EvoAgent(
        cfg=config, 
        root_dir=str(config.project_root), 
        generator_llm=generator_llm, 
        exp_pool=pool,
        planner=planner
    )
    
    # 运行进化
    best_info = agent.evolve()
    logging.info("最佳目标值: %s", best_info.get("objective"))
    
    # 运行验证
    run_validation(config, best_info)
    
    # ===== 新增：输出规划器统计信息 =====
    if planner is not None:
        planner_summary = planner.get_summary()
        logging.info("\n" + "="*60)
        logging.info("规划器工作摘要")
        logging.info("="*60)
        logging.info(f"总决策次数: {planner_summary['total_decisions']}")
        logging.info(f"策略分布: {planner_summary['strategy_distribution']}")
        logging.info(f"算子使用统计: {planner_summary['operator_usage']}")
        if planner_summary.get('population_trend'):
            trend = planner_summary['population_trend']
            if trend.get('improvement') is not None:
                logging.info(f"种群改进: {trend['improvement']:.6f}")
        logging.info("="*60 + "\n")


if __name__ == "__main__":
    main()
