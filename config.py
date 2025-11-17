"""
统一的配置管理模块
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.utils import AttrDict, load_config, instantiate_from_target


@dataclass
class RuntimePaths:
    """运行时产物路径集合，便于在其他模块中复用。"""
    run_dir: Path
    iter_dir: Path
    logs_dir: Path
    best_dir: Path
    summary_path: Path
    config_snapshot: Path


class ReevoConfig:
    """
    同时负责创建结果目录、打印摘要、持久化配置等。
    """

    def __init__(self, config_path="cfg/config.yaml"):
        self.config_path = config_path

        # 基础字段
        self.raw = None
        self.problem = None
        self.llm_client = None

        # 实验参数
        self.algorithm = "reevo"
        self.n_pop = 10
        self.pop_size = 10
        self.init_pop_size = 30
        self.timeout = 60
        self.diversify_init_pop = False

        # 日志/输出控制
        self.logging_level = "INFO"
        self.output_root = Path.cwd() / "Results"
        self.run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_filename = "summary.json"

        # 路径信息
        self.paths = None
        self.project_root = Path(__file__).resolve().parent

    # ------------------------------------------------------------------ #
    # 加载与初始化
    # ------------------------------------------------------------------ #
    def load(self):
        """加载配置文件并准备运行时所需的字段。"""
        self.raw = load_config(self.config_path)

        # 实验基础参数
        self.algorithm = self.raw.get("algorithm", self.algorithm)
        self.n_pop = int(self.raw.get("n_pop", self.n_pop))
        self.pop_size = int(self.raw.get("pop_size", self.pop_size))
        self.init_pop_size = int(self.raw.get("init_pop_size", self.init_pop_size))
        self.timeout = int(self.raw.get("timeout", self.timeout))
        self.diversify_init_pop = bool(self.raw.get("diversify_init_pop", self.diversify_init_pop))

        # 问题与模型配置
        self.problem = self.raw.problem
        self.generator_llm = self.raw.generator_llm
        self.reflector_llm = self.raw.reflector_llm
        self.planner_llm = self.raw.planner_llm

        # 日志与输出配置
        logging_cfg = AttrDict(self.raw.get("logging", {}))
        self.logging_level = logging_cfg.get("level", self.logging_level)

        exp_cfg = AttrDict(self.raw.get("exp", {}))
        self.output_root = Path(exp_cfg.get("output_root", "Results"))
        run_name =  exp_cfg.get("run_name")
        if run_name in ("auto", None, ""):
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = run_name
        self.summary_filename = exp_cfg.get("summary_name", self.summary_filename)

        # 构建运行目录
        run_dir = (self.output_root / self.problem.problem_name / self.run_name).resolve()
        iter_dir = run_dir / "generations"
        logs_dir = run_dir / "logs"
        best_dir = run_dir / "best"
        summary_path = run_dir / self.summary_filename
        config_snapshot = run_dir / "config_snapshot.json"
        for path in (run_dir, iter_dir, logs_dir, best_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.paths = RuntimePaths(
            run_dir=run_dir,
            iter_dir=iter_dir,
            logs_dir=logs_dir,
            best_dir=best_dir,
            summary_path=summary_path,
            config_snapshot=config_snapshot,
        )

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #
    def summary(self):
        """打印当前配置的关键信息，便于快速核对。"""
        print("\n" + "=" * 60)
        print("Reevo 配置摘要".center(60))
        print("=" * 60)

        print("\n【基础设置】")
        print(f"  算法: {self.algorithm}")
        print(f"  问题: {self.problem.problem_name} ({self.problem.problem_type})")
        print(f"  目标类型: {self.problem.obj_type}")
        print(f"  迭代轮数: {self.n_pop}")
        print(f"  种群规模: {self.pop_size}, 初始规模: {self.init_pop_size}")

        print("\n【LLM设置】")
        print(f"  主模型: {self.generator_llm.get('model')}")
        print(f"  反思模型: {self.reflector_llm.get('model')}")

        print("\n【输出设置】")
        print(f"  输出根目录: {str(self.output_root)}")
        print(f"  本次运行目录: {str(self.paths.run_dir)}")
        print(f"  日志目录: {str(self.paths.logs_dir)}")

        print("=" * 60 + "\n")

    def save_summary(self, summary):
        """保存运行摘要到JSON文件。"""
        with open(self.paths.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def snapshot_config(self):
        """保存当前实际生效的配置，方便复现实验。"""
        payload = self.to_dict()
        with open(self.paths.config_snapshot, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def create_llm_client(self, section = "generator_llm"):
        """基于配置实例化指定的LLM客户端。"""
        cfg_map = {
            "generator_llm": self.generator_llm,
            "reflector_llm": self.reflector_llm,
            "planner_llm": self.planner_llm
        }
        cfg = cfg_map.get(section)
        if not cfg:
            return None
        return instantiate_from_target(cfg)

    def to_dict(self) -> Dict[str, Any]:
        """导出当前配置字典，包含运行时信息。"""
        return {
            "config_path": self.config_path,
            "algorithm": self.algorithm,
            "n_pop": self.n_pop,
            "pop_size": self.pop_size,
            "init_pop_size": self.init_pop_size,
            "timeout": self.timeout,
            "diversify_init_pop": self.diversify_init_pop,
            "problem": dict(self.problem) if self.problem else None,
            "generator_llm": self.generator_llm,
            "reflector_llm": self.reflector_llm,
            "logging": {"level": self.logging_level},
            "exp": {
                "output_root": str(self.output_root),
                "run_name": self.run_name,
                "summary_name": self.summary_filename
            },
            "paths": {
                "run_dir": str(self.paths.run_dir),
                "iter_dir": str(self.paths.iter_dir),
                "logs_dir": str(self.paths.logs_dir),
                "best_dir": str(self.paths.best_dir),
                "summary_path": str(self.paths.summary_path),
            },
        }
