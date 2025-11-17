"""
规划代理接口
负责动态决策探索/利用策略，并根据种群状态改进算子
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from utils.utils import file_to_string, _extract_json_block


logger = logging.getLogger(__name__)


class PlannerInterface:
    """
    规划代理，负责：
    1. 分析种群状态（多样性、收敛度等）
    2. 决策当前轮次的探索/利用策略
    3. 动态改进算子以适应种群变化
    """

    # 算子分类
    EXPLORATION_OPERATORS = ["op1", "op2"]  # 探索型：创建全新算法
    EXPLOITATION_OPERATORS = ["op3", "op4", "op5"]  # 利用型：改进现有算法

    def __init__(self, llm, problem_cfg, prompt_dir=None, operator_descriptions={}):
        """
        初始化规划代理

        Args:
            operator_descriptions: 算子描述字典，从EvolutionOperators传入
        """
        self.llm = llm
        self.problem_cfg = problem_cfg
        self.prompt_dir = prompt_dir or Path("prompts/common")

        # 历史记录
        self.history = {
            "decisions": [],  # 决策历史
            "population_stats": [],  # 种群统计历史
            "operator_performance": {op: [] for op in self.EXPLORATION_OPERATORS + self.EXPLOITATION_OPERATORS}
        }

        # 算子描述（从EvolutionOperators传入）
        self.operator_descriptions = operator_descriptions

        # 加载系统提示词
        self.system_planner = file_to_string(str(self.prompt_dir / "system_planner.txt"))
        self.system_improver = file_to_string(str(self.prompt_dir / "system_operator_improver.txt"))

        # 连续利用阈值：若最近连续 k 轮策略为"利用"，则本轮强制改为"探索"
        self.explore_after_k_exploit = 5

        # 算子改进配置
        self.no_improvement_threshold = 5  # 连续5次无提升则触发改进
        self.improvement_tolerance = 1e-6  # 改进阈值（相对变化）

    def analyze_population(self, population):
        """
        分析种群状态，计算关键指标
        """

        # 提取有效目标值
        objectives = [ind["objective"] for ind in population if ind["objective"] is not None]        
        objectives = np.array(objectives)
        
        # 基础统计
        stats = {
            "size": len(population),
            "mean_objective": float(np.mean(objectives)),
            "std_objective": float(np.std(objectives)),
            "best_objective": float(np.min(objectives)),
            "worst_objective": float(np.max(objectives)),
        }
        
        # 多样性：归一化标准差
        obj_range = stats["worst_objective"] - stats["best_objective"]
        if obj_range > 1e-9:
            stats["diversity"] = float(stats["std_objective"] / obj_range)
        else:
            stats["diversity"] = 0.0
        
        # 收敛率：基于历史数据计算
        if len(self.history["population_stats"]) > 0:
            prev_best = self.history["population_stats"][-1]["best_objective"]
            if prev_best is not None and prev_best != 0:
                improvement = (prev_best - stats["best_objective"]) / abs(prev_best)
                stats["convergence_rate"] = float(improvement)
            else:
                stats["convergence_rate"] = 0.0
        else:
            stats["convergence_rate"] = 0.0
        
        # 改进潜力：基于多样性和收敛率
        stats["improvement_potential"] = float(
            0.6 * stats["diversity"] + 0.4 * (1.0 if stats["convergence_rate"] > 0.01 else 0.5)
        )
        
        return stats

    def decide_strategy(self, population, generation):
        """
        决策当前轮次应该探索还是利用
        """
        # 分析种群状态
        stats = self.analyze_population(population)
        self.history["population_stats"].append(stats)
        
        # 调用LLM做决策
        decision = self._query_llm_for_strategy(stats, generation)

        # 若最近连续 k 轮都是“利用”，则本轮强制改为“探索”
        k = self.explore_after_k_exploit
        prev = self.history.get("decisions", [])
        if k > 0 and len(prev) >= k:
            last_k = prev[-k:]
            if all(d.get("decision", {}).get("strategy") == "exploitation" for d in last_k):
                logger.info(f"连续 {k} 轮选择'利用'，本轮自动切换为'探索'")
                decision = {
                    "strategy": "exploration",
                    "reasoning": f"Auto-switch due to {k} consecutive exploitation",
                    "selected_operators": self.EXPLORATION_OPERATORS,
                    "confidence": 0.7,
                }

        # 记录决策
        self.history["decisions"].append({
            "generation": generation,
            "stats": stats,
            "decision": decision
        })
        
        return decision

    def _build_pop_stats(self, stats, generation):
        """构建用于LLM决策的上下文信息"""
        context_parts = [
            f"## Generation {generation} Population Statistics:",
            f"- Population Size: {stats['size']}",
            f"- Best Objective: {stats['best_objective']:.6f}" if stats['best_objective'] is not None else "- Best Objective: N/A",
            f"- Mean Objective: {stats['mean_objective']:.6f}" if stats['mean_objective'] is not None else "- Mean Objective: N/A",
            f"- Std Deviation: {stats['std_objective']:.6f}" if stats['std_objective'] is not None else "- Std Deviation: N/A",
            f"- Diversity Score: {stats['diversity']:.4f}",
            f"- Convergence Rate: {stats['convergence_rate']:.4f}",
            f"- Improvement Potential: {stats['improvement_potential']:.4f}",
            f"",
        ]
        
        # 添加历史趋势
        if len(self.history["population_stats"]) > 1:
            context_parts.append("## Historical Trend (last 3 generations):")
            for hist_stat in self.history["population_stats"][-3:]:
                context_parts.append(
                    f"- Best: {hist_stat['best_objective']:.6f}, "
                    f"Diversity: {hist_stat['diversity']:.4f}"
                )
            context_parts.append("")
        
        return "\n".join(context_parts)
        
    def _build_operator_context(self):
        """构建算子相关的上下文信息"""
        context_parts = []
        context_parts.append("- Exploration Operators (Create new search directions):")
        for op in self.EXPLORATION_OPERATORS:
            desc = self.operator_descriptions.get(op, "No description available")
            context_parts.append(f"- **{op}**: {desc}")
        context_parts.append("")

        context_parts.append("- Exploitation Operators (Refine existing solutions):")
        for op in self.EXPLOITATION_OPERATORS:
            desc = self.operator_descriptions.get(op, "No description available")
            context_parts.append(f"- **{op}**: {desc}")
        context_parts.append("")

        return "\n".join(context_parts)

    def _query_llm_for_strategy(self, stats, generation):
        """
        调用LLM做出探索/利用决策
        """
        pop_stats = self._build_pop_stats(stats, generation)
        operator_context = self._build_operator_context()

        user_prompt = file_to_string(str(self.prompt_dir/"user_sel_ops.txt")).format(
            pop_stats=pop_stats,
            ops_desc=operator_context,
            exploration_ops=self.EXPLORATION_OPERATORS,
            exploitation_ops=self.EXPLOITATION_OPERATORS
        )

        messages = [
            {"role": "system", "content": self.system_planner},
            {"role": "user", "content": user_prompt}
        ]
            
        response = self.llm._chat_completion_api(messages, temperature=0.7)
            
        # 解析JSON响应
        json_text = _extract_json_block(response)
        decision_list = json.loads(json_text)
        decision = decision_list[0]
            
        # 验证决策格式
        if "strategy" not in decision or "selected_operators" not in decision:
            return self._heuristic_decision()
            
        # 确保选择的算子有效
        valid_ops = self.EXPLORATION_OPERATORS + self.EXPLOITATION_OPERATORS
        decision["selected_operators"] = [
            op for op in decision["selected_operators"] if op in valid_ops
        ]
            
        if not decision["selected_operators"]:
            return self._heuristic_decision()
            
        return decision

    def _heuristic_decision(self):
        """基于启发式规则的决策（备用方案）"""
        logging.info(f"LLM决策失败，使用启发式规则")
        if not self.history["population_stats"]:
            return {
                "strategy": "exploration",
                "reasoning": "Initial generation, need exploration",
                "selected_operators": ["op1", "op2"],
                "confidence": 0.8
            }
        
        stats = self.history["population_stats"][-1]
        diversity = stats["diversity"]
        convergence = stats["convergence_rate"]
        
        # 简单规则：
        # - 多样性低 或 收敛慢 -> 探索
        # - 多样性高 且 收敛快 -> 利用
        if diversity < 0.4 or convergence < 0.03:
            return {
                "strategy": "exploration",
                "reasoning": f"Low diversity ({diversity:.3f}) or slow convergence ({convergence:.3f})",
                "selected_operators": ["op1", "op2"],
                "confidence": 0.7
            }
        else:
            return {
                "strategy": "exploitation",
                "reasoning": f"High diversity ({diversity:.3f}) and good convergence ({convergence:.3f})",
                "selected_operators": ["op3", "op4"],
                "confidence": 0.7
            }

    def improve_operator(self, operator_id: str, population: List[Dict]) -> Dict:
        """
        根据种群特征和性能历史动态改进算子的描述

        Args:
            operator_id: 算子ID (op1-op5)
            population: 当前种群

        Returns:
            改进结果字典，包含：
            - operator: 算子ID
            - improvement_type: "incremental" 或 "radical"
            - new_description: 新的算子描述
            - reasoning: 改进理由
            - confidence: 置信度
        """
        if operator_id not in (self.EXPLORATION_OPERATORS + self.EXPLOITATION_OPERATORS):
            logger.warning(f"未知算子 {operator_id}，返回None")
            return None

        # 分析种群特征
        stats = self.analyze_population(population)

        # 构建改进上下文
        context_dict = self._build_improvement_context(operator_id, stats, population)

        # 调用LLM改进算子
        improvement_result = self._query_llm_for_improvement(operator_id, context_dict)

        if improvement_result is None:
            logger.warning(f"算子 {operator_id} 改进失败")
            return None

        # 更新算子描述
        self.operator_descriptions[operator_id] = improvement_result["new_description"]

        logger.info(
            f"算子 {operator_id} 改进完成 ({improvement_result['improvement_type']}): "
            f"{improvement_result['reasoning']}"
        )

        return {
            "operator": operator_id,
            "improvement_type": improvement_result["improvement_type"],
            "new_description": improvement_result["new_description"],
            "reasoning": improvement_result["reasoning"],
            "confidence": improvement_result["confidence"],
            "population_context": stats
        }

    def _build_improvement_context(self, operator_id: str, stats: Dict, population: List[Dict]) -> Dict[str, str]:
        """构建用于改进算子的上下文"""
        # 收集种群中表现最好的几个个体的特征
        valid_pop = [ind for ind in population if ind["objective"] is not None]
        valid_pop.sort(key=lambda x: x["objective"])
        top_individuals = valid_pop[:5] if len(valid_pop) >= 5 else valid_pop

        # 构建性能历史字符串
        perf_history = self.history["operator_performance"].get(operator_id, [])
        recent_history = perf_history[-10:] if len(perf_history) > 10 else perf_history

        history_lines = []
        for i, perf in enumerate(recent_history):
            gen = perf.get("generation", i)
            mean = perf.get("mean")
            best = perf.get("best")
            success_rate = perf.get("success_rate", 0.0)

            if mean is not None:
                history_lines.append(
                    f"Gen {gen}: Mean={mean:.6f}, Best={best:.6f}, Success Rate={success_rate:.2%}"
                )
            else:
                history_lines.append(
                    f"Gen {gen}: No valid offspring (Success Rate={success_rate:.2%})"
                )

        performance_history = "\n".join(history_lines) if history_lines else "No performance history available"

        # 判断性能状态
        if len(recent_history) >= 3:
            recent_means = [p["mean"] for p in recent_history[-3:] if p["mean"] is not None]
            if len(recent_means) >= 2:
                if recent_means[-1] < recent_means[0]:
                    performance_status = "improving performance"
                elif recent_means[-1] > recent_means[0] * 1.05:
                    performance_status = "degrading performance"
                else:
                    performance_status = "stagnating performance"
            else:
                performance_status = "insufficient valid results"
        else:
            performance_status = "insufficient history"

        # 构建种群上下文
        pop_context_parts = [
            f"Current Population Size: {stats['size']}",
            f"Best Objective: {stats['best_objective']:.6f}" if stats['best_objective'] is not None else "Best Objective: N/A",
            f"Mean Objective: {stats['mean_objective']:.6f}" if stats['mean_objective'] is not None else "Mean Objective: N/A",
            f"Diversity Score: {stats['diversity']:.4f}",
            f"Convergence Rate: {stats['convergence_rate']:.4f}",
        ]
        population_context = "\n".join(pop_context_parts)

        # 构建顶级算法字符串
        top_algo_parts = []
        for i, ind in enumerate(top_individuals, 1):
            algo_desc = ind.get("algorithm", "No description")
            # 截取算法描述的前150个字符
            algo_summary = algo_desc[:150] + "..." if len(algo_desc) > 150 else algo_desc
            top_algo_parts.append(f"**Algorithm {i}** (Objective: {ind['objective']:.6f})")
            top_algo_parts.append(f"{algo_summary}")
            top_algo_parts.append("")

        top_algorithms = "\n".join(top_algo_parts) if top_algo_parts else "No successful algorithms yet"

        return {
            "performance_history": performance_history,
            "performance_status": performance_status,
            "history_length": len(recent_history),
            "population_context": population_context,
            "top_algorithms": top_algorithms
        }

    def _query_llm_for_improvement(self, operator_id: str, context_dict: Dict[str, str]) -> Dict:
        """调用LLM改进算子描述"""

        current_desc = self.operator_descriptions.get(operator_id, "")
        if not current_desc:
            logger.warning(f"No current description for {operator_id}")
            return None

        user_prompt = file_to_string(str(self.prompt_dir / "user_improve_operator.txt")).format(
            operator_id=operator_id,
            current_description=current_desc,
            performance_status=context_dict["performance_status"],
            history_length=context_dict["history_length"],
            performance_history=context_dict["performance_history"],
            population_context=context_dict["population_context"],
            top_algorithms=context_dict["top_algorithms"]
        )

        if self.llm is None:
            logger.warning("No LLM available for operator improvement")
            return None

        try:
            messages = [
                {"role": "system", "content": self.system_improver},
                {"role": "user", "content": user_prompt}
            ]

            response = self.llm._chat_completion_api(messages, temperature=0.7)

            # 提取JSON响应
            json_text = _extract_json_block(response)
            improvement_result = json.loads(json_text)

            # 验证响应格式
            required_fields = ["improvement_type", "reasoning", "new_description", "confidence"]
            if not all(field in improvement_result for field in required_fields):
                logger.warning(f"Invalid improvement response format: {improvement_result}")
                return None

            logger.info(f"Operator {operator_id} improvement: {improvement_result['improvement_type']}")
            logger.info(f"Reasoning: {improvement_result['reasoning']}")
            logger.info(f"New description: {improvement_result['new_description'][:100]}...")

            return improvement_result

        except Exception as e:
            logger.warning(f"算子改进失败: {e}")
            return None

    def update_operator_performance(self, operator_id: str, offsprings: List[Dict], generation: int = None):
        """
        更新算子性能统计

        Args:
            operator_id: 算子ID
            offsprings: 该算子生成的后代列表
            generation: 当前代数（可选）
        """
        if operator_id not in self.history["operator_performance"]:
            return

        # 计算该算子的性能指标
        valid_objs = [off["objective"] for off in offsprings if off["objective"] is not None]

        if valid_objs:
            performance = {
                "generation": generation,
                "mean": float(np.mean(valid_objs)),
                "best": float(np.min(valid_objs)),
                "worst": float(np.max(valid_objs)),
                "std": float(np.std(valid_objs)),
                "success_rate": len(valid_objs) / len(offsprings) if offsprings else 0.0,
                "num_valid": len(valid_objs),
                "num_total": len(offsprings)
            }
        else:
            performance = {
                "generation": generation,
                "mean": None,
                "best": None,
                "worst": None,
                "std": None,
                "success_rate": 0.0,
                "num_valid": 0,
                "num_total": len(offsprings)
            }

        self.history["operator_performance"][operator_id].append(performance)

        logger.info(f"Updated {operator_id} performance: mean={performance['mean']}, "
                    f"best={performance['best']}, success_rate={performance['success_rate']:.2f}")

    def check_operator_needs_improvement(self, operator_id: str) -> Tuple[bool, str]:
        """
        检查算子是否需要改进（连续N次无提升）

        Args:
            operator_id: 算子ID

        Returns:
            (needs_improvement, reason): 是否需要改进及原因
        """
        if operator_id not in self.history["operator_performance"]:
            return False, ""

        perf_history = self.history["operator_performance"][operator_id]

        # 至少需要有 threshold+1 次记录才能判断
        if len(perf_history) < self.no_improvement_threshold + 1:
            return False, "Insufficient performance history"

        # 获取最近的记录
        recent_perfs = perf_history[-self.no_improvement_threshold-1:]

        # 过滤出有效的mean值
        valid_means = [(i, p["mean"]) for i, p in enumerate(recent_perfs) if p["mean"] is not None]

        if len(valid_means) < self.no_improvement_threshold:
            return False, "Too many failed generations"

        # 检查是否连续无提升
        no_improvement_count = 0
        baseline_mean = valid_means[0][1]

        for i, (idx, mean_val) in enumerate(valid_means[1:], 1):
            # 计算相对改进
            if baseline_mean != 0:
                relative_improvement = (baseline_mean - mean_val) / abs(baseline_mean)
            else:
                relative_improvement = 0.0 if mean_val == 0 else -1.0

            if relative_improvement >= self.improvement_tolerance:
                # 有显著改进，重置计数
                no_improvement_count = 0
                baseline_mean = mean_val  # 更新基准
            else:
                # 无改进
                no_improvement_count += 1

        if no_improvement_count >= self.no_improvement_threshold:
            reason = (f"No improvement in last {no_improvement_count} generations. "
                      f"Baseline mean: {baseline_mean:.6f}, Latest mean: {valid_means[-1][1]:.6f}")
            logger.info(f"Operator {operator_id} needs improvement: {reason}")
            return True, reason

        return False, ""

    def get_summary(self) -> Dict:
        """获取规划代理的运行摘要"""
        return {
            "total_decisions": len(self.history["decisions"]),
            "strategy_distribution": self._count_strategies(),
            "operator_usage": self._count_operator_usage(),
            "population_trend": self._get_population_trend()
        }

    def _count_strategies(self) -> Dict[str, int]:
        """统计各策略的使用次数"""
        counts = {"exploration": 0, "exploitation": 0}
        for decision in self.history["decisions"]:
            strategy = decision["decision"].get("strategy")
            counts[strategy] = counts.get(strategy, 0) + 1
        return counts

    def _count_operator_usage(self) -> Dict[str, int]:
        """统计各算子的使用次数"""
        counts = {}
        for decision in self.history["decisions"]:
            for op in decision["decision"].get("selected_operators", []):
                counts[op] = counts.get(op, 0) + 1
        return counts

    def _get_population_trend(self) -> Dict:
        """获取种群演化趋势"""
        if not self.history["population_stats"]:
            return {}
        
        stats = self.history["population_stats"]
        return {
            "initial_best": stats[0]["best_objective"],
            "final_best": stats[-1]["best_objective"],
            "improvement": (
                stats[0]["best_objective"] - stats[-1]["best_objective"]
                if stats[0]["best_objective"] is not None and stats[-1]["best_objective"] is not None
                else None
            ),
            "diversity_trend": [s["diversity"] for s in stats[-5:]]  # 最近5代
        }
