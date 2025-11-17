"""
融合 经验池 + 进化算法 + 规划器 的智能体
"""

from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
import numpy as np

from utils.utils import save_json
from evo_interface import EvolutionInterface


logger = logging.getLogger(__name__)


class EvoAgent:

    def __init__(self, cfg, root_dir,
                 generator_llm=None, exp_pool=None, planner=None):
        """
        初始化进化智能体

        Args:
            cfg: 配置对象
            root_dir: 根目录
            generator_llm: 生成器LLM
            exp_pool: 经验池
            planner: 规划智能体(新增)
        """
        self.cfg = cfg
        self.root_dir = Path(root_dir).resolve()

        self.evo = EvolutionInterface(
            pool=exp_pool,
            init_pop_size = cfg.init_pop_size,
            pop_size = cfg.pop_size,
            m = 2,
            llm_interface = generator_llm,
            problem_cfg = cfg.problem,
            selection_method = "prob_rank",
            management_method = "pop_greedy",
            n_proc = 4,
            timeout = 60,
            project_root=Path(root_dir).resolve()
        )
        self.pop_size = cfg.pop_size
        self.n_pop = cfg.n_pop

        self.runtime_paths = cfg.paths
        self.iter_root = self.runtime_paths.iter_dir
        self.best_dir = self.runtime_paths.best_dir

        # 将算子描述传递给planner
        if planner is not None:
            planner.operator_descriptions = self.evo.operators.op_desc_dict.copy()
            logging.info("已将算子描述传递给规划器")
        self.planner = planner  # 新增：规划智能体

        # 新增：策略决策历史
        self.strategy_history = []
        # 新增：算子改进历史
        self.improvement_history = []

        self.generation = 0
        self.init_problem()

    def init_problem(self):
        """加载问题提示词"""
        problem_cfg = self.cfg.problem
        self.problem = problem_cfg.problem_name
        self.problem_desc = problem_cfg.description
        self.problem_size = problem_cfg.problem_size
        self.func_name = problem_cfg.func_name
        self.obj_type = problem_cfg.obj_type
        self.problem_type = problem_cfg.problem_type

        logging.info("任务名称: %s", self.problem)
        logging.info("问题描述: %s", self.problem_desc)
        logging.info("目标函数: %s", self.func_name)

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def evolve(self):
        time_start = time.time()
        # 创建新种群
        logging.info("创建初始种群...")
        population = self.evo.population_init()
        # 种群管理
        population = self.evo.manage(population, self.pop_size)
        logging.info("初始种群:")
        for ind in population:
            logging.info(f"  个体{ind['id']['uid']}: Obj={ind['objective']}")
        self.evo.save_population(population, self.generation, self.iter_root)

        # 迭代进化
        for pop_id in range(self.n_pop):

            # =====使用规划器决定策略和选择算子=====
            if self.planner is not None:
                # 调用规划器决定探索/利用策略
                decision = self.planner.decide_strategy(
                    population=population,
                    generation=self.generation
                )
                
                # 获取选中的算子
                selected_operators = decision.get("selected_operators")
                strategy = decision.get("strategy")
                reason = decision.get("reasoning")
                
                # 记录决策
                self.strategy_history.append({
                    "generation": self.generation,
                    "strategy": strategy,
                    "operators": selected_operators,
                    "reasoning": reason
                })
                
                logging.info(f"=== 第 {self.generation} 代策略决策 ===")
                logging.info(f"策略: {strategy}")
                logging.info(f"选择的算子: {selected_operators}")
                logging.info(f"决策理由: {reason}")

                # 动态改进算子（基于性能停滞检测）
                for op in selected_operators:
                    needs_improvement, stagnation_reason = self.planner.check_operator_needs_improvement(op)
                    if needs_improvement:
                        logging.info(f"--- 第 {self.generation} 代：检测到算子 {op} 需要改进 ---")
                        logging.info(f"停滞原因: {stagnation_reason}")

                        improvement = self.planner.improve_operator(op, population)

                        if improvement:
                            improvement_type = improvement.get("improvement_type", "unknown")
                            new_description = improvement.get("new_description")
                            reasoning = improvement.get("reasoning", "")
                            confidence = improvement.get("confidence", 0.0)

                            if new_description:
                                # 同步更新算子描述到 evo.operators
                                self.evo.operators.op_desc_dict[op] = new_description

                                # 记录改进历史
                                self.improvement_history.append({
                                    "generation": self.generation,
                                    "operator_id": op,
                                    "improvement_type": improvement_type,
                                    "reasoning": reasoning,
                                    "confidence": confidence,
                                    "new_description": new_description
                                })

                                logging.info(f"算子 {op} 已改进:")
                                logging.info(f"  改进类型: {improvement_type}")
                                logging.info(f"  置信度: {confidence:.2f}")
                                logging.info(f"  改进原因: {reasoning}")
                                logging.info(f"  新描述: {new_description}")
                            else:
                                logging.warning(f"算子 {op} 改进失败：未获取到新描述")
            else:
                # 如果没有规划器，使用所有算子（原始行为）
                selected_operators = ["op1","op2","op3","op4","op5"]
                logging.info("未使用规划器，使用全部算子")

            # 每代统计
            offsprings_gen = []
            dup_obj_skips_gen = 0
            gen_all_objs = []
            self.generation += 1
            
            # ===== 算子依次作用（使用选中的算子）=====
            for op in selected_operators:
                
                parents, offsprings = self.evo.get_algorithm(population, op)
                
                # 更新规划器的算子性能统计
                if self.planner is not None:
                    self.planner.update_operator_performance(op, offsprings, generation=self.generation)
                
                # 算子级统计（r1、l、均值、方差)
                objs_op_valid, stas_dict = self.evo.compute_off_stas(op, offsprings, self.generation, self.iter_root)
                
                # 累加到本代汇总
                offsprings_gen.extend(offsprings)
                gen_all_objs.extend(objs_op_valid)

                # 每算子小结
                logging.info(
                    f"  算子 {op} 产生 {len(offsprings)} 个体"
                    f"r1={stas_dict['r1']}，l={stas_dict['l']}"
                    f"均值={stas_dict['mean']}，方差={stas_dict['var']}"
                )

            # 本代所有新个体适应度统计与保存
            mean_gen = float(np.mean(gen_all_objs)) if len(gen_all_objs) > 0 else None
            var_gen = float(np.var(gen_all_objs)) if len(gen_all_objs) > 0 else None
            save_json(
                {
                    "generation": self.generation,
                    "objectives": gen_all_objs,
                    "mean": mean_gen,
                    "var": var_gen,
                },
                os.path.join(self.iter_root, "stats", f"stats_gen_{self.generation}.json")
            )

            # 种群管理
            population, dup_obj_skips_gen = self.evo.add2pop(population, offsprings_gen)
            size_act = min(len(population)+len(offsprings_gen), self.pop_size)
            population = self.evo.manage(population, size_act)
            # 保存当前代的种群
            self.evo.save_population(population, self.generation, self.iter_root)
            
            # 每代汇总输出
            logging.info(
                f"第 {self.generation} 代汇总：新增 {len(offsprings_gen)} 个体"
                f"因目标值重复跳过 {dup_obj_skips_gen} 个"
                f"本代新个体适应度统计：均值={mean_gen:.6f}，方差={var_gen:.6f}"
            )

            # 保存最优个体
            best_individual = self.evo.get_best_individual(population)
            self.evo.save_best_individual(best_individual, self.generation, self.best_dir)
            
            # 统计信息
            stats = self.evo.get_statistics(population)
            elapsed_time = (time.time() - time_start) / 60
            
            # 保存种群信息
            save_json(stas_dict, os.path.join(self.iter_root, "stats", f"stats_pop_{self.generation}.json"))

            logging.info(f"--- 第 {self.generation}/{self.n_pop} 代完成")
            logging.info(f"--- 耗时: {elapsed_time:.1f} 分钟")
            logging.info(f"--- 种群统计:")
            logging.info(f"    最优: {stats['best']:.6f}")
            logging.info(f"    平均: {stats['mean']:.6f}")
            logging.info(f"    标准差: {stats['std']:.6f}")

        # ===== 保存规划器摘要 =====
        if self.planner is not None:
            planner_summary = self.planner.get_summary()
            save_json(
                planner_summary,
                os.path.join(self.runtime_paths.run_dir, "planner_summary.json")
            )
            logging.info("规划器摘要已保存")
            
            # 保存策略决策历史
            save_json(
                self.strategy_history,
                os.path.join(self.runtime_paths.run_dir, "strategy_history.json")
            )
            logging.info("策略决策历史已保存")

            # 保存算子改进历史
            save_json(
                self.improvement_history,
                os.path.join(self.runtime_paths.run_dir, "improvement_history.json")
            )
            logging.info("算子改进历史已保存")

        return {
            "code": best_individual["code"],
            "objective": best_individual["objective"]
        }
