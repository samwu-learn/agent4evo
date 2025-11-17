"""
进化算法
1. 选择策略 Selection Strategies
2. 种群管理策略 Population Management
3. 进化算子 Evolution Operators
"""

import json
import logging
import numpy as np
import random
import re
import time
import concurrent.futures
from pathlib import Path
import os
from joblib import Parallel, delayed
import subprocess
import uuid
import shutil
from utils.utils import file_to_string, save_json, compute_r1, compute_correlation_length, _ensure_required_imports, _parse_objective

# ============= 第一部分：选择策略 =============

class SelectionStrategies:
    """所有选择策略的集合"""
    
    @staticmethod
    def prob_rank(population, m):
        """
        基于排名的概率选择
        排名越高，被选中的概率越大
        """
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        parents = random.choices(population, weights=probs, k=m)
        return parents
    
    @staticmethod
    def equal(population, m):
        """等概率随机选择"""
        parents = random.choices(population, k=m)
        return parents
    
    @staticmethod
    def roulette_wheel(population, m):
        """
        轮盘赌选择
        fitness越好，被选中的概率越大
        """
        fitness_values = [1 / (fit['objective'] + 1e-6) for fit in population]
        fitness_sum = sum(fitness_values)
        probs = [fit / fitness_sum for fit in fitness_values]
        parents = random.choices(population, weights=probs, k=m)
        return parents
    
    @staticmethod
    def tournament(population, m, tournament_size=2):
        """
        锦标赛选择
        每次从种群中随机选择tournament_size个个体，选择其中最优的
        """
        parents = []
        while len(parents) < m:
            tournament = random.sample(population, tournament_size)
            tournament_fitness = [fit['objective'] for fit in tournament]
            winner = tournament[tournament_fitness.index(min(tournament_fitness))]
            parents.append(winner)
        return parents
    
    @classmethod
    def get_selection_method(cls, method_name):
        """根据方法名获取选择策略"""
        methods = {
            'prob_rank': cls.prob_rank,
            'equal': cls.equal,
            'roulette_wheel': cls.roulette_wheel,
            'tournament': cls.tournament
        }
        
        if method_name not in methods:
            raise ValueError(f"Unknown selection method: {method_name}")
        
        return methods[method_name]


# ============= 第二部分：种群管理策略 =============

class PopulationManagement:
    """所有种群管理策略的集合"""
    
    @staticmethod
    def pop_greedy(population, size):
        """
        贪心种群管理
        保留fitness最好的size个个体
        """
        import heapq
        
        # 过滤掉objective为None的个体
        pop = [individual for individual in population if individual['objective'] is not None]
        if size > len(pop):
            size = len(pop)
        
        # 去重：相同objective的个体只保留一个
        unique_pop = [] 
        unique_objectives = []
        for individual in pop:
            if individual['objective'] not in unique_objectives:
                unique_pop.append(individual)
                unique_objectives.append(individual['objective'])
        
        # 选择最优的size个
        pop_new = heapq.nsmallest(size, unique_pop, key=lambda x: x['objective'])
        return pop_new
    
    @staticmethod
    def ls_greedy(population, new, temperature=None):
        """
        局部搜索贪心管理
        如果新个体更好，则替换
        """
        if (new['objective'] is not None) and (len(population) == 0 or new['objective'] < population[0]['objective']):
            population[0] = new
        return
    
    @staticmethod
    def ls_sa(population, new, temperature):
        """
        局部搜索模拟退火管理
        使用Metropolis准则接受新解
        """
        import math
        
        def acceptance_probability(old_cost, new_cost, temp):
            if new_cost < old_cost:
                return 1.0
            return math.exp(((old_cost - new_cost) / old_cost) / temp)
        
        current_best = population[0]
        
        if (new['objective'] is not None) and (
            len(population) == 0 or 
            acceptance_probability(current_best['objective'], new['objective'], temperature) > random.random()
        ):
            population[0] = new
        
        return
    
    @classmethod
    def get_management_method(cls, method_name):
        """根据方法名获取种群管理策略"""
        methods = {
            'pop_greedy': cls.pop_greedy,
            'ls_greedy': cls.ls_greedy,
            'ls_sa': cls.ls_sa
        }
        
        if method_name not in methods:
            raise ValueError(f"Unknown management method: {method_name}")
        
        return methods[method_name]


# ============= 第三部分：进化算子 =============

class EvolutionOperators:
    """
    统一的进化算子类
    """
    
    def __init__(self, llm, problem_cfg):
        """
        初始化进化算子
        """
        self.llm = llm
        self.problem = problem_cfg.problem_name
        self.problem_desc = problem_cfg.description
        self.problem_size = problem_cfg.problem_size
        self.func_name = problem_cfg.func_name
        self.obj_type = problem_cfg.obj_type
        self.problem_type = problem_cfg.problem_type
        self.init_prompt()

    def init_prompt(self):
        """从prompts文件初始化提示词"""

        prompts_path = Path("prompts")
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = prompts_path / f"{self.problem}{prompt_path_suffix}"

        self.func_signature = file_to_string(str(problem_prompt_path / "func_signature.txt"))
        self.func_desc = file_to_string(str(problem_prompt_path / "func_desc.txt"))
        self.seed_func = file_to_string(str(problem_prompt_path / "seed_func.txt"))

        external_path = problem_prompt_path / "external_knowledge.txt"
        if external_path.exists():
            self.external_knowledge = file_to_string(str(external_path))
        else:
            self.external_knowledge = ""

        self.common_dir = prompts_path / "common"
        self.system_generator = file_to_string(str(self.common_dir / "system_generator.txt")).format(
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc
        )
        self.op_desc_dict = {"op1": "create a new algorithm that has a totally different form from the given ones",
                             "op2": "create a new algorithm that has a totally different form from the given ones but can be motivated from them",
                             "op3": "create a new algorithm that has a different form but can be a modified version of the algorithm provided",
                             "op4": "identify the main algorithm parameters and create a new algorithm that has a different parameter settings of the score function provided",
                             "op5": "identify the main components in the function and simplify the components to enhance the generalization to potential out-of-distribution instances"}
        self.op_nparents_dict = {"op1": 2, "op2": 2, "op3": 1, "op4": 1, "op5": 1}

    def _extract_algorithm_and_code(self, response):
        """从LLM响应中提取算法描述和代码"""
        # 提取算法描述
        algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        if not algorithm_match:
            raise ValueError("LLM response missing algorithm description block.")
        algorithm = algorithm_match.group(1)
        
        # 提取代码
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            raise ValueError("LLM response missing python code block.")
        code = match.group(1)

        if len(algorithm) == 0 or len(code) == 0:
            raise ValueError("Failed to extract algorithm or code from LLM response.")
        
        return code, algorithm

    def init(self, seed=True):
        # Prepare external knowledge section
        ext_knowledge = self.external_knowledge if self.external_knowledge else "No additional domain knowledge provided."

        if seed:
            user = file_to_string(str(self.common_dir / "user_init.txt")).format(
                seed_func=self.seed_func,
                external_knowledge=ext_knowledge
            )
        else:
            user = file_to_string(str(self.common_dir / "user_init.txt")).format(
                seed_func=self.func_signature,
                external_knowledge=ext_knowledge
            )

        messages = [{"role": "system", "content": self.system_generator}, {"role": "user", "content": user}]
        response = self.llm._chat_completion_api(
            messages,
            temperature=1.0
        )
        code_all, algorithm = self._extract_algorithm_and_code(response)

        return code_all, algorithm
    

    def operate(self, id, parents, processed_memories):
        """
        算子1：创建完全不同形式的新算法
        算子2：基于共同骨干思想创建新算法
        算子3：创建修改版本
        算子4：参数调整
        算子5：简化以增强泛化能力

        参数:
            id: 算子ID
            parents: 父代个体列表
            processed_memories: 已处理的经验字典，包含 'success_exp' 和 'fail_exp' 字段
        """
        # Format parent algorithms with clear structure
        prompt_indiv = ""
        for i in range(len(parents)):
            prompt_indiv += f"\n**Parent Algorithm {i+1}**\n"
            prompt_indiv += f"Description: {{{parents[i]['algorithm']}}}\n"
            prompt_indiv += f"Code:\n```python\n{parents[i]['code']}\n```\n"

        user_op_prompt = file_to_string(str(self.common_dir / "user_operate.txt")).format(
            n_parents=len(parents),
            prompt_indiv=prompt_indiv,
            success_exp=processed_memories['success_exp'],
            fail_exp=processed_memories['fail_exp'],
            op_desc=self.op_desc_dict[id]
        )

        messages = [{"role": "system", "content": self.system_generator}, {"role": "user", "content": user_op_prompt}]
        response = self.llm._chat_completion_api(
            messages,
            temperature=1.0
        )
        code_all, algorithm = self._extract_algorithm_and_code(response)

        return code_all, algorithm
    
# ============= 第四部分：评估接口 =============

class EvolutionInterface:
    """
    进化算法的统一接口
    负责种群初始化、算子调用、并行评估等
    """
    
    def __init__(
        self,
        pool,
        init_pop_size,
        pop_size,
        m,
        llm_interface,
        problem_cfg,
        selection_method,
        management_method,
        n_proc=1,
        timeout=60,
        project_root=None
    ):
        """
        初始化进化接口
        Args:
            pop_size: 种群大小
            m: 交叉时的父代数量
            llm_interface: LLM接口
            problem_cfg: 问题配置
            selection_method: 选择策略
            management_method: 种群管理策略
            n_proc: 并行进程数
            timeout: 评估超时时间
            project_root: 项目根目录，用于定位问题脚本
        """
        self.init_pop_size = init_pop_size
        self.pop_size = pop_size
        self.m = m
        self.problem_cfg = problem_cfg
        self.n_proc = n_proc
        self.timeout = timeout
        self.project_root = project_root if project_root else Path(__file__).resolve().parent
        self.problem_name = problem_cfg.problem_name
        self.problem_type = problem_cfg.problem_type
        self.problem_size = problem_cfg.problem_size
        self.problem_dir = self.project_root / "problems" / self.problem_name
        self.gpt_file = self.problem_dir / "gpt.py"
        self.eval_mode = "train"
        self.id_counter = 0

        # 初始化进化算子
        self.operators = EvolutionOperators(llm_interface, problem_cfg)
        
        # 获取选择和管理策略
        self.select = SelectionStrategies.get_selection_method(selection_method)
        self.manage = PopulationManagement.get_management_method(management_method)

        # 经验池实例
        self.exp_pool = pool
    
    def _run_code(self, code: str) -> float:
        """
        将生成的代码写入独立的临时工作目录并调用评估脚本，返回目标值。
        使用临时目录可以避免多线程/多进程间的文件冲突。
        """
        if not code or not code.strip():
            raise ValueError("生成的代码为空，无法评估。")

        code = _ensure_required_imports(code)

        # 创建唯一的临时工作目录
        temp_id = uuid.uuid4().hex[:8]
        temp_dir = self.problem_dir / f"temp_eval_{temp_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 将代码写入临时目录的 gpt.py
            temp_gpt_file = temp_dir / "gpt.py"
            temp_gpt_file.write_text(code.rstrip() + "\n", encoding="utf-8")

            # 复制评估脚本到临时目录
            eval_script_name = "eval_black_box.py" if self.problem_type == "black_box" else "eval.py"
            eval_script = self.problem_dir / eval_script_name
            if not eval_script.exists():
                raise FileNotFoundError(f"评估脚本不存在: {eval_script}")

            temp_eval_script = temp_dir / eval_script_name
            shutil.copy2(eval_script, temp_eval_script)

            # 如果存在 dataset 目录，创建符号链接（避免复制大量数据）
            dataset_dir = self.problem_dir / "dataset"
            if dataset_dir.exists():
                temp_dataset_link = temp_dir / "dataset"
                if not temp_dataset_link.exists():
                    temp_dataset_link.symlink_to(dataset_dir.resolve())

            # 复制可能需要的其他文件（如 gen_inst.py）
            for support_file in ["gen_inst.py", "__init__.py", "gls.py"]:
                src_file = self.problem_dir / support_file
                if src_file.exists():
                    shutil.copy2(src_file, temp_dir / support_file)

            env = os.environ.copy()
            existing_py = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{self.project_root}:{existing_py}" if existing_py else str(self.project_root)

            command = [
                "python",
                "-u",
                str(temp_eval_script),
                str(self.problem_size),
                str(self.project_root),
                self.eval_mode,
            ]

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=str(temp_dir),  # 在临时目录中运行
                check=False,
            )
            output = process.stdout or ""
            if process.returncode != 0:
                raise RuntimeError(
                    f"评估脚本执行失败 (exit {process.returncode}). 输出:\n{output.strip()}"
                )

            return _parse_objective(output)

        finally:
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def check_duplicate(self, population, code):
        """检查代码是否重复"""
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def check_improvement(self, parents, offspring):
        """
        判断offspring是否相比parents带来了改进

        参数:
            parents: 父代个体列表
            offspring: 子代个体

        返回:
            bool: True表示有改进，False表示无改进或无法判断
        """
        if not parents or not offspring:
            return False

        offspring_obj = offspring.get('objective')
        if offspring_obj is None:
            return False

        # 获取所有有效的父代objective
        parent_objectives = []
        for parent in parents:
            if isinstance(parent, dict) and parent.get('objective') is not None:
                try:
                    parent_objectives.append(float(parent['objective']))
                except (TypeError, ValueError):
                    continue

        if not parent_objectives:
            return False

        # 根据优化类型判断是否改进
        try:
            offspring_obj_val = float(offspring_obj)
        except (TypeError, ValueError):
            return False

        obj_type = getattr(self.problem_cfg, 'obj_type', 'min').lower()

        if obj_type == 'max':
            # 最大化：offspring比所有parents都大
            return offspring_obj_val > max(parent_objectives)
        else:
            # 最小化：offspring比所有parents都小
            return offspring_obj_val < min(parent_objectives)

    def _make_individual_id(self, operator, parents=None):
        self.id_counter += 1
        parent_ids = []
        for p in parents or []:
            if not isinstance(p, dict):
                parent_ids.append(p)
                continue
            pid = p.get('id')
            if isinstance(pid, dict) and 'uid' in pid:
                parent_ids.append(pid['uid'])
            else:
                parent_ids.append(pid)
        return {
            'uid': self.id_counter,
            'operator': operator,
            'parents': parent_ids
        }

    def build_query(self, opertor_desc, parents):
        """
        构造用于经验检索的查询文本，综合算子意图与父代核心信息
        """
        # 父代摘要
        parent_summaries = []
        for idx, parent in enumerate(parents or [], 1):
            obj_val = parent.get("objective")
            algorithm_desc = parent.get("algorithm") or ""
            algorithm_desc = re.sub(r"\s+", " ", algorithm_desc).strip()
            if len(algorithm_desc) > 400:
                algorithm_desc = algorithm_desc[:400] + " ..."

            meta_bits = []
            if obj_val is not None:
                meta_bits.append(f"objective={obj_val}")

            meta_prefix = "; ".join(meta_bits) if meta_bits else "meta=unknown"
            parent_summaries.append(f"Parent {idx} ({meta_prefix}): {algorithm_desc}")

        if not parent_summaries:
            parent_summaries.append("Parents: none selected")

        query_lines = parent_summaries
        return "\n".join(query_lines)

    def process_memories(self, memories):
        """
        处理检索到的经验，将成功经验和失败经验分开

        参数:
            memories: 检索到的记忆列表

        返回:
            dict: 包含 'success_exp' 和 'fail_exp' 两个字段的字典
        """
        if not memories:
            return {
                'success_exp': "No successful experience retrieved.",
                'fail_exp': "No failed experience retrieved."
            }

        success_summaries = []
        fail_summaries = []

        for memory in memories:
            distilled_raw = memory.get("distilled_items")
            is_success = memory.get("is_success", False)

            if not distilled_raw:
                continue

            # 序列化经验内容
            if isinstance(distilled_raw, (list, dict)):
                try:
                    serialized = json.dumps(distilled_raw, ensure_ascii=False)
                except TypeError:
                    serialized = str(distilled_raw)
            else:
                serialized = str(distilled_raw)

            serialized = serialized.strip()
            if not serialized:
                continue

            # 解析列表格式的经验
            if serialized.startswith("[") and serialized.endswith("]"):
                try:
                    parsed = json.loads(serialized)
                    for item in parsed:
                        if isinstance(item, (dict, list)):
                            item_str = json.dumps(item, ensure_ascii=False)
                        else:
                            item_str = str(item)

                        if is_success:
                            success_summaries.append(item_str)
                        else:
                            fail_summaries.append(item_str)
                    continue
                except json.JSONDecodeError:
                    pass

            # 添加到对应列表
            if is_success:
                success_summaries.append(serialized)
            else:
                fail_summaries.append(serialized)

        success_exp = "\n".join(success_summaries) if success_summaries else "No successful experience retrieved."
        fail_exp = "\n".join(fail_summaries) if fail_summaries else "No failed experience retrieved."

        return {
            'success_exp': success_exp,
            'fail_exp': fail_exp
        }

    def build_trajectory(self, parents, offspring):
        if not isinstance(offspring, dict):
            return None

        objective_type = getattr(self.problem_cfg, "obj_type", "min")
        parent_infos = []
        parent_objectives = []

        for parent in parents or []:
            if not isinstance(parent, dict):
                continue

            parent_obj = parent.get("objective")
            try:
                parent_obj_val = float(parent_obj) if parent_obj is not None else None
            except (TypeError, ValueError):
                parent_obj_val = None

            if parent_obj_val is not None:
                parent_objectives.append(parent_obj_val)

            parent_infos.append({
                "objective": parent_obj_val,
                "algorithm": (parent.get("algorithm") or "")[:400]
            })

        offspring_obj = offspring.get("objective")
        try:
            offspring_obj_val = float(offspring_obj) if offspring_obj is not None else None
        except (TypeError, ValueError):
            offspring_obj_val = None

        raw_id = offspring.get("id") or {}
        operator_name = raw_id.get("operator")
        operator_desc = self.operators.op_desc_dict.get(operator_name, "")

        if parent_objectives:
            if str(objective_type).lower() == "max":
                baseline = max(parent_objectives)
                improvement = offspring_obj_val - baseline if offspring_obj_val is not None else None
            else:
                baseline = min(parent_objectives)
                improvement = baseline - offspring_obj_val if offspring_obj_val is not None else None
        else:
            baseline = None
            improvement = None

        trajectory = {
            "operator_goal": operator_desc,
            "parents": parent_infos,
            "offspring": {
                "objective": offspring_obj_val,
                "algorithm": (offspring.get("algorithm") or "")[:400]
            },
            "objective_type": objective_type,
            "best_parent_objective": baseline,
            "improvement": improvement,
            "problem_name": self.problem_name
        }
        return trajectory
    
    def add_experience(self, query, parents, offspring):
        if not self.exp_pool or not query or not query.strip():
            return

        trajectory = self.build_trajectory(parents, offspring)

        if not trajectory:
            return

        try:
            self.exp_pool.add_experience(query, trajectory)
        except Exception as exc:
            logging.warning("Failed to add experience: %s", exc)
        return

    def update_retrieved_memories_score(self, memories, parents, offspring):
        """
        更新检索到的memories的score

        参数:
            memories: 检索到的记忆列表
            parents: 父代个体列表
            offspring: 子代个体
        """
        if not self.exp_pool or not memories:
            return

        # 判断是否有改进
        improved = self.check_improvement(parents, offspring)

        # 更新每条记忆的score
        for memory in memories:
            try:
                self.exp_pool.update_memory_score(memory, improved)
            except Exception as exc:
                logging.warning("Failed to update memory score: %s", exc)

    def _get_algorithm(self, pop, operator):
        """
        调用指定的进化算子生成新算法
        """
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'id':None,
            'other_inf': None
        }

        memories = []  # 初始化memories

        # 根据方法类型和算子名称调用相应的函数
        if operator == "init":
            parents = None
            query = None
            offspring['code'], offspring['algorithm'] = self.operators.init()

        elif operator in ["op1","op2","op3","op4","op5"]:
            parents = self.select(pop, self.operators.op_nparents_dict[operator])
            op_desc = self.operators.op_desc_dict[operator]
            query = self.build_query(op_desc, parents)

            if self.exp_pool and query:
                memories = self.exp_pool.retrieve_memories(query, operator_goal=op_desc)

            # 处理经验，将成功和失败经验分开
            processed_memories = self.process_memories(memories)
            offspring['code'], offspring['algorithm'] = self.operators.operate(operator, parents, processed_memories)

        else:
            raise ValueError(f"Unknown EoH operator: {operator}")

        parent_records = parents if parents is not None else []
        offspring['id'] = self._make_individual_id(operator, parent_records)

        return parents, query, offspring, memories
    
    def get_offspring(self, pop, operator):
        """
        获取单个子代并评估
        """
        # 生成算法，处理偶发解析失败或重复
        max_generation_attempts = 5
        max_duplicate_retries = 3
        generation_attempts = 0
        duplicate_attempts = 0
        last_generation_error = None
        p = []
        q = None
        offspring = {}
        retrieved_memories = []  # 保存检索到的memories

        while True:
            generation_attempts += 1
            try:
                p, q, offspring, retrieved_memories = self._get_algorithm(pop, operator)
                last_generation_error = None
            except ValueError as exc:
                last_generation_error = str(exc)
                if generation_attempts >= max_generation_attempts:
                    offspring = {
                        'code': '',
                        'objective': None,
                        'algorithm': '',
                        'parents': [],
                        'id': self._make_individual_id(operator, []),
                        'other_inf': {}
                    }
                    retrieved_memories = []
                    break
                else:
                    continue

            code = offspring.get('code', '')
            if self.check_duplicate(pop, code) and duplicate_attempts < max_duplicate_retries:
                duplicate_attempts += 1
                continue
            break
        
        code = offspring.get('code', '')
        
        # 评估代码
        timed_out = False
        eval_error = None
        evaluation_skipped = False
        if code and code.strip():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_code, code)
                try:
                    fitness = future.result(timeout=self.timeout)
                    offspring['objective'] = np.round(fitness, 5) if fitness is not None else None
                except concurrent.futures.TimeoutError:
                    timed_out = True
                    future.cancel()
                    offspring['objective'] = None
                except Exception as exc:
                    eval_error = str(exc)
                    if isinstance(exc, subprocess.TimeoutExpired):
                        timed_out = True
                    offspring['objective'] = None
        else:
            offspring['objective'] = None
            evaluation_skipped = True

        # 记录代码重复尝试次数到 other_inf
        if offspring.get('other_inf') is None:
            offspring['other_inf'] = {}
        offspring['other_inf']['duplicate_attempts'] = duplicate_attempts
        offspring['other_inf']['generation_attempts'] = generation_attempts
        if last_generation_error:
            offspring['other_inf']['generation_error'] = last_generation_error
        if evaluation_skipped:
            offspring['other_inf']['evaluation_skipped'] = True

        if timed_out:
            offspring['other_inf']['timeout'] = True
        if eval_error:
            offspring['other_inf']['error'] = eval_error

        self.add_experience(q, p, offspring)

        # 更新检索到的memories的score
        if retrieved_memories and p:
            self.update_retrieved_memories_score(retrieved_memories, p, offspring)

        return p, offspring
    
    def get_algorithm(self, pop, operator):
        """
        并行生成和评估多个算法
        """
        # print(f"当前种群数: {len(pop)}，生成算子: {operator}")
        results = []
        results = Parallel(n_jobs=self.n_proc, prefer="threads")(
            delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size)
        )
        time.sleep(2)
        
        out_p = []
        out_off = []
        for p, off in results:
            out_p.append(p)
            out_off.append(off)
        
        return out_p, out_off
    
    def population_init(self):
        """
        生成初始种群
        """
        n_create = int(self.init_pop_size/self.pop_size)
        population = []
        
        for i in range(n_create):
            _, pop = self.get_algorithm([], 'init')
            for p in pop:
                population.append(p)
        
        return population

    def save_population(self, population, generation, output_path):
        """
        保存种群到文件
        """
        filepath = os.path.join(
            output_path,
            f"population_generation_{generation}.json"
        )
        save_json(population, filepath)

    def compute_off_stas(self, op, offs, generation, output_path):
        """
        计算种群统计信息
        """
        objs_op = [off.get('objective') for off in offs]
        objs_op_valid = [v for v in objs_op if v is not None]

        r1_val = compute_r1(objs_op_valid)
        l_val = compute_correlation_length(r1_val)
        mean_op = float(np.mean(objs_op_valid)) if len(objs_op_valid) > 0 else None
        var_op = float(np.var(objs_op_valid)) if len(objs_op_valid) > 0 else None

        stats_dict = {"generation": generation,
                    "operator": op,
                    "objectives": objs_op,
                    "r1": r1_val,
                    "l": l_val,
                    "mean": mean_op,
                    "var": var_op}

        save_json(stats_dict, os.path.join(output_path, "history", f"stats_gen_{generation}_{op}.json"))
        
        return objs_op_valid, stats_dict
    
    def add2pop(self, population, offspring):
        """
        将子代添加到种群中
        自动检查重复的objective值
        """
        dup_obj_count = 0
        for off in offspring:
            # 检查是否有重复的objective
            is_duplicate = False
            for ind in population:
                if ind['objective'] == off['objective']:
                    is_duplicate = True
                    dup_obj_count += 1
                    break
            
            if not is_duplicate:
                population.append(off)
                
        return population, dup_obj_count

    def get_best_individual(self, population):
        """
        获取种群中的最优个体
        """
        valid_pop = [ind for ind in population if ind['objective'] is not None]
        if not valid_pop:
            return None
        return min(valid_pop, key=lambda x: x['objective'])

    def save_best_individual(self, individual, generation, output_path):
        """
        保存最优个体
        """
        filepath = os.path.join(
            output_path,
            f"best_generation_{generation}.json"
        )
        save_json(individual, filepath)


    def get_statistics(self, population):
        """
        计算种群统计信息
        """
        objectives = [ind['objective'] for ind in population if ind['objective'] is not None]
        
        if len(objectives) == 0:
            return {
                'best': None,
                'worst': None,
                'mean': None,
                'std': None,
                'count': 0
            }
        
        return {
            'best': float(np.min(objectives)),
            'worst': float(np.max(objectives)),
            'mean': float(np.mean(objectives)),
            'std': float(np.std(objectives)),
            'count': len(objectives)
        }
