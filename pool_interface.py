import json
from typing import Any, Dict, List
import abc
import chromadb
import uuid
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import _extract_json_block, file_to_string

class ExperiencePool:
    """
    负责协调记忆的存储、蒸馏与检索，内部整合了记忆后端、向量嵌入模型以及大语言模型，
    为管理与利用智能体经验提供一套完整方案。
    """

    def __init__(self, memory_backend, llm, embedding_model, problem_cfg=None):

        self.memory_backend = memory_backend
        self.llm = llm
        self.embedding_model = embedding_model
        self.prompt_dir = Path("prompts/common")
        self.problem_cfg = problem_cfg

    def judge_trajectory(self, trajectory):
        """
        根据生成的轨迹信息判断算子是否成功。
        """
        if not trajectory:
            return False

        offspring = (trajectory or {}).get("offspring") or {}
        objective = offspring.get("objective")
        if objective is None:
            return False

        parents = trajectory.get("parents") or []
        parent_objs = [
            p.get("objective")
            for p in parents
            if isinstance(p, dict) and p.get("objective") is not None
        ]

        # 无父代信息时，认为只要有目标值即成功。
        if not parent_objs:
            return True

        try:
            objective_type = (trajectory.get("objective_type") or "min").lower()
        except AttributeError:
            objective_type = "min"

        try:
            objective_val = float(objective)
        except (TypeError, ValueError):
            return False

        try:
            parent_vals = [float(val) for val in parent_objs]
        except (TypeError, ValueError):
            return False

        tolerance = 1e-9
        if objective_type == "max":
            baseline = max(parent_vals)
            return objective_val > baseline + tolerance
        else:
            baseline = min(parent_vals)
            return objective_val < baseline - tolerance
    
    def distill_trajectory(self, query, trajectory):
        is_success = self.judge_trajectory(trajectory)
        status_text = "successful" if is_success else "unsuccessful"
        trajectory_json = json.dumps(trajectory, ensure_ascii=False, indent=2)

        if self.llm is None:
            distilled_items = [{
                "status": status_text,
                "reason": "LLM unavailable; storing raw trajectory summary.",
                "trajectory": trajectory
            }]
            return is_success, json.dumps(distilled_items, ensure_ascii=False)
        
        system_prompt = file_to_string(self.prompt_dir/"system_reflector.txt")

        # 如果有问题配置，格式化 system_prompt
        if self.problem_cfg:
            # 从问题配置目录读取函数描述
            problem_prompt_path = Path("prompts") / self.problem_cfg.problem_name
            func_desc_path = problem_prompt_path / "func_desc.txt"
            func_desc = file_to_string(str(func_desc_path)) if func_desc_path.exists() else "No function description available."

            system_prompt = system_prompt.format(
                problem_desc=self.problem_cfg.description,
                func_desc=func_desc
            )

        if is_success:
            user_prompt = file_to_string(self.prompt_dir/"user_reflector_success.txt").format(
                query=query,
                trajectory=trajectory_json
            )
        else:
            user_prompt = file_to_string(self.prompt_dir/"user_reflector_fail.txt").format(
                query=query,
                trajectory=trajectory_json
            )


        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm._chat_completion_api(
            messages,
            temperature=1.0
        )
        # 提取 JSON 内容
        json_text = _extract_json_block(response)
        if isinstance(json_text, str):
            json_text = json_text.strip()
        else:
            json_text = json.dumps(json_text, ensure_ascii=False)

        # 验证 JSON 格式
        try:
            parsed_json = json.loads(json_text)
            # 确保是列表格式
            if not isinstance(parsed_json, list):
                parsed_json = [parsed_json]
            # 重新序列化以确保格式正确
            json_text = json.dumps(parsed_json, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError) as e:
            # JSON 解析失败，使用降级方案
            print(f"Warning: Failed to parse LLM reflection output: {e}")
            print(f"Raw response: {response[:500]}...")
            print(f"Extracted JSON: {json_text[:500]}...")

            fallback = [{
                "summary": "Trajectory analysis unavailable due to invalid LLM output.",
                "recommendations": ["Review LLM output format", "Check system_reflector.txt prompt"],
                "applicable_when": "LLM parsing failure - please check the reflection prompt"
            }]
            json_text = json.dumps(fallback, ensure_ascii=False)

        return is_success, json_text

    def add_experience(self, query, trajectory):
        """
        向记忆库中新增一条经验
        """
        if not query or not query.strip() or not trajectory:
            return

        if self.embedding_model is None or self.memory_backend is None:
            return

        clean_query = query.strip()

        # 从 trajectory 中提取 operator_goal
        operator_goal = trajectory.get("operator_goal", "unknown")

        is_success, distilled_memories = self.distill_trajectory(clean_query, trajectory)

        # 解析向量表示
        try:
            query_embedding = self.embedding_model.encode(clean_query)
        except Exception:
            return
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        distilled_payload = distilled_memories
        if not isinstance(distilled_payload, str):
            try:
                distilled_payload = json.dumps(distilled_payload, ensure_ascii=False)
            except TypeError:
                distilled_payload = str(distilled_payload)

        trajectory_payload = json.dumps(trajectory, ensure_ascii=False, default=str)

        # 准备存储体验数据。
        experience_to_add = {
            "embedding": query_embedding,
            "metadata": {
                "query": clean_query,
                "trajectory": trajectory_payload,
                "distilled_items": distilled_payload,
                "is_success": is_success,
                "operator_goal": operator_goal,  # 记录此条经验由哪个算子得到
                "n_retrieves": 0,  # 记录该条memory被使用的次数
                "score": 0,  # 记录该条memory的分数
            },
            "document": clean_query,
        }

        self.memory_backend.add([experience_to_add])

    def retrieve_memories(self, query, operator_goal, k=3):
        """
        针对给定的问题检索最相关的前 k 条记忆。
        按照 operator_goal 强制缩小检索范围，根据 query 返回最相关的前 k 条经验。

        参数:
            query (str): 用来检索相关记忆的问题（不包含 operator_goal 信息）。
            operator_goal (str): 算子目标，用于过滤记忆。
            k (int): 需要返回的记忆条数。

        返回:
            List[Dict]: 最相关的 k 条记忆组成的列表。
        """
        if not query or not query.strip() or self.embedding_model is None or self.memory_backend is None:
            return []

        clean_query = query.strip()
        try:
            query_embedding = self.embedding_model.encode(clean_query)
        except Exception:
            return []
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        # 检索足够多的记忆以便过滤
        initial_k = max(k * 10, 50)
        all_memories = self.memory_backend.query(query_embedding, initial_k)

        if not all_memories:
            return []

        # 按 operator_goal 过滤记忆
        filtered_memories = []
        for memory in all_memories:
            memory_operator_goal = memory.get("operator_goal", "unknown")
            if memory_operator_goal == operator_goal:
                filtered_memories.append(memory)

        # 如果没有匹配的记忆，返回空列表
        if not filtered_memories:
            return []

        # 取前 k 个记忆（已经按相似度排序）
        selected_memories = filtered_memories[:k]

        # 更新选中记忆的 n_retrieves 计数
        for memory in selected_memories:
            memory_id = memory.get("_id")
            if memory_id:
                current_n_retrieves = memory.get("n_retrieves", 0)
                self.memory_backend.update(memory_id, {"n_retrieves": current_n_retrieves + 1})

        return selected_memories

    def update_memory_score(self, memory, improved):
        """
        更新指定记忆的分数。

        参数:
            memory (Dict): 从 retrieve_memories 返回的记忆对象，需包含 _id 和 score 字段。
            improved (bool): 该记忆是否带来了算法改进。True 表示 score+1，False 表示 score+0（不变）。
        """
        if not memory or self.memory_backend is None:
            return

        memory_id = memory.get("_id")
        if not memory_id:
            return

        current_score = memory.get("score", 0)
        new_score = current_score + (1 if improved else 0)

        self.memory_backend.update(memory_id, {"score": new_score})



class MemoryBackend(abc.ABC):
    """
    记忆后端的抽象基类。

    该类定义了所有记忆后端必须实现的接口，提供统一的新增与查询方式，
    以便不同存储方案可以互换使用。
    """

    @abc.abstractmethod
    def add(self, items: List[Dict]):
        """
        向后端新增一批记忆条目。

        参数:
            items (List[Dict]): 由字典组成的列表，每个字典代表一条记忆。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        查询与给定向量最相似的前 k 条记忆。

        参数:
            query_embedding (List[float]): 查询的向量表示。
            k (int): 需要返回的条目数量。

        返回:
            List[Dict]: 最相似的 k 条记忆条目列表。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, memory_id: str, updates: Dict):
        """
        更新指定ID的记忆条目的metadata。

        参数:
            memory_id (str): 记忆条目的唯一标识符。
            updates (Dict): 要更新的字段字典。
        """
        raise NotImplementedError

"""ChromaDB 记忆后端实现。"""
class ChromaMemoryBackend(MemoryBackend):
    """
    使用 ChromaDB 作为存储的记忆后端。

    该实现基于 ChromaDB 存储与检索记忆，适用于需要可扩展且高效向量数据库的生产环境。
    """

    def __init__(self, collection_name: str = "reasoning_bank"):
        """
        初始化 ChromaMemoryBackend。

        参数:
            collection_name (str): 指定要使用的 ChromaDB 集合名称。
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(self, items: List[Dict]):
        """
        将一批记忆条目写入 ChromaDB 集合。

        每个条目需包含以下字段:
        - embedding: 记忆条目的向量表示。
        - metadata: 包含标题、描述与正文的字典。
        - document: 记忆条目的文本内容。

        参数:
            items (List[Dict]): 待写入的记忆条目列表。
        """
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in items],
            embeddings=[item["embedding"] for item in items],
            metadatas=[item["metadata"] for item in items],
            documents=[item["document"] for item in items],
        )

    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        在 ChromaDB 集合中检索与查询最相似的前 k 条条目。

        参数:
            query_embedding (List[float]): 查询的向量表示。
            k (int): 需要返回的结果数量。

        返回:
            List[Dict]: 最相似的 k 条条目的元数据列表，每个字典包含 _id 字段和原始 metadata。
        """
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=k
        )
        # 查询结果中的 metadatas 是按查询向量分组的列表；
        # 由于这里只传入单个向量，因此取第一个结果列表即可。
        if not results["metadatas"] or not results["ids"]:
            return []

        metadatas = results["metadatas"][0]
        ids = results["ids"][0]

        # 将 id 添加到每个 metadata 中
        return [{"_id": id_val, **metadata} for id_val, metadata in zip(ids, metadatas)]

    def update(self, memory_id: str, updates: Dict):
        """
        更新指定ID的记忆条目的metadata。

        参数:
            memory_id (str): 记忆条目的唯一标识符。
            updates (Dict): 要更新的字段字典。
        """
        # 获取当前的 metadata
        current = self.collection.get(ids=[memory_id])
        if not current["metadatas"]:
            return

        current_metadata = current["metadatas"][0]
        # 更新 metadata
        current_metadata.update(updates)

        # ChromaDB 的 update 方法
        self.collection.update(
            ids=[memory_id],
            metadatas=[current_metadata]
        )


"""基于 JSON 文件的记忆后端。"""
class JSONMemoryBackend(MemoryBackend):
    """
    使用 JSON 文件存储记忆的轻量级后端。

    该实现主要用于测试与开发，将记忆写入简单的 JSON 文件，并通过余弦相似度执行查询。
    因性能限制，不推荐在生产环境中使用。
    """

    def __init__(self, filepath: str):
        """
        初始化 JSONMemoryBackend。

        参数:
            filepath (str): 记忆数据所对应的 JSON 文件路径。
        """
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> List[Dict]:
        """从 JSON 文件加载记忆。"""
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def _save(self):
        """将记忆写入 JSON 文件。"""
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=4)

    def add(self, items: List[Dict]):
        """
        将一批记忆条目追加到 JSON 文件。

        参数:
            items (List[Dict]): 待写入的记忆列表。
        """
        # 为每个item添加唯一ID
        for item in items:
            if "_id" not in item:
                item["_id"] = str(uuid.uuid4())
        self.data.extend(items)
        self._save()

    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        利用余弦相似度，在 JSON 文件中检索最相似的前 k 条条目。

        参数:
            query_embedding (List[float]): 查询的向量表示。
            k (int): 返回的结果数量。

        返回:
            List[Dict]: 最相似的 k 条记忆条目的元数据，每个字典包含 _id 字段和原始 metadata。
        """
        if not self.data:
            return []

        embeddings = np.array([item["embedding"] for item in self.data])
        query_embedding_np = np.array(query_embedding).reshape(1, -1)

        similarities = cosine_similarity(query_embedding_np, embeddings)[0]

        # 取出相似度最高的前 k 个索引。
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # 返回包含 _id 和 metadata 的结果
        return [{"_id": self.data[i]["_id"], **self.data[i]["metadata"]} for i in top_k_indices]

    def update(self, memory_id: str, updates: Dict):
        """
        更新指定ID的记忆条目的metadata。

        参数:
            memory_id (str): 记忆条目的唯一标识符。
            updates (Dict): 要更新的字段字典。
        """
        for item in self.data:
            if item.get("_id") == memory_id:
                # 更新 metadata
                item["metadata"].update(updates)
                self._save()
                return
