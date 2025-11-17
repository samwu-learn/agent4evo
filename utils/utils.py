import logging
import re
import inspect
import os
import yaml
import importlib
from pathlib import Path
import json
import numpy as np

class AttrDict(dict):
    def __getattr__(self, item):
        try:
            value = self[item]
            if isinstance(value, dict) and not isinstance(value, AttrDict):
                value = AttrDict(value)
                self[item] = value
            return value
        except KeyError:
            raise AttributeError(item)
    def __setattr__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        self[key] = value


def instantiate_from_target(cfg_dict: dict):
    """Instantiate a class from a dict containing `_target_` and params."""
    assert isinstance(cfg_dict, dict), "Configuration must be a dict"
    target = cfg_dict.get("_target_")
    assert target, "Missing `_target_` in configuration"
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    params = {k: v for k, v in cfg_dict.items() if k != "_target_"}
    return cls(**params)


_OC_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]+))?\}")


def _convert_literal(value):
    if value is None:
        return None
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return text


def _resolve_oc_env(value):
    if isinstance(value, dict):
        return {k: _resolve_oc_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_oc_env(item) for item in value]
    if isinstance(value, str):
        full_match = _OC_ENV_PATTERN.fullmatch(value)
        if full_match:
            var_name = full_match.group(1).strip()
            default_raw = full_match.group(2)
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            return _convert_literal(default_raw)

        def _substitute(match):
            var_name = match.group(1).strip()
            default_raw = match.group(2)
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            default_value = _convert_literal(default_raw)
            if default_value is None:
                return ""
            return str(default_value)

        return _OC_ENV_PATTERN.sub(_substitute, value)
    return value

def _write_text(path, content):
    """写入文本内容，自动创建上级目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

def load_json(filepath):
    """从JSON文件加载数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, filepath):
    """保存数据到JSON文件"""
    # 确保目录存在
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    # 保存JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _safe_load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _resolve_oc_env(data)


def load_config(config_source: str = "cfg") -> AttrDict:
    """
    Load config.yaml and compose problem / llm_client similar to a simplified Hydra workflow.
    
    Args:
        config_source (str): Directory containing config.yaml or direct path to a yaml file.
    """
    config_path = Path(config_source)
    if config_path.is_dir():
        config_dir = config_path
        config_path = config_dir / "config.yaml"
    else:
        config_dir = config_path.parent

    base = _safe_load_yaml(config_path)

    problem_name = base.get("problem")
    problem_cfg = _safe_load_yaml(config_dir / "problem" / f"{problem_name}.yaml")
    generator_cfg = _safe_load_yaml(config_dir / "llm_client" / "generator_llm.yaml")
    reflector_cfg = _safe_load_yaml(config_dir / "llm_client" / "reflector_llm.yaml")
    planner_cfg = _safe_load_yaml(config_dir / "llm_client" / "planner_llm.yaml")

    composed = AttrDict({
        "algorithm": base.get("algorithm"),
        "n_pop": base.get("n_pop"),
        "pop_size": base.get("pop_size"),
        "init_pop_size": base.get("init_pop_size"),
        "timeout": base.get("timeout"),
        "diversify_init_pop": base.get("diversify_init_pop", False),
        "exp": base.get("exp"),
        "logging": base.get("logging"),
        "problem": problem_cfg,
        "generator_llm": generator_cfg,
        "reflector_llm": reflector_cfg,
        "planner_llm": planner_cfg
    })
    return composed

def setup_logging(config):
    """初始化日志，兼顾终端与文件输出。"""
    log_level = getattr(logging, config.logging_level.upper(), logging.INFO)
    log_file = config.paths.logs_dir / "run.log"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    # Silence verbose HTTP client logging that clutters stdout.
    for noisy_logger in ("httpx", "httpcore", "dashscope"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.info("日志文件: %s", str(log_file))
    return log_file

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if  len(log) > 0:
            if log_status and "Traceback" in log:
                logging.warning(
                    f"Iteration {iter_num}: Code Run {response_id} execution error! (see {stdout_filepath})"
                )
            else:
                logging.info(
                    f"Iteration {iter_num}: Code Run {response_id} successful! (see {stdout_filepath})"
                )
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
    
    if code_string is None:
        return None
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


def compute_r1(values):
    """
    计算序列的lag-1自相关系数 r1。
    r1 = sum_t (f_t - \bar f)(f_{t+1} - \bar f) / sum_t (f_t - \bar f)^2
    仅使用非None的值；当样本不足或方差为0时返回None。
    Args:
        values: list/array of objective values
    Returns:
        float or None
    """
        
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    mean = np.mean(arr)
    dev = arr - mean
    denom = np.dot(dev, dev)
    if denom == 0:
        return None
    numer = np.dot(dev[:-1], dev[1:])
    r1 = float(numer / denom)
    # 限制到[-1, 1] 范围（数值稳定）
    if r1 > 1:
        r1 = 1.0
    elif r1 < -1:
        r1 = -1.0
    return r1
    
def compute_correlation_length(r1):
    """
    根据 r1 计算相关长度 l = -1 / ln(|r1|)。
    规则：
    - r1 为 None -> 返回 None
    - |r1| == 0 -> 返回 0
    - |r1| >= 1 -> 返回 None（不可定义/无限）
    Args:
        r1: float, lag-1 autocorrelation coefficient
    Returns:
        float or None
    """
    if r1 is None:
        return None
    a = abs(float(r1))
    if a == 0:
        return 0.0
    if a >= 1.0:
        return None
    return float(-1.0 / np.log(a))


def _extract_json_block(text):
    """
    从文本中提取JSON内容，支持多种格式：
    1. ```json ... ``` 代码块
    2. 直接的 JSON 数组或对象
    """
    if not text or not isinstance(text, str):
        return text

    # 尝试提取 ```json ... ``` 代码块
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 尝试提取 ```..``` 代码块（没有json标记）
    match = re.search(r"```\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 尝试直接查找 JSON 数组或对象
    # 查找以 [ 开头，] 结尾的数组
    match = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 查找以 { 开头，} 结尾的对象
    match = re.search(r'(\{\s*".*?\})', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 如果都没找到，返回原文本
    return text.strip()

def _parse_objective(output):
    """从评估脚本输出中提取数值目标。"""
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate)
        if match:
            return float(match.group(0))
    raise ValueError("无法从评估输出中解析目标值。")

def _ensure_required_imports(code):
    """根据代码内容自动补齐常见缺失的依赖。"""
    header_lines = []
    stripped_code = code.lstrip()

    needs_numpy = bool(re.search(r"\bnp\.", code))
    has_numpy_import = bool(re.search(r"import\s+numpy\s+as\s+np", code))
    if needs_numpy and not has_numpy_import:
        header_lines.append("import numpy as np")

    if header_lines:
        return "\n".join(header_lines) + "\n\n" + stripped_code
    return code
