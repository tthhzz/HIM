from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union
import json
from datetime import datetime
import uuid
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import requests
import json as json_lib
import time
import math
from dataclasses import dataclass, field

# ========== 创新点配置类 ==========
@dataclass
class CognitiveMemoryConfig:
    """Human-like Memory (HLM) 类人记忆策略配置

    三大模块:
    1. Cognitive Memory Encoding (认知记忆编码)
    2. Dynamic Memory Consolidation (动态记忆巩固)
    3. Reliable Memory Retrieval (可靠记忆检索)

    用于消融实验，可以单独启用/禁用每个创新点
    """

    # ========== Module 1: Cognitive Memory Encoding ==========
    # 源监控 (Source Monitoring) - 记录说话人身份
    enable_source_monitoring: bool = True

    # 注意力门控 (Attention Gating) - 只影响写入分级，不影响检索排序
    enable_attention_gating: bool = True
    importance_threshold_low: float = 0.3   # 低于此值进入 STM
    importance_threshold_high: float = 0.7  # 高于此值进入 LTM

    # 时间上下文编码 (Temporal Context Encoding)
    enable_temporal_encoding: bool = True

    # ========== Module 2: Dynamic Memory Consolidation ==========
    # 轻量巩固 (Sleep Consolidation) - 提升高频访问记忆
    enable_consolidation: bool = True
    consolidation_interval: int = 50  # 每 N 条记忆触发一次巩固
    consolidation_boost: float = 0.1  # 高频访问记忆的 importance 提升幅度

    # ========== Module 3: Reliable Memory Retrieval ==========
    # ACT-R 激活记录 (Activation Logging) - 只记录不排序，用于可解释性
    enable_activation_logging: bool = True
    activation_alpha: float = 0.70   # 相似度权重
    activation_beta: float = 0.10    # 频率加强权重
    activation_gamma: float = 0.05   # 时间衰减系数
    activation_delta: float = 0.10   # 重要性权重
    activation_epsilon: float = 0.05 # 记忆层级权重

    # 检索置信度评分 (Retrieval Confidence)
    enable_confidence_scoring: bool = True

    # 跨说话人引用追踪 (Cross-Speaker Reference)
    enable_cross_speaker_tracking: bool = True

    # ========== 温度配置 ==========
    keyword_generation_temperature: float = 0.2  # 关键词生成温度（应稳定）
    content_analysis_temperature: float = 0.2    # 内容分析温度
    evolution_temperature: float = 0.2           # 演化决策温度

    # ========== 可复现性配置 ==========
    random_seed: int = 42

    # ========== 日志配置 ==========
    enable_cognitive_logging: bool = True  # 启用认知日志输出

# 默认配置实例
DEFAULT_COGNITIVE_CONFIG = CognitiveMemoryConfig()

# ========== 时间解析工具函数 ==========
def parse_timestamp(timestamp_str: str) -> datetime:
    """解析各种格式的时间戳字符串"""
    if isinstance(timestamp_str, datetime):
        return timestamp_str

    # 尝试多种格式
    formats = [
        "%Y%m%d%H%M",  # 默认格式: 202301091430
        "%I:%M %p on %d %B, %Y",  # 1:56 pm on 8 May, 2023
        "%H:%M %p on %d %B, %Y",  # 13:56 pm on 8 May, 2023
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except (ValueError, TypeError):
            continue

    # 如果都失败，返回当前时间
    return datetime.now()

def calculate_time_delta_days(timestamp_str: str, reference_time: datetime = None) -> float:
    """计算时间差（天数）"""
    if reference_time is None:
        reference_time = datetime.now()

    parsed_time = parse_timestamp(timestamp_str)
    delta = reference_time - parsed_time
    return max(delta.total_seconds() / 86400.0, 0.001)  # 至少 0.001 天避免除零

def simple_tokenize(text):
    return word_tokenize(text)

def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        flattened = []
        for item in value:
            if isinstance(item, list):
                flattened.extend(str(sub_item) for sub_item in item)
            else:
                flattened.append(str(item))
        return flattened
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=True)]
    if isinstance(value, str):
        return [value]
    return [str(value)]

def is_llama_model(model_name: Optional[str]) -> bool:
    return bool(model_name and "llama" in model_name.lower())

def _extract_json_block(text: str) -> str:
    response_cleaned = text.strip()
    if not response_cleaned.startswith('{'):
        start_idx = response_cleaned.find('{')
        if start_idx != -1:
            response_cleaned = response_cleaned[start_idx:]
    if not response_cleaned.endswith('}'):
        end_idx = response_cleaned.rfind('}')
        if end_idx != -1:
            response_cleaned = response_cleaned[:end_idx + 1]
    return response_cleaned

def _normalize_json_literals(text: str) -> str:
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text

def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    cleaned = _extract_json_block(response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return json.loads(_normalize_json_literals(cleaned))
        except json.JSONDecodeError:
            return None

def _coerce_indices(indices: Any) -> List[int]:
    if indices is None:
        return []
    coerced = []
    for idx in indices:
        try:
            coerced.append(int(idx))
        except (TypeError, ValueError):
            continue
    return coerced

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            # Allow custom OpenAI-compatible endpoints (e.g., DeepSeek) via base_url or OPENAI_BASE_URL
            if base_url is None:
                base_url = os.getenv('OPENAI_BASE_URL')
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if max_tokens is None:
            max_tokens = 1000
        extra_args = {}
        if top_p is not None:
            extra_args["top_p"] = top_p
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_args
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"],
                                                            prop_schema.get("items"))

        return result

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            completion_args = {
                "model": "ollama_chat/{}".format(self.model),
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": response_format,
                "temperature": temperature,
            }
            if top_p is not None:
                completion_args["top_p"] = top_p
            if max_tokens is not None:
                completion_args["max_tokens"] = max_tokens
            response = completion(
                **completion_args
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class SGLangController(BaseLLMController):
    def __init__(self, model: str = "llama2", sglang_host: str = "http://localhost", sglang_port: int = 30000):
        self.model = model
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.base_url = f"{sglang_host}:{sglang_port}"

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"],
                                                            prop_schema.get("items"))

        return result

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            # Extract JSON schema from response_format and convert to string format
            json_schema = response_format.get("json_schema", {}).get("schema", {})
            json_schema_str = json.dumps(json_schema)
            max_new_tokens = max_tokens if max_tokens is not None else 1000

            # Prepare SGLang request with correct format
            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "json_schema": json_schema_str  # SGLang expects JSON schema as string
                }
            }
            if top_p is not None:
                payload["sampling_params"]["top_p"] = top_p

            # Make request to SGLang server
            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                # SGLang returns the generated text in 'text' field
                generated_text = result.get("text", "")
                return generated_text
            else:
                print(f"SGLang server returned status {response.status_code}: {response.text}")
                raise Exception(f"SGLang server error: {response.status_code}")

        except Exception as e:
            print(f"SGLang completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LiteLLMController(BaseLLMController):
    """LiteLLM controller for universal LLM access including Ollama and SGLang"""
    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"],
                                                            prop_schema.get("items"))

        return result

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            # Prepare completion arguments
            completion_args = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": response_format,
                "temperature": temperature
            }
            if top_p is not None:
                completion_args["top_p"] = top_p
            if max_tokens is not None:
                completion_args["max_tokens"] = max_tokens

            # Add API base and key if provided
            if self.api_base:
                completion_args["api_base"] = self.api_base
            if self.api_key:
                completion_args["api_key"] = self.api_key

            response = completion(**completion_args)
            return response.choices[0].message.content

        except Exception as e:
            print(f"LiteLLM completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        self.backend = backend
        self.model = model
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, api_base)
        elif backend == "ollama":
            # Use LiteLLM to control Ollama with JSON output
            ollama_model = f"ollama/{model}" if not model.startswith("ollama/") else model
            self.llm = LiteLLMController(
                model=ollama_model,
                api_base="http://localhost:11434",
                api_key="EMPTY"
            )
        elif backend == "sglang":
            # Direct SGLang API calls (better performance, no proxy)
            self.llm = SGLangController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', or 'sglang'")

class MemoryNote:
    """Basic memory unit with metadata

    创新点扩展字段:
    - source_speaker: 来源说话人（策略3：源监控）
    - memory_level: STM/MTM/LTM 记忆分级（策略6：注意力门控）
    """
    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None,
                 # ===== 创新点新增字段 =====
                 source_speaker: Optional[str] = None,  # 策略3：源监控
                 memory_level: str = "MTM",  # 策略6：记忆分级 STM/MTM/LTM
                 cognitive_config: CognitiveMemoryConfig = None):
        self.content = content
        self._cognitive_config = cognitive_config or DEFAULT_COGNITIVE_CONFIG

        # ===== 策略3：源监控 - 解析 source_speaker =====
        if source_speaker:
            self.source_speaker = source_speaker
        elif self._cognitive_config.enable_source_monitoring:
            # 尝试从 content 中解析 speaker
            self.source_speaker = self._extract_speaker_from_content(content)
        else:
            self.source_speaker = "Unknown"

        # Generate metadata using LLM if not provided and controller is available
        analysis = None
        if llm_controller and any(param is None for param in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller, self._cognitive_config)
            print("analysis", analysis)
            keywords = keywords or analysis.get("keywords", [])
            context = context or analysis.get("context", "General")
            tags = tags or analysis.get("tags", [])
            # ===== 策略6：注意力门控 - 从 LLM 获取 importance =====
            if importance_score is None and self._cognitive_config.enable_attention_gating:
                importance_score = analysis.get("importance", 0.5)

        # Set default values for optional parameters
        self.id = id or str(uuid.uuid4())
        self.keywords = normalize_list(keywords)
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        # Handle context that can be either string or list
        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(str(item) for item in self.context)
        elif isinstance(self.context, dict):
            self.context = json.dumps(self.context, ensure_ascii=True)
        elif not isinstance(self.context, str):
            self.context = str(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = normalize_list(tags)

        # ===== 策略6：注意力门控 - 计算 memory_level =====
        if self._cognitive_config.enable_attention_gating:
            if self.importance_score < self._cognitive_config.importance_threshold_low:
                self.memory_level = "STM"  # 短期记忆
            elif self.importance_score >= self._cognitive_config.importance_threshold_high:
                self.memory_level = "LTM"  # 长期记忆
            else:
                self.memory_level = "MTM"  # 中期记忆
        else:
            self.memory_level = memory_level

    def _extract_speaker_from_content(self, content: str) -> str:
        """策略3：从内容中提取 speaker 信息"""
        # 常见模式: "Speaker X says:" 或 "Speaker Xsays :"
        patterns = [
            r"Speaker\s+([A-Za-z]+)\s*says\s*:",  # Speaker Caroline says:
            r"Speaker\s+([A-Za-z]+)says\s*:",     # Speaker Carolinesays :
            r"^([A-Za-z]+)\s*:\s*",               # Caroline: ...
            r"\[([A-Za-z]+)\]",                    # [Caroline]
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "Unknown"

    def update_access(self):
        """更新访问信息（策略2：ACT-R 激活）"""
        self.last_accessed = datetime.now().strftime("%Y%m%d%H%M")
        self.retrieval_count += 1

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController, cognitive_config: CognitiveMemoryConfig = None) -> Dict:
        """Analyze content to extract keywords, context, and other metadata

        创新点：添加 importance 评分（策略6：注意力门控）
        """
        if cognitive_config is None:
            cognitive_config = DEFAULT_COGNITIVE_CONFIG

        # 基础 prompt
        base_prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags"""

        # 策略6：注意力门控 - 添加 importance 评分说明
        if cognitive_config.enable_attention_gating:
            base_prompt += """
            4. Evaluating importance score (0.0-1.0) based on:
               - Is this a key event, decision, or action? (high: 0.7-1.0)
               - Is this opinion expression or emotional reaction? (medium: 0.4-0.7)
               - Is this greeting, confirmation, or repetitive info? (low: 0.1-0.4)"""

        prompt = base_prompt + """

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context":
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]"""

        if cognitive_config.enable_attention_gating:
            prompt += """,
                "importance": // float 0.0-1.0, how important is this memory for future recall"""

        prompt += """
            }

            Content for analysis:
            """ + content
        response = None
        try:
            model_name = getattr(llm_controller, "model", None)
            is_llama = is_llama_model(model_name)

            # 构建 JSON schema
            schema_properties = {
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "context": {
                    "type": "string",
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
            }
            required_fields = ["keywords", "context", "tags"]

            # 策略6：注意力门控 - 添加 importance 字段
            if cognitive_config.enable_attention_gating:
                schema_properties["importance"] = {
                    "type": "number",
                }
                required_fields.append("importance")

            response = llm_controller.llm.get_completion(
                prompt,
                response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": schema_properties,
                                "required": required_fields,
                                "additionalProperties": False
                            },
                            "strict": not is_llama
                    }
                },
                temperature=0.2 if is_llama else 0.7,
                top_p=0.9 if is_llama else None,
                max_tokens=512 if is_llama else None,
            )

            analysis = _parse_json_response(response)
            if analysis is None:
                print("JSON parsing error in analyze_content")
                if response is not None:
                    print(f"Raw response: {response}")
                analysis = {
                    "keywords": [],
                    "context": "General",
                    "tags": [],
                    "importance": 0.5
                }

            # 确保 importance 在有效范围内
            if "importance" in analysis:
                analysis["importance"] = max(0.1, min(1.0, float(analysis.get("importance", 0.5))))

            return analysis

        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            if response is not None:
                print(f"Raw response: {response}")
            return {
                "keywords": [],
                "context": "General",
                "category": "Uncategorized",
                "tags": [],
                "importance": 0.5
            }

class HybridRetriever:
    """Hybrid retrieval system combining BM25 and semantic search."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        """Initialize the hybrid retriever.

        Args:
            model_name: Name of the SentenceTransformer model to use
            alpha: Weight for combining BM25 and semantic scores (0 = only BM25, 1 = only semantic)
        """
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index


    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""

        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)

        # Save everything else using pickle
        state = {
            'alpha': self.alpha,
            'bm25': self.bm25,
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'model_name': 'all-MiniLM-L6-v2'  # Default value for model name
        }

        # Try to get the actual model name if possible
        try:
            state['model_name'] = self.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            pass

        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        # Load the pickled state
        with open(retriever_cache_file, 'rb') as f:
            state = pickle.load(f)

        # Create new instance
        retriever = cls(model_name=state['model_name'], alpha=state['alpha'])
        retriever.bm25 = state['bm25']
        retriever.corpus = state['corpus']
        retriever.document_ids = state.get('document_ids', {})

        # Load embeddings from numpy file if it exists
        if retriever_cache_embeddings_file.exists():
            retriever.embeddings = np.load(retriever_cache_embeddings_file)

        return retriever

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str, alpha: float) -> bool:
        """Load retriever state from memory"""
        all_docs = [", ".join(m.keywords) for m in memories.values()] #[m.content for m in memories.values()]
        retriever = cls(model_name, alpha)
        retriever.add_documents(all_docs)
        return retriever

    def add_documents(self, documents: List[str]) -> bool:
        """One-time Add documents to both BM25 and semantic index"""
        if not documents:
            return

        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Create embeddings
        self.embeddings = self.model.encode(documents)
        self.corpus = documents
        doc_idx = 0
        for document in documents:
            self.document_ids[document] = doc_idx
            doc_idx += 1

        return True

    def add_document(self, document: str) -> bool:
        """Add a single document to the retriever.

        Args:
            document: Text content to add

        Returns:
            bool: True if document was added, False if it was already present
        """
        # Check if document already exists
        if document in self.document_ids:
            return False

        # Add to corpus and get index
        doc_idx = len(self.corpus)
        self.corpus.append(document)
        self.document_ids[document] = doc_idx

        # Update BM25
        if self.bm25 is None:
            # First document, initialize BM25
            tokenized_corpus = [simple_tokenize(document)]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            # Add to existing BM25
            tokenized_doc = simple_tokenize(document)
            self.bm25.add_document(tokenized_doc)

        # Update embeddings
        doc_embedding = self.model.encode([document], convert_to_tensor=True)
        if self.embeddings is None:
            self.embeddings = doc_embedding
        else:
            self.embeddings = torch.cat([self.embeddings, doc_embedding])

        return True

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """Retrieve documents using hybrid scoring"""
        if not self.corpus:
            return []

        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Normalize BM25 scores if they exist
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

        # Get semantic scores
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]

        # Combine scores
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores

        # Get top k indices
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return top_k_indices.tolist()

class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the simple embedding retriever.

        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index

    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        # Reset if no existing documents
        if not self.corpus:
            self.corpus = documents
            # print("documents", documents, len(documents))
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append new documents
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        """Search for similar documents using cosine similarity.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of dicts with document text and score
        """
        if not self.corpus:
            return []
        # print("corpus", len(self.corpus), self.corpus)
        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]


        return top_k_indices

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)

        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")

        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")

        # Load other attributes
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")

        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        """Load retriever state from memory"""
        # Create documents combining content and metadata for each memory
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)

        # Create and initialize retriever
        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever

class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval

    创新点扩展:
    - 策略3：源监控（Source Monitoring）
    - 策略2：ACT-R 激活式检索
    - 策略6：注意力门控（选择性写入）
    - 轻量巩固（Sleep Consolidation）
    """
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 # ===== 创新点配置 =====
                 cognitive_config: CognitiveMemoryConfig = None):
        self.memories = {}  # id -> MemoryNote
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base, sglang_host, sglang_port)

        # 创新点配置
        self.cognitive_config = cognitive_config or DEFAULT_COGNITIVE_CONFIG

        # 记忆计数器（用于巩固触发）
        self.memory_counter = 0

        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"],
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    def add_note(self, content: str, time: str = None, source_speaker: str = None, **kwargs) -> str:
        """Add a new memory note

        创新点:
        - 策略3：接受 source_speaker 参数
        - 策略6：传递 cognitive_config 给 MemoryNote
        - 轻量巩固：每 N 条记忆触发巩固
        """
        note = MemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            source_speaker=source_speaker,  # 策略3：源监控
            cognitive_config=self.cognitive_config,  # 策略6：注意力门控
            **kwargs
        )

        # Update retriever with all documents
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note

        # 策略3：源监控 - 在检索文档中包含 source_speaker
        doc_text = "content:" + note.content + " context:" + note.context
        doc_text += " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)
        if self.cognitive_config.enable_source_monitoring and note.source_speaker:
            doc_text += " speaker: " + note.source_speaker

        self.retriever.add_documents([doc_text])

        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        # ===== 轻量巩固：定期触发 =====
        self.memory_counter += 1
        if (self.cognitive_config.enable_consolidation and
            self.memory_counter % self.cognitive_config.consolidation_interval == 0):
            self._perform_consolidation()

        return note.id

    def _perform_consolidation(self):
        """===== 轻量巩固（Sleep Consolidation）=====

        巯查高频访问的记忆，提升其 importance 并将 STM 转为 MTM/LTM
        """
        if not self.cognitive_config.enable_consolidation:
            return

        print(f"[Consolidation] Performing memory consolidation at {self.memory_counter} memories")

        for note_id, note in self.memories.items():
            # 根据 retrieval_count 提升 importance
            if note.retrieval_count >= 3:
                # 高频访问记忆，提升 importance
                boost = self.cognitive_config.consolidation_boost * math.log(1 + note.retrieval_count)
                note.importance_score = min(1.0, note.importance_score + boost)

                # STM -> MTM/LTM 升级
                if note.memory_level == "STM":
                    if note.importance_score >= self.cognitive_config.importance_threshold_high:
                        note.memory_level = "LTM"
                    else:
                        note.memory_level = "MTM"
                    print(f"[Consolidation] Memory {note_id[:8]}... upgraded to {note.memory_level}")

    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents

        This function re-initializes the retriever and updates it with all memory documents,
        including their context, keywords, and tags to ensure the retrieval system has the
        latest state of all memories.
        """
        # Reset the retriever with the same model
        try:
            # Try to get model name through get_config_dict if available
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            # Fallback: use the model name from the class initialization
            model_name = 'all-MiniLM-L6-v2'

        self.retriever = SimpleEmbeddingRetriever(model_name)

        # Re-add all memory documents with their metadata
        for memory in self.memories.values():
            # Combine memory metadata into a single searchable document
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            # Add both the content and metadata as separate documents for better retrieval
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def process_memory(self, note: MemoryNote) -> bool:
        """Process a memory note and return an evolution label"""
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        fallback_links = _coerce_indices(indices)
        prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories=neighbor_memory,neighbor_number=len(indices))
        print("prompt_memory", prompt_memory)
        model_name = getattr(self.llm_controller, "model", None)
        is_llama = is_llama_model(model_name)
        response = self.llm_controller.llm.get_completion(
            prompt_memory,
            response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean",
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve","actions","suggested_connections","tags_to_update","new_context_neighborhood","new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": not is_llama
                    }},
            temperature=0.2 if is_llama else 0.7,
            top_p=0.9 if is_llama else None,
            max_tokens=512 if is_llama else None,
        )
        print("response", response, type(response))
        response_json = _parse_json_response(response)
        if response_json is None:
            print("JSON parsing error in process_memory")
            print(f"Raw response: {response}")
            if fallback_links:
                note.links.extend(fallback_links)
            return False, note
        print("response_json", response_json, type(response_json))
        should_evolve = bool(response_json.get("should_evolve", False))
        if should_evolve:
            actions = response_json.get("actions", [])
            if not isinstance(actions, list):
                actions = [actions]
            for action in actions:
                if action == "strengthen":
                    suggest_connections = _coerce_indices(response_json.get("suggested_connections", []))
                    if not suggest_connections and fallback_links:
                        suggest_connections = fallback_links
                    note.links.extend(suggest_connections)
                    new_tags = response_json.get("tags_to_update", [])
                    if isinstance(new_tags, list):
                        note.tags = [str(tag) for tag in new_tags]
                    elif new_tags is None:
                        note.tags = []
                    else:
                        note.tags = [str(new_tags)]
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                    new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    print("indices", indices)
                    # if slms output less than the number of neighbors, use the sequential order of new tags and context.
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        # find some memory
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        # add tag to memory
                        notetmp.tags = normalize_list(tag)
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp
        return should_evolve,note

    def find_related_memories(self, query: str, k: int = 5) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return "",[]

        # Get indices of related memories
        # indices = self.retriever.retrieve(query_note.content, k)
        indices = self.retriever.search(query, k)

        # Convert to list of memories
        all_memories = list(self.memories.values())
        memory_str = ""
        # print("indices", indices)
        # print("all_memories", all_memories)
        for i in indices:
            memory_str += (
                "memory index:" + str(i)
                + "\t talk start time:" + str(all_memories[i].timestamp)
                + "\t memory content: " + str(all_memories[i].content)
                + "\t memory context: " + str(all_memories[i].context)
                + "\t memory keywords: " + str(all_memories[i].keywords)
                + "\t memory tags: " + str(all_memories[i].tags)
                + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5,
                                    category: int = None) -> str:
        """Find related memories using semantic retrieval

        Human-like Memory (HLM) 创新点:
        - Module 1: Source Monitoring - 输出包含说话人信息
        - Module 3: ACT-R Activation Logging - 计算并记录激活值（不影响排序）
        - Module 3: Retrieval Confidence - 计算检索置信度
        - Module 3: Cross-Speaker Tracking - 跨说话人引用追踪

        Args:
            query: 查询文本
            k: 返回记忆数量
            category: 问题类别

        Returns:
            memory_str: 检索到的记忆文本
        """
        if not self.memories:
            return ""

        # ===== 核心检索：使用原始相似度排序，不做任何重排 =====
        indices = self.retriever.search(query, k)

        # Convert to list of memories
        all_memories = list(self.memories.values())
        memory_str = ""
        j = 0

        # ===== Module 3: ACT-R Activation Logging =====
        activation_logs = []
        speaker_counts = {}  # 用于跨说话人追踪

        for rank, i in enumerate(indices):
            if i >= len(all_memories):
                continue

            mem = all_memories[i]

            # 计算 ACT-R 激活值（仅用于日志，不影响排序）
            if self.cognitive_config.enable_activation_logging:
                activation = self._calculate_activation_score(mem, rank, len(indices))
                activation_logs.append({
                    'rank': rank,
                    'memory_id': mem.id[:8],
                    'activation': round(activation, 4),
                    'memory_level': getattr(mem, 'memory_level', 'MTM'),
                    'retrieval_count': getattr(mem, 'retrieval_count', 0),
                    'speaker': getattr(mem, 'source_speaker', 'Unknown')
                })

            # ===== Module 3: Cross-Speaker Tracking =====
            if self.cognitive_config.enable_cross_speaker_tracking:
                speaker = getattr(mem, 'source_speaker', 'Unknown')
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            # ===== Module 1: Source Monitoring =====
            speaker_prefix = ""
            if self.cognitive_config.enable_source_monitoring and hasattr(mem, 'source_speaker') and mem.source_speaker:
                speaker_prefix = f"({mem.source_speaker} said) "

            memory_str += (
                "talk start time:" + str(mem.timestamp)
                + " memory content: " + speaker_prefix + str(mem.content)
                + " memory context: " + str(mem.context)
                + " memory keywords: " + str(mem.keywords)
                + " memory tags: " + str(mem.tags)
                + "\n"
            )

            # 更新访问信息（用于巩固）
            if hasattr(mem, 'update_access'):
                mem.update_access()

            # 处理邻居链接
            neighborhood = mem.links if hasattr(mem, 'links') else []
            for neighbor in neighborhood:
                if not isinstance(neighbor, int):
                    try:
                        neighbor = int(neighbor)
                    except (TypeError, ValueError):
                        continue

                if neighbor >= len(all_memories):
                    continue

                neighbor_mem = all_memories[neighbor]

                # Source Monitoring for neighbors
                neighbor_speaker_prefix = ""
                if self.cognitive_config.enable_source_monitoring and hasattr(neighbor_mem, 'source_speaker') and neighbor_mem.source_speaker:
                    neighbor_speaker_prefix = f"({neighbor_mem.source_speaker} said) "

                memory_str += (
                    "talk start time:" + str(neighbor_mem.timestamp)
                    + " memory content: " + neighbor_speaker_prefix + str(neighbor_mem.content)
                    + " memory context: " + str(neighbor_mem.context)
                    + " memory keywords: " + str(neighbor_mem.keywords)
                    + " memory tags: " + str(neighbor_mem.tags)
                    + "\n"
                )

                if j >= k:
                    break
                j += 1

        # ===== 认知日志输出 =====
        if self.cognitive_config.enable_cognitive_logging and activation_logs:
            print(f"[HLM-Retrieval] Category: {category}, Query: {query[:50]}...")
            print(f"[HLM-Activation] Logs: {activation_logs[:3]}...")  # 只显示前3个
            if self.cognitive_config.enable_cross_speaker_tracking:
                print(f"[HLM-CrossSpeaker] Distribution: {speaker_counts}")

        return memory_str

    def _calculate_activation_score(self, mem, rank: int, total: int) -> float:
        """计算 ACT-R 激活值（仅用于日志记录和可解释性分析）

        激活值公式: A = α·sim + β·freq + δ·importance + ε·level

        Args:
            mem: 记忆对象
            rank: 相似度排名
            total: 候选总数

        Returns:
            activation: 激活值分数
        """
        cfg = self.cognitive_config

        # 相似度分数（排名越靠前分数越高）
        sim_score = math.exp(-0.1 * rank)

        # 频率分数
        retrieval_count = getattr(mem, 'retrieval_count', 0)
        freq_score = math.log(1 + retrieval_count) / math.log(10)
        freq_score = min(1.0, freq_score)

        # 重要性分数
        importance = getattr(mem, 'importance_score', 0.5)

        # 记忆层级分数
        level = getattr(mem, 'memory_level', 'MTM')
        level_score = {'LTM': 1.0, 'MTM': 0.5, 'STM': 0.2}.get(level, 0.5)

        # 综合激活值
        activation = (
            cfg.activation_alpha * sim_score +
            cfg.activation_beta * freq_score +
            cfg.activation_delta * importance +
            cfg.activation_epsilon * level_score
        )

        return activation



def run_tests():
    """Run system tests"""
    print("Starting Memory System Tests...")

    # Initialize memory system with OpenAI backend
    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',
        llm_backend='openai',
        llm_model='gpt-4o-mini'
    )

    print("\nAdding test memories...")

    # Add test memories - only content is required
    memory_ids = []
    memory_ids.append(memory_system.add_note(
        "Neural networks are composed of layers of neurons that process information."
    ))

    memory_ids.append(memory_system.add_note(
        "Data preprocessing involves cleaning and transforming raw data for model training."
    ))

    print("\nQuerying for related memories...")
    query = MemoryNote(
        content="How do neural networks process data?",
        llm_controller=memory_system.llm_controller
    )

    related = memory_system.find_related_memories(query.content, k=2)
    print("related", related)
    print("\nResults:")
    for i, memory in enumerate(related, 1):
        print(f"\n{i}. Memory:")
        print(f"Content: {memory.content}")
        print(f"Category: {memory.category}")
        print(f"Keywords: {memory.keywords}")
        print(f"Tags: {memory.tags}")
        print(f"Context: {memory.context}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
