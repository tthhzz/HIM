import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.him_memory import (
    AgenticMemorySystem,
    CognitiveMemoryConfig,
    DEFAULT_COGNITIVE_CONFIG,
    LLMController,
    is_llama_model,
)

import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
from src.dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from src.metrics import calculate_metrics, aggregate_metrics
from datetime import datetime

# ===== 可复现性：固定随机种子 =====
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===== 按类别设置的 k 值 (Table 8) =====
# Qwen2.5-3B: MultiHop 10 / Temporal 10 / OpenDomain 50 / SingleHop 10 / Adversarial 10
CATEGORY_K_VALUES = {
    1: 10,   # Multi-hop
    2: 10,   # Temporal
    3: 10,   # Open-domain
    4: 10,   # Single-hop
    5: 10,   # Adversarial
}

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

class advancedMemAgent:
    """高级记忆代理

    创新点支持:
    - 策略3：源监控 - 传递 source_speaker
    - 策略2：ACT-R 激活 - 传递 category 给检索
    - 策略6：注意力门控 - 通过 cognitive_config
    - 轻量巩固 - 自动触发
    """
    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000,
                 cognitive_config: CognitiveMemoryConfig = None):
        # 使用认知配置
        self.cognitive_config = cognitive_config or DEFAULT_COGNITIVE_CONFIG

        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            cognitive_config=self.cognitive_config  # 传递配置
        )
        self.retriever_llm = LLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None, source_speaker=None):
        """===== 策略3：源监控 - 接受 source_speaker ====="""
        self.memory_system.add_note(content, time=time, source_speaker=source_speaker)

    def retrieve_memory(self, content, k=10, category=None):
        """===== 策略2：ACT-R 激活 - 传递 category ====="""
        return self.memory_system.find_related_memories_raw(content, k=k, category=category)

    def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text.
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""

            # Get LLM response
        model_name = getattr(self.retriever_llm, "model", None)
        is_llama = is_llama_model(model_name)
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "relevant_parts": {
                                        "type": "string",
                                    }
                                },
                                "required": ["relevant_parts"],
                                "additionalProperties": False
                            },
                            "strict": not is_llama
                        }}, temperature=0.2)
        # print("response:{}".format(response))
        return response

    def generate_query_llm(self, question):
        """生成检索关键词

        HLM 创新点: 使用低温度(0.2)确保关键词生成的稳定性和可复现性
        """
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text.

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""

            # Get LLM response - 使用 temperature=0.2 确保可复现性
        model_name = getattr(self.retriever_llm, "model", None)
        is_llama = is_llama_model(model_name)
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": not is_llama
                        }}, temperature=0.2)  # HLM: 关键词生成使用低温度
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    def answer_question(self, question: str, category: int, answer: str) -> str:
        """Generate answer for a question given the conversation context.

        创新点:
        - 策略2：ACT-R 激活 - 传递 category 给检索方法
        - 按类别设置 k 值 (Table 8)
        """
        keywords = self.generate_query_llm(question)
        # ===== 按类别设置 k 值 =====
        k = CATEGORY_K_VALUES.get(category, self.retrieve_k)
        # ===== 策略2：传递 category 用于决定是否启用时间增强 =====
        raw_context = self.retrieve_memory(keywords, k=k, category=category)
        context = raw_context
        assert category in [1,2,3,4,5]
        user_prompt = f"""Context:
                {context}

                Question: {question}

                Answer the question based only on the information provided in the context above."""
        # ===== 恢复原始温度设置: 类别 1/2/3/4 用 0.7，类别 5 用 0.5 =====
        temperature = 0.7
        if category == 5: # adversial question, follow the initial paper.
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. {question}

                            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                            """
            temperature = self.temperature_c5  # Category 5 用 0.5
        elif category == 2:
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

                            Question: {question} Short answer:
                            """
        elif category == 3:
            user_prompt = f"""
                            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        model_name = getattr(self.memory_system.llm_controller, "model", None)
        is_llama = is_llama_model(model_name)
        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": not is_llama
                    }},temperature=temperature
        )
        # print(response)
        return response,user_prompt,raw_context

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     # ===== 创新点配置参数 =====
                     enable_source_monitoring: bool = True,
                     enable_activation_retrieval: bool = True,
                     enable_attention_gating: bool = True,
                     enable_consolidation: bool = True):
    """Evaluate the agent on the LoComo dataset.

    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        output_path: Path to save results
        ratio: Ratio of dataset to evaluate
        enable_source_monitoring: 策略3 - 源监控开关
        enable_activation_retrieval: 策略2 - ACT-R 激活检索开关
        enable_attention_gating: 策略6 - 注意力门控开关
        enable_consolidation: 轻量巩固开关
    """
    # ===== 创建认知配置 =====
    cognitive_config = CognitiveMemoryConfig(
        enable_source_monitoring=enable_source_monitoring,
        enable_activation_logging=enable_activation_retrieval,
        enable_attention_gating=enable_attention_gating,
        enable_consolidation=enable_consolidation
    )

    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    safe_model = model.replace(":", "_").replace("/", "_")

    # 在日志文件名中标记启用的创新点
    innov_flags = []
    if enable_source_monitoring:
        innov_flags.append("src")
    if enable_activation_retrieval:
        innov_flags.append("act")
    if enable_attention_gating:
        innov_flags.append("att")
    if enable_consolidation:
        innov_flags.append("con")
    innov_str = "_".join(innov_flags) if innov_flags else "baseline"

    log_filename = f"eval_innov_{safe_model}_{backend}_ratio{ratio}_{innov_str}_{timestamp}.log"
    log_path = str(REPO_ROOT / "runtime" / "logs" / log_filename)

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")

    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")

    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    # Store results
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    # Evaluate each sample
    i = 0
    error_num = 0
    # 缓存目录包含创新点标记
    memories_dir = str(REPO_ROOT / "runtime" / f"cached_memories_him_{backend}_{safe_model}_{innov_str}")
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1,2,3,4,5]

    # 日志记录创新点配置
    logger.info(f"\n===== 创新点配置 =====")
    logger.info(f"策略3 源监控 (Source Monitoring): {enable_source_monitoring}")
    logger.info(f"策略2 ACT-R 激活检索: {enable_activation_retrieval}")
    logger.info(f"策略6 注意力门控: {enable_attention_gating}")
    logger.info(f"轻量巩固: {enable_consolidation}")
    logger.info(f"========================\n")

    for sample_idx, sample in enumerate(samples):
        # 传递 cognitive_config 给 agent
        agent = advancedMemAgent(
            model, backend, retrieve_k, temperature_c5,
            sglang_host, sglang_port,
            cognitive_config=cognitive_config
        )
        # Create memory cache filename based on sample and session indices
        memory_cache_file = os.path.join(
            memories_dir,
            f"memory_cache_sample_{sample_idx}.pkl"
        )
        retriever_cache_file = os.path.join(
            memories_dir,
            f"retriever_cache_sample_{sample_idx}.pkl"
        )
        retriever_cache_embeddings_file = os.path.join(
            memories_dir,
            f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        # Check if cached memories exist
        if os.path.exists(memory_cache_file):
            logger.info(f"Loading cached memories for sample {sample_idx}")
            # try:
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            # Restore memories to agent
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                print(f"Found retriever cache files:")
                print(f"  - Retriever cache: {retriever_cache_file}")
                print(f"  - Embeddings cache: {retriever_cache_embeddings_file}")
                agent.memory_system.retriever = agent.memory_system.retriever.load(retriever_cache_file,retriever_cache_embeddings_file)
            else:
                print(f"No retriever cache found at {retriever_cache_file}, loading from memory")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(cached_memories, 'all-MiniLM-L6-v2')
            print(agent.memory_system.retriever.corpus)
            logger.info(f"Successfully loaded {len(cached_memories)} memories")
            # except Exception as e:
            #     logger.info(f"Error loading cached memories: {e}. Will recreate memories.")
            #     cached_memories = None
        else:
            logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")
            cached_memories = None

            for session_idx, (_, turns) in enumerate(sample.conversation.sessions.items()):
                logger.info(f"Sample {sample_idx}: building memories for session {session_idx} with {len(turns.turns)} turns")
                for turn_idx, turn in enumerate(turns.turns):
                    if turn_idx % 20 == 0:
                        logger.info(f"Sample {sample_idx}: session {session_idx} turn {turn_idx+1}/{len(turns.turns)}")
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker "+ turn.speaker + "says : " + turn.text
                    # ===== 策略3：源监控 - 传递 source_speaker =====
                    agent.add_memory(conversation_tmp, time=turn_datatime, source_speaker=turn.speaker)
                logger.info(f"Sample {sample_idx}: session {session_idx} done, total memories so far {len(agent.memory_system.memories)}")
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file,retriever_cache_embeddings_file)
            logger.info(f"\nSuccessfully cached {len(memories_to_cache)} memories")

        logger.info(f"\nProcessing sample {sample_idx + 1}/{len(samples)}")

        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1

                # Generate prediction
                prediction, user_prompt,raw_context = agent.answer_question(qa.question,qa.category,qa.final_answer)
                try:
                    prediction = json.loads(prediction)["answer"]
                except:
                    prediction = prediction
                    logger.info(f"Failed to parse prediction as JSON: {prediction}")
                    error_num += 1
                # Log results
                logger.info(f"\nQuestion {total_questions}: {qa.question}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Reference: {qa.final_answer}")
                logger.info(f"User Prompt: {user_prompt}")
                logger.info(f"Category: {qa.category}")
                logger.info(f"Raw Context: {raw_context}")

                # Calculate metrics
                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }

                all_metrics.append(metrics)
                all_categories.append(qa.category)

                # Store individual result
                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics
                }
                results.append(result)

                # Log progress
                if total_questions % 10 == 0:
                    logger.info(f"Processed {total_questions} questions")

    # Calculate aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    # Prepare final results
    final_results = {
        "model": model,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results
    }
    logger.info(f"Error number: {error_num}")
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    # Log summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value:.4f}")

    return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset with cognitive innovations")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                      help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                      help="OpenAI model to use")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                      help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="sglang",
                      help="Backend to use (openai, ollama, or sglang)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                      help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=10,
                      help="Retrieve k")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                      help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                      help="SGLang server port (for sglang backend)")

    # ===== 创新点开关参数 =====
    parser.add_argument("--enable_source_monitoring", type=lambda x: x.lower() == 'true',
                      default=True, help="策略3: 源监控 (default: True)")
    parser.add_argument("--enable_activation_retrieval", type=lambda x: x.lower() == 'true',
                      default=True, help="策略2: ACT-R 激活检索 (default: True)")
    parser.add_argument("--enable_attention_gating", type=lambda x: x.lower() == 'true',
                      default=True, help="策略6: 注意力门控 (default: True)")
    parser.add_argument("--enable_consolidation", type=lambda x: x.lower() == 'true',
                      default=True, help="轻量巩固 (default: True)")

    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    # Convert relative path to absolute path
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    evaluate_dataset(
        str(dataset_path), args.model, str(output_path) if output_path else None, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port,
        # 创新点开关
        enable_source_monitoring=args.enable_source_monitoring,
        enable_activation_retrieval=args.enable_activation_retrieval,
        enable_attention_gating=args.enable_attention_gating,
        enable_consolidation=args.enable_consolidation
    )

if __name__ == "__main__":
    main()
