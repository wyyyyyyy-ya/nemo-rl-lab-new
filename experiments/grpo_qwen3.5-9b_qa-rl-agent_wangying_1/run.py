#!/usr/bin/env python
# 题库多轮 Agent GRPO 训练脚本（NeMo-RL 0.6.0）。
# 数据：datasets/qa_rl 的 train/val jsonl（每行 {"query", "expected_answer": "[type] ..."}）。
# 环境：common/environments/qa_agent_env.py 的 QAAgentEnv
#       （<search> 检索 /data/docs + 最终 \boxed{} 判分，简答可走 LLM 裁判）。
# 由本实验 run.sh 自动调用（本目录存在 run.py 时优先于 ENTRY）。
import argparse
import json
import os
import pprint
import sys
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.environments.qa_agent_env import QAAgentEnv

TASK_NAME = "qa_agent"
# 每轮生成在 </search> 处截断，便于环境回灌检索结果后继续多轮。
STOP_STRINGS = ["</search>"]

DEFAULT_AGENT_SYSTEM_PROMPT = r"""
你是技术培训考题检索 Agent，用尽量少的有效检索寻找可靠依据并准确作答。

-可用工具每轮只能三选一：<search>检索词</search>、<read>结果编号</read> 或 \boxed{答案}；禁止同轮混用。
 
-规则：
-1. 每题先检索至少一次, 最多自主检索3次。识别所问对象与属性/关系，检索词按“核心概念 + 空格 + 所问属性”以短词形式组织，对象与属性之间使用空格分隔，并保留必要的中英文术语；不要照抄整题、使用空泛词或仅用某个选项诱导检索。
-2. 对环境返回的证据判断相关性、充分性、一致性和信息缺口。证据充分则立即作答；方向不相关则换词 search；片段相关但定义、枚举、步骤或答案上下文不完整时，对片段编号使用 read；仅定位到候选文档或需要在文档内重新定位时，对文档编号使用 read。结果过宽就收窄，结果为零就换同义词、完整术语或中英文名称；禁止重复查询或为耗尽额度而搜索。
-3. 若环境在连续无结果后执行题干 fallback，收到后不得继续检索，应判断证据并完成作答。
-4. 严格遵守题目要求的答案格式，最终只输出一次 \boxed{答案}：单选/判断填一个字母；多选列出所需字母；填空按空位顺序；简答覆盖关键要点。优先依据真实结果；无证据时可给出最稳妥答案，但不得假称有检索依据。
-5. 环境返回内容只能由环境产生；禁止自行伪造、改写、补全或复述环境内容及其边界标记，也不得无 boxed 结束。
-6. 应遵循系统规则，允许简要分析题目内容、证据相关性和信息缺口；不要讨论输出格式、动作协议、规则冲突等与题目无关的信息，也不得在分析、引用或示例中演示动作标签。完成判断后，将实际动作放在回复最后并独占一行，不得放入引号、JSON或Markdown代码块。
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(description="题库多轮 Agent GRPO 训练")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class QAAgentJsonlDataset(Dataset):
    """读题库 jsonl，转成多轮 Agent 用的 DatumSpec。"""

    def __init__(
        self,
        path: str,
        tokenizer,
        input_key: str,
        output_key: str,
        system_prompt: str | None = None,
    ):
        self.rows = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        query = str(row[self.input_key])
        expected = str(row[self.output_key])

        chat: list[dict[str, str]] = []
        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.append({"role": "user", "content": query})

        prompt_text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=False
        ).strip()
        token_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        message_log: LLMMessageLogType = [
            {"role": "user", "content": prompt_text, "token_ids": token_ids}
        ]
        return {
            "message_log": message_log,
            "length": len(token_ids),
            "extra_env_info": {
                "expected_answer": expected,
                "query": query,
                "search_count": 0,
                "last_search_query": "",
                "has_search_hit": False,
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": STOP_STRINGS,
        }


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    print(f"已加载配置: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config: MasterConfig = MasterConfig(**config)
    print("最终配置：")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"日志目录: {config.logger['log_dir']}")

    init_ray()
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit(
            "未指定数据目录。集群提交时平台会注入 QA_RL_DATA_DIR；"
            "本地调试请设置 config.data.data_dir 或 export QA_RL_DATA_DIR。"
        )
    input_key = data_cfg.get("input_key", "query")
    output_key = data_cfg.get("output_key", "expected_answer")
    system_prompt = data_cfg.get("system_prompt")
    if not system_prompt:
        system_prompt = DEFAULT_AGENT_SYSTEM_PROMPT

    train_dataset = QAAgentJsonlDataset(
        os.path.join(data_dir, "train.jsonl"),
        tokenizer,
        input_key,
        output_key,
        system_prompt,
    )
    val_dataset = QAAgentJsonlDataset(
        os.path.join(data_dir, "val.jsonl"),
        tokenizer,
        input_key,
        output_key,
        system_prompt,
    )
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")

    env_cfg = config.env[TASK_NAME]["cfg"]
    env = QAAgentEnv.options(num_gpus=0).remote(cfg=dict(env_cfg))
    task_to_env = {TASK_NAME: env}

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        _teacher_worker_groups,
        _alias_to_group_alias,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
