# Portable API Environment Eval

这是从原项目中抽出的独立评测包，只保留：

- ALFWorld 文本环境的 OOD `valid_unseen` 测试；
- ScienceWorld L1 的 1,684 个 test 变体；
- OpenAI-compatible `/v1/chat/completions` 客户端；
- 50 步上限、10 步历史、断点续跑和 JSONL 结果；
- API、环境和小规模 smoke 检查。

它不依赖 VERL、Ray、Torch、Transformers，也不会读取仓库根目录的
`api.txt`。建议在 Linux 服务器上使用 Python 3.10。

## 1. 创建环境

Ubuntu/Debian 先安装 Python 和 Java。ScienceWorld 官方要求 Java 8+，
这里推荐无界面的 OpenJDK 17：

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv openjdk-17-jre-headless
```

进入本目录后，可直接运行：

```bash
cd portable_eval
PYTHON_BIN=python3.10 bash scripts/bootstrap.sh
source .venv/bin/activate
```

也可以手动创建：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

锁定的环境版本是 `alfworld==0.4.2`、`scienceworld==1.2.3` 和
`PyYAML==6.0.2`。上游安装说明见
[ALFWorld](https://github.com/alfworld/alfworld) 与
[ScienceWorld](https://github.com/allenai/ScienceWorld)。

## 2. 下载测试环境

### ALFWorld OOD

ALFWorld 需要单独下载游戏数据。建议放在大容量数据盘：

```bash
export ALFWORLD_DATA=/data/alfworld
bash scripts/download_alfworld.sh "$ALFWORLD_DATA"
```

脚本会调用官方 `alfworld-download`，并检查
`$ALFWORLD_DATA/json_2.1.1/valid_unseen`。本评测固定使用这个
`valid_unseen` OOD split，完整规模通常是 134 个游戏。

### ScienceWorld L1

`scienceworld==1.2.3` 的 Python wheel 已包含运行所需 JAR，不需要再下载
数据集。确认 Java 可用即可：

```bash
java -version
python -c "from scienceworld import ScienceWorldEnv; print('ScienceWorld import OK')"
```

本包的 L1 test 集合与原仓库
`variations_idx/L1_idx.json["test"]` 完全相同，共 1,684 个唯一
`(task_id, variation_id)`。这里只把每个 task 的连续测试区间紧凑编码，
并改为稳定顺序，便于跨服务器断点续跑。

## 3. 需要的 OpenAI 协议

模型服务至少要实现：

```text
POST {OPENAI_BASE_URL}/chat/completions
Authorization: Bearer {OPENAI_API_KEY}
Content-Type: application/json
```

请求体：

```json
{
  "model": "your-served-model-name",
  "messages": [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "environment state"}
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 512
}
```

响应至少包含：

```json
{
  "choices": [
    {
      "message": {
        "content": "<action>look</action>"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 10,
    "total_tokens": 110
  }
}
```

`GET {OPENAI_BASE_URL}/models` 是可选接口：`preflight` 会尝试访问，但即使
该接口不存在，只要真实 chat completion 成功，API 检查仍会通过。

可先用 curl 验证你的服务器：

```bash
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$OPENAI_MODEL"'",
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

例如用 vLLM 暴露模型时，服务名必须与 `OPENAI_MODEL` 一致：

```bash
vllm serve /models/your-model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name your-served-model-name \
  --api-key your-token
```

评测机配置：

```bash
export OPENAI_BASE_URL=http://MODEL_SERVER_IP:8000/v1
export OPENAI_API_KEY=your-token
export OPENAI_MODEL=your-served-model-name
```

若模型需要关闭 thinking，可把服务端扩展字段传到请求顶层：

```bash
export EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
agent-eval preflight --extra-body-json "$EXTRA_BODY"
```

密钥只放环境变量，不要写进配置、日志或 Git。

## 4. 运行前检查

先发送一次真实 completion，确认网络、密钥、model name 和响应结构：

```bash
agent-eval preflight
```

再各跑两个 episode：

```bash
bash scripts/smoke.sh
```

如果只想单独检查：

```bash
agent-eval alfworld \
  --output results/smoke_alfworld \
  --limit 2 \
  --max-steps 50 \
  --history-steps 10

agent-eval sciworld \
  --output results/smoke_sciworld \
  --limit 2 \
  --max-steps 50 \
  --history-steps 10
```

## 5. 正式评测

ALFWorld 完整 OOD：

```bash
agent-eval alfworld \
  --output results/alfworld_ood_full \
  --max-steps 50 \
  --history-steps 10 \
  --temperature 0
```

ScienceWorld 完整 L1：

```bash
agent-eval sciworld \
  --output results/sciworld_l1_full \
  --max-steps 50 \
  --history-steps 10 \
  --temperature 0
```

复用同一个输出目录时加 `--resume`，已写入 `episodes.jsonl` 的 episode 会被
跳过：

```bash
agent-eval alfworld \
  --output results/alfworld_ood_full \
  --max-steps 50 \
  --history-steps 10 \
  --resume
```

可用 `--start-index` 和 `--limit` 做固定切片，例如只跑 ScienceWorld 的
第 128 到 255 个样本：

```bash
agent-eval sciworld \
  --output results/sciworld_l1_128_255 \
  --start-index 128 \
  --limit 128
```

默认只保存 episode 指标，减少磁盘占用。调试模型行为时加
`--save-transcripts` 保存每一步 observation、原始模型输出和实际动作。

## 6. 结果和指标

每个输出目录包含：

```text
episodes.jsonl   # 每个 episode 完成后立即追加，可用于恢复
summary.json     # 每次追加后原子重算的汇总
```

主要字段：

- `success_rate`：严格任务成功率；
- `average_score`：环境最终分数均值；
- `average_steps`：平均实际交互步数；
- `invalid_actions`：缺失或重复 `<action>` 标签的次数；
- `api_total_tokens`：服务返回的 token 总数。

ALFWorld 使用环境的 `won` 标记。ScienceWorld 的 `done` 也会在步数耗尽或
负分时触发，因此不能把 `done && score > 0` 当成功；本包只在最终
`score >= 100` 时计为成功，同时保留原始平均分。这可以避免 L1 成功率被
部分得分错误抬高。

模型每一步必须输出且只输出一组完整动作标签，例如：

```text
<think>I should inspect the room first.</think>
<action>look</action>
```

缺失或多组 `<action>` 不再截取文本尾部作为伪动作，而会记录为
`invalid_actions` 并向环境发送明确的无效命令。

## 7. 本地回归

不启动环境、不请求外部 API 的单元测试：

```bash
python -m unittest discover -s tests -v
python -m compileall -q agent_eval
```
