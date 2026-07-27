from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .alfworld_eval import run_alfworld_ood
from .api import APIError, APISettings, OpenAICompatibleClient
from .sciworld_eval import run_sciworld_l1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Portable OpenAI-compatible ALFWorld OOD and ScienceWorld L1 evaluator.",
    )
    api_parent = argparse.ArgumentParser(add_help=False)
    api_parent.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL, including /v1 (or OPENAI_BASE_URL).",
    )
    api_parent.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key (or OPENAI_API_KEY).",
    )
    api_parent.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL"),
        help="Served model name (or OPENAI_MODEL).",
    )
    api_parent.add_argument("--timeout", type=float, default=120)
    api_parent.add_argument("--retries", type=int, default=3)
    api_parent.add_argument("--temperature", type=float, default=0)
    api_parent.add_argument("--top-p", type=float, default=1)
    api_parent.add_argument("--max-tokens", type=int, default=512)
    api_parent.add_argument(
        "--extra-body-json",
        default="{}",
        help='Extra top-level JSON fields, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'.',
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "preflight",
        parents=[api_parent],
        help="Verify model listing (when supported) and one chat completion.",
    )

    for name, help_text in (
        ("alfworld", "Run the official valid_unseen ALFWorld OOD split."),
        ("sciworld", "Run the 1,684-instance ScienceWorld L1 test split."),
    ):
        subparser = subparsers.add_parser(name, parents=[api_parent], help=help_text)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--start-index", type=int, default=0)
        subparser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Number of episodes; omit for the remaining full split.",
        )
        subparser.add_argument("--max-steps", type=int, default=50)
        subparser.add_argument("--history-steps", type=int, default=10)
        subparser.add_argument("--resume", action="store_true")
        subparser.add_argument("--save-transcripts", action="store_true")

    alfworld = subparsers.choices["alfworld"]
    alfworld.add_argument("--seed", type=int, default=42)

    sciworld = subparsers.choices["sciworld"]
    sciworld.add_argument(
        "--simplifications",
        default="",
        help="ScienceWorld simplification string. Empty is the benchmark default.",
    )
    sciworld.add_argument(
        "--jar-path",
        default=None,
        help="Optional custom ScienceWorld JAR path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = OpenAICompatibleClient(_api_settings(args))

    if args.command == "preflight":
        return _preflight(client)
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.history_steps < 0:
        raise SystemExit("--history-steps must be non-negative")

    if args.command == "alfworld":
        summary = run_alfworld_ood(
            client=client,
            output_dir=args.output,
            start_index=args.start_index,
            limit=args.limit,
            max_steps=args.max_steps,
            history_steps=args.history_steps,
            seed=args.seed,
            resume=args.resume,
            save_transcripts=args.save_transcripts,
        )
    else:
        summary = run_sciworld_l1(
            client=client,
            output_dir=args.output,
            start_index=args.start_index,
            limit=args.limit,
            max_steps=args.max_steps,
            history_steps=args.history_steps,
            resume=args.resume,
            save_transcripts=args.save_transcripts,
            simplifications=args.simplifications,
            jar_path=args.jar_path,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _api_settings(args: argparse.Namespace) -> APISettings:
    try:
        extra_body: dict[str, Any] = json.loads(args.extra_body_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra-body-json is invalid JSON: {exc}") from exc
    if not isinstance(extra_body, dict):
        raise SystemExit("--extra-body-json must decode to a JSON object")
    if not args.base_url:
        raise SystemExit("Set --base-url or OPENAI_BASE_URL")
    if not args.model:
        raise SystemExit("Set --model or OPENAI_MODEL")
    return APISettings(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_seconds=args.timeout,
        retries=args.retries,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
    )


def _preflight(client: OpenAICompatibleClient) -> int:
    models: list[str] | None = None
    model_list_error: str | None = None
    try:
        models = client.list_models()
    except APIError as exc:
        model_list_error = str(exc)

    result = client.chat(
        [
            {
                "role": "user",
                "content": "Reply with exactly the word OK.",
            }
        ]
    )
    print(
        json.dumps(
            {
                "chat_completions": "ok",
                "model": client.settings.model,
                "models_endpoint": models,
                "models_endpoint_warning": model_list_error,
                "latency_seconds": round(result.latency_seconds, 3),
                "response_preview": result.content[:120],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
