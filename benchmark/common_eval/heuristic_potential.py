"""Heuristic progress-potential baseline for text-environment trajectories."""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "cause",
    "complete",
    "completely",
    "done",
    "first",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "part",
    "some",
    "state",
    "task",
    "that",
    "the",
    "then",
    "to",
    "up",
    "use",
    "when",
    "will",
    "with",
    "you",
    "your",
}

_FAILURE_FRAGMENTS = (
    "no known action matches",
    "ambiguous request",
    "nothing happens",
    "can't",
    "cannot",
    "not possible",
    "not valid",
)

_RESET_FRAGMENTS = (
    "reset task",
    "reset the goal progress",
)

_POSITIVE_ACTION_PREFIXES = (
    "activate",
    "clean",
    "close",
    "cool",
    "dunk",
    "focus on",
    "go to",
    "heat",
    "look",
    "mix",
    "move",
    "open",
    "pick up",
    "put",
    "take",
    "turn on",
    "use",
    "wait",
)

_TRANSFORM_WORDS = {
    "clean": ("clean", "washed", "rinsed"),
    "cool": ("cool", "cooled", "cold", "freezer", "fridge", "freeze", "frozen"),
    "heat": ("heat", "heated", "hot", "microwave", "stove", "oven", "burner"),
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _phrase_tokens(phrase: Any) -> List[str]:
    text = _normalize_text(phrase)
    words = [word for word in text.split() if word and word not in {"a", "an", "the", "some"}]
    return [_singularize(word) for word in words if word]


def _canonical_phrase(phrase: Any) -> str:
    return " ".join(_phrase_tokens(phrase))


def _phrase_in_text(phrase: Any, text: Any) -> bool:
    tokens = _phrase_tokens(phrase)
    if not tokens:
        return False
    haystack = _normalize_text(text)
    if not haystack:
        return False
    haystack_tokens = set(_singularize(word) for word in haystack.split())
    canonical = " ".join(tokens)
    if canonical and canonical in " ".join(_singularize(word) for word in haystack.split()):
        return True
    return all(token in haystack_tokens for token in tokens)


def _any_phrase_in_text(phrases: Sequence[str], text: Any) -> bool:
    return any(_phrase_in_text(phrase, text) for phrase in phrases if phrase)


def _keyword_set(text: Any) -> List[str]:
    words = []
    for word in _normalize_text(text).split():
        word = _singularize(word)
        if len(word) <= 2 or word in _STOPWORDS:
            continue
        if word not in words:
            words.append(word)
    return words


def _step_value(step: Any, key: str) -> str:
    if isinstance(step, dict):
        return str(step.get(key) or "")
    return str(getattr(step, key, "") or "")


def _clean_env_output(text: str) -> str:
    cleaned_lines: List[str] = []
    for line in str(text or "").splitlines():
        if re.search(r"\byour task is to\s*:", line, flags=re.IGNORECASE):
            continue
        if "Welcome to TextWorld" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _as_steps(steps: Optional[Sequence[Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for step in steps or []:
        rows.append(
            {
                "obs_before": _clean_env_output(_step_value(step, "obs_before")),
                "action": _step_value(step, "action"),
                "obs_after": _clean_env_output(_step_value(step, "obs_after")),
            }
        )
    return rows


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class TaskHints:
    """Lightweight parsed task intent used by the heuristic solver."""

    task_kind: str
    targets: List[str] = field(default_factory=list)
    places: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    required_transforms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_task(cls, task_description: str, env: str = "") -> "TaskHints":
        normalized = _normalize_text(task_description)
        required_transforms: List[str] = []
        if re.search(r"\b(clean|cleaned|wash|washed|rinse|rinsed)\b", normalized):
            required_transforms.append("clean")
        if re.search(r"\b(cool|cooled|cold|freeze|frozen)\b", normalized):
            required_transforms.append("cool")
        if re.search(r"\b(heat|heated|hot|warm|warmed)\b", normalized):
            required_transforms.append("heat")

        look_match = re.search(
            r"(?:look at|examine)\s+(?:a|an|the|some)?\s*(.+?)\s+(?:under|with|using|in)\s+(?:a|an|the)?\s*(.+?)(?:$|\.|,)",
            task_description,
            flags=re.IGNORECASE,
        )
        if look_match:
            return cls(
                task_kind="look_at",
                targets=[_canonical_phrase(look_match.group(1))],
                tools=[_canonical_phrase(look_match.group(2))],
                required_transforms=required_transforms,
                keywords=_keyword_set(task_description),
            )

        put_match = re.search(
            r"(?:put|place)\s+(?:a|an|the|some|two|all)?\s*(.+?)\s+(?:on|in|into|inside|onto)\s+(?:a|an|the)?\s*(.+?)(?:$|\.|,)",
            task_description,
            flags=re.IGNORECASE,
        )
        if put_match:
            return cls(
                task_kind="put",
                targets=[_canonical_phrase(put_match.group(1))],
                places=[_canonical_phrase(put_match.group(2))],
                required_transforms=required_transforms,
                keywords=_keyword_set(task_description),
            )

        state_match = re.search(
            r"state of matter of\s+(?:a|an|the|some)?\s*(.+?)(?:$|\.|,)",
            task_description,
            flags=re.IGNORECASE,
        )
        if state_match:
            return cls(
                task_kind="sciworld_state_change",
                targets=[_canonical_phrase(state_match.group(1)), "substance"],
                places=[],
                required_transforms=required_transforms,
                keywords=_keyword_set(task_description),
            )

        create_match = re.search(
            r"(?:create|make)\s+(?:a|an|the|some)?\s*(.+?)(?:$|\.|,)",
            task_description,
            flags=re.IGNORECASE,
        )
        if create_match:
            target = _canonical_phrase(create_match.group(1))
            return cls(
                task_kind="sciworld_create",
                targets=[target],
                required_transforms=required_transforms,
                keywords=_keyword_set(task_description),
            )

        focus_matches = re.findall(
            r"focus on\s+(?:a|an|the|some)?\s*([^.,]+)",
            task_description,
            flags=re.IGNORECASE,
        )
        targets = [_canonical_phrase(match) for match in focus_matches if _canonical_phrase(match)]
        return cls(
            task_kind="generic",
            targets=targets[:2],
            required_transforms=required_transforms,
            keywords=_keyword_set(task_description),
        )


@dataclass
class TrajectoryFacts:
    steps: List[Dict[str, str]]
    history_text: str
    final_text: str
    final_action: str
    final_result: str
    current_location: str
    inventory_text: str
    inventory_objects: List[str]
    failure_count: int
    reset_seen: bool
    completion_seen: bool
    useful_action_count: int

    @classmethod
    def from_steps(cls, steps: Sequence[Any]) -> "TrajectoryFacts":
        rows = _as_steps(steps)
        history_parts: List[str] = []
        final_action = ""
        final_result = ""
        inventory_text_parts: List[str] = []
        inventory_objects: List[str] = []
        failure_count = 0
        reset_seen = False
        completion_seen = False
        useful_action_count = 0
        current_location = ""

        for row in rows:
            action = row["action"]
            result = row["obs_after"]
            before = row["obs_before"]
            joined = "\n".join([before, action, result])
            history_parts.append(joined)
            final_action = action
            final_result = result

            normalized_joined = _normalize_text(joined)
            normalized_action = _normalize_text(action)
            normalized_result = _normalize_text(result)
            if any(fragment in normalized_joined for fragment in _FAILURE_FRAGMENTS):
                failure_count += 1
            if any(fragment in normalized_joined for fragment in _RESET_FRAGMENTS):
                reset_seen = True
            if any(fragment in normalized_joined for fragment in ("task complete", "task is complete", "you won")):
                completion_seen = True
            if normalized_action.startswith(_POSITIVE_ACTION_PREFIXES):
                useful_action_count += 1

            for obj in cls._extract_inventory_additions(action, result):
                if obj and obj not in inventory_objects:
                    inventory_objects.append(obj)

            if "in your inventory" in normalized_result:
                inventory_text_parts.append(result)

            location = cls._extract_location(result) or cls._extract_location(before)
            if location:
                current_location = location

        history_text = "\n".join(history_parts)
        final_text = rows[-1]["obs_after"] if rows else ""
        inventory_text = "\n".join(inventory_text_parts + inventory_objects)
        return cls(
            steps=rows,
            history_text=history_text,
            final_text=final_text,
            final_action=final_action,
            final_result=final_result,
            current_location=current_location,
            inventory_text=inventory_text,
            inventory_objects=inventory_objects,
            failure_count=failure_count,
            reset_seen=reset_seen,
            completion_seen=completion_seen,
            useful_action_count=useful_action_count,
        )

    @staticmethod
    def _extract_inventory_additions(action: str, result: str) -> List[str]:
        additions: List[str] = []
        texts = [action, result]
        patterns = [
            r"(?:pick up|take)\s+(?:a|an|the)?\s*([a-z0-9_\-\s]+?)(?:\s+from|\s+on|\s+in|\s+under|\s+with|$|\.|,)",
            r"(?:move|moved)\s+(?:a|an|the)?\s*([a-z0-9_\-\s]+?)\s+to\s+(?:the\s+)?inventory",
        ]
        for text in texts:
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    obj = _canonical_phrase(match.group(1))
                    if obj and obj not in {"inventory"} and obj not in additions:
                        additions.append(obj)
        return additions

    @staticmethod
    def _extract_location(text: str) -> str:
        patterns = [
            r"you arrive at\s+(?:a|an|the)?\s*([^.\n]+)",
            r"you move to\s+(?:a|an|the)?\s*([^.\n]+)",
            r"this room is called\s+(?:a|an|the)?\s*([^.\n]+)",
            r"you are (?:at|in|on)\s+(?:a|an|the)?\s*([^.\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return _canonical_phrase(match.group(1))
        return ""


@dataclass
class PotentialResult:
    phi: float
    stage: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phi": self.phi,
            "stage": self.stage,
            "evidence": self.evidence,
        }


class HeuristicPotentialSolver:
    """A model-free potential estimator that reads only task text and env outputs."""

    def __init__(self, same_threshold: float = 0.05, milestone_threshold: float = 0.55, profile: str = "coarse"):
        self.same_threshold = float(same_threshold)
        self.milestone_threshold = float(milestone_threshold)
        self.profile = str(profile or "coarse").strip().lower()
        if self.profile not in {"coarse", "generic", "task_aware"}:
            raise ValueError("profile must be one of: coarse, generic, task_aware")

    def score_trajectory(
        self,
        task_description: str,
        steps: Optional[Sequence[Any]],
        *,
        env: str = "",
        is_completed: bool = False,
    ) -> PotentialResult:
        hints = TaskHints.from_task(task_description, env=env)
        facts = TrajectoryFacts.from_steps(steps or [])

        if is_completed or facts.completion_seen:
            return PotentialResult(1.0, "completed", {"task_kind": hints.task_kind})
        if facts.reset_seen and self.profile != "coarse":
            return PotentialResult(0.0, "reset", {"task_kind": hints.task_kind, "reset_seen": True})

        if self.profile == "coarse":
            result = self._score_generic(hints, facts)
        elif hints.task_kind == "put":
            result = self._score_put_task(hints, facts)
        elif hints.task_kind == "look_at":
            result = self._score_look_task(hints, facts)
        elif hints.task_kind == "sciworld_state_change" and self.profile == "task_aware":
            result = self._score_sciworld_state_change(hints, facts)
        elif hints.task_kind == "sciworld_create" and self.profile == "task_aware":
            result = self._score_sciworld_create(hints, facts)
        else:
            result = self._score_generic(hints, facts)

        penalty = min(0.18, 0.03 * facts.failure_count)
        if facts.failure_count and result.phi < 0.25:
            penalty = min(penalty, 0.06)
        if facts.reset_seen and self.profile == "coarse":
            penalty = max(penalty, 0.12)
        phi = _clamp(result.phi - penalty)
        if self.profile == "coarse" and not (is_completed or facts.completion_seen):
            phi = self._coarse_phi(phi)
        evidence = dict(result.evidence)
        evidence.update(
            {
                "task_kind": hints.task_kind,
                "failure_count": facts.failure_count,
                "penalty": penalty,
                "current_location": facts.current_location,
            }
        )
        return PotentialResult(phi, result.stage, evidence)

    def compare_pair(
        self,
        task_description: str,
        trajectory_a: Any,
        trajectory_b: Any,
        *,
        env: str = "",
    ) -> Tuple[str, PotentialResult, PotentialResult]:
        steps_a = getattr(trajectory_a, "steps", None)
        steps_b = getattr(trajectory_b, "steps", None)
        completed_a = bool(getattr(trajectory_a, "is_completed", False))
        completed_b = bool(getattr(trajectory_b, "is_completed", False))

        score_a = self.score_trajectory(task_description, steps_a, env=env, is_completed=completed_a)
        score_b = self.score_trajectory(task_description, steps_b, env=env, is_completed=completed_b)
        margin = 0.12 if self.profile == "coarse" else 1e-9
        if score_a.phi > score_b.phi + margin:
            return "A", score_a, score_b
        if score_b.phi > score_a.phi + margin:
            return "B", score_a, score_b
        if completed_a != completed_b:
            return ("A" if completed_a else "B"), score_a, score_b
        fail_a = int(score_a.evidence.get("failure_count", 0) or 0)
        fail_b = int(score_b.evidence.get("failure_count", 0) or 0)
        if abs(fail_a - fail_b) >= 2:
            return ("A" if fail_a < fail_b else "B"), score_a, score_b
        len_a = len(steps_a or [])
        len_b = len(steps_b or [])
        return self._stable_tie_break(task_description, len_a, len_b), score_a, score_b

    def classify_delta(
        self,
        task_description: str,
        trajectory_prefix: Sequence[Any],
        added_steps: Sequence[Any],
        *,
        env: str = "",
    ) -> Tuple[str, PotentialResult, PotentialResult, float]:
        before = self.score_trajectory(task_description, trajectory_prefix, env=env)
        after_steps = list(trajectory_prefix or []) + list(added_steps or [])
        after = self.score_trajectory(task_description, after_steps, env=env)
        delta = after.phi - before.phi
        added_facts = TrajectoryFacts.from_steps(added_steps or [])
        regression = self._detect_regression(task_description, trajectory_prefix, after_steps, added_steps, env=env)
        if added_facts.reset_seen:
            return "decrease", before, after, delta
        if regression:
            return "decrease", before, after, delta
        threshold = max(self.same_threshold, 0.12) if self.profile == "coarse" else self.same_threshold
        if delta > threshold:
            return "increase", before, after, delta
        if delta < -threshold:
            return "decrease", before, after, delta
        return "same", before, after, delta

    def localize_milestone(
        self,
        task_description: str,
        trajectory: Sequence[Any],
        milestones: Sequence[Any],
        *,
        env: str = "",
    ) -> Tuple[str, int, float, PotentialResult, List[Dict[str, Any]]]:
        hints = TaskHints.from_task(task_description, env=env)
        facts = TrajectoryFacts.from_steps(trajectory or [])
        scored: List[Dict[str, Any]] = []
        best_label = ""
        best_index = 0

        for milestone in milestones:
            milestone_id = str(getattr(milestone, "id", "") or "")
            description = str(getattr(milestone, "description", "") or "")
            milestone_index = int(getattr(milestone, "milestone_index", 0) or 0)
            confidence, reason = self._score_milestone_description(description, milestone_index, hints, facts)
            scored.append(
                {
                    "id": milestone_id,
                    "milestone_index": milestone_index,
                    "confidence": confidence,
                    "reason": reason,
                }
            )
            if milestone_index == 0 and not best_label:
                best_label = milestone_id
                best_index = milestone_index
            if milestone_index > best_index and confidence >= self.milestone_threshold:
                best_label = milestone_id
                best_index = milestone_index

        max_index = max([int(getattr(item, "milestone_index", 0) or 0) for item in milestones] or [0])
        phi = (best_index / float(max_index)) if max_index > 0 else 0.0
        overall = self.score_trajectory(task_description, trajectory, env=env)
        if phi > overall.phi:
            overall = PotentialResult(phi, "milestone_index", {"milestone_index": best_index})
        return best_label, best_index, phi, overall, scored

    def _score_put_task(self, hints: TaskHints, facts: TrajectoryFacts) -> PotentialResult:
        target = hints.targets[0] if hints.targets else ""
        place = hints.places[0] if hints.places else ""
        target_seen = _phrase_in_text(target, facts.history_text)
        target_current = _phrase_in_text(target, facts.final_text)
        target_in_inventory = self._target_currently_held(target, facts)
        place_seen = _phrase_in_text(place, facts.history_text)
        place_current = _phrase_in_text(place, facts.final_text) or _phrase_in_text(place, facts.current_location)
        target_at_place = self._target_currently_at_place(target, place, facts)
        put_done = target_at_place or bool(facts.completion_seen)
        transform_done = self._required_transform_done(hints, facts)

        score = 0.03 if facts.steps else 0.0
        stage = "started" if facts.steps else "empty"
        if target_seen:
            score, stage = max(score, 0.18), "target_seen_before"
        if target_current:
            score, stage = max(score, 0.34), "target_currently_visible"
        if place_seen:
            score, stage = max(score, 0.20), "place_seen_before"
        if place_current:
            score, stage = max(score, 0.30), "at_destination"
        if target_in_inventory:
            score, stage = max(score, 0.52), "target_in_inventory"
        if transform_done:
            score, stage = max(score, 0.66), "target_transformed"
        if target_in_inventory and place_current:
            score, stage = max(score, 0.78), "target_at_destination"
        if target_at_place:
            score, stage = max(score, 0.92), "target_currently_placed"
        if put_done:
            score, stage = 1.0, "placed"

        return PotentialResult(
            score,
            stage,
            {
                "target": target,
                "place": place,
                "target_seen": target_seen,
                "target_current": target_current,
                "target_in_inventory": target_in_inventory,
                "place_seen": place_seen,
                "place_current": place_current,
                "target_at_place": target_at_place,
                "put_done": put_done,
                "transform_done": transform_done,
            },
        )

    @staticmethod
    def _coarse_phi(phi: float) -> float:
        if phi < 0.20:
            return 0.0
        if phi < 0.45:
            return 0.30
        if phi < 0.70:
            return 0.55
        if phi < 0.90:
            return 0.75
        return 0.85

    @staticmethod
    def _stable_tie_break(task_description: str, len_a: int, len_b: int) -> str:
        key = f"{task_description}|{len_a}|{len_b}".encode("utf-8", errors="ignore")
        digest = hashlib.md5(key).digest()
        return "A" if digest[0] % 2 == 0 else "B"

    def _score_look_task(self, hints: TaskHints, facts: TrajectoryFacts) -> PotentialResult:
        target = hints.targets[0] if hints.targets else ""
        tool = hints.tools[0] if hints.tools else ""
        target_seen = _phrase_in_text(target, facts.history_text)
        tool_seen = _phrase_in_text(tool, facts.history_text)
        target_current = _phrase_in_text(target, facts.final_text)
        tool_current = _phrase_in_text(tool, facts.final_text)
        both_current = target_current and tool_current
        tool_activated = bool(
            re.search(r"(turn on|activate|switch on).{0,80}" + re.escape(tool), _normalize_text(facts.history_text))
        ) or ("desklamp" in tool and "is on" in _normalize_text(facts.history_text))
        inspected = _phrase_in_text(target, facts.final_action) and _normalize_text(facts.final_action).startswith(
            ("look", "examine", "use")
        )

        score = 0.03 if facts.steps else 0.0
        stage = "started" if facts.steps else "empty"
        if target_seen:
            score, stage = max(score, 0.20), "target_seen_before"
        if target_current:
            score, stage = max(score, 0.34), "target_currently_visible"
        if target_seen and tool_seen:
            score, stage = max(score, 0.42), "target_and_tool_seen_before"
        if target_current and tool_seen:
            score, stage = max(score, 0.52), "target_current_tool_known"
        if both_current:
            score, stage = max(score, 0.62), "target_and_tool_current"
        if tool_activated:
            score, stage = max(score, 0.72), "tool_activated"
        if inspected and (tool_seen or tool_activated):
            score, stage = max(score, 0.92), "target_inspected"

        return PotentialResult(
            score,
            stage,
            {
                "target": target,
                "tool": tool,
                "target_seen": target_seen,
                "tool_seen": tool_seen,
                "target_current": target_current,
                "tool_current": tool_current,
                "both_current": both_current,
                "tool_activated": tool_activated,
                "inspected": inspected,
            },
        )

    def _score_sciworld_state_change(self, hints: TaskHints, facts: TrajectoryFacts) -> PotentialResult:
        text = _normalize_text(facts.history_text)
        target_seen = _any_phrase_in_text(hints.targets, facts.history_text)
        focus_target = "focus on" in text and target_seen
        apparatus = any(word in text for word in ("sink", "freezer", "fridge", "stove", "oven", "burner", "thermometer"))
        moved_to_apparatus = bool(re.search(r"move .{0,60} to (freezer|fridge|stove|oven|sink)", text))
        waited = "wait" in text or "you decide to wait" in text
        transformed = any(word in text for word in ("melt", "freeze", "frozen", "boil", "steam", "solid", "liquid", "gas"))

        score = 0.03 if facts.steps else 0.0
        stage = "started" if facts.steps else "empty"
        if target_seen:
            score, stage = max(score, 0.20), "target_seen"
        if apparatus:
            score, stage = max(score, 0.34), "apparatus_found"
        if moved_to_apparatus:
            score, stage = max(score, 0.55), "substance_moved_to_apparatus"
        if focus_target:
            score, stage = max(score, 0.68), "substance_focused"
        if waited and (moved_to_apparatus or apparatus):
            score, stage = max(score, 0.80), "waited_for_state_change"
        if transformed:
            score, stage = max(score, 0.90), "state_changed"
        return PotentialResult(
            score,
            stage,
            {
                "targets": hints.targets,
                "target_seen": target_seen,
                "apparatus": apparatus,
                "moved_to_apparatus": moved_to_apparatus,
                "focus_target": focus_target,
                "waited": waited,
                "transformed": transformed,
            },
        )

    def _score_sciworld_create(self, hints: TaskHints, facts: TrajectoryFacts) -> PotentialResult:
        text = _normalize_text(facts.history_text)
        target_seen = _any_phrase_in_text(hints.targets, facts.history_text)
        color_or_product = any(keyword in text for keyword in hints.keywords if keyword not in {"create", "make"})
        mixed = any(word in text for word in ("mix", "mend", "pour", "dunk", "paint"))
        focused_product = "focus on" in text and (target_seen or color_or_product)

        score = 0.03 if facts.steps else 0.0
        stage = "started" if facts.steps else "empty"
        if color_or_product:
            score, stage = max(score, 0.25), "ingredients_or_product_seen"
        if mixed:
            score, stage = max(score, 0.50), "mixing_action"
        if target_seen:
            score, stage = max(score, 0.70), "target_product_seen"
        if focused_product:
            score, stage = max(score, 0.86), "target_product_focused"
        return PotentialResult(
            score,
            stage,
            {
                "targets": hints.targets,
                "target_seen": target_seen,
                "color_or_product": color_or_product,
                "mixed": mixed,
                "focused_product": focused_product,
            },
        )

    def _score_generic(self, hints: TaskHints, facts: TrajectoryFacts) -> PotentialResult:
        if not facts.steps:
            return PotentialResult(0.0, "empty", {"keywords": hints.keywords})
        keyword_hits = [keyword for keyword in hints.keywords if _phrase_in_text(keyword, facts.history_text)]
        target_seen = _any_phrase_in_text(hints.targets, facts.history_text)
        coverage = len(keyword_hits) / float(max(1, len(hints.keywords)))
        action_score = min(0.30, 0.04 * facts.useful_action_count)
        score = min(0.55, 0.15 + 0.35 * coverage + action_score)
        stage = "generic_progress"
        if target_seen:
            score, stage = max(score, 0.45), "target_seen"
        if facts.inventory_objects:
            score, stage = max(score, 0.58), "inventory_progress"
        return PotentialResult(
            score,
            stage,
            {
                "keywords": hints.keywords,
                "keyword_hits": keyword_hits,
                "targets": hints.targets,
                "target_seen": target_seen,
                "inventory_objects": facts.inventory_objects,
                "useful_action_count": facts.useful_action_count,
            },
        )

    def _score_milestone_description(
        self,
        description: str,
        milestone_index: int,
        hints: TaskHints,
        facts: TrajectoryFacts,
    ) -> Tuple[float, str]:
        desc_norm = _normalize_text(description)
        if milestone_index == 0:
            return 1.0, "initial_milestone"

        event_confidence, event_reason = self._score_milestone_event(description, facts)
        if event_confidence >= self.milestone_threshold:
            return event_confidence, event_reason

        if "task is complete" in desc_norm or "task has been completed" in desc_norm:
            if facts.completion_seen:
                return 0.98, "completion_signal"
            if self._put_event_seen(hints.targets[0] if hints.targets else "", hints.places[0] if hints.places else "", facts):
                return 0.90, "final_put_event"
            return 0.0, "completion_not_seen"

        found_match = re.search(r"(?:found|reached|picked up|put|placed|focused on)\s+(?:a|an|the)?\s*([a-z0-9_\-\s]+?)(?:\.|,|$)", description, flags=re.IGNORECASE)
        mentioned_obj = _canonical_phrase(found_match.group(1)) if found_match else ""

        if "found" in desc_norm and mentioned_obj and _phrase_in_text(mentioned_obj, facts.history_text):
            return 0.75, "object_found"
        if "picked up" in desc_norm and mentioned_obj and self._target_ever_held(mentioned_obj, facts):
            return 0.90, "object_in_inventory"
        if "reached" in desc_norm:
            place = hints.places[0] if hints.places else mentioned_obj
            target = hints.targets[0] if hints.targets else ""
            if self._reached_place_with_target(target, place, facts):
                return 0.82, "reached_place_with_target"
        if ("put" in desc_norm or "placed" in desc_norm) and hints.targets and hints.places:
            if self._put_event_seen(hints.targets[0], hints.places[0], facts):
                return 0.95, "put_event"
        if "focus" in desc_norm:
            focus_targets = hints.targets + ([mentioned_obj] if mentioned_obj else [])
            if "focus on" in _normalize_text(facts.history_text) and _any_phrase_in_text(focus_targets, facts.history_text):
                return 0.78, "focus_seen"

        if any(verb in desc_norm for verb in ("found", "picked up", "reached", "put", "placed", "complete")):
            return 0.0, "structured_milestone_not_matched"

        desc_keywords = [word for word in _keyword_set(description) if word not in {"agent", "trajectory", "milestone"}]
        if desc_keywords:
            hits = [word for word in desc_keywords if _phrase_in_text(word, facts.history_text)]
            ratio = len(hits) / float(len(desc_keywords))
            if ratio >= 0.65:
                return min(0.80, ratio), "keyword_overlap"
        return 0.0, "not_matched"

    def _score_milestone_event(self, description: str, facts: TrajectoryFacts) -> Tuple[float, str]:
        action_match = re.search(r"action\s+'([^']+)'", description, flags=re.IGNORECASE)
        report_match = re.search(r"reports:\s*(.+)$", description, flags=re.IGNORECASE)
        if not action_match and not report_match:
            return 0.0, "no_event_template"

        action = _normalize_text(action_match.group(1)) if action_match else ""
        report = str(report_match.group(1)).strip() if report_match else ""
        action_seen = bool(action) and any(_normalize_text(row["action"]) == action for row in facts.steps)
        report_seen = bool(report) and _phrase_in_text(report, facts.history_text)

        if action_seen and report_seen:
            return 0.98, "event_action_and_report"
        if report_seen:
            return 0.90, "event_report"
        if action_seen and self.profile == "task_aware":
            return 0.85, "exact_action"
        if action_seen and not report:
            return 0.75, "event_action"
        return 0.0, "event_not_matched"

    def _target_in_inventory(self, target: str, facts: TrajectoryFacts) -> bool:
        return self._target_currently_held(target, facts)

    def _target_ever_held(self, target: str, facts: TrajectoryFacts) -> bool:
        if not target:
            return False
        if _phrase_in_text(target, facts.inventory_text):
            return True
        return any(_phrase_in_text(target, obj) or _phrase_in_text(obj, target) for obj in facts.inventory_objects)

    def _target_currently_held(self, target: str, facts: TrajectoryFacts) -> bool:
        if not target:
            return False
        holding = False
        for row in facts.steps:
            if self._step_picks_target(target, row):
                holding = True
            elif self._step_removes_target_from_inventory(target, row):
                holding = False
        if "in your inventory" in _normalize_text(facts.final_result):
            return _phrase_in_text(target, facts.final_result)
        return holding

    def _put_event_seen(self, target: str, place: str, facts: TrajectoryFacts) -> bool:
        if not target or not place:
            return False
        for row in facts.steps:
            text = "\n".join([row["action"], row["obs_after"]])
            normalized = _normalize_text(text)
            if not any(verb in normalized for verb in ("put", "place", "move")):
                continue
            if _phrase_in_text(target, text) and _phrase_in_text(place, text):
                return True
        if _phrase_in_text(target, facts.final_text) and _phrase_in_text(place, facts.final_text):
            if any(marker in _normalize_text(facts.final_text) for marker in (" on ", " in ", " you see ")):
                return True
        return False

    def _target_currently_at_place(self, target: str, place: str, facts: TrajectoryFacts) -> bool:
        if not target or not place:
            return False
        at_place = False
        for row in facts.steps:
            text = "\n".join([row["action"], row["obs_after"]])
            normalized = _normalize_text(text)
            if self._step_picks_target(target, row):
                at_place = False
                continue
            if self._step_drops_or_moves_target(target, row):
                at_place = _phrase_in_text(place, text)
                continue
            if any(verb in normalized for verb in ("put", "place", "move")):
                if _phrase_in_text(target, text):
                    at_place = _phrase_in_text(place, text)
        final_result_norm = _normalize_text(facts.final_result)
        if (
            not self._target_currently_held(target, facts)
            and _phrase_in_text(target, facts.final_result)
            and _phrase_in_text(place, facts.final_result)
            and any(marker in final_result_norm for marker in ("you see", "on the", "in the"))
        ):
            return True
        return at_place

    def _reached_place_with_target(self, target: str, place: str, facts: TrajectoryFacts) -> bool:
        if not target or not place:
            return False
        carrying_target = False
        for row in facts.steps:
            if self._step_picks_target(target, row):
                carrying_target = True
            if carrying_target:
                location = TrajectoryFacts._extract_location(row["obs_after"])
                if _phrase_in_text(place, location) or (
                    "you arrive at" in _normalize_text(row["obs_after"]) and _phrase_in_text(place, row["obs_after"])
                ):
                    return True
            if self._put_event_seen(target, place, TrajectoryFacts.from_steps([row])):
                return True
        return False

    def _step_picks_target(self, target: str, row: Dict[str, str]) -> bool:
        if not target:
            return False
        text = "\n".join([row.get("action", ""), row.get("obs_after", "")])
        normalized = _normalize_text(text)
        if not any(marker in normalized for marker in ("pick up", "take", "to inventory")):
            return False
        return _phrase_in_text(target, text)

    def _step_removes_target_from_inventory(self, target: str, row: Dict[str, str]) -> bool:
        if not target:
            return False
        text = "\n".join([row.get("action", ""), row.get("obs_after", "")])
        normalized = _normalize_text(text)
        if not _phrase_in_text(target, text):
            return False
        if any(marker in normalized for marker in ("put", "place", "drop", "move")):
            return "to inventory" not in normalized
        return False

    def _step_drops_or_moves_target(self, target: str, row: Dict[str, str]) -> bool:
        if not target:
            return False
        text = "\n".join([row.get("action", ""), row.get("obs_after", "")])
        normalized = _normalize_text(text)
        if not _phrase_in_text(target, text):
            return False
        return any(marker in normalized for marker in ("put", "place", "drop", "move"))

    def _detect_regression(
        self,
        task_description: str,
        trajectory_prefix: Sequence[Any],
        after_steps: Sequence[Any],
        added_steps: Sequence[Any],
        *,
        env: str = "",
    ) -> bool:
        hints = TaskHints.from_task(task_description, env=env)
        prefix_facts = TrajectoryFacts.from_steps(trajectory_prefix or [])
        after_facts = TrajectoryFacts.from_steps(after_steps or [])
        added_facts = TrajectoryFacts.from_steps(added_steps or [])
        target = hints.targets[0] if hints.targets else ""
        place = hints.places[0] if hints.places else ""

        if added_facts.reset_seen:
            return True
        if added_facts.failure_count >= max(2, len(added_facts.steps)):
            return True

        if target and place:
            was_held = self._target_currently_held(target, prefix_facts)
            now_held = self._target_currently_held(target, after_facts)
            was_at_place = self._target_currently_at_place(target, place, prefix_facts)
            now_at_place = self._target_currently_at_place(target, place, after_facts)
            was_place_current = _phrase_in_text(place, prefix_facts.final_text) or _phrase_in_text(
                place, prefix_facts.current_location
            )
            now_place_current = _phrase_in_text(place, after_facts.final_text) or _phrase_in_text(
                place, after_facts.current_location
            )
            was_target_current = _phrase_in_text(target, prefix_facts.final_text)
            now_target_current = _phrase_in_text(target, after_facts.final_text)

            if was_at_place and not now_at_place:
                return True
            if was_held and not now_held and not now_at_place:
                return True
            if was_held and was_place_current and now_held and not now_place_current:
                return True
            if was_target_current and not now_target_current and not now_held:
                return True

        normalized_added_actions = [_normalize_text(row["action"]) for row in added_facts.steps]
        navigation_only = bool(normalized_added_actions) and all(
            action.startswith(("go to ", "open door to ", "close door to ", "look", "examine"))
            for action in normalized_added_actions
        )
        if navigation_only and prefix_facts.current_location and after_facts.current_location:
            if prefix_facts.current_location != after_facts.current_location:
                useful_before = _any_phrase_in_text(hints.targets + hints.places + hints.tools, prefix_facts.final_text)
                useful_after = _any_phrase_in_text(hints.targets + hints.places + hints.tools, after_facts.final_text)
                if useful_before and not useful_after:
                    return True

        return False

    def _required_transform_done(self, hints: TaskHints, facts: TrajectoryFacts) -> bool:
        if not hints.required_transforms:
            return False
        text = _normalize_text(facts.history_text)
        for transform in hints.required_transforms:
            aliases = _TRANSFORM_WORDS.get(transform, (transform,))
            if any(alias in text for alias in aliases):
                return True
        return False


__all__ = ["HeuristicPotentialSolver", "PotentialResult", "TaskHints"]
