from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PromptAssets:
    system_prompt_template: str
    retrieval_instruction: str
    tool_spec_template: str

    @classmethod
    def from_files(
        cls,
        system_prompt_path: Path,
        retrieval_instruction_path: Path,
        tool_spec_template_path: Path,
    ) -> "PromptAssets":
        return cls(
            system_prompt_template=system_prompt_path.read_text(encoding="utf-8").strip(),
            retrieval_instruction=retrieval_instruction_path.read_text(encoding="utf-8").strip(),
            tool_spec_template=tool_spec_template_path.read_text(encoding="utf-8").strip(),
        )


def render_system_prompt(
    *,
    assets: PromptAssets,
    tool_spec: str,
    extra_system_instruction: str | None = None,
) -> str:
    sections = [
        assets.system_prompt_template,
    ]
    if extra_system_instruction:
        sections.append(extra_system_instruction)
    sections.extend(
        [
            assets.tool_spec_template.format(tool_spec=tool_spec),
            assets.retrieval_instruction,
        ]
    )
    return "\n\n".join(section for section in sections if section)


def build_initial_messages(
    *,
    query: str,
    system_prompt: str,
    retrieval_instruction: str,
) -> list[dict[str, str]]:
    user_prompt = f"Query: {query}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_mode_system_instruction(*, sentinel: str, answer_mode: str) -> str | None:
    if answer_mode == "legacy":
        return (
            "Legacy answer mode: during ordinary turns, you may return a final answer or "
            f"{sentinel} when the retrieved evidence does not support a single best answer."
        )
    if answer_mode == "deferred_insufficient_evidence":
        return (
            f"Deferred insufficient-evidence mode: during ordinary turns, do not answer with {sentinel}. "
            "Either keep searching or return a supported non-sentinel answer. "
            f"Use {sentinel} only after you are explicitly told that retrieval stops here and you must decide."
        )
    return None


def build_retry_user_message(*, reason: str, sentinel: str) -> str:
    if reason == "duplicate_search_blocked":
        return (
            "That search is too similar to a recent search. Do not repeat it. "
            "Try a different search angle or answer with the best supported non-sentinel answer if the evidence is already sufficient."
        )
    if reason == "no_progress":
        return (
            "Recent searches are not adding new evidence. Try a different grounded search angle. "
            f"Do not answer with {sentinel} during ordinary turns; wait for the final decision prompt if needed."
        )
    if reason == "early_insufficient_evidence":
        return (
            f"Do not answer with {sentinel} during ordinary turns. "
            "Continue searching for more evidence or return a supported non-sentinel answer if you already have one."
        )
    return (
        "Continue searching with a different grounded search angle or answer with the best supported non-sentinel answer if the evidence is already sufficient."
    )


def build_final_user_message(
    *,
    sentinel: str,
    escalated: bool = False,
) -> str:
    if escalated:
        return (
            "You must answer NOW. Do NOT emit a tool call. "
            "Give your single best-supported plain-text answer based on everything retrieved so far. "
            "If the evidence chain is almost complete — e.g. you found a person connected to the subject "
            "as a partner, collaborator, or close associate rather than with the exact relationship word "
            "in the question — provide that person's name as your answer. "
            "If the retrieved evidence supports a single best answer candidate, return that candidate now. "
            f"Use {sentinel} ONLY if the retrieved evidence still does not support a single best answer."
        )
    return (
        "Retrieval stops here. Give your single best plain-text answer using only what was already retrieved. "
        "Rules:\n"
        "1. If you found a name or entity strongly connected to the subject, state it — even if the "
        "relationship word is a near-synonym (e.g. 'partner' for 'spouse', 'performed by' for 'performed'). "
        "2. If the evidence chain is incomplete but points toward a specific answer, give that answer.\n"
        # f"3. Use {sentinel} ONLY if the retrieved evidence contains no entity names or facts "
        # "relevant to the question at all. Do NOT use it just because the chain is imperfect."
    )
