"""Prompt templates for the cold-start data generation pipeline.

Two LLM roles:

- **Tutor**: sees gold answer + decomposition + downstream docs. Used for
  decomposition, bridge-entity extraction, backward command discovery, judging
  whether a tool output confirms an expected answer, and editing the planner's
  near-miss commands, and editing the planner's draft think to align with the
  known-working command.

- **Planner**: sees only the original question and the tool_responses of steps
  already in the trace. Generates draft (think, command) pairs forward, and
  drafts each step from the question and observed history only. Never sees the
  gold answer, the decomposition, or the target command.

Hard rules baked into tutor prompts:
  - Backward commands MUST search via terms in the sub-question, NOT via the
    expected answer. The agent at inference time does not know the answer.
  - Tutor edits to the planner's think must not introduce facts the agent
    couldn't observe at this point in the trajectory.
"""

# ============================================================
# Corpus description used wherever a prompt needs to refer to the file format.
# ============================================================

CORPUS_DESCRIPTION = """Corpus file: corpus.jsonl

Treat the corpus as a large text file with one Wikipedia passage per line. The file has 21 million lines, so always cap output (use `| head -n 3` for narrow searches, up to `| head -n 8` when you need to scan more chunks of the same article).

Useful command patterns:
- Substring search:
    rg -F "distinctive phrase" corpus.jsonl | head -n 3
- AND-narrow with a second grep:
    rg -F "phrase1" corpus.jsonl | rg -i -F "phrase2" | head -n 3
- Count first, then narrow if too many results:
    rg -F "pattern" corpus.jsonl | wc -l
- Useful flags: -F (fixed string, no regex), -i (case-insensitive), -w (whole word).

Allowed shell tools: rg, grep, find, sed, awk, head, tail, cat, ls, wc, sort, cut, uniq, tr.
You MAY pipe with |. You MAY NOT use redirection (>, <), command chaining (; && ||), or command substitution ($(...), `...`)."""


# ============================================================
# Phase A: Decomposition (tutor)
# ============================================================

DECOMPOSE_PROMPT = """You are decomposing a multi-hop question into the ordered chain of single-hop sub-questions an agent would need to solve in order to reach the answer.

Question: {question}
Final answer: {answer}

Rules:
- Output 1 to 3 ordered sub-questions; each one's answer is what an agent would need before it can solve the next one.
- Order MUST match the order an agent would solve the question (sub_q[i+1] depends on sub_q[i]'s answer).
- The LAST sub-question's answer is the final answer above.
- Do NOT include expected_answers — only the sub-questions themselves. The intermediate answers will be derived later.
- Do NOT make a sub-question whose answer is already given in the question (e.g., do not ask "Who is X?" when the question already names X).

Output ONLY a JSON array, no prose:
[{{"sub_question": "..."}}, {{"sub_question": "..."}}, ...]"""


# ============================================================
# Phase B: Bridge-entity extraction (tutor)
# Used to derive expected_answer_i from doc_{i+1} when i < N.
# ============================================================

BRIDGE_EXTRACT_PROMPT = """You are reading a Wikipedia passage that was retrieved to help answer a multi-hop question. Your job is to identify the entity the passage uses as the bridge — the answer to the previous sub-question.

Original multi-hop question: {question}

Earlier sub-question (whose answer we are trying to identify): {sub_q_prev}
Later sub-question (already solved): {sub_q_next}
Answer to the later sub-question: {expected_next}

Passage retrieved for the later sub-question:
---
{doc_next}
---

Read the passage. The passage answers the later sub-question; somewhere in it, the entity named in the answer to the EARLIER sub-question appears (since the later sub-question depends on it). Extract that entity exactly.

Also collect:
- "aliases" — alternative names of the SAME entity that the passage explicitly mentions (e.g., stage names, real names, abbreviations, parenthetical "also known as"/"born ..." forms). Only include forms that literally appear in the passage. Do NOT invent aliases.
- "alternates" — at most 2 OTHER plausible candidate entities from the passage that could also be the bridge if your primary pick is wrong. Use this only when the passage genuinely supports multiple readings.

Output a JSON object, nothing else:
{{"bridge_entity": "<short noun phrase>", "aliases": ["<alt-name-1>", ...], "alternates": ["<other-candidate-1>", ...], "evidence": "<the exact short snippet from the passage that names the primary entity>"}}

If you cannot extract a clear bridge entity (the passage truly does not contain enough to derive the previous answer), output:
{{"bridge_entity": null, "aliases": [], "alternates": [], "evidence": null}}"""


# ============================================================
# Phase B: Backward command discovery (tutor)
# Iterative — proposes and refines until the LLM judge confirms.
# ============================================================

BACKWARD_COMMAND_SYSTEM = """You are an expert at writing shell commands to retrieve specific Wikipedia passages from a large JSONL corpus.

{corpus_description}

You are working BACKWARD from a known answer to construct a tool command that, when executed, returns a passage that confirms the answer to a sub-question.

CRITICAL — do not violate these:

(1) ANSWER-LEAK RULE.
Your shell command must NOT use the expected_answer string itself (or a near-identical paraphrase of it) as a search term. The agent at inference time will not know the answer; the answer must EMERGE from the document the command retrieves, not be an input to the search.

You are however ENCOURAGED to use:
- Words and phrases from the sub-question.
- Synonyms, paraphrases, and reformulations of those words. Pure keyword search often misses; trying near-synonyms is a real agent skill (e.g., "head office" / "headquartered in" / "based in" / "located in", "founded" / "established" / "started", "directed by" / "directed", "wife" / "spouse" / "married").
- Related concepts that an agent could plausibly guess from the sub-question alone.

The hard line: don't paste the literal answer string into the command.

Example (sub_question = "Which hotel company is the Oberoi family part of?", expected_answer = "The Oberoi Group"):
  WRONG (literal answer):           rg -F 'Oberoi Group' corpus.jsonl | head -n 3
  WRONG (near-identical paraphrase): rg -F 'The Oberoi' corpus.jsonl | head -n 3
  RIGHT (sub-question term):        rg -F 'Oberoi family' corpus.jsonl | rg -i -F 'hotel' | head -n 3
  RIGHT (synonym variation):        rg -F 'Oberoi family' corpus.jsonl | rg -i -F 'hospitality' | head -n 3
  RIGHT (broader, then narrow):     rg -F 'Oberoi family' corpus.jsonl | head -n 5

If keyword search keeps coming up empty, switch to a different paraphrase of the sub-question's key concept rather than reaching for the answer.

(2) OUTPUT SIZE.
Keep output short — typically `| head -n 3`. You may go up to `| head -n 8` when you need to scan more chunks of the same article (e.g., the article's intro is in chunk 1 but a specific fact is in a later chunk). If you're hitting clearly-irrelevant articles, prefer refining the QUERY before enlarging the head cap.

(3) STRUCTURE.
A single shell pipeline. No redirection (>), no chaining (; && ||), no command substitution ($(...), `...`). Allowed tools: rg, grep, find, sed, awk, head, tail, cat, ls, wc, sort, cut, uniq, tr."""


BACKWARD_COMMAND_USER_INITIAL = """Sub-question to find a supporting passage for: {sub_question}

Expected answer (known to you for verification only — do NOT use any of these as search terms):
- primary: {expected_answer}
- alternative forms (also forbidden as search terms): {forbidden_forms}

Other passages already retrieved for later sub-questions in this chain (for context only):
{downstream_docs}

Reason briefly about which Wikipedia article would most directly answer this sub-question, and what distinguishing terms from the SUB-QUESTION (not the answer or any of its alternative forms) you'd use to find it. Then output your single shell command.

Output exactly:
<reasoning>
your reasoning (2-4 sentences); explicitly check that your command does NOT include the expected answer or any of its alternative forms as a search term
</reasoning>
<command>
your single-pipeline shell command
</command>"""


BACKWARD_COMMAND_USER_REFINE = """Your previous command did not retrieve a passage that confirms the expected answer.

Sub-question: {sub_question}
Expected answer (do NOT search for this string): {expected_answer}
Alternative forms (also forbidden as search terms): {forbidden_forms}

Previous attempts:
{prior_attempts}

Most recent attempt:
Command: {last_command}
Output:
---
{last_output}
---
Judge said it does NOT confirm the answer because: {judge_reasoning}

Propose a different command. Common fixes:
- Your phrase may not appear verbatim — try a shorter or differently-worded substring from the sub-question.
- The query may be too broad — add a second piped rg with another distinctive phrase to AND-narrow.
- The query may be too narrow — drop a filter or shorten the phrase to broaden.
- Try synonym variants of words from the sub-question (e.g., "headquartered in" vs "head office", "based in", "located in").

Same hard rules as before (do NOT use the expected answer as a search term; keep output ≤ 5 lines via head; one pipeline; whitelisted tools only).

Output exactly:
<reasoning>
why the previous command failed and what is different about your new approach
</reasoning>
<command>
your new single-pipeline shell command
</command>"""


# ============================================================
# Phase B: LLM judge (tutor)
# ============================================================

JUDGE_PROMPT = """You are judging whether a tool output confirms the expected answer to a sub-question.

Sub-question: {sub_question}
Expected answer (primary): {expected_answer}
Acceptable alternative forms (any of these counts as confirming): {acceptable_forms}

Tool output:
---
{tool_output}
---

Read the tool output. Does it contain a passage that clearly answers the sub-question with the primary expected answer OR any of the acceptable alternative forms? Be strict — a tangential mention or a different entity with a coincidentally similar name does NOT count. But if the same underlying entity is referenced under any of the alternative forms (e.g., a stage name, a real name, or an alias listed above), that DOES count as confirming.

Output exactly one JSON object:
{{"verdict": "YES" or "NO", "reasoning": "<one short sentence>"}}"""


# ============================================================
# Phase C: Planner draft (planner)
# Sees only (question, prior_history). No answer, no decomposition, no target.
# ============================================================

PLANNER_SYSTEM = """You are a research agent that searches a Wikipedia corpus to answer multi-hop questions by issuing shell commands.

{corpus_description}

For every step, write a short paragraph of reasoning in plain prose (2-5 sentences) — what you have learned from prior tool outputs, what's still missing, what you'll search for next. Then output exactly one of:

<tool_call>
{{"name": "shell", "arguments": {{"command": "your single-pipeline shell command, no newlines"}}}}
</tool_call>

OR (only when you have enough information to answer):

<answer>
the final answer (concise — typically a short noun phrase, name, or date)
</answer>

Always reason first, then exactly one action block. Do not skip the reasoning."""


PLANNER_USER = """Question: {question}

Trace so far:
{history}

Produce the next step."""


# ============================================================
# Phase C: Tutor edits the planner's draft think (tutor)
# ============================================================

TUTOR_EDIT_PROMPT = """You are editing a research agent's draft reasoning so that it leads naturally to a known-correct next command. Your goal is to produce a final reasoning paragraph that:

1. Is grounded ONLY in what the agent has actually observed so far (the trace below). Do NOT add facts the agent could not know at this point.
2. Concludes with a clear motivation for the target command.
3. Reads as natural forward reasoning by a careful agent — not as a justification written after the fact. Hedge where genuine uncertainty exists.
4. Stays close to the agent's draft when the draft is already on track; rewrite freely when the draft proposes something the wrong direction.

Original question (the agent knows this): {question}

Trace observed so far by the agent:
{history}

Agent's draft reasoning:
---
{draft_think}
---

Agent's draft command (may be discarded):
---
{draft_command}
---

The known-correct next command (this is what the trace will use):
---
{target_command}
---

What this command will return when executed (for your context only — the agent only sees this AFTER it runs the command, NOT while reasoning):
---
{target_doc_preview}
---

FORBIDDEN — your edited reasoning must NOT contain:

(a) The expected answer of the current step, the final answer, or any answer to be discovered in a future step. The agent should not know these yet.

(b) Any factual claim the agent could not derive from (i) the user's question, (ii) prior tool_responses in the trace, or (iii) common knowledge.

ALLOWED — the agent's reasoning may freely include:
- Words from the question and synonyms/paraphrases of them ("hotel" → "hospitality", "head office" → "headquarters" / "based in").
- Tentative hypotheses about WHERE the answer might be found, phrased as guesses — e.g., "the article likely uses the phrase 'headquartered in' rather than 'head office'", or "this is probably described in a Wikipedia article about the family's business empire".
- Common-knowledge inference — e.g., "Delhi is the capital of India" once Delhi has appeared, or "World Cup 2002 was the FIFA tournament" given a sports question.

Concrete pass/fail examples:

  Question: "The Oberoi family is part of a hotel company that has a head office in what city?"
  Trace so far: (no commands run yet)
  Target command: rg -F 'Oberoi family' corpus.jsonl | rg -F 'hotel' | head -n 3

    LEAK (names the bridge answer "Oberoi Group" before it has been seen):
      "I need to locate the article on the Oberoi Group to find its headquarters."

    OK (describes the search in question terms with hypotheses):
      "I need to find which hotel company the Oberoi family is part of. The corpus likely contains passages mentioning the 'Oberoi family' alongside hotel-business context. I'll grep for 'Oberoi family' and narrow with 'hotel' to surface the relevant article."

  Question: same as above
  Trace so far: tool_response showed "...the Oberoi family is the majority shareholder in EIH Ltd, parent of The Oberoi Group..."
  Target command: rg -F 'Oberoi Group' corpus.jsonl | rg -F 'head office' | head -n 3

    OK ("Oberoi Group" is now derivable from the previous tool_response):
      "The previous result identifies the family's hotel company as The Oberoi Group. To find its head office, I'll search for that company alongside 'head office', or a synonym like 'headquartered'."

When the planner's draft is already valid under these rules, keep it close to the draft. When the draft proposes a substantively different direction, rewrite the reasoning so it naturally leads to the target command — using hypotheses, synonyms, and search-strategy reasoning, never by naming entities the agent shouldn't know yet.

Output exactly:
<edited_reasoning>
the final reasoning paragraph (2-5 sentences)
</edited_reasoning>"""


# ============================================================
# Phase D: Final answer (planner)
# ============================================================

FINAL_ANSWER_USER = """Question: {question}

Trace so far:
{history}

You now have enough information. Produce a brief reasoning paragraph synthesizing the answer from the trace, then output exactly:

<answer>
the final answer (concise — just a name, date, or short noun phrase)
</answer>"""


# ============================================================
# Phase E: Quality judge (post-hoc filter on the assembled trajectory)
# Uses turn-numbered plain-text rendering (no <think>/<tool_call>/etc tags)
# so the judge LLM never has to disambiguate its own reasoning from the
# trajectory's reasoning.
# ============================================================

QUALITY_JUDGE_PROMPT = """You are an expert reviewer of multi-hop QA trajectories produced by a research agent. The agent answers a question by issuing shell commands against a Wikipedia corpus and reasoning about the results across multiple turns. Your job is NOT to solve the question. Your job is to verify that the trajectory is INTERNALLY COHERENT.

INFORMATION FRONTIER

At any TURN k, the agent could legitimately know only:
  (a) the text of the original QUESTION,
  (b) anything that appeared in the OUTPUT of any earlier turn (1 ... k-1),
  (c) generic common knowledge that any educated reader would have without consulting Wikipedia. Examples of (c):
        - "Delhi is in India", "Brazil is a country", "WW II ended in 1945".
        - "Headquarters" / "head office" / "based in" mean roughly the same.
        - Synonyms and paraphrases of words from the question.
      Counter-examples (NOT (c)):
        - "Cafu was Brazil's 2002 World Cup captain" - too specific.
        - "The Oberoi Group is based in Delhi" - too specific.
        - The bridge answer of any sub-question.

CHECKS

The trajectory FAILS if ANY of the following holds for ANY turn. Report the FIRST turn where it fails and which check failed.

CHECK 1 - REASONING leaks future facts.
  TURN k's REASONING names a specific entity, number, date, or fact not derivable from the information frontier.
  FAIL example: TURN 1 REASONING (no prior turns) says "I'll look up the Oberoi Group's headquarters" when the question only mentions "Oberoi family" - the company name has not been observed.

CHECK 2 - COMMAND leaks future facts.
  TURN k's COMMAND uses a search term not derivable from the information frontier.
  FAIL example: question asks for a track length, no prior OUTPUT has named a number, and the COMMAND contains rg -F "6.213" - that number is the answer the agent should still be searching for.

CHECK 3 - ACTION does not match REASONING.
  TURN k's REASONING explicitly states "the answer is X", "no further search is needed", "the question is answered", "I have the answer", and the action on this turn is a COMMAND rather than a FINAL ANSWER.
  FAIL example: REASONING ends with "the answer is clearly Amy Jo Johnson, no further tool execution is needed" and the action is a shell command.

CHECK 4 - FINAL ANSWER not supported.
  The final turn's FINAL ANSWER is not stated in, or clearly inferrable from, any earlier OUTPUT.
  FAIL example: FINAL ANSWER is "Roseau, Minnesota" but no earlier OUTPUT contains "Roseau" or supports that location.

If none of CHECK 1-4 fails on any turn, the trajectory PASSES.

When in doubt, prefer FAIL over PASS - we want clean training data, not high yield.

OUTPUT

Return EXACTLY one JSON object, no surrounding prose, no markdown fences:
{{"verdict": "PASS" or "FAIL", "failing_check": 1 or 2 or 3 or 4 or null, "first_failing_turn": <int or null>, "reasoning": "<one short sentence, max ~30 words>"}}

QUESTION
{question}

TRAJECTORY
{trajectory_text}"""
