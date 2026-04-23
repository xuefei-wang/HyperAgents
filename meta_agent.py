# Copyright (c) Meta Platforms, Inc. and affiliates.

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.git_utils import diff_versus_commit

MAX_EMPTY_DIFF_RETRIES = 2


class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None, base_commit=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which
                the meta agent will be invoked in future. Defaults to None.
            base_commit (str, optional): Git commit used to detect whether the repo changed.
        """
        instruction = f"""You are improving a writable git repository.

Repository root:
`{repo_path}`

Task requirements:
- Inspect the repository and modify files inside `{repo_path}` directly.
- Use the available tools to read files, edit files, and run shell commands.
- Do not create a new project or write files outside `{repo_path}`.
- Do not stop after analysis. Keep working until you have made at least one real code change in `{repo_path}`.
- When you are finished, provide a short JSON response summarizing the edits.

Respond in JSON format with the following schema:
<json>
{{
    "response": "brief summary of the edits you made"
}}
</json>"""

        msg_history = []
        current_instruction = instruction
        for attempt in range(MAX_EMPTY_DIFF_RETRIES + 1):
            msg_history = chat_with_agent(
                current_instruction,
                model=self.model,
                msg_history=msg_history,
                logging=self.log,
                tools_available=["bash", "editor"],
                require_tool_call_before_final=True,
            )
            if base_commit is None or diff_versus_commit(repo_path, base_commit).strip():
                return msg_history
            if attempt == MAX_EMPTY_DIFF_RETRIES:
                raise RuntimeError(
                    f"No repository changes were detected in `{repo_path}` after repeated retries."
                )
            current_instruction = f"""No repository changes were detected in `{repo_path}`.

Continue the task now:
- Inspect the repository and make at least one concrete code change inside `{repo_path}`.
- Use the available tools to edit files directly.
- Do not return the final JSON response until the repository diff is non-empty.
- Do not create a new project or write files outside `{repo_path}`."""

        return msg_history
