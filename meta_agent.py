# Copyright (c) Meta Platforms, Inc. and affiliates.

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which
                the meta agent will be invoked in future. Defaults to None.
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

        chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available=["bash", "editor"],
        )
