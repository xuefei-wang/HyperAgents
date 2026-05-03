from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons

class TaskAgent(AgentSystem):
    def _build_instruction(self, inputs):
        domain = inputs['domain']

        if domain == 'swebench_pro':
            git_tempdir = inputs['git_tempdir']
            requirements = inputs.get('requirements') or "No additional requirements provided."
            interface = inputs.get('interface') or "No new interfaces are introduced."
            return f"""You are solving a coding problem.

Repository:
`{git_tempdir}`

Problem statement:
{inputs['problem_statement']}

Requirements:
{requirements}

New interfaces introduced:
{interface}

Task requirements:
- You must inspect the repository and modify files in `{git_tempdir}` directly.
- Use the available tools to read files, edit files, and run shell commands.
- Leave your final code changes in the repository.
- You may use one tool per response, repeatedly across multiple responses, until you have edited the repository.
- Do not merely describe a solution or offer code for the user to paste.
- Do not stop after analysis. Keep working until you have made the code changes you believe solve the task.
- Do not apologize that you cannot access files; use the provided tools to inspect and edit `{git_tempdir}`.
- Your final response is only appropriate after repository files have actually been modified, or after you have determined no code change is needed.
- Prefer targeted edits over broad rewrites.
- When you are finished, provide a short JSON response confirming what you changed.

Respond in JSON format with the following schema:
<json>
{{
    "response": "brief summary of the edits you made"
}}
</json>"""

        if domain == 'arc_ui':
            git_tempdir = inputs['git_tempdir']
            return f"""You are solving one ARC visual reasoning task.

Workspace:
`{git_tempdir}`

Task:
{inputs['problem_statement']}

Public puzzle data:
- `{git_tempdir.rstrip('/')}/payload.json`: canonical JSON with `train` examples and top-level `test_input`.
- `{git_tempdir.rstrip('/')}/grid_summary.md`: readable rendering of the same public grids.
- The hidden test output is not present.

Goal:
Infer the transformation from the training input/output examples, apply it to `test_input`, and write the predicted output grid.

Required output file:
`{git_tempdir.rstrip('/')}/prediction.json`

Use this compact schema:
`{{"attempt_1": <2D integer grid>, "attempt_2": <2D integer grid>}}`

Rules:
- Use only integer colors 0 through 9.
- Each attempt must be a non-empty rectangular grid with side lengths at most 30.
- Provide exactly two attempts. If you have one credible answer, repeat it.
- Do not copy the schema placeholder as an answer; each attempt must match your inferred test output grid.
- Do not use `workspace_state.json` as the final answer unless it genuinely contains your inferred grid.
- Do not change the public puzzle data files, and do not use the task id as a shortcut.
- Prefer `editor` for inspecting workspace files.
- Do not use `bash` to print full grids, full payloads, or full prediction files into the conversation.
- Use `bash` only for short file commands, creating `prediction.json`, and validation.

Reasoning flow:
1. Observe each training input/output pair: dimensions, background color, objects, colors, positions, and what changed.
2. Infer one transformation rule that explains all training pairs.
3. Check the rule against every training pair using only public information.
4. Apply the same rule to `test_input`.
5. Before writing, check output dimensions, color palette, object positions/counts, symmetry, cropping/expansion, movement, recoloring, and consistency across all training examples.

Validate when done:
`cd {git_tempdir} && python3 tools/validate_prediction.py prediction.json`

Leave `prediction.json` in the workspace. A short final confirmation is enough."""

        return f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""

    def forward(self, inputs):
        """
        An agent that solves a given task.

        Args:
            inputs (dict): A dictionary with input data for the task.

        Returns:
            tuple:
                - prediction (str): The prediction made by the agent.
                - new_msg_history (list): A list of messages representing the message history of the interaction.
        """
        instruction = self._build_instruction(inputs)
        domain = inputs.get('domain')
        tools_available = ['bash', 'editor'] if domain in {'swebench_pro', 'arc_ui'} else 'all'
        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available=tools_available,
        )

        # Extract the response
        prediction = "None"
        try:
            extracted_jsons = extract_jsons(new_msg_history[-1]['text'])
            if extracted_jsons is not None and "response" in extracted_jsons[-1]:
                prediction = extracted_jsons[-1]['response']
        except Exception as e:
            self.log(f"Error extracting prediction: {e}")
            prediction = "None"

        return prediction, new_msg_history
