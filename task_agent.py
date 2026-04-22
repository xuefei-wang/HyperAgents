from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons
from utils.git_utils import diff_versus_commit
import json
import os

class TaskAgent(AgentSystem):
    def _validate_arc_grid(self, grid, label):
        if not isinstance(grid, list) or not grid:
            return False, f"{label} must be a non-empty 2D list"
        if not all(isinstance(row, list) and row for row in grid):
            return False, f"{label} must contain non-empty rows"
        width = len(grid[0])
        if len(grid) > 30 or width > 30:
            return False, f"{label} dimensions must be at most 30x30"
        for row_index, row in enumerate(grid):
            if len(row) != width:
                return False, f"{label} row {row_index} has inconsistent width"
            for col_index, value in enumerate(row):
                if not isinstance(value, int) or value < 0 or value > 9:
                    return False, f"{label}[{row_index}][{col_index}] must be an integer 0..9"
        return True, ""

    def _validate_arc_prediction(self, prediction):
        if not isinstance(prediction, dict):
            return False, "prediction must be a JSON object"
        if 'attempts' in prediction:
            attempts = prediction['attempts']
            if not isinstance(attempts, list) or not attempts:
                return False, "attempts must be a non-empty list"
            for attempt_index, attempt in enumerate(attempts[:2], start=1):
                grid = attempt.get('prediction', attempt.get('output', attempt.get('answer'))) if isinstance(attempt, dict) else attempt
                ok, reason = self._validate_arc_grid(grid, f"attempt {attempt_index} prediction")
                if not ok:
                    return False, reason
            return True, ""
        if 'attempt_1' in prediction or 'attempt_2' in prediction:
            found = False
            for key in ['attempt_1', 'attempt_2']:
                if key not in prediction:
                    continue
                found = True
                ok, reason = self._validate_arc_grid(prediction[key], key)
                if not ok:
                    return False, reason
            return (True, "") if found else (False, "prediction must contain attempt_1 or attempt_2")
        if 'prediction' in prediction:
            return self._validate_arc_grid(prediction['prediction'], 'prediction')
        return False, "prediction must contain attempts, attempt_1/attempt_2, or prediction"

    def _arc_prediction_file_valid(self, prediction_path):
        try:
            with open(prediction_path, 'r', encoding='utf-8') as handle:
                prediction = json.load(handle)
        except Exception as exc:
            return False, f"prediction.json is not readable JSON: {exc}"
        return self._validate_arc_prediction(prediction)

    def _extract_arc_prediction(self, msg_history):
        for msg in reversed(msg_history):
            text = msg.get('text', '')
            for candidate in reversed(extract_jsons(text) or []):
                if not isinstance(candidate, dict):
                    continue
                if 'tool_name' in candidate or 'tool_input' in candidate:
                    continue
                if (
                    'attempts' in candidate
                    or 'prediction' in candidate
                    or 'attempt_1' in candidate
                    or 'attempt_2' in candidate
                ):
                    return candidate
        return None

    def _materialize_arc_prediction(self, inputs, msg_history):
        prediction_path = os.path.join(inputs['git_tempdir'], 'prediction.json')
        if os.path.exists(prediction_path) and os.path.getsize(prediction_path) > 0:
            valid, reason = self._arc_prediction_file_valid(prediction_path)
            if valid:
                return True
            self.log(f"Invalid ARC prediction.json removed: {reason}")
            try:
                os.remove(prediction_path)
            except OSError:
                pass
            return False

        prediction = self._extract_arc_prediction(msg_history)
        if prediction is None:
            return False

        valid, reason = self._validate_arc_prediction(prediction)
        if not valid:
            self.log(f"Skipping invalid ARC prediction from message history: {reason}")
            return False

        with open(prediction_path, 'w', encoding='utf-8') as handle:
            json.dump(prediction, handle, indent=2)
            handle.write('\n')
        valid, reason = self._arc_prediction_file_valid(prediction_path)
        if not valid:
            self.log(f"Invalid materialized ARC prediction removed: {reason}")
            try:
                os.remove(prediction_path)
            except OSError:
                pass
            return False
        return True

    def _build_instruction(self, inputs):
        domain = inputs['domain']
        if domain == 'polyglot':
            git_tempdir = inputs['git_tempdir']
            return f"""You are solving a coding task in a writable git repository.

Repository:
`{git_tempdir}`

Problem statement:
{inputs['problem_statement']}

Testing guidance:
{inputs['test_description']}

Task requirements:
- You must inspect the repository and modify files in `{git_tempdir}` directly.
- Use the available tools to read files, edit files, and run shell commands.
- Do not merely describe a solution or offer code for the user to paste.
- Do not stop after analysis. Keep working until you have made the code changes you believe solve the task.
- Prefer targeted edits over broad rewrites.
- When you are finished, provide a short JSON response confirming what you changed.

Respond in JSON format with the following schema:
<json>
{{
    "response": "brief summary of the edits you made"
}}
</json>"""

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
        tools_available = ['bash', 'editor'] if domain in {'polyglot', 'swebench_pro', 'arc_ui'} else 'all'
        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available=tools_available,
            return_on_error=domain == 'arc_ui',
        )

        if domain == 'swebench_pro':
            for _ in range(2):
                patch = diff_versus_commit(inputs['git_tempdir'], inputs['base_commit']).strip()
                if patch:
                    break
                retry_instruction = f"""No repository changes were detected in `{inputs['git_tempdir']}`.

Continue the task now. Use the available tools to inspect and edit files in `{inputs['git_tempdir']}`. You may use one tool per response repeatedly. Do not provide a final answer until you have made a concrete code change, unless you can justify that no code change is necessary for this issue."""
                new_msg_history = chat_with_agent(
                    retry_instruction,
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    tools_available=tools_available,
                )
            patch = diff_versus_commit(inputs['git_tempdir'], inputs['base_commit']).strip()
            if patch:
                review_instruction = f"""A git diff is now present in `{inputs['git_tempdir']}`.

Do one final generic code-review pass before finishing:
- Inspect the current diff.
- If the diff edits tests without the issue explicitly requiring test edits, revert those test edits or justify why they are necessary.
- Check for obvious compile/syntax problems introduced by your changes, such as unused imports, wrong package names, missing functions, or malformed code.
- Run one cheap relevant validation command if it is practical in this repository. If validation is unavailable or too expensive, use static inspection.
- If you find a problem caused by your patch, fix it.
- If the current diff is incomplete or would likely fail to compile and you cannot complete it, revert the broken hunks instead of leaving known-bad partial edits.
- Do not provide a final JSON response while knowingly leaving an incomplete or uncompilable patch in the repository."""
                new_msg_history = chat_with_agent(
                    review_instruction,
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    tools_available=tools_available,
                )

        if domain == 'arc_ui':
            prediction_path = os.path.join(inputs['git_tempdir'], 'prediction.json')
            for _ in range(2):
                if self._materialize_arc_prediction(inputs, new_msg_history):
                    break
                retry_instruction = f"""No `prediction.json` was detected in `{inputs['git_tempdir']}`.

Continue the ARC visual reasoning task now. Use `payload.json` / `grid_summary.md` to infer one transformation that explains the public training pairs, apply it to top-level `test_input`, and write `{prediction_path}`.

Use the compact schema `{{"attempt_1": <2D integer grid>, "attempt_2": <2D integer grid>}}`. Each attempt must be non-empty, rectangular, and at most 30x30. Do not copy the placeholder. If uncertain, write your best non-empty guess rather than leaving the task unanswered. Validate with `cd {inputs['git_tempdir']} && python3 tools/validate_prediction.py prediction.json`. Do not append `cat prediction.json`."""
                new_msg_history = chat_with_agent(
                    retry_instruction,
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    tools_available=tools_available,
                    return_on_error=True,
                )
            if not self._materialize_arc_prediction(inputs, new_msg_history):
                final_json_instruction = f"""No `prediction.json` was created in `{inputs['git_tempdir']}`.

Stop using tools. Based on the ARC task context already shown, return only the final prediction JSON object. Do not wrap it in a tool call, do not include prose, and do not include Markdown.

Required schema:
{{"attempt_1": <2D integer grid>, "attempt_2": <2D integer grid>}}

Use integer colors 0 through 9. Each attempt must be non-empty, rectangular, and at most 30x30. Do not copy the placeholder. If you have only one credible grid, repeat it for both attempts."""
                new_msg_history = chat_with_agent(
                    final_json_instruction,
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    tools_available=[],
                    return_on_error=True,
                )
                self._materialize_arc_prediction(inputs, new_msg_history)
            if os.path.exists(prediction_path) and os.path.getsize(prediction_path) > 0:
                review_instruction = f"""A prediction file now exists at `{prediction_path}`.

Review it as an ARC visual reasoning answer using only the public training pairs and `test_input`.

Consistency check:
- Does one transformation rule explain every training input/output pair?
- Does your prediction apply that same rule to `test_input`?
- Are the output dimensions, colors, object positions/counts, symmetry, cropping/expansion, movement, and recoloring consistent with the training pairs?
- Is either attempt just a copied schema placeholder or arbitrary tiny grid?
- Is each attempt non-empty, rectangular, and at most 30x30?

If the current prediction is inconsistent, revise `{prediction_path}`. If it is consistent, leave it unchanged. Validate with `cd {inputs['git_tempdir']} && python3 tools/validate_prediction.py prediction.json`. Do not print the full prediction file."""
                new_msg_history = chat_with_agent(
                    review_instruction,
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    tools_available=tools_available,
                    return_on_error=True,
                )
                self._materialize_arc_prediction(inputs, new_msg_history)
            if not self._materialize_arc_prediction(inputs, new_msg_history):
                final_json_instruction = f"""No valid `prediction.json` remains in `{inputs['git_tempdir']}`.

Stop using tools. Based on the ARC task context already shown, return only a valid final prediction JSON object. Do not wrap it in a tool call, do not include prose, and do not include Markdown.

Required schema:
{{"attempt_1": <2D integer grid>, "attempt_2": <2D integer grid>}}

Use integer colors 0 through 9. Each attempt must be non-empty, rectangular, and at most 30x30. Do not copy the placeholder. If you have only one credible grid, repeat it for both attempts."""
                new_msg_history = chat_with_agent(
                    final_json_instruction,
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    tools_available=[],
                    return_on_error=True,
                )
                self._materialize_arc_prediction(inputs, new_msg_history)

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
