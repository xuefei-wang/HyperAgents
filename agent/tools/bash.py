import os
import subprocess

TIMEOUT_SECONDS = 120.0


def tool_info():
    return {
        "name": "bash",
        "description": """Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* You do have access to a mirror of common linux and python packages via apt and pip.
* Each call runs in a fresh shell. State does not persist across command calls, so chain related steps with `&&`.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Do not use shell heredocs such as python <<'PY' or cat <<'EOF'. They can hang this tool. Use python -c, existing scripts, or the editor tool instead.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": ["command"]
        }
    }


def filter_error(error):
    filtered_lines = []
    i = 0
    error_lines = error.splitlines()
    while i < len(error_lines):
        line = error_lines[i]

        if "Inappropriate ioctl for device" in line:
            i += 3
            if i < len(error_lines) and "<<exit>>" in error_lines[i]:
                i += 1
            while i < len(error_lines) - 1:
                filtered_lines.append(error_lines[i])
                i += 1
            i += 1
            continue

        filtered_lines.append(line)
        i += 1
    return "\n".join(filtered_lines).strip()


def tool_function(command):
    """Execute a command in a fresh bash shell."""
    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=os.environ.copy(),
        )
        output = result.stdout.strip()
        error = filter_error(result.stderr.strip())
        response = output if output else ""
        if error:
            response += ("\n" if response else "") + "Error:\n" + error
        return response.strip()
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "").strip()
        error = filter_error((exc.stderr or "").strip())
        response = output if output else ""
        if error:
            response += ("\n" if response else "") + "Error:\n" + error
        timeout_error = f"Timed out: bash has not returned in {TIMEOUT_SECONDS} seconds."
        response += ("\n" if response else "") + "Error:\n" + timeout_error
        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        input_command = " ".join(sys.argv[1:])
        result = tool_function(input_command)
        print(result)
