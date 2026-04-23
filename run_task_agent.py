import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from task_agent import TaskAgent
from utils.git_utils import diff_versus_commit


def _load_shared_env() -> None:
    task_agent_path = Path(__file__).resolve()
    parent_candidates = list(task_agent_path.parents)
    env_paths = [
        root / relative_path
        for root in [*(parent_candidates[idx] for idx in (1, 2) if idx < len(parent_candidates)), Path.cwd()]
        for relative_path in (
            Path("configs") / "providers" / ".env.shared",
            Path("configs") / "models" / "shared.env",
        )
    ]
    loaded = set()
    for env_path in env_paths:
        if env_path.exists() and env_path not in loaded:
            load_dotenv(env_path, override=True)
            loaded.add(env_path)


def _default_model_for_domain(domain):
    if domain == "polyglot":
        return os.getenv(
            "HYPERAGENTS_POLYGLOT_MODEL",
            os.getenv("HYPERAGENTS_TASK_MODEL", "openai/gpt-5.4-mini"),
        )
    return os.getenv("HYPERAGENTS_TASK_MODEL", "openai/gpt-5.4-mini")


def main():
    _load_shared_env()
    parser = argparse.ArgumentParser(description='Run task agent on a coding benchmark task.')
    parser.add_argument('--domain', default='polyglot', help='Benchmark domain name')
    parser.add_argument('--problem_statement', required=True, help='The problem statement to process')
    parser.add_argument('--requirements', default=None, help='Task requirements text')
    parser.add_argument('--interface', default=None, help='New interfaces introduced text')
    parser.add_argument('--git_dir', required=True, help='Path to git repository directory')
    parser.add_argument('--base_commit', required=True, help='Base commit hash to compare against')
    parser.add_argument('--chat_history_file', required=True, help='Path to chat history file')
    parser.add_argument('--outdir', required=False, default="/dgm/", help='Output directory')
    parser.add_argument('--test_description', default=None, required=False, help='Description of how to test the repository')
    parser.add_argument('--language', default=None, required=False, help='Coding language of the repository')
    parser.add_argument(
        '--model',
        required=False,
        default=None,
        help='LLM model to use',
    )
    args = parser.parse_args()
    model = args.model or _default_model_for_domain(args.domain)

    # Process the repository
    agentic_system = TaskAgent(
        model=model,
        chat_history_file=args.chat_history_file,
    )
    inputs = {
        "domain": args.domain,
        "problem_statement": args.problem_statement,
        "requirements": args.requirements,
        "interface": args.interface,
        "git_tempdir": args.git_dir,
        "base_commit": args.base_commit,
        "test_description": args.test_description,
        "language": args.language,
    }

    # Run the agentic system to try to solve the problem
    agentic_system.forward(inputs)

    # Get code diff and save to model_patch.diff
    model_patch = diff_versus_commit(args.git_dir, args.base_commit)
    model_patch_outfile = os.path.join(args.outdir, 'model_patch.diff') if args.outdir else 'model_patch.diff'
    with open(model_patch_outfile, 'w') as f:
        f.write(model_patch)

if __name__ == "__main__":
    main()
