
import os
import asyncio
import argparse
import subprocess

from agent import MUSE
from prompt.system_prompt import MUSE_sys_prompt

"""
This Python script needs to be run in the TAC task container.
"""

def get_tac_evaluation(task_name: str, agent_name: str, mode: str, round: int) -> str:
    cmd = [
        "python_default", "/utils/eval.py",
        "--trajectory_path", f"/MUSE/outputs/{agent_name}/{mode}/{task_name}/round_{round}/history.txt",
        "--result_path", f"/MUSE/outputs/{agent_name}/{mode}/{task_name}/round_{round}/agent_eval_output.json"
    ]
    env = {"DECRYPTION_KEY": "theagentcompany is all you need", **os.environ}
    proc = subprocess.run(cmd, capture_output=True, env=env, text=True, cwd="/workspace")

    evaluation = proc.stdout + "\n" + proc.stderr
    with open("/instruction/checkpoints.md", "r") as f:
        checkpoints = f.read()

    result = f"<checkpoints>\n{checkpoints}\n</checkpoints>\n<evaluation>\n{evaluation}\n<evaluation>\n"

    return result

async def main():
    parser = argparse.ArgumentParser(description="Interact with Agent")
    parser.add_argument("--agent_name", type=str, help="Agent name", default="test_agent")
    parser.add_argument("--task_name", type=str, help="Task name", default="test_task")
    parser.add_argument("--task", type=str, help="Please enter your instructions")
    parser.add_argument("--mode", type=str, help="Training mode", default="test")
    parser.add_argument("--round", type=int, help="Training round", default=1)
    parser.add_argument("--llm", type=str, help="Base LLM", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    args = parser.parse_args()

    mode = args.mode

    if mode == "train":
        agent = MUSE(
            init_model_name=args.llm,
            sys_prompt_template=MUSE_sys_prompt,
            memory_dir="memory",
            agent_name=args.agent_name,
            task_name=args.task_name,
            output_dir="outputs",
            mode_label="train",
            task_round=args.round,
            use_memory=True,
            update_memory=True,
            # lang="zh"
            # env_feedback_func=get_tac_evaluation,
            # env_feedback_args={"task_name": args.task_name, "agent_name": agent_name, "mode": args.mode, "round": args.round}
        )
        agent.logger.log_task(args.task, subtitle="STARTING······", title="Task")
        await agent.run(args.task, subtask_action_limit=20, num_actions_scale=8, time_limit=2400, verbose=False)
    else:
        agent = MUSE(
            init_model_name=args.llm,
            sys_prompt_template=MUSE_sys_prompt,
            memory_dir="memory",
            agent_name=args.agent_name,
            task_name=args.task_name,
            output_dir="outputs",
            mode_label="test",
            task_round=args.round,
            use_memory=False,
            update_memory=False,
            # lang="zh"
        )
        agent.logger.log_task(args.task, subtitle="STARTING······", title="Task")
        await agent.run(args.task, subtask_action_limit=20, num_actions_scale=8, time_limit=2400, verbose=False)

    eval_log = get_tac_evaluation(args.task_name, args.agent_name, args.mode, args.round)
    agent.logger.log_task(eval_log, subtitle="TEST", title="Final Score")
    with open(agent._get_output_dir() / "eval_log.txt", mode="w") as f:
        f.write(eval_log)

if __name__ == "__main__":
    asyncio.run(main())
