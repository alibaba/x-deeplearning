find . -name "*.py" | awk '{cmd="python "$0" --run_mode=local";print(cmd);system(cmd)}'
