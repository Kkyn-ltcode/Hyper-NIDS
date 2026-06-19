import re
from pathlib import Path

names = [
    "ta1-theia-e3-official-6r.json.10",
    "ta1-theia-e3-official-6r.json.2",
    "ta1-theia-e3-official-1r.json",
    "ta1-theia-e3-official-1r.json.10",
    "ta1-theia-e3-official-3.json.1",
    "ta1-theia-e3-official-1r.json.2",
    "ta1-theia-e3-official-5m.json",
    "ta1-trace-e3-official.json.10",
    "ta1-trace-e3-official.json.2",
]

def get_shard_sort_key(filepath_str: str):
    name = filepath_str
    match = re.search(r'\.json(?:\.(\d+))?$', name)
    suffix_num = int(match.group(1)) if match and match.group(1) else 0
    base_name = re.sub(r'\.json.*$', '', name)
    group_match = re.search(r'-(\d+)[a-zA-Z]*$', base_name)
    group_num = int(group_match.group(1)) if group_match else 0
    return (group_num, base_name, suffix_num)

sorted_names = sorted(names, key=get_shard_sort_key)
for n in sorted_names:
    print(n)
