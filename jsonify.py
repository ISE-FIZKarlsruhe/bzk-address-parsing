import json
import sys
from pathlib import Path

in_path = Path(sys.argv[1])
out_path = in_path.with_suffix("").with_suffix(".json")

objects = []

with open(in_path, "r") as in_f:
    for line in in_f:
        obj = json.loads(line)
        objects.append(obj)

with open(out_path, "w") as out_f:
    json.dump(objects, out_f, indent=4)

