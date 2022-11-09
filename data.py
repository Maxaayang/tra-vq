from utils import *
import glob

for orig_file in glob.glob("./events/*.pkl"):
    out_file = orig_file.replace('/events/','/data/')
    events = pickle_load(orig_file)
    for event in events:
        if event["name"] == "Note_Velocity":
            event["value"] = min(max(40,event["value"]),80)
    bar_idx = []
    for idx, event in enumerate(events):
        if event["name"] == "Bar":
            bar_idx.append(idx)

    result = (bar_idx,events)
    pickle_dump(result,out_file)