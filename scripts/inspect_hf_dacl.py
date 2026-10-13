import requests
from collections import Counter
for repo in ["rajsiri/dacl10k", "jhoncrad/dacl10k", "smallopen1145141919810/dacl10k"]:
    url = f"https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true&limit=500"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(repo, "ERROR", exc)
        continue
    files = [item for item in data if item.get("type") == "file"]
    ext = Counter()
    for item in files:
        name = item["path"].lower()
        if name.endswith((".jpg",".jpeg")): ext["jpg"] += 1
        elif name.endswith(".png"): ext["png"] += 1
        elif name.endswith(".json"): ext["json"] += 1
        elif name.endswith(".txt"): ext["txt"] += 1
        else: ext["other"] += 1
    masks = [item["path"] for item in files if "mask" in item["path"].lower()][:5]
    labels = [item["path"] for item in files if "label" in item["path"].lower()][:5]
    print(repo, "shown", len(files), dict(ext), "masks", masks, "labels", labels)
