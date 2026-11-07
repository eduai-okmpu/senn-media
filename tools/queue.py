#!/usr/bin/env python3
"""Tiny helper for content/queue.json so nobody hand-edits JSON.

  queue.py next                      -> print the next item to publish (JSON) or exit 3 if none
  queue.py pending                   -> print number of pending items
  queue.py mark ID STATUS [MEDIA_ID] -> set status (pending|publishing|published|failed|skipped)
  queue.py add FILE.json             -> append items from a JSON list file (status forced to pending)
  queue.py show                      -> compact table
"""
import json, os, sys, datetime

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content", "queue.json")


def load():
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


def save(d):
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, d = sys.argv[1], load()
    items = d["items"]
    if cmd == "next":
        # a stuck 'publishing' item must be resolved by the runbook before anything else
        for it in items:
            if it["status"] == "publishing":
                print(json.dumps(it, ensure_ascii=False)); return
        for it in items:
            if it["status"] == "pending":
                print(json.dumps(it, ensure_ascii=False)); return
        sys.exit(3)
    elif cmd == "pending":
        print(sum(1 for it in items if it["status"] == "pending"))
    elif cmd == "mark":
        iid, status = sys.argv[2], sys.argv[3]
        assert status in ("pending", "publishing", "published", "failed", "skipped"), status
        for it in items:
            if it["id"] == iid:
                it["status"] = status
                it["updated_at"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
                if len(sys.argv) > 4:
                    it["media_id"] = sys.argv[4]
                save(d); print("ok", iid, status); return
        sys.exit("unknown id " + iid)
    elif cmd == "add":
        with open(sys.argv[2], encoding="utf-8") as f:
            new = json.load(f)
        have = {it["id"] for it in items}
        for it in new:
            assert it["id"] not in have, "duplicate id " + it["id"]
            assert 1 <= len(it["lines"]) <= 4 and it["palette"] in ("dusk", "dawn", "night", "mist")
            it["status"] = "pending"
            items.append(it)
        save(d); print("added", len(new))
    elif cmd == "show":
        for it in items:
            print(f'{it["id"]:<22} {it["status"]:<11} {it["palette"]:<6} {" / ".join(it["lines"])}')
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
