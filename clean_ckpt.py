import os
import re
import shutil
import argparse
from pathlib import Path

def delete_old_checkpoints(root_dir, dry_run=False):
    root = Path(root_dir)

    # 递归查找所有名为 'checkpoints' 的文件夹
    for checkpoints_dir in root.rglob('checkpoints'):
        if not checkpoints_dir.is_dir():
            continue

        print(f"Processing checkpoints directory: {checkpoints_dir}")

        # 找出所有符合 iters_XXXXXXX（或 iters_任意数字）格式的子目录
        iter_dirs = []
        for item in checkpoints_dir.iterdir():
            if item.is_dir() and re.match(r'^iter_\d+$', item.name):
                try:
                    num = int(item.name.split('_', 1)[1])  # 支持 iters_123, iters_0003000 等
                    iter_dirs.append((num, item))
                except (IndexError, ValueError):
                    continue  # 忽略格式异常的
        
        if len(iter_dirs) <= 1:
            print(f"  → No old checkpoints to delete (found {len(iter_dirs)}).")
            continue

        # 按数字排序，保留最大的
        iter_dirs.sort(key=lambda x: x[0])
        to_keep = iter_dirs[-1][1]
        to_delete = [d[1] for d in iter_dirs[:-1]]

        print(f"  → Keeping: {to_keep.name}")
        for d in to_delete:
            if dry_run:
                print(f"  → [DRY-RUN] Would delete: {d}")
            else:
                print(f"  → Deleting: {d}")
                shutil.rmtree(d)

    print("Cleanup finished.")

def main():
    parser = argparse.ArgumentParser(description="Delete old checkpoint folders, keeping only the latest iters_XXXXXX in each 'checkpoints' directory.")
    parser.add_argument("root_directory", help="Root directory to scan for checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting anything")

    args = parser.parse_args()

    root_directory = args.root_directory
    if not os.path.isdir(root_directory):
        print(f"Error: '{root_directory}' is not a valid directory.", file=os.sys.stderr)
        exit(1)

    delete_old_checkpoints(root_directory, dry_run=args.dry_run)

if __name__ == "__main__":
    main()