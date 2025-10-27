import os

def print_file_tree(start_path, indent=""):
    try:
        items = sorted(os.listdir(start_path))
    except Exception:
        return

    for item in items:
        item_path = os.path.join(start_path, item)

        # 제외 대상
        if item in ["dataset", ".git", "outputs", "__pycache__", "assets", "build", "logs"]:
            continue

        if os.path.isdir(item_path):
            print(f"{indent}[{item}]")

            if item == "thirdparty":
                try:
                    sub_items = sorted(os.listdir(item_path))
                except Exception:
                    sub_items = []
                for sub in sub_items:
                    sub_path = os.path.join(item_path, sub)
                    if os.path.isdir(sub_path):
                        print(f"{indent}  [{sub}]")
                continue 

            print_file_tree(item_path, indent + "  ")
        else:
            print(f"{indent}{item}")

current_directory = "."
print_file_tree(current_directory)
