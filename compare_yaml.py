import yaml

def compare_yaml(file1_path, file2_path):
    def load_yaml(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def compare_dicts(d1, d2, path=''):
        diffs = []

        all_keys = set(d1.keys()) | set(d2.keys())
        for key in all_keys:
            full_path = f"{path}.{key}" if path else key

            if key not in d1:
                diffs.append(f"{full_path} only in second file: {d2[key]}")
            elif key not in d2:
                diffs.append(f"{full_path} only in first file: {d1[key]}")
            else:
                val1, val2 = d1[key], d2[key]
                if isinstance(val1, dict) and isinstance(val2, dict):
                    diffs.extend(compare_dicts(val1, val2, full_path))
                elif val1 != val2:
                    diffs.append(f"{full_path} differs: {val1} != {val2}")

        return diffs

    yaml1 = load_yaml(file1_path)
    yaml2 = load_yaml(file2_path)

    differences = compare_dicts(yaml1, yaml2)

    return differences

diffs = compare_yaml("/hdd0/ljn/new_sam2/my_fssam/ori_sam2_configs/sam2_hiera_s.yaml", "/hdd0/ljn/new_sam2/my_fssam/sam2_configs/sam2.1_hiera_s.yaml")
for diff in diffs:
    print(diff)