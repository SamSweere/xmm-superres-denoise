import yaml


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


a = read_yaml('configs/dataset/xmm_dev_ff.yaml')
print(a)