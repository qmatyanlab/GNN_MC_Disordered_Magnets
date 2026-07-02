import yaml

def generate_one_hot_dict(n: int) -> dict[int, list[int]]:
    """
    Generate one-hot embeddings for integers 1..n.

    Returns:
        {i: one_hot_vector} with length n
    """
    one_hot = {}
    for i in range(1, n + 1):
        vec = [0] * n
        vec[i - 1] = 1
        one_hot[i] = vec
    return one_hot

def write_one_hot_to_yaml(
        filename: str,
        max_key: int = 150,
):
    one_hot_dict = generate_one_hot_dict(max_key)

    with open(filename, "w") as f:
        yaml.safe_dump(
            one_hot_dict,
            f,
            default_flow_style=False,
            sort_keys=True,
        )

def write_atomic_number_to_yaml(
        filename: str,
        max_key: int = 150,
):
    atomic_number_dict = {n: n for n in range(1, max_key + 1)}
    with open(filename, "w") as f:
        yaml.safe_dump(
            atomic_number_dict,
            f,
            default_flow_style=False,
            sort_keys=True,
        )

if __name__ == "__main__":
    max_key = 150
    write_one_hot_to_yaml("one_hot.yaml", max_key=max_key)
    write_atomic_number_to_yaml("atomic_number.yaml", max_key=max_key)