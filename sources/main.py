from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def main():
    ds = load_dataset("mteb/stsbenchmark-sts")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    left = model.encode(ds["train"]["sentence1"])
    right = model.encode(ds["train"]["sentence2"])
    left.tofile("left.le")
    right.tofile("right.le")


if __name__ == "__main__":
    main()
