import argparse
import numpy as np
from  gensim.models.keyedvectors import FastTextKeyedVectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    embeddings = FastTextKeyedVectors.load(args.model)
    vectors = [np.zeros(300, dtype=np.float32)]
    vocab = ["*UNK*"]
    with open(f"{args.output_dir}/static_word_vocabulary.txt", "w") as f1, \
        open(f"{args.output_dir}/tuned_word_vocabulary.txt", "w") as f2:
        f1.write("*UNK*\n")
        f2.write("*UNK*\n")
        for key in embeddings.vocab.keys():
            vector = embeddings.get_vector(key)
            f.write(f"{key}\n")
            vectors.append(vector)
    np.save(f"{args.output_dir}/static_word_embeddings.npy", np.array(vectors))
    np.save(f"{args.output_dir}/tuned_word_embeddings.npy", np.array(vectors))
