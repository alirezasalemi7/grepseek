"""Download the Wikipedia-2018 corpus used by Search-R1 (~21M passages) and place
it as `wiki_corpus.jsonl` for the agent to search.

It fetches `wiki-18.jsonl.gz` (~5 GB) from the `PeterJinGo/wiki-18-corpus` dataset
repo and decompresses it to `<dest>/wiki_corpus.jsonl` (~14 GB). Each line is a
passage: {"id": ..., "contents": ...}.

Usage:
    python download_corpus.py --dest data/wiki_18
    # then point the generator at it:
    #   python create_data.py --corpus_dir data/wiki_18 ...   (or export CORPUS_DIR=data/wiki_18)

Notes:
  - Make sure --dest (and your HF cache, via HF_HOME) live on a roomy filesystem
    with ~20 GB free — NOT your home directory.
  - The download resumes if interrupted (huggingface_hub caching).
"""
import argparse
import gzip
import os
import shutil
import tarfile

from huggingface_hub import hf_hub_download

REPO = "PeterJinGo/wiki-18-corpus"
GZ_IN_REPO = "wiki-18.jsonl.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True,
                    help="directory to place wiki_corpus.jsonl in (needs ~20 GB free)")
    ap.add_argument("--keep_gz", action="store_true",
                    help="keep the downloaded .gz after decompressing")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, "wiki_corpus.jsonl")
    if os.path.exists(out):
        print(f"already present: {out}  ({os.path.getsize(out) / 1e9:.1f} GB) — nothing to do")
        return

    print(f"[1/2] downloading {GZ_IN_REPO} from {REPO} (~5 GB) ...")
    gz_local = hf_hub_download(repo_id=REPO, filename=GZ_IN_REPO,
                              repo_type="dataset", local_dir=dest)
    print(f"      downloaded -> {gz_local}")

    print("[2/2] extracting -> wiki_corpus.jsonl ...")
    tmp = out + ".part"
    # The published file is actually a gzipped *tar* whose member is the JSONL
    # (e.g. .../wiki_dump.jsonl). Extract that member; fall back to plain gzip
    # in case the format changes to a bare gzipped JSONL.
    extracted = False
    try:
        with tarfile.open(gz_local, "r:gz") as tf:
            m = tf.next()
            while m is not None and not m.name.endswith(".jsonl"):
                m = tf.next()
            if m is not None:
                with tf.extractfile(m) as src, open(tmp, "wb") as fout:
                    shutil.copyfileobj(src, fout, length=16 * 1024 * 1024)
                extracted = True
    except tarfile.ReadError:
        pass  # not a tar -> treat as plain gzip below
    if not extracted:
        with gzip.open(gz_local, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    os.replace(tmp, out)

    if not args.keep_gz:
        try:
            os.remove(gz_local)
        except OSError:
            pass
        shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)  # local_dir lock cache

    print(f"\nDone -> {out}  ({os.path.getsize(out) / 1e9:.1f} GB)")
    print(f"Use it with:  --corpus_dir {dest}   (or export CORPUS_DIR={dest})")


if __name__ == "__main__":
    main()
