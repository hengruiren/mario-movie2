import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2"))
from mario_llm_stage2.subset import make_manifest, save_manifest


def main():
    parser = argparse.ArgumentParser(description="Create or verify fixed Stage-2 Movies training targets")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset-seed", type=int, default=42)
    args = parser.parse_args()
    import dgl

    directory = Path(args.data_root).resolve() / "Movies"
    graphs, _ = dgl.load_graphs(str(directory / "MoviesGraph.pt"))
    graph = graphs[0]
    labels = graph.ndata["label"].cpu().numpy().reshape(-1)
    with (directory / "Movies.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != graph.num_nodes():
        raise ValueError("CSV and graph node counts differ")
    for i, row in enumerate(rows):
        if int(row["id"]) != i or int(row["label"]) != int(labels[i]):
            raise ValueError("CSV row, node ID and graph label do not align at row {}".format(i))
    manifest = make_manifest(labels, directory, args.subset_seed)
    save_manifest(args.output, manifest)
    print("Manifest: {}".format(Path(args.output).resolve()))
    print("Stage 1 targets: {}; Stage 2 targets: {}; validation: {}; test: {}".format(
        len(manifest["splits"]["train"]), len(manifest["targets"]["train"]),
        len(manifest["splits"]["val"]), len(manifest["splits"]["test"])))


if __name__ == "__main__":
    main()
