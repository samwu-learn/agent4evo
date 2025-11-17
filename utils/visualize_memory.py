#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_memory(mem_path: Path) -> List[Dict[str, Any]]:
    if not mem_path.exists():
        raise FileNotFoundError(f"memory.json not found: {mem_path}")
    with mem_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("memory.json format invalid: expected a list of entries")
    return data


def extract_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for it in items:
        meta = (it or {}).get("metadata") or {}

        rec = {
            "is_success": bool(meta.get("is_success")),
            "embedding": it.get("embedding"),
            "n_retrieves": meta.get("n_retrieves"),
            "score": meta.get("score"),
        }
        records.append(rec)
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_retrieves_list = [r.get("n_retrieves") for r in records if isinstance(r.get("n_retrieves"), (int, float))]
    scores = [r.get("score") for r in records if isinstance(r.get("score"), (int, float))]
    embeddings = [r.get("embedding") for r in records if r.get("embedding") is not None and isinstance(r.get("embedding"), list)]
    is_success_list = [r.get("is_success") for r in records]

    return {
        "n_retrieves": n_retrieves_list,
        "scores": scores,
        "embeddings": embeddings,
        "is_success_list": is_success_list,
    }


def try_plot(output_dir: Path, summary: Dict[str, Any], records: List[Dict[str, Any]]):
    """Create charts with matplotlib if available; return list of image filenames."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return []

    imgs: List[str] = []

    # is_success distribution (pie chart)
    is_success_list = summary.get("is_success_list") or []
    if is_success_list:
        success_count = sum(is_success_list)
        fail_count = len(is_success_list) - success_count
        plt.figure(figsize=(8, 6))
        colors = ['#4C78A8', '#E45756']
        plt.pie([success_count, fail_count], labels=['Success', 'Failed'],
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title("is_success Distribution")
        plt.tight_layout()
        fn1 = "is_success_dist.png"
        plt.savefig(output_dir / fn1)
        imgs.append(fn1)
        plt.close()

    # n_retrieves distribution
    n_retrieves = summary.get("n_retrieves") or []
    if n_retrieves:
        plt.figure(figsize=(10, 4))
        plt.hist(n_retrieves, bins=max(20, int(max(n_retrieves)) + 1), color="#E15759", edgecolor="white")
        plt.xlabel("n_retrieves")
        plt.ylabel("Frequency")
        plt.title("n_retrieves Distribution")
        plt.tight_layout()
        fn2 = "n_retrieves_hist.png"
        plt.savefig(output_dir / fn2)
        imgs.append(fn2)
        plt.close()

    # score distribution
    scores = summary.get("scores") or []
    if scores:
        plt.figure(figsize=(10, 4))
        plt.hist(scores, bins=20, color="#FF9DA7", edgecolor="white")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Score Distribution")
        plt.tight_layout()
        fn3 = "score_hist.png"
        plt.savefig(output_dir / fn3)
        imgs.append(fn3)
        plt.close()

    # Embedding visualization with t-SNE
    embeddings = summary.get("embeddings") or []
    if embeddings and len(embeddings) > 1:
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA

            # Convert to numpy array
            X = np.array(embeddings)
            is_success_colors = [r.get("is_success") for r in records if r.get("embedding") is not None]

            # Use PCA first if high dimensional to speed up t-SNE
            if X.shape[1] > 50:
                pca = PCA(n_components=50)
                X_pca = pca.fit_transform(X)
            else:
                X_pca = X

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
            X_tsne = tsne.fit_transform(X_pca)

            # Create scatter plot colored by is_success
            plt.figure(figsize=(10, 8))
            colors_map = np.array(['#E15759' if not s else '#4C78A8' for s in is_success_colors])
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors_map, alpha=0.6, s=50)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#4C78A8', label='Success'),
                             Patch(facecolor='#E15759', label='Failed')]
            plt.legend(handles=legend_elements, loc='best')

            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.title("Embedding Visualization (t-SNE)")
            plt.tight_layout()
            fn4 = "embedding_tsne.png"
            plt.savefig(output_dir / fn4)
            imgs.append(fn4)
            plt.close()

            # Also create PCA visualization
            pca = PCA(n_components=2)
            X_pca_2d = pca.fit_transform(X)

            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=colors_map, alpha=0.6, s=50)
            plt.legend(handles=legend_elements, loc='best')
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            plt.title("Embedding Visualization (PCA)")
            plt.tight_layout()
            fn5 = "embedding_pca.png"
            plt.savefig(output_dir / fn5)
            imgs.append(fn5)
            plt.close()

        except Exception as e:
            print(f"Warning: Could not create embedding visualization: {e}")

    return imgs


def main():
    parser = argparse.ArgumentParser(description="Visualize memory.json into charts and HTML page.")
    parser.add_argument("--memory", type=str, default="/Users/pwzzy/Documents/Program/agentevo/Results/memory.json", help="Path to memory.json")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for HTML and images (default=memory.json's directory)")
    args = parser.parse_args()

    mem_path = Path(args.memory).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else mem_path.parent
    vis_dir = outdir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    items = load_memory(mem_path)
    records = extract_records(items)
    summary = summarize(records)
    images = try_plot(vis_dir, summary, records)
    for img in images:
        print(f"Chart: {vis_dir / img}")


if __name__ == "__main__":
    main()