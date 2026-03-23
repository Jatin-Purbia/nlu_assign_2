import os
import sys
import argparse
import traceback

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_CORPUS_PATH = os.path.join(SCRIPT_DIR, "data", "raw_corpus.txt")
sys.path.insert(0, SCRIPT_DIR)


def step(title):
    print(f"\n{'#'*64}\n  {title}\n{'#'*64}")


def run_step1_scrape(skip, no_pdf=False):
    step("STEP 1 -- Data Collection (Web Scraping + PDF Download)")
    if skip or os.path.exists(RAW_CORPUS_PATH):
        print(f"  Raw corpus exists at {RAW_CORPUS_PATH}. Skipping.")
        return
    from scraper import IITJScraper
    scraper = IITJScraper(download_pdfs=not no_pdf)
    scraper.crawl()
    scraper.save()


def run_step2_preprocess():
    step("STEP 2 -- Preprocessing, Statistics & Word Cloud")
    from preprocessor import (
        build_processed_corpus, save_processed_corpus,
        report_statistics, generate_wordcloud,
        RAW_CORPUS, PROCESSED_FILE, WORDCLOUD_FILE,
    )
    sentences = build_processed_corpus(RAW_CORPUS)
    save_processed_corpus(sentences, PROCESSED_FILE)
    report_statistics(sentences)
    generate_wordcloud(sentences, WORDCLOUD_FILE)


def run_step3_train():
    step("STEP 3 -- Word2Vec Training (CBOW & Skip-gram, from scratch)")
    from train_word2vec import (
        load_corpus, run_experiments, save_models,
        print_results_table, compare_with_pretrained,
        PROCESSED_FILE, PROBE_WORDS,
    )
    sentences    = load_corpus(PROCESSED_FILE)
    results_dict = run_experiments(sentences)
    print_results_table(results_dict["results_table"])
    save_models(results_dict["best_cbow"], results_dict["best_skipgram"])
    compare_with_pretrained(results_dict["best_cbow"], results_dict["best_skipgram"], PROBE_WORDS)


def run_step4_semantic():
    step("STEP 4 -- Semantic Analysis (Nearest Neighbours & Analogies)")
    from train_word2vec    import Word2VecResult, BEST_CBOW_PATH, BEST_SG_PATH
    from semantic_analysis import run_analysis, semantic_discussion, RESULTS_FILE

    cbow_result = Word2VecResult.load(BEST_CBOW_PATH)
    sg_result   = Word2VecResult.load(BEST_SG_PATH)

    full_report  = run_analysis(cbow_result, "CBOW (from scratch)")
    full_report += run_analysis(sg_result,   "Skip-gram (from scratch)")
    full_report += semantic_discussion({})

    print(full_report)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"  Results saved: {RESULTS_FILE}")


def run_step5_visualize():
    step("STEP 5 -- Embedding Visualisation (PCA & t-SNE)")
    from train_word2vec import Word2VecResult, BEST_CBOW_PATH, BEST_SG_PATH
    from visualize import visualize_model, plot_side_by_side_comparison

    cbow_result = Word2VecResult.load(BEST_CBOW_PATH)
    sg_result   = Word2VecResult.load(BEST_SG_PATH)

    visualize_model(cbow_result, "CBOW")
    visualize_model(sg_result,   "Skip-gram")
    plot_side_by_side_comparison(cbow_result, sg_result, method="pca")
    plot_side_by_side_comparison(cbow_result, sg_result, method="tsne")


def main():
    parser = argparse.ArgumentParser(description="Problem 1 pipeline.")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--no-pdf",      action="store_true")
    args = parser.parse_args()

    steps = [
        ("Scrape",     lambda: run_step1_scrape(args.skip_scrape, args.no_pdf)),
        ("Preprocess", run_step2_preprocess),
        ("Train",      run_step3_train),
        ("Semantic",   run_step4_semantic),
        ("Visualize",  run_step5_visualize),
    ]

    failed = []
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:
            print(f"\n  Step '{name}' failed: {exc}")
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'#'*64}")
    if failed:
        print(f"  Pipeline complete with errors in: {failed}")
    else:
        print("  Problem 1 pipeline completed successfully!")
    print(f"{'#'*64}\n")


if __name__ == "__main__":
    main()
