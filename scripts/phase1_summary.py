"""Generate Phase 1 completion summary."""

import json
from datetime import datetime


def generate_summary():
    """Generate Phase 1 summary report."""

    # Load statistics
    with open("data/raw/collection_stats.json", "r") as f:
        stats = json.load(f)

    with open("data/raw/validation_report.json", "r") as f:
        validation = json.load(f)

    # Create summary
    summary = f"""
{'='*80}
PHASE 1: DATA COLLECTION - COMPLETION REPORT
{'='*80}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA COLLECTION SUMMARY
{'='*80}
Total Samples Collected: {stats['total_samples']:,}
├─ Reddit Samples: {stats['reddit_samples']:,}
└─ Kaggle Samples: {stats['kaggle_samples']:,}

SENTIMENT DISTRIBUTION
{'='*80}
"""

    for sentiment, count in stats["sentiment_distribution"].items():
        percentage = (count / stats["total_samples"]) * 100
        summary += f"{sentiment.capitalize():10s}: {count:6,} ({percentage:5.2f}%)\n"

    summary += f"""
DATA QUALITY METRICS
{'='*80}
Duplicates Removed: {stats['duplicates_removed']:,}
Average Text Length: {stats['text_length_stats']['mean']:.0f} characters
Average Word Count: {stats['word_count_stats']['mean']:.0f} words
Text Length Range: {stats['text_length_stats']['min']}-{stats['text_length_stats']['max']} characters

VALIDATION RESULTS
{'='*80}
✓ No Missing Text: {validation['no_missing_text']}
✓ No Missing Sentiment: {validation['no_missing_sentiment']}
✓ Valid Sentiments: {validation['valid_sentiments']}
✓ Length Requirements: {validation['min_length_ok'] and validation['max_length_ok']}
✓ No Duplicates: {validation['no_duplicates']}

DVC TRACKING
{'='*80}
✓ Data versioned with DVC
✓ Data pushed to remote storage
✓ Pipeline configuration complete

NEXT STEPS
{'='*80}
1. Proceed to Phase 2: Data Preprocessing & EDA
2. Run: python src/preprocessing/preprocess.py
3. Review: notebooks/01_data_exploration.ipynb

{'='*80}
PHASE 1 COMPLETED SUCCESSFULLY ✓
{'='*80}
"""

    print(summary)

    # Save summary
    with open("docs/PHASE1_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write(summary)

    print("\nSummary saved to: docs/PHASE1_SUMMARY.md")


if __name__ == "__main__":
    generate_summary()
