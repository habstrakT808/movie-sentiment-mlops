#!/usr/bin/env python3
"""
Generate sample feedback for testing continuous learning.

This script adds feedback to existing predictions in the database.
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.config import DATABASE_PATH  # noqa: E402


def generate_sample_feedback(
    positive_count: int = 5,
    negative_count: int = 5,
    random_feedback: bool = False,
    total_count: int = None,
):
    """
    Generate sample feedback for testing.

    Args:
        positive_count: Number of positive feedback (feedback = 1)
        negative_count: Number of negative feedback (feedback = -1)
        random_feedback: If True, randomly assign feedback to predictions
        total_count: Total number of feedback to generate (if random_feedback=True)
    """
    db_path = DATABASE_PATH

    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        print("   Please make sure the database exists or create predictions first.")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        if total_predictions == 0:
            print("❌ No predictions found in database.")
            print("   Please create some predictions first via API or dashboard.")
            conn.close()
            return

        print(f"📊 Total predictions in database: {total_predictions}")

        if random_feedback and total_count:
            # Random feedback assignment
            import random

            cursor.execute(
                "SELECT id FROM predictions WHERE feedback IS NULL LIMIT ?",
                (total_count,),
            )
            ids = [row[0] for row in cursor.fetchall()]

            if len(ids) < total_count:
                print(
                    f"⚠️ Only {len(ids)} predictions available (requested: {total_count})"
                )

            for pred_id in ids:
                feedback = random.choice([1, -1])  # Random positive/negative
                cursor.execute(
                    "UPDATE predictions SET feedback = ? WHERE id = ?",
                    (feedback, pred_id),
                )

            conn.commit()
            print(f"✅ Added random feedback to {len(ids)} predictions")

        else:
            # Specific positive/negative feedback
            # Get predictions without feedback
            cursor.execute(
                "SELECT id FROM predictions WHERE feedback IS NULL ORDER BY id LIMIT ?",
                (positive_count + negative_count,),
            )
            ids = [row[0] for row in cursor.fetchall()]

            if len(ids) < (positive_count + negative_count):
                print(
                    f"⚠️ Only {len(ids)} predictions available without feedback "
                    f"(requested: {positive_count + negative_count})"
                )
                # Adjust counts
                available = len(ids)
                positive_count = min(positive_count, available)
                negative_count = available - positive_count

            # Add positive feedback
            positive_ids = ids[:positive_count]
            if positive_ids:
                placeholders = ",".join("?" * len(positive_ids))
                cursor.execute(
                    f"UPDATE predictions SET feedback = 1 WHERE id IN ({placeholders})",
                    positive_ids,
                )
                print(
                    f"✅ Added positive feedback (1) to {len(positive_ids)} predictions"
                )

            # Add negative feedback
            negative_ids = ids[positive_count : positive_count + negative_count]
            if negative_ids:
                placeholders = ",".join("?" * len(negative_ids))
                cursor.execute(
                    f"UPDATE predictions SET feedback = -1 WHERE id IN ({placeholders})",
                    negative_ids,
                )
                print(
                    f"✅ Added negative feedback (-1) to {len(negative_ids)} predictions"
                )

            conn.commit()

        # Check feedback statistics
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE feedback IS NOT NULL")
        total_feedback = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE feedback = 1")
        positive_feedback = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE feedback = -1")
        negative_feedback = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE feedback IS NOT NULL AND used_for_training = 0"
        )
        unused_feedback = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE feedback IS NOT NULL AND used_for_training = 1"
        )
        used_feedback = cursor.fetchone()[0]

        print("\n📊 Feedback Statistics:")
        print(f"   Total feedback: {total_feedback}")
        print(f"   Positive (1): {positive_feedback}")
        print(f"   Negative (-1): {negative_feedback}")
        print(f"   Unused for training: {unused_feedback}")
        print(f"   Used for training: {used_feedback}")

        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        if conn:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate sample feedback for testing continuous learning"
    )
    parser.add_argument(
        "--positive",
        type=int,
        default=5,
        help="Number of positive feedback (default: 5)",
    )
    parser.add_argument(
        "--negative",
        type=int,
        default=5,
        help="Number of negative feedback (default: 5)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly assign feedback",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Total number of feedback to generate (for --random)",
    )

    args = parser.parse_args()

    if args.random:
        if not args.count:
            print("❌ --count is required when using --random")
            sys.exit(1)
        generate_sample_feedback(random_feedback=True, total_count=args.count)
    else:
        generate_sample_feedback(
            positive_count=args.positive, negative_count=args.negative
        )
