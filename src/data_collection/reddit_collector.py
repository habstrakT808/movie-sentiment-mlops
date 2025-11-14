"""
Reddit data collector for movie reviews and discussions.
Uses PRAW (Python Reddit API Wrapper) to collect data from movie-related subreddits.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import praw

from src.utils.config import Config
from src.utils.helpers import compute_hash, retry, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedditCollector:
    """Collector for Reddit movie reviews and discussions."""

    def __init__(self):
        """Initialize Reddit API client."""
        try:
            self.reddit = praw.Reddit(
                client_id=Config.REDDIT_CLIENT_ID,
                client_secret=Config.REDDIT_CLIENT_SECRET,
                user_agent=Config.REDDIT_USER_AGENT,
            )
            # Test connection
            self.reddit.user.me()
            logger.info("Successfully connected to Reddit API")
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            raise

        self.collected_ids = set()  # Track collected items to avoid duplicates

    @retry(max_attempts=3, delay=2, backoff=2)
    def search_movie_posts(
        self,
        movie_title: str,
        subreddit_name: str,
        limit: int = 100,
        time_filter: str = "all",
    ) -> List[Dict]:
        """
        Search for posts about a specific movie in a subreddit.

        Args:
            movie_title: Title of the movie to search for
            subreddit_name: Name of the subreddit
            limit: Maximum number of posts to retrieve
            time_filter: Time filter (all, year, month, week, day)

        Returns:
            List of post dictionaries
        """
        logger.info(
            f"Searching for '{movie_title}' in r/{subreddit_name} (limit={limit})"
        )

        posts = []
        subreddit = self.reddit.subreddit(subreddit_name)

        try:
            for submission in subreddit.search(
                movie_title, limit=limit, time_filter=time_filter
            ):
                # Skip if already collected
                if submission.id in self.collected_ids:
                    continue

                # Only collect posts with substantial content
                if len(submission.selftext) < 50:
                    continue

                post_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "text": submission.selftext,
                    "score": submission.score,
                    "upvote_ratio": submission.upvote_ratio,
                    "num_comments": submission.num_comments,
                    "created_utc": datetime.fromtimestamp(submission.created_utc),
                    "author": (
                        str(submission.author) if submission.author else "[deleted]"
                    ),
                    "subreddit": subreddit_name,
                    "url": f"https://reddit.com{submission.permalink}",
                    "movie_title": movie_title,
                    "source_type": "post",
                    "hash": compute_hash(submission.selftext),
                }

                posts.append(post_data)
                self.collected_ids.add(submission.id)

                # Rate limiting
                time.sleep(1)

            logger.info(
                f"Collected {len(posts)} posts for '{movie_title}' from r/{subreddit_name}"
            )
            return posts

        except Exception as e:
            logger.error(f"Error searching posts: {e}")
            return []

    @retry(max_attempts=3, delay=2, backoff=2)
    def get_post_comments(
        self, submission_id: str, max_comments: int = 20
    ) -> List[Dict]:
        """
        Get comments from a specific post.

        Args:
            submission_id: Reddit submission ID
            max_comments: Maximum number of comments to retrieve

        Returns:
            List of comment dictionaries
        """
        comments = []

        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)  # Remove "load more" comments

            for comment in submission.comments.list()[:max_comments]:
                # Skip if already collected or deleted
                if comment.id in self.collected_ids or comment.body == "[deleted]":
                    continue

                # Only collect comments with substantial content
                if len(comment.body) < 30:
                    continue

                comment_data = {
                    "id": comment.id,
                    "title": submission.title,
                    "text": comment.body,
                    "score": comment.score,
                    "upvote_ratio": None,  # Not available for comments
                    "num_comments": None,
                    "created_utc": datetime.fromtimestamp(comment.created_utc),
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "subreddit": submission.subreddit.display_name,
                    "url": f"https://reddit.com{comment.permalink}",
                    "movie_title": None,  # Will be filled later
                    "source_type": "comment",
                    "hash": compute_hash(comment.body),
                }

                comments.append(comment_data)
                self.collected_ids.add(comment.id)

            return comments

        except Exception as e:
            logger.error(f"Error getting comments: {e}")
            return []

    def infer_sentiment_from_score(
        self, score: int, upvote_ratio: Optional[float] = None
    ) -> str:
        """
        Infer sentiment label from Reddit score and upvote ratio.

        Args:
            score: Reddit score (upvotes - downvotes)
            upvote_ratio: Ratio of upvotes (0-1)

        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        # Use upvote ratio if available
        if upvote_ratio is not None:
            if upvote_ratio >= 0.7:
                return "positive"
            elif upvote_ratio <= 0.4:
                return "negative"
            else:
                return "neutral"

        # Fallback to score
        if score >= 10:
            return "positive"
        elif score <= -5:
            return "negative"
        else:
            return "neutral"

    @timer
    def collect_movie_data(
        self,
        movie_title: str,
        subreddits: List[str],
        posts_per_subreddit: int = 50,
        comments_per_post: int = 20,
    ) -> pd.DataFrame:
        """
        Collect comprehensive data for a specific movie.

        Args:
            movie_title: Title of the movie
            subreddits: List of subreddit names
            posts_per_subreddit: Number of posts to collect per subreddit
            comments_per_post: Number of comments to collect per post

        Returns:
            DataFrame with collected data
        """
        all_data = []

        for subreddit in subreddits:
            # Collect posts
            posts = self.search_movie_posts(
                movie_title=movie_title,
                subreddit_name=subreddit,
                limit=posts_per_subreddit,
            )

            all_data.extend(posts)

            # Collect comments from each post
            for post in posts[:10]:  # Limit to first 10 posts to avoid rate limits
                comments = self.get_post_comments(
                    submission_id=post["id"], max_comments=comments_per_post
                )

                # Add movie title to comments
                for comment in comments:
                    comment["movie_title"] = movie_title

                all_data.extend(comments)

                # Rate limiting between posts
                time.sleep(2)

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        if len(df) > 0:
            # Infer sentiment labels
            df["sentiment"] = df.apply(
                lambda row: self.infer_sentiment_from_score(
                    row["score"], row.get("upvote_ratio")
                ),
                axis=1,
            )

            # Add metadata
            df["collection_date"] = datetime.now()
            df["text_length"] = df["text"].str.len()
            df["word_count"] = df["text"].str.split().str.len()

            logger.info(
                f"Collected {len(df)} items for '{movie_title}' "
                f"({df['sentiment'].value_counts().to_dict()})"
            )

        return df

    def _collect_sentiment_data(
        self,
        movies: List[str],
        sentiment: str,
        subreddits: List[str],
        target_count: int,
        all_data: list,
        sentiment_counts: dict,
    ) -> None:
        """Collect data for a specific sentiment."""
        logger.info(f"Collecting {sentiment} reviews...")
        for movie in movies:
            if sentiment_counts[sentiment] >= target_count:
                break

            df = self.collect_movie_data(
                movie_title=movie,
                subreddits=subreddits,
                posts_per_subreddit=100,
                comments_per_post=20,
            )

            if len(df) > 0:
                filtered_df = df[df["sentiment"] == sentiment]
                all_data.append(filtered_df)
                sentiment_counts[sentiment] += len(filtered_df)
                logger.info(
                    f"{sentiment.capitalize()}: {sentiment_counts[sentiment]}/{target_count}"
                )

    @timer
    def collect_balanced_dataset(
        self,
        positive_movies: List[str],
        negative_movies: List[str],
        neutral_movies: List[str],
        subreddits: List[str],
        target_per_sentiment: int = 10000,
    ) -> pd.DataFrame:
        """
        Collect a balanced dataset across different sentiments.

        Args:
            positive_movies: List of movies expected to have positive reviews
            negative_movies: List of movies expected to have negative reviews
            neutral_movies: List of movies expected to have neutral/mixed reviews
            subreddits: List of subreddit names
            target_per_sentiment: Target number of samples per sentiment

        Returns:
            Balanced DataFrame
        """
        logger.info("Starting balanced dataset collection...")

        all_data = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        # Collect positive reviews
        self._collect_sentiment_data(
            positive_movies,
            "positive",
            subreddits,
            target_per_sentiment,
            all_data,
            sentiment_counts,
        )

        # Collect negative reviews
        self._collect_sentiment_data(
            negative_movies,
            "negative",
            subreddits,
            target_per_sentiment,
            all_data,
            sentiment_counts,
        )

        # Collect neutral reviews
        self._collect_sentiment_data(
            neutral_movies,
            "neutral",
            subreddits,
            target_per_sentiment,
            all_data,
            sentiment_counts,
        )

        # Combine all data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)

            # Remove duplicates based on hash
            final_df = final_df.drop_duplicates(subset=["hash"])

            # Balance the dataset
            min_count = min(sentiment_counts.values())
            balanced_dfs = []

            for sentiment in ["positive", "negative", "neutral"]:
                sentiment_df = final_df[final_df["sentiment"] == sentiment]
                if len(sentiment_df) > min_count:
                    sentiment_df = sentiment_df.sample(n=min_count, random_state=42)
                balanced_dfs.append(sentiment_df)

            final_df = pd.concat(balanced_dfs, ignore_index=True)

            # Shuffle
            final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

            logger.info(f"Final dataset: {len(final_df)} samples")
            logger.info(
                f"Sentiment distribution:\n{final_df['sentiment'].value_counts()}"
            )

            return final_df
        else:
            logger.warning("No data collected!")
            return pd.DataFrame()

    def save_collection_stats(self, df: pd.DataFrame, output_path: Path):
        """
        Save collection statistics.

        Args:
            df: Collected DataFrame
            output_path: Path to save statistics
        """
        stats = {
            "total_samples": len(df),
            "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
            "source_type_distribution": df["source_type"].value_counts().to_dict(),
            "subreddit_distribution": df["subreddit"].value_counts().to_dict(),
            "avg_text_length": df["text_length"].mean(),
            "avg_word_count": df["word_count"].mean(),
            "avg_score": df["score"].mean(),
            "date_range": {
                "start": df["created_utc"].min().isoformat(),
                "end": df["created_utc"].max().isoformat(),
            },
            "collection_date": datetime.now().isoformat(),
        }

        import json

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Collection statistics saved to {output_path}")


def main():
    """Main function for testing."""
    collector = RedditCollector()

    # Test with a single movie
    df = collector.collect_movie_data(
        movie_title="Inception",
        subreddits=["movies"],
        posts_per_subreddit=10,
        comments_per_post=5,
    )

    print(f"\nCollected {len(df)} samples")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nSample data:\n{df.head()}")


if __name__ == "__main__":
    main()
