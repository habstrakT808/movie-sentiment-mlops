"""
Load testing for the API.
"""

import asyncio
import time
from typing import Dict

import aiohttp


async def make_request(session: aiohttp.ClientSession, url: str, data: Dict) -> Dict:
    """Make async HTTP request."""
    async with session.post(url, json=data) as response:
        return {
            "status": response.status,
            "time": time.time(),
            "data": await response.json(),
        }


async def load_test_single_predictions(base_url: str, num_requests: int = 100):
    """Load test single predictions."""
    print(f"🔥 Load testing single predictions ({num_requests} requests)...")

    test_texts = [
        "Amazing movie with great acting!",
        "Terrible film, waste of time.",
        "It was okay, nothing special.",
        "Brilliant cinematography!",
        "Boring and predictable.",
    ]

    url = f"{base_url}/predict"

    async with aiohttp.ClientSession() as session:
        start_time = time.time()

        # Create tasks
        tasks = []
        for i in range(num_requests):
            text = test_texts[i % len(test_texts)]
            data = {"text": text}
            tasks.append(make_request(session, url, data))

        # Execute all requests
        results = await asyncio.gather(*tasks)

        end_time = time.time()

    # Analyze results
    total_time = end_time - start_time
    successful_requests = sum(1 for r in results if r["status"] == 200)
    failed_requests = num_requests - successful_requests

    print(f"✅ Total time: {total_time:.2f}s")
    print(f"✅ Requests per second: {num_requests / total_time:.2f}")
    print(f"✅ Average time per request: {total_time / num_requests:.3f}s")
    print(f"✅ Successful requests: {successful_requests}")
    print(f"❌ Failed requests: {failed_requests}")

    return {
        "total_time": total_time,
        "rps": num_requests / total_time,
        "avg_time": total_time / num_requests,
        "success_rate": successful_requests / num_requests,
    }


async def load_test_batch_predictions(base_url: str, num_batches: int = 20):
    """Load test batch predictions."""
    print(f"🔥 Load testing batch predictions ({num_batches} batches)...")

    batch_texts = [
        "Amazing movie with incredible acting and storyline!",
        "Terrible film, complete waste of time and money.",
        "It was okay, nothing too special but watchable.",
        "Brilliant cinematography and outstanding performances!",
        "Boring and predictable plot with poor character development.",
    ]

    url = f"{base_url}/predict/batch"

    async with aiohttp.ClientSession() as session:
        start_time = time.time()

        # Create tasks
        tasks = []
        for _ in range(num_batches):
            data = {"texts": batch_texts}
            tasks.append(make_request(session, url, data))

        # Execute all requests
        results = await asyncio.gather(*tasks)

        end_time = time.time()

    # Analyze results
    total_time = end_time - start_time
    successful_requests = sum(1 for r in results if r["status"] == 200)
    total_predictions = num_batches * len(batch_texts)

    print(f"✅ Total time: {total_time:.2f}s")
    print(f"✅ Batches per second: {num_batches / total_time:.2f}")
    print(f"✅ Predictions per second: {total_predictions / total_time:.2f}")
    print(f"✅ Successful batches: {successful_requests}")

    return {
        "total_time": total_time,
        "batches_per_second": num_batches / total_time,
        "predictions_per_second": total_predictions / total_time,
        "success_rate": successful_requests / num_batches,
    }


async def main():
    """Run load tests."""
    base_url = "http://localhost:8000"

    print("🚀 Starting load tests...")
    print("=" * 60)

    # Test single predictions
    single_results = await load_test_single_predictions(base_url, 50)

    print("\n" + "=" * 60)

    # Test batch predictions
    batch_results = await load_test_batch_predictions(base_url, 10)

    print("\n" + "=" * 60)
    print("📊 LOAD TEST SUMMARY")
    print("=" * 60)
    print("Single Predictions:")
    print(f"  - RPS: {single_results['rps']:.2f}")
    print(f"  - Avg Time: {single_results['avg_time']:.3f}s")
    print(f"  - Success Rate: {single_results['success_rate']:.1%}")

    print("\nBatch Predictions:")
    print(f"  - Predictions/sec: {batch_results['predictions_per_second']:.2f}")
    print(f"  - Success Rate: {batch_results['success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
