from locust import HttpUser, task, between
import random
import json

class SearchUser(HttpUser):
    wait_time = between(1, 3)  # Simulate user think time between 1-3 seconds

    # Generate a random query from a predefined set of example queries
    def get_random_query(self):
        queries = [
            "best camera",
            "waterproof phone under $500",
            "4k camera with image stabilization",
            "laptop with 16GB RAM",
            "smartphone with the best camera"
        ]
        return random.choice(queries)

    @task
    def search(self):
        query = self.get_random_query()  # Pick a random query
        response = self.client.post("/search", json={"query": query})  # Send POST request to /search
        assert response.status_code == 200  # Ensure the request is successful
        print(f"Query: {query}, Response Status: {response.status_code}")

