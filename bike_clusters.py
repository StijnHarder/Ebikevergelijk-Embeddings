import os
import uuid
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm
import json
from collections import Counter

class BikeClusterUpdater:
    def __init__(self, similarity_threshold: float = 0.96):
        load_dotenv()
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.similarity_threshold = similarity_threshold
        
        # Internal data structures
        self.bike_ids = []
        self.embeddings = []
        self.bike_webshop_map = {}
        self.clusters = {}  # cluster_id -> {"representative": embedding, "webshops": set}
        self.bike_to_cluster = {}  # bike_id -> cluster_id

    def load_data(self, only_new: bool = False):
        """
        Load embeddings and webshop info.
        If only_new=True, only load bikes not yet clustered.
        """
        resp_vectors = self.supabase.table("vector_storage").select("bike_id, embedding").execute()
        vector_data = resp_vectors.data

        resp_bikes = self.supabase.table("scraped_bikes").select("id, webshop").execute()
        self.bike_webshop_map = {row["id"]: row["webshop"] for row in resp_bikes.data}

        if only_new:
            resp_master = self.supabase.table("master_bikes").select("bike_id").execute()
            clustered_ids = {row["bike_id"] for row in resp_master.data}
            vector_data = [row for row in vector_data if row["bike_id"] not in clustered_ids]

        self.bike_ids = []
        self.embeddings = []
        for row in tqdm(vector_data, desc="Processing embeddings"):
            self.bike_ids.append(row["bike_id"])
            emb_list = json.loads(row["embedding"])
            self.embeddings.append(np.array(emb_list, dtype=np.float32))
        
        if self.embeddings:
            self.embeddings = np.stack(self.embeddings)
        else:
            self.embeddings = np.array([])

        print(f"Loaded {len(self.bike_ids)} bikes for clustering.")

    def cluster_bikes(self):
        if not len(self.bike_ids):
            print("No bikes to cluster.")
            return

        self.clusters = {}
        self.bike_to_cluster = {}

        # baseline_webshop = list(set(self.bike_webshop_map.values()))[0]
        
        # baseline webshop logic
        PREFERRED_BASELINE = "fietsvoordeelshop.nl"

        all_webshops = set(self.bike_webshop_map.values())

        if PREFERRED_BASELINE in all_webshops:
            baseline_webshop = PREFERRED_BASELINE
        else:
            raise ValueError(f'{PREFERRED_BASELINE} is not in all_webshops: {','.join(all_webshops)}')
        # baseline webshop logic

        baseline_indices = [i for i, bid in enumerate(self.bike_ids) if self.bike_webshop_map[bid] == baseline_webshop]

        print(f"Using {baseline_webshop} as baseline webshop.")

        for idx in tqdm(baseline_indices, desc="Clustering baseline webshop"):
            bike_id = self.bike_ids[idx]
            cluster_id = str(uuid.uuid4())
            self.clusters[cluster_id] = {
                "representative": self.embeddings[idx],
                "webshops": {baseline_webshop}
            }
            self.bike_to_cluster[bike_id] = cluster_id

        other_indices = [i for i, bid in enumerate(self.bike_ids) if self.bike_webshop_map[bid] != baseline_webshop]

        for idx in tqdm(other_indices, desc="Clustering other webshops"):
            bike_id = self.bike_ids[idx]
            bike_emb = self.embeddings[idx]
            bike_webshop = self.bike_webshop_map[bike_id]

            best_cluster = None
            best_similarity = -1

            for cid, info in self.clusters.items():
                if bike_webshop in info["webshops"]:
                    continue
                cluster_emb = info["representative"]
                sim = np.dot(bike_emb, cluster_emb) / (np.linalg.norm(bike_emb) * np.linalg.norm(cluster_emb))
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster = cid

            if best_similarity >= self.similarity_threshold:
                self.bike_to_cluster[bike_id] = best_cluster
                self.clusters[best_cluster]["webshops"].add(bike_webshop)
            else:
                new_cluster_id = str(uuid.uuid4())
                self.clusters[new_cluster_id] = {
                    "representative": bike_emb,
                    "webshops": {bike_webshop}
                }
                self.bike_to_cluster[bike_id] = new_cluster_id

        print(f"Total clusters created/updated: {len(self.clusters)}")

    def write_clusters(self, batch_size: int = 500):
        if not self.bike_to_cluster:
            print("No clusters to write.")
            return

        insert_rows = [{"cluster_id": cid, "bike_id": bid} for bid, cid in self.bike_to_cluster.items()]

        for i in tqdm(range(0, len(insert_rows), batch_size), desc="Inserting clusters"):
            try:
                batch = insert_rows[i:i+batch_size]
                # self.supabase.table("master_bikes").insert(batch, upsert=True).execute()
                self.supabase.table("master_bikes").upsert(batch, on_conflict="bike_id").execute()
            except Exception as e:
                print(f'Upserting cluster failed : {e}')
                continue

        print("Clusters successfully inserted/updated in master_bikes.")

    def validate_clusters(self, top_n: int = 10):
        resp = self.supabase.table("master_bikes").select("cluster_id, bike_id").execute()
        all_rows = resp.data
        cluster_counts = Counter(row["cluster_id"] for row in all_rows)

        print(f"Top {top_n} largest clusters:")
        for cluster_id, count in cluster_counts.most_common(top_n):
            print(f"Cluster {cluster_id}: {count} bikes")

if __name__ == "__main__":
    clusterer = BikeClusterUpdater(similarity_threshold=0.95)
    
    # Full baseline backfill
    clusterer.load_data(only_new=False)
    clusterer.cluster_bikes()
    clusterer.write_clusters()
    clusterer.validate_clusters()
