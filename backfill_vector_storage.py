import os
from supabase import create_client
from dotenv import load_dotenv
from pe_core_meta import PeCoreModelLoader
from tqdm import tqdm
from pprint import pprint

load_dotenv()

class VectorBackfill:
    def __init__(self):
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        self.supabase = create_client(url, key)
        self.model = PeCoreModelLoader()

    def get_missing_bike_ids(self) -> list[str]:
        """Return IDs of scraped_bikes that do not exist in vector_storage."""

        scraped_ids = (
            self.supabase.table("scraped_bikes")
            .select("id")
            .neq("image_url", None)
            .execute()
            .data
        )

        vector_ids = (
            self.supabase.table("vector_storage")
            .select("bike_id")
            .execute()
            .data
        )

        scraped_set = [row["id"] for row in scraped_ids]
        vector_set = [row["bike_id"] for row in vector_ids]

        missing_ids = set(scraped_set) - set(vector_set)

        return list(missing_ids)

    def get_missing_bikes(self) -> list[dict]:
        """Return full scraped_bikes rows for bikes missing embeddings."""
        missing_ids = self.get_missing_bike_ids()

        if not missing_ids:
            print(f'No missing bike IDs found.')
            return

        # Fetch FULL records only for missing IDs
        data = (
            self.supabase.table("scraped_bikes")
            .select("id, name, image_url")
            .in_("id", missing_ids)
            .execute()
            .data
        )

        return data

    def backfill_vector_storage(self):
        bikes = self.get_missing_bikes()

        if not bikes:
            print("No missing bikes found, all bikes have embeddings.")
            return
        
        print(f'Found {len(bikes)} bikes missing embeddings. Starting backfill...')

        for row in tqdm(bikes):
            title = row.get('name')
            img = row.get('image_url')

            if not img or not title:
                continue

            embedding = self.model.generate_joint_embedding(title, img)
            if embedding is None:
                continue

            # print(f'Length embedding: {len(embedding)}')

            self.supabase.table("vector_storage").insert({
                "bike_id": row["id"],
                "embedding": embedding
            }).execute()

            # break  # remove this break to process all rows

        print("Phase 1 complete.")

if __name__ == "__main__":
    backfill = VectorBackfill()
    # pprint(backfill.get_missing_bikes())
    backfill.backfill_vector_storage()