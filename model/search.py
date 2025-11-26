import faiss
import numpy as np
import json
import re

from dotenv import load_dotenv
from const import get_embedding_model
from datetime import datetime

load_dotenv()

class IndexSearch():
    
    def __init__(self, index_path, metadata_path):
        
        self.index = faiss.read_index(index_path)
        self.model = get_embedding_model()
        self.metadata = self.load_metadata(metadata_path)
        
    def load_metadata(self, metadata_path: str) -> dict:
        """
        Loads a metadata json into a python dictionary.
        
        Args:
            metadata_path (str): The path to the metadata json file.
        
        Returns:
            metadata (dict): The metadata file as a python dict.
        
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        return metadata
    
    def search_index(self, query: str, constraints: dict,  top_k: int = 10) -> list:
        """
        Performs a semantic similarity search on the index. Filters output by temporal constraints.
        
        Args:
            metadata_path (str): The path to the metadata json file.
            constraints (dict): A dictionary of constraints, keys are 'after', 'before', 'on'.
        
        Returns:
            filtered_docs (list): The list containing the top_k matches, ordered by similarity, and filtered by constraints.
        """
        model = self.model
        
        embeddings = model.encode(query, convert_to_numpy=True, normalize_embeddings=True,)
        embeddings = np.array([embeddings]).astype("float32")
        
        _, ids = self.index.search(embeddings, top_k)
        
        filtered_docs = []
        retrieved_docs = [self.metadata[i] for i in ids[0]]

        for doc in retrieved_docs:
            cand_ts = doc["timestamp"]
            cand_dt, _ = self.parse_date_auto(cand_ts)

            keep = True

            for constraint_type in constraints:

                constraint_str = constraints[constraint_type]
                constraint_dt, constraint_grain = self.parse_date_auto(constraint_str)

                if constraint_type == "after":
                    if not self.is_after(cand_dt, constraint_dt, constraint_grain):
                        keep = False

                elif constraint_type == "before":
                    if not self.is_before(cand_dt, constraint_dt, constraint_grain):
                        keep = False

                elif constraint_type == "on":
                    if not self.is_on(cand_dt, constraint_dt, constraint_grain):
                        keep = False

            if keep:
                filtered_docs.append(doc)

        return filtered_docs

    def parse_date_auto(self, date: str) -> datetime | str:
        """
        Parses a string representing a time, day, month, or year, and returns a datetime object and a granularity string.
        
        Args:
            date (str): A string representing a time.
        
        Returns:
            dt (datetime): A datetime object representing the date.
            gran_s (str): A string specifying the granularity of the date.
        """
        s = date.strip()

        # FULL DATE: YYYY-MM-DD → day-level
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return datetime.strptime(s, "%Y-%m-%d"), "day"

        # YEAR + MONTH: YYYY-MM → month-level
        if re.fullmatch(r"\d{4}-\d{2}", s):
            return datetime.strptime(s + "-01", "%Y-%m-%d"), "month"

        # YEAR ONLY: YYYY → year-level
        if re.fullmatch(r"\d{4}", s):
            return datetime.strptime(s + "-01-01", "%Y-%m-%d"), "year"
        
    def is_after(self, cand_dt: datetime , constraint_dt: datetime, constraint_grain: str) -> bool:
        """
        Returns whether cand_dt occurs strictly after constraint_dt,
        interpreted at the proper granularity.

        Args:
            cand_dt (datetime): The candidate timestamp.
            constraint_dt (datetime): The constraint timestamp.
            constraint_grain (str): Granularity {'year', 'month', 'day'}.

        Returns:
            bool: True if the candidate occurs after the constraint.
        """

        # YEAR-level comparison: only compare cand_dt.year
        if constraint_grain == "year":
            return cand_dt.year > constraint_dt.year

        # MONTH-level comparison: compare year first, then month
        if constraint_grain == "month":
            return (cand_dt.year > constraint_dt.year) or \
                (cand_dt.year == constraint_dt.year and cand_dt.month > constraint_dt.month)

        # DAY-level comparison: compare full datetime objects directly
        if constraint_grain == "day":
            return cand_dt > constraint_dt


    def is_before(self, cand_dt: datetime, constraint_dt: datetime, constraint_grain: str) -> bool:
        """
        Returns whether cand_dt occurs strictly before constraint_dt,
        interpreted at the appropriate granularity.

        Args:
            cand_dt (datetime): The candidate timestamp.
            constraint_dt (datetime): The constraint timestamp.
            constraint_grain (str): Granularity {'year', 'month', 'day'}.

        Returns:
            bool: True if the candidate occurs before the constraint.
        """

        # YEAR-level comparison
        if constraint_grain == "year":
            return cand_dt.year < constraint_dt.year

        # MONTH-level comparison
        if constraint_grain == "month":
            return (cand_dt.year < constraint_dt.year) or \
                (cand_dt.year == constraint_dt.year and cand_dt.month < constraint_dt.month)

        # DAY-level comparison
        if constraint_grain == "day":
            return cand_dt < constraint_dt


    def is_on(self, cand_dt: datetime, constraint_dt: datetime, constraint_grain: str) -> bool:
        """
        Returns whether cand_dt occurs exactly ON the constraint date,
        interpreted at the appropriate granularity.

        Args:
            cand_dt (datetime): The candidate timestamp.
            constraint_dt (datetime): The constraint timestamp.
            constraint_grain (str): Granularity {'year', 'month', 'day'}.

        Returns:
            bool: True if the candidate matches the constraint exactly at
                its granularity (same year, same year+month, or full date).
        """

        # YEAR-level: same year only
        if constraint_grain == "year":
            return cand_dt.year == constraint_dt.year

        # MONTH-level: same year and same month
        if constraint_grain == "month":
            return cand_dt.year == constraint_dt.year and cand_dt.month == constraint_dt.month

        # DAY-level: full exact match
        if constraint_grain == "day":
            return cand_dt == constraint_dt