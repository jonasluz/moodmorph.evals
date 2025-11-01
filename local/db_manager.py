# Database manager for MoodMorph evaluations
# ###############################################
# This module provides functionality to save evaluation results to a SQLite database.
# @author: Jonas de Araújo Luz Jr.
# @date: October 2025
# ###############################################
import sqlite3

import pandas as pd


class DBManager:
    """Class to manage database operations for MoodMorph evaluations."""

    def __init__(self, db_path: str):

        self._db_path = db_path


    def save_dataframe(self, df: pd.DataFrame, table_name: str, 
                       if_exists: str = 'replace', index: bool = False):
        """Saves a pandas DataFrame to the specified table in the SQLite database."""
        
        with sqlite3.connect(self._db_path) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=index)
            conn.commit()


    def load_dataframe(self, table_name: str, index_col: str = 'id') -> pd.DataFrame:
        """Loads a pandas DataFrame from the specified table in the SQLite database."""

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            df.set_index(index_col, inplace=True)
        return df


    def list_tables(self) -> list[str]:
        """Lists all table names in the SQLite database."""

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        return tables


__all__ = ['DBManager']