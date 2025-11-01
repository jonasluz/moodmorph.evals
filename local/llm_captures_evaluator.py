# LLM Evaluation classes, prompt and schema
# ########################################################
# FILE: llm_processor.py
# @author: Jonas de Araújo Luz Jr.
# @date: October 2025
# This file contains the prompt and output schema used for evaluating
# 3D character animation expressions using a large language model (LLM).
# ########################################################

from pathlib import Path
import base64, json

import pandas as pd
import litellm

from local.db_manager import DBManager

# region PROMPT DEFINITION -------------------------------

# Original prompt, by Prof. Andreia Formico Rodrigues.
#
# prompt = """
# You are presented with the peak frame from a 3D character animation sequence. Based on facial cues visible in this single frame,
# identify the most likely emotion being expressed, using Ekman's seven universal emotions as reference. Describe the key visual cues that support your interpretation. Then, rate the expression's realism on a scale from 1 (poor) to 5 (highly realistic).
# """

# Revised prompt requiring the detection of Akman's FACS Action Units (AUs) and changing the realism scale to 1-4.
# _prompt = """
# You are presented with the peak frame from a 3D character animation sequence. Based on facial cues visible in this single frame, identify the most likely emotions being expressed, using Ekman's seven universal emotions as reference (happiness, sadness, anger, fear, surprise, disgust, contempt). 
# Describe the key visual cues that support your interpretation. 
# Then, rate the expression's realism on a scale from 1 to 4, where 1 → Poor, 2 → Fair, 3 → Good, and 4 → Excellent.
# """

# Revised prompt requiring the detection of Akman's FACS Action Units (AUs) and changing the realism scale to 1-4.
_prompt = """
You are presented with the peak frame from a 3D character animation sequence. Based on facial cues visible in this single frame, identify the most likely emotions being expressed, using Ekman's seven universal emotions as reference (happiness, sadness, anger, fear, surprise, disgust, contempt). 
Describe the key visual cues that support your interpretation.
"""
# endregion PROMPT DEFINITION

# region OUTPUT SCHEMA DEFINITION ------------------------
# _output_schema = {
#     "type": "object",
#     "properties": {
#         "emotion": {
#             "type": "string", 
#             "description": "Identified emotions from the image, separated by commas if multiple."
#         },
#         "key_visual_cues": {
#             "type": "string",
#             "description": "Key visual cues supporting the emotion interpretation."
#         },
#         "realism_rating": {
#             "type": "integer",
#             "minimum": 1,
#             "maximum": 4
#         }
#     },
#     "required": ["emotion", "key_visual_cues", "realism_rating"]
# }

_output_schema = {
    "type": "object",
    "properties": {
        "emotion": {
            "type": "string", 
            "description": "Identified emotions from the image, separated by commas if multiple."
        },
        "key_visual_cues": {
            "type": "string",
            "description": "Key visual cues supporting the emotion interpretation."
        }
    },
    "required": ["emotion", "key_visual_cues"]
}
# endregion OUTPUT SCHEMA DEFINITION

# region Processor Class --------------------------------
class LLMCapturesEvaluator:
    """Class to process captures using a specified LLM model."""

    def __init__(self, model: str, 
                 df_captures: pd.DataFrame, captures_path: str | Path):
        self._model = model
        self._df_captures = df_captures
        self._captures_path = Path(captures_path) \
            if isinstance(captures_path, str) else captures_path

        self.df_evaluations = pd.DataFrame(
            columns=['expected_emotion', 'identified_emotion',
                     'key_visual_cues', 'tokens_used', 'evaluator', 
                     'timestamp']
        )
        self.df_evaluations.index.name = 'test_id'


    def evaluate_capture(self, idx:int) -> pd.Series:
        """Evaluate a single capture and store results in df_evaluations."""

        rec = self._df_captures.loc[idx]
        #print(f"Processing file: {rec['path']} with expected emotion: {rec['emotion']}")

        result = self._process_capture(self._captures_path / rec['path'])
        response = json.loads(result['response'])

        self.df_evaluations.loc[idx, 'expected_emotion'] = rec['emotion']
        self.df_evaluations.loc[idx, 'identified_emotion'] = response['emotion']
        self.df_evaluations.loc[idx, 'key_visual_cues'] = response['key_visual_cues']
        #self.df_evaluations.loc[idx, 'realism_rating'] = response['realism_rating']
        self.df_evaluations.loc[idx, 'tokens_used'] = result['tokens_used']
        self.df_evaluations.loc[idx, 'evaluator'] = self._model
        self.df_evaluations.loc[idx, 'timestamp'] = pd.Timestamp.now()

        return self.df_evaluations.loc[idx]


    def save_to_db(self, db: DBManager, table_name: str = 'evaluations', 
                       override_rule: str = 'append'):
        """Save the evaluations DataFrame to a SQLite database."""

        self.df_evaluations['timestamp'] = self.df_evaluations['timestamp'].astype('datetime64[ns]')
        
        df_data = self.df_evaluations.reset_index()
        db.save_dataframe(df_data, table_name, override_rule)


    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """Function to encode the image to base64 string using pathlib.Path.open."""
        with image_path.open('rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def _process_capture(self, capture_path: Path) -> dict:
        """Process a single capture file with the LLM."""
        
        # Getting the Base64 string
        base64_image = self._encode_image(capture_path)

        completion = litellm.completion(
            model=self._model,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": _prompt },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "emotion_response",
                    "schema": _output_schema
                }
            }
        )

        return {
            "response": completion.choices[0].message.content,
            "tokens_used": completion.usage.total_tokens
        }
# endregion Processor Class

__all__ = ["LLMCapturesEvaluator"]