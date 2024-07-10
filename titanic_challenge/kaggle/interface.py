from datetime import datetime
from titanic_challenge.params import *
import pandas as pd
import kaggle
import os

class SubmissionInterface:

    def __init__(self) -> None:
        self.test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
        self.kapi = kaggle.api

    def push_submission(self, pipeline):
        y_pred = pipeline.predict(self.test_df)
        submission_df = pd.DataFrame(
            {"Survived": y_pred},
            index=self.test_df.PassengerId)
        file_name = os.path.join(
            DATA_PATH,
            "submissions",
            f'sub_{datetime.today().strftime("%d-%m-%Y@%H:%M:%S")}.csv')
        submission_df.to_csv(file_name)

        self.kapi.competition_submit(
            file_name,
            file_name.split('/')[-1],
            "titanic")
