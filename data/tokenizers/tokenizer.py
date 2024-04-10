from sklearn.preprocessing import OneHotEncoder
import numpy as np


class Tokenizer:
    def __call__(self, data: str):
        # TODO HOE should account for 2 additional tokens <s> and </s> (see Sturm et al.)
        split_data = data.split()
        split_data = np.array(split_data).reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(split_data)
        encoded_categories = encoder.transform(split_data)
        return encoded_categories
