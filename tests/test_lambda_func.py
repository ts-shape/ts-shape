import numpy as np
import pandas as pd  # type: ignore
from ts_shape.transform.functions.lambda_func import LambdaProcessor


def test_lambda_processor_apply_function_vectorized():
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = LambdaProcessor.apply_function(df.copy(), "x", lambda arr: arr + 10)
    assert out["x"].tolist() == [11, 12, 13]


def test_lambda_processor_raises_on_missing_column():
    df = pd.DataFrame({"y": [1, 2, 3]})
    try:
        LambdaProcessor.apply_function(df, "x", lambda arr: arr)
        assert False, "Expected ValueError for missing column"
    except ValueError as e:
        assert "Column 'x' not found" in str(e)
