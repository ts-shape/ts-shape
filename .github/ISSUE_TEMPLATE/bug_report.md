---
name: Bug report
about: Something behaves incorrectly or raises unexpectedly
title: ''
labels: bug
assignees: ''
---

## What happened

<!-- The behaviour you observed, and what you expected instead. If a computed
     value is wrong, say what the correct value is and why. -->

## Reproduction

<!-- A minimal, self-contained snippet. ts_shape.make_timeseries() generates a
     sample frame, so a reproduction rarely needs your real data. -->

```python
import pandas as pd
import ts_shape

df = ts_shape.make_timeseries(["sensor:temp"], n_points=100)
# ...
```

## Output

<!-- The full traceback, or the wrong output next to the expected output. -->

```text

```

## Environment

- ts-shape version: <!-- ts_shape.__version__ -->
- Python version:
- pandas / numpy version:
- Operating system:

## Data shape (if relevant)

<!-- Column names and dtypes, the timestamp frequency, whether timestamps are
     tz-aware, and roughly how many rows and signals. Do not paste real
     production data or credentials. -->
