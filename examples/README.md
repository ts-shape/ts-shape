# ts-shape examples

One runnable script per area of the library. Each script generates its own
synthetic data, so it needs no input files.

The list of scripts, with what each one shows, is on the documentation site:
[Example scripts](https://ts-shape.github.io/ts-shape/examples/)
(source: [`docs/examples.md`](../docs/examples.md)).

## Running a script

From a clone of the repository:

```bash
pip install -e .
python examples/<name>.py
```

For example, `python examples/quality_events_demo.py`. A few scripts use
optional extras (for example `unit_runtime_demo.py` uses `ts-shape[units]`
for unit conversion); `pip install -e ".[all]"` installs them all.
