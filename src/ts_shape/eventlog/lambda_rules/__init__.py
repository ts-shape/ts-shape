"""Lambda-rule subsystem — declarative, YAML-driven detectors that share
all canonical-event-log plumbing with the 290 built-in detector methods.

Typical use::

    from ts_shape.eventlog import register_lambda_rule, RuleSpec, TriggerSpec

    spec = RuleSpec(
        id="high_torque",
        class_name="LambdaToolWear",
        method_name="high_torque",
        pack="maintenance",
        shape="point",
        archetype="threshold",
        template="maintenance.tool.high_torque",
        trigger=TriggerSpec(expression="torque > 75 & state == 'run'"),
        standard_attrs={
            "ts_shape:method": "lambda_threshold",
            "ts_shape:direction": "above",
            "ts_shape:threshold_high": 75.0,
        },
    )
    detector = register_lambda_rule(spec)
    log = detector.to_event_log(df)

See :doc:`/guides/lambda-rules` for the full walkthrough including a
threshold case and an interval-with-hysteresis case.
"""
from .backtest import BacktestResult, run_backtest
from .detector import LambdaDetector
from .expression import UnsafeExpression, compile_expression
from .loader import load_dicts, load_yaml, register_lambda_rule, unregister_lambda_rule
from .spec import RuleSpec, TriggerSpec

__all__ = [
    "BacktestResult",
    "LambdaDetector",
    "RuleSpec",
    "TriggerSpec",
    "UnsafeExpression",
    "compile_expression",
    "load_dicts",
    "load_yaml",
    "register_lambda_rule",
    "run_backtest",
    "unregister_lambda_rule",
]
