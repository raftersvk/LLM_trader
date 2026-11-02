"""LLM-focus v2 preparation package.

The modules in this package are pure, side-effect free building blocks for
the staged v2 rollout. They are intentionally not wired into the live trading
path yet; callers opt in through the feature flags in
:mod:`src.llm_focus.flags`.
"""
