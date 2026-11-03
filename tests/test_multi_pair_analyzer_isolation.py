"""Per-pair analyzer collaborators must not be shared between engines."""
from unittest.mock import MagicMock

from start import CompositionRoot
from src.config.regime_taxonomy import CostsSettings


def _root() -> CompositionRoot:
    root = object.__new__(CompositionRoot)
    config = MagicMock()
    config.TIMEFRAME = "1h"
    config.AI_CHART_CANDLE_LIMIT = 125
    root.config = config
    root.logger = MagicMock()
    return root


def _inputs():
    infra = {"session": MagicMock()}
    apis = {"alternative_me": MagicMock(), "market": MagicMock()}
    utils = {
        "format_utils": MagicMock(),
        "token_counter": MagicMock(),
        "timeframe_validator": MagicMock(),
        "parser": MagicMock(),
    }
    rag = MagicMock(name="rag")
    models = {"manager": MagicMock(name="model_manager")}
    return infra, apis, utils, rag, models


def test_collaborators_are_fresh_for_every_engine():
    """Two engines must not share collector/prompt/calculator instances."""
    root = _root()
    infra, apis, utils, rag, models = _inputs()

    first = root.build_analyzer_engine(infra, apis, utils, rag, models)
    second = root.build_analyzer_engine(infra, apis, utils, rag, models)

    assert first is not second
    for attribute in (
        "technical_calculator",
        "pattern_analyzer",
        "prompt_builder",
        "data_collector",
        "metrics_calculator",
        "result_processor",
        "chart_generator",
    ):
        assert getattr(first, attribute) is not getattr(second, attribute), attribute


def test_collaborator_builder_returns_new_instances():
    """The raw collaborator builder allocates new objects on every call."""
    root = _root()
    infra, apis, utils, rag, models = _inputs()

    first = root._build_analyzer_collaborators(infra, apis, utils, rag, models)
    second = root._build_analyzer_collaborators(infra, apis, utils, rag, models)

    for key in first:
        assert first[key] is not second[key], key


def test_analyzer_processors_receive_the_startup_costs():
    """The v2 fee gate must see the config.ini > taxonomy > defaults costs."""
    root = _root()
    root._startup_costs = CostsSettings(
        venue="spot", tier=1, entry_order_type="taker",
        maker_pct=0.40, taker_pct=0.80,
    )
    infra, apis, utils, rag, models = _inputs()

    collaborators = root._build_analyzer_collaborators(
        infra, apis, utils, rag, models
    )

    assert collaborators["result_processor"]._costs is root._startup_costs
