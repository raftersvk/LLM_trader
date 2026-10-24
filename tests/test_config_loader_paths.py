"""Config loader path overrides used by the signal CLI and smoke script."""
from pathlib import Path

from src.config.loader import Config


def _write_ini(path: Path, pair: str) -> None:
    path.write_text(
        "[general]\n"
        f"crypto_pair = {pair}\n"
        "timeframe = 1h\n"
        "\n"
        "[ai_providers]\n"
        "provider = googleai\n"
        "\n"
        "[model_config]\n"
        "max_tokens = 32768\n"
        "google_max_tokens = 65536\n",
        encoding="utf-8",
    )


def test_config_paths_honour_environment_overrides(tmp_path, monkeypatch):
    """LLM_TRADER_CONFIG_INI / LLM_TRADER_KEYS_ENV select the loaded files."""
    ini = tmp_path / "custom.ini"
    _write_ini(ini, "ETH/USDC")
    keys = tmp_path / "custom.env"
    keys.write_text("CUSTOM_ENV_TOKEN=from-custom-keys\n", encoding="utf-8")

    monkeypatch.setenv("LLM_TRADER_CONFIG_INI", str(ini))
    monkeypatch.setenv("LLM_TRADER_KEYS_ENV", str(keys))

    config = Config()

    assert config.CRYPTO_PAIR == "ETH/USDC"
    assert config.get_env("CUSTOM_ENV_TOKEN") == "from-custom-keys"


def test_reload_picks_up_a_new_config_path(tmp_path, monkeypatch):
    """reload() re-resolves the overrides instead of keeping the first paths."""
    ini = tmp_path / "custom.ini"
    _write_ini(ini, "ETH/USDC")
    keys = tmp_path / "custom.env"
    keys.write_text("CUSTOM_ENV_TOKEN=first\n", encoding="utf-8")

    monkeypatch.setenv("LLM_TRADER_CONFIG_INI", str(ini))
    monkeypatch.setenv("LLM_TRADER_KEYS_ENV", str(keys))
    config = Config()
    assert config.CRYPTO_PAIR == "ETH/USDC"

    other_ini = tmp_path / "other.ini"
    _write_ini(other_ini, "SOL/USDC")
    other_keys = tmp_path / "other.env"
    other_keys.write_text("CUSTOM_ENV_TOKEN=second\n", encoding="utf-8")
    monkeypatch.setenv("LLM_TRADER_CONFIG_INI", str(other_ini))
    monkeypatch.setenv("LLM_TRADER_KEYS_ENV", str(other_keys))

    config.reload()

    assert config.CRYPTO_PAIR == "SOL/USDC"
    assert config.get_env("CUSTOM_ENV_TOKEN") == "second"


def test_removing_the_override_restores_the_default_path(monkeypatch):
    """Unsetting the env var must not keep the import-time override forever."""
    from src.config import loader

    monkeypatch.setenv("LLM_TRADER_CONFIG_INI", "/tmp/custom.ini")
    assert loader._resolve_config_ini_path() == Path("/tmp/custom.ini")

    monkeypatch.delenv("LLM_TRADER_CONFIG_INI")
    assert loader._resolve_config_ini_path() == loader.CONFIG_DIR / "config.ini"


def _write_litellm_ini(path: Path) -> None:
    path.write_text(
        "[general]\n"
        "crypto_pair = BTC/USDC\n"
        "timeframe = 1h\n"
        "\n"
        "[ai_providers]\n"
        "provider = googleai\n"
        "litellm_context_window = 123456\n"
        "vision_context_window = 8192\n"
        "\n"
        "[model_config]\n"
        "max_tokens = 32768\n"
        "google_max_tokens = 65536\n"
        "\n"
        "[model_context_windows]\n"
        "litellm/deepseek-v4.1-flash = 64000\n"
        "vision_litellm/vision-test = 8192\n",
        encoding="utf-8",
    )


def test_litellm_context_windows_are_configurable(tmp_path, monkeypatch):
    """litellm/vision keys and windows must be real, validated config."""
    ini = tmp_path / "litellm.ini"
    _write_litellm_ini(ini)
    monkeypatch.setenv("LLM_TRADER_CONFIG_INI", str(ini))

    config = Config()

    assert config.LITELLM_CONTEXT_WINDOW == 123456
    assert config.VISION_CONTEXT_WINDOW == 8192
    assert config.MODEL_CONTEXT_WINDOWS["litellm/deepseek-v4.1-flash"] == 64000
    assert config.MODEL_CONTEXT_WINDOWS["vision_litellm/vision-test"] == 8192


def test_context_window_defaults_when_not_configured(tmp_path, monkeypatch):
    """Vision falls back to the LiteLLM window; LiteLLM has a documented default."""
    ini = tmp_path / "defaults.ini"
    _write_ini(ini, "BTC/USDC")
    monkeypatch.setenv("LLM_TRADER_CONFIG_INI", str(ini))

    config = Config()

    assert config.LITELLM_CONTEXT_WINDOW == 1_000_000
    assert config.VISION_CONTEXT_WINDOW == config.LITELLM_CONTEXT_WINDOW


def _write_timeout_ini(path: Path, timeout_line: str) -> None:
    path.write_text(
        "[general]\n"
        "crypto_pair = BTC/USDC\n"
        "timeframe = 1h\n"
        "\n"
        "[ai_providers]\n"
        "provider = googleai\n"
        "\n"
        "[model_config]\n"
        "max_tokens = 32768\n"
        "google_max_tokens = 65536\n"
        f"{timeout_line}\n",
        encoding="utf-8",
    )


def test_model_config_timeout_is_honoured(tmp_path, monkeypatch):
    """A configured 600s timeout must survive into get_model_config as a number."""
    ini = tmp_path / "timeout.ini"
    _write_timeout_ini(ini, "timeout = 600")
    monkeypatch.setenv("LLM_TRADER_CONFIG_INI", str(ini))

    config = Config()

    assert config.get_model_config("some-litellm-model")["timeout"] == 600


def test_model_config_omits_timeout_when_not_configured(tmp_path, monkeypatch):
    """An omitted timeout must not impose a global default on provider clients."""
    ini = tmp_path / "no-timeout.ini"
    _write_timeout_ini(ini, "# timeout omitted on purpose")
    monkeypatch.setenv("LLM_TRADER_CONFIG_INI", str(ini))

    config = Config()

    assert "timeout" not in config.get_model_config("some-litellm-model")
