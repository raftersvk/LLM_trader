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
