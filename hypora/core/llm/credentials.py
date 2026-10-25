"""Provider API keys, read from the environment or the repo's git-ignored `.env` (PLAN.md S0.5).

Keys are held as `SecretStr`, so `repr()`, `str()`, logs, tracebacks, and validation errors show
`**********` instead of the value. They are unwrapped only at the moment a provider SDK client is
built. Error messages name the variable, never a value. Environment variables win over `.env`.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from hypora.core.llm.errors import MissingCredentialError
from hypora.core.schemas.events import Provider

DOTENV_PATH = Path(__file__).resolve().parents[3] / ".env"
"""The repo-root `.env` (git-ignored; see .env.example for the variable names)."""

API_KEY_VARIABLES: dict[Provider, str] = {
    Provider.GEMINI: "GEMINI_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
}


class ProviderCredentials(BaseSettings):
    """API keys per provider. Missing or empty means "not configured"."""

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds keys for other steps (e.g. HF_TOKEN)
        frozen=True,
    )

    gemini_api_key: SecretStr | None = Field(None, validation_alias="GEMINI_API_KEY")
    openai_api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY")

    def _secret(self, provider: Provider) -> SecretStr | None:
        if provider is Provider.GEMINI:
            return self.gemini_api_key
        if provider is Provider.OPENAI:
            return self.openai_api_key
        return None

    def has(self, provider: Provider) -> bool:
        """Whether a non-empty key is configured for `provider`."""
        secret = self._secret(provider)
        return secret is not None and secret.get_secret_value() != ""

    def api_key(self, provider: Provider) -> str:
        """The raw key, for building that provider's SDK client only. Never log the result."""
        secret = self._secret(provider)
        if secret is None or not secret.get_secret_value():
            variable = API_KEY_VARIABLES.get(provider, f"{provider.value.upper()}_API_KEY")
            raise MissingCredentialError(
                f"{variable} is not set: add it to the git-ignored .env file at the repo root "
                "(names in .env.example), or export it. Never commit or paste the value."
            )
        return secret.get_secret_value()

    def secret_values(self) -> tuple[str, ...]:
        """Every configured key, so stored transcripts can be scrubbed of them."""
        return tuple(self.api_key(p) for p in API_KEY_VARIABLES if self.has(p))


def load_credentials(env_file: Path | None = DOTENV_PATH) -> ProviderCredentials:
    """Read keys from the environment, then `env_file` (None: environment only)."""
    return ProviderCredentials(_env_file=env_file)
