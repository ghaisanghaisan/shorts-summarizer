from .base import Analyzer, Evidence


def get_analyzer(name: str = "claude", model: str | None = None) -> Analyzer:
    if name == "claude":
        from .claude import ClaudeAnalyzer
        return ClaudeAnalyzer(model=model)
    if name == "gemini":
        from .gemini import GeminiAnalyzer
        return GeminiAnalyzer(model=model)
    raise ValueError(f"unknown analyzer {name!r}")


__all__ = ["Analyzer", "Evidence", "get_analyzer"]
