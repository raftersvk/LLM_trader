"""Analysis Result Processor.

Handles the processing and formatting of analysis results from the AI models.
"""
from __future__ import annotations

import io
import json
import re
from typing import TYPE_CHECKING, Any

from src.analyzer.decision_contract import validate_decision_payload
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.analyzer.risk_reward_validator import RiskRewardValidator
from src.analyzer.trend_validator import TrendValidator
from src.config.regime_taxonomy import (
    CostsSettings,
    TaxonomyConfigError,
    load_taxonomy_config,
    resolve_costs_config,
)
from src.llm_focus.deterministic_risk import build_risk_plan, compute_position_size
from src.llm_focus.prompts import to_v1_compatible
from src.logger.logger import Logger
from src.utils.indicator_classifier import resolve_scalar

if TYPE_CHECKING:
    from src.analyzer.analysis_context import AnalysisContext
    from src.managers.model_manager import ModelManager


class AnalysisResultProcessor:
    """Processes and formats market analysis results from AI models"""

    def __init__(
        self,
        model_manager: ModelManager,
        logger: Logger,
        unified_parser,
        trend_validator: TrendValidator,
        quality_scorer: PatternQualityScorer,
        risk_reward_validator: RiskRewardValidator,
        costs: CostsSettings | None = None,
    ):
        """Initialize the processor"""
        self.model_manager = model_manager
        self.logger = logger
        self.unified_parser = unified_parser
        self.context: AnalysisContext | None = None
        self._trend_validator = trend_validator
        self._quality_scorer = quality_scorer
        self._risk_reward_validator = risk_reward_validator
        self._costs = costs

    async def process_analysis(self, system_prompt: str, prompt: str,
                              chart_image: io.BytesIO | bytes | str | None = None,
                              provider: str | None = None, model: str | None = None,
                              response_format: dict | None = None) -> dict[str, Any]:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Process analysis by sending prompts to AI model and formatting response.
        Returns:
            Dictionary containing formatted analysis results
        """
        self.logger.debug("Sending prompt to AI model for analysis")

        use_chart_analysis = chart_image is not None and self.model_manager.supports_image_analysis(provider)
        if chart_image is not None and use_chart_analysis:
            prov_name, model_name = self.model_manager.describe_provider_and_model(provider, model, chart=True)
            prov_label = prov_name.upper() if prov_name else "UNKNOWN"
            self.logger.info(
                "Using chart image analysis via %s (model: %s)",
                prov_label,
                model_name
            )
            try:
                complete_response = await self.model_manager.send_prompt_with_chart_analysis(
                    prompt=prompt,
                    chart_image=chart_image,
                    system_message=system_prompt,
                    provider=provider,
                    model=model,
                    response_format=response_format,
                )
            except (ValueError, Exception) as chart_error:  # pylint: disable=broad-exception-caught  # noqa: BLE001
                self.logger.warning("Chart analysis failed: %s. Falling back to text-only analysis.", chart_error)
                complete_response = await self.model_manager.send_prompt_streaming(
                    prompt=prompt,
                    system_message=system_prompt,
                    provider=provider,
                    model=model,
                    response_format=response_format,
                )
        else:
            complete_response = await self.model_manager.send_prompt_streaming(
                prompt=prompt,
                system_message=system_prompt,
                provider=provider,
                model=model,
                response_format=response_format,
            )

        self.logger.debug("Received response from AI model")
        cleaned_response = self._clean_response(complete_response)

        parsed_response = self.unified_parser.parse_ai_response(cleaned_response)
        parsed_response = self._normalize_v2_decision(parsed_response)

        if not self.unified_parser.validate_ai_response(parsed_response):
            self.logger.warning("Invalid response format from AI model")
            return {
                "error": "Invalid response format",
                "raw_response": cleaned_response
            }
        parse_failed = bool(parsed_response.get("parse_error"))
        problems = (
            validate_decision_payload(parsed_response)
            if isinstance(parsed_response.get("analysis"), dict) and not parse_failed else []
        )
        if problems:
            self.logger.warning(
                "AI response contract violation: %s — asking once more with the violations",
                "; ".join(problems[:3]),
            )
            correction = (
                "\n\n## CONTRACT CORRECTION\n"
                "Your previous response failed these Python checks:\n- "
                + "\n- ".join(problems[:5])
                + "\nReturn the FULL response again, with one valid decision object and complete "
                "execution fields. Do not repeat the violation.\n"
            )
            complete_response = await self.model_manager.send_prompt_streaming(
                prompt=prompt + correction,
                system_message=system_prompt,
                provider=provider,
                model=model,
                response_format=response_format,
            )
            cleaned_response = self._clean_response(complete_response)
            parsed_response = self.unified_parser.parse_ai_response(cleaned_response)
            parsed_response = self._normalize_v2_decision(parsed_response)
            if not self.unified_parser.validate_ai_response(parsed_response):
                self.logger.error("Contract retry returned an invalid response shape")
                return {
                    "error": "Invalid response format after contract retry",
                    "raw_response": cleaned_response
                }
            remaining = validate_decision_payload(parsed_response)
            if remaining:
                self.logger.error(
                    "Contract still violated after one retry: %s", "; ".join(remaining[:3])
                )
        self._log_response_validation(parsed_response.get("response_validation"))
        if "narrative" not in parsed_response and "```json" in cleaned_response:
            parsed_response["narrative"] = cleaned_response.split("```json", 1)[0].strip()
        self._validate_llm_claims(parsed_response)
        for issue in self._risk_reward_validator.validate(parsed_response):
            self.logger.warning("R/R validation: %s", issue)
        self._log_analysis_result(parsed_response)

        return self._format_analysis_response(parsed_response, cleaned_response)

    def _normalize_v2_decision(self, parsed_response: dict[str, Any]) -> dict[str, Any]:
        """Convert an LLM-focus v2 payload into the v1 decision contract.

        ``llm_focus_v2`` prompts the model for ``action`` plus proposed
        ``stop_loss``/``take_profit`` instead of ``signal`` and the v1 execution
        fields. ``to_v1_compatible`` is the single conversion point; the current
        market price and the deterministic position size fill the fields the
        parser/decision contract requires. Responses that already carry a v1
        ``signal`` are returned untouched.
        """
        analysis = parsed_response.get("analysis")
        if not isinstance(analysis, dict) or "signal" in analysis or "action" not in analysis:
            return parsed_response

        entry_price = self.context.current_price if self.context is not None else None
        converted = to_v1_compatible(
            parsed_response,
            entry_price=entry_price,
            position_size=compute_position_size(analysis.get("confidence")),
        )
        normalized = {**parsed_response, **converted}
        self._apply_v2_risk_validation(normalized)
        validate = getattr(self.unified_parser, "validate_trading_response", None)
        if callable(validate):
            normalized["response_validation"] = validate(normalized)
        return normalized

    def _apply_v2_risk_validation(self, normalized: dict[str, Any]) -> None:
        """Run policy-C risk validation on a converted v2 decision.

        ``to_v1_compatible`` carries the LLM's absolute levels into the v1
        contract, but policy C also requires those levels to pass the
        deterministic target validation (correct side, R/R floor), the
        ATR-relative stop band and the fee-aware entry gate. The resulting
        ``risk_approved`` flag and explicit rejection reason are written back
        into ``analysis`` so an invalid signal cannot be executed silently.
        """
        analysis = normalized.get("analysis")
        if not isinstance(analysis, dict):
            return
        signal = str(analysis.get("signal") or "").strip().upper()
        if signal not in ("BUY", "SELL"):
            return

        stop_loss = analysis.get("stop_loss")
        take_profit = analysis.get("take_profit")
        if stop_loss is None or take_profit is None:
            analysis["risk_approved"] = False
            analysis["risk_rejection_reason"] = (
                "LLM stop_loss and take_profit must both be provided"
            )
            return

        technical = getattr(self.context, "technical_data", None) or {}
        atr = resolve_scalar(technical.get("atr"))
        atr_pct = resolve_scalar(
            technical.get("atr_percent", technical.get("atr_percentage"))
        )
        entry_price = (
            getattr(self.context, "current_price", None)
            if self.context is not None
            else None
        )
        plan = build_risk_plan(
            direction="LONG" if signal == "BUY" else "SHORT",
            entry_price=entry_price,
            atr=atr if atr > 0 else None,
            confidence=analysis.get("confidence"),
            atr_pct=atr_pct if atr_pct > 0 else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            costs=self._v2_costs(),
        )
        analysis["risk_approved"] = plan.risk_approved
        if plan.rejection_reason:
            analysis["risk_rejection_reason"] = plan.rejection_reason
        if plan.rr_ratio is not None:
            analysis["risk_reward_ratio"] = plan.rr_ratio
        normalized["risk_plan"] = {
            "direction": plan.direction,
            "target_source": plan.target_source,
            "risk_approved": plan.risk_approved,
            "rejection_reason": plan.rejection_reason,
            "sl_distance_pct": plan.sl_distance_pct,
            "tp_distance_pct": plan.tp_distance_pct,
            "rr_ratio": plan.rr_ratio,
        }

    def _v2_costs(self) -> CostsSettings:
        """Resolve effective costs for the v2 fee gate, falling back safely."""
        if self._costs is not None:
            return self._costs
        try:
            return resolve_costs_config(load_taxonomy_config())
        except (TaxonomyConfigError, OSError):
            return CostsSettings()

    def _log_analysis_result(self, parsed_response: dict[str, Any]) -> None:
        """Log analysis result information"""
        if "analysis" in parsed_response:
            analysis = parsed_response["analysis"]

            if "signal" in analysis:
                signal = analysis.get("signal", "UNKNOWN")
                confidence = analysis.get("confidence", 0)
                trend_info = analysis.get("trend", {})
                direction = trend_info.get("direction", "UNKNOWN")

                strength = trend_info.get("strength")
                if strength is None:
                    strength = trend_info.get("strength_daily", trend_info.get("strength_4h", 0))

                confluence_factors = analysis.get("confluence_factors", {})
                if confluence_factors:
                    cf_str = ", ".join([f"{k}={v}" for k, v in confluence_factors.items()])
                    self.logger.debug("Trading analysis complete: Signal %s, Confidence %s, Trend %s (%s%% strength) | Confluence: %s", signal, confidence, direction, strength, cf_str)
                else:
                    self.logger.debug("Trading analysis complete: Signal %s, Confidence %s, Trend %s (%s%% strength)", signal, confidence, direction, strength)
            else:
                bias = analysis.get("technical_bias", "UNKNOWN")
                trend = analysis.get("observed_trend", "UNKNOWN")
                confidence = analysis.get("confidence_score", 0)
                self.logger.debug("Analysis complete: Technical bias %s with %s trend (%s%% confidence)", bias, trend, confidence)
        else:
            self.logger.warning("Analysis complete but response format may be incomplete")

    def _log_response_validation(self, validation: dict[str, Any] | None) -> None:
        """Log response-contract validation metadata."""
        if not validation:
            return
        status = validation.get("status")
        if status == "valid":
            self.logger.debug("AI response contract validation passed: %s", validation.get("schema"))
            return
        if status == "invalid":
            errors = validation.get("errors", [])
            self.logger.warning("AI response contract validation failed: %s", errors[:3])
            return
        self.logger.debug("AI response contract validation skipped: no trading signal found")

    def _format_analysis_response(self, parsed_response: dict[str, Any],
                                cleaned_response: str) -> dict[str, Any]:
        """Format the final analysis response."""
        parsed_response["raw_response"] = self._render_response_text(parsed_response, cleaned_response)
        if self.context is not None:
            parsed_response["current_price"] = self.context.current_price
        return parsed_response

    @staticmethod
    def _render_response_text(parsed_response: dict[str, Any], cleaned_response: str) -> str:
        """Render the validated narrative and decision for either provider contract."""
        narrative = parsed_response.get("narrative")
        analysis = parsed_response.get("analysis")
        if not isinstance(narrative, str) or not isinstance(analysis, dict):
            return cleaned_response
        block = json.dumps({"analysis": analysis}, indent=4, ensure_ascii=False)
        return f"{narrative.strip()}\n\n```json\n{block}\n```"

    @staticmethod
    def _clean_response(text: str) -> str:
        """Remove thinking sections and extra whitespace from AI responses"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _validate_llm_claims(self, parsed_response: dict[str, Any]) -> None:
        """Validate LLM-reported ADX strengths and pattern quality against computed data.

        Runs after parsing but before the response is returned. Overwrites
        LLM values with computed ones when discrepancies exceed thresholds.
        Logs warnings for any significant gaps found.
        """
        if self.context is None:
            return

        analysis = parsed_response.get("analysis")
        if not analysis:
            return

        trend = analysis.get("trend", {})
        tech_data = self.context.technical_data or {}
        long_term_data = self.context.long_term_data or {}

        trend_result = self._trend_validator.validate(
            strength_4h=trend.get("strength_4h"),
            strength_daily=trend.get("strength_daily"),
            computed_adx=tech_data.get("adx"),
            computed_daily_adx=long_term_data.get("daily_adx") if long_term_data else None,
        )

        if trend_result.has_computed_data:
            self._trend_validator.overwrite_llm_trend(analysis, trend_result)

            if not trend_result.passed:
                for d in trend_result.discrepancies:
                    self.logger.warning("ADX validation: %s", d)
            else:
                self.logger.debug(
                    "ADX validation passed: 4H=%s daily=%s",
                    trend_result.validated_4h, trend_result.validated_daily,
                )

        llm_quality_raw = analysis.get("pattern_quality")
        try:
            llm_quality = float(llm_quality_raw) if llm_quality_raw is not None else None
        except (TypeError, ValueError):
            llm_quality = None

        quality = self._quality_scorer.score(
            patterns=self.context.technical_patterns,
            tech_data=tech_data,
            llm_quality=llm_quality,
        )
        self._quality_scorer.overwrite_llm_quality(analysis, quality)

        if not quality.passed:
            for d in quality.discrepancies:
                self.logger.warning("Pattern quality validation: %s", d)
        else:
            self.logger.debug(
                "Pattern quality: computed=%s (%s)",
                round(quality.overall), quality.label,
            )

