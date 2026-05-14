"""Drafter — turns a typed Bundle into a structured narrative draft.

The cloud Drafter (`LLMDrafter`) is provider-agnostic: it speaks to its
configured `Provider` (Anthropic or Gemini today) through a thin
interface, so swapping models is a one-argument change. The privacy
tier is enforced at the Drafter level — any cloud provider refuses
`LOCAL_ONLY` bundles.

Adding a new provider means writing one `generate()` method. Adding a
new model on an existing provider is `LLMDrafter(provider=...
(model="..."))`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Literal

import anthropic

from pgx_digest.bundle import Bundle, DrugRec, PGxFinding, PrivacyTier


class PrivacyViolation(RuntimeError):
    """Raised when a cloud Drafter is invoked on LOCAL_ONLY data."""


@dataclass(frozen=True)
class DraftedCard:
    """One claim card produced by the Drafter. Verifier checks each field."""

    gene: str
    diplotype: str
    phenotype: str
    drug: str
    recommendation: str
    cited_pmids: tuple[int, ...]


@dataclass(frozen=True)
class Draft:
    cards: tuple[DraftedCard, ...]
    raw_text: str


SYSTEM_PROMPT = """\
You write patient-facing pharmacogenomic summaries from a typed evidence
bundle. Each card is for exactly one (gene, drug) pair. The
`recommendation` field is a plain-English paraphrase of the source CPIC
guideline text for that pair — nothing more.

A downstream verifier rejects drafts that violate the TYPED rules. The
PROSE rules are enforced by clinical review; violations are common and
costly.

TYPED rules (verifier-enforced — output will be rejected):
1. Every `gene`, `diplotype`, `phenotype`, `drug`, and `cited_pmids`
   value must come from the input Bundle. Do not invent any of these.

PROSE rules for the `recommendation` field:
2. SINGLE GENE PER CARD. Mention ONLY this card's gene by name. A card
   for CYP2D6 must not reference CYP2C19, and vice versa. If the source
   guideline mentions multiple genes, paraphrase using only this card's
   gene.
3. NO METABOLISM-DIRECTION CLAIMS IN YOUR OWN WORDS. Direction errors
   are common — Poor Metabolizers of parent drugs typically have
   INCREASED drug levels (accumulation/toxicity risk), NOT reduced
   response. Do not say "reduced response", "sub-optimal response",
   "less effective", "more effective", "faster metabolism", or "slower
   metabolism" unless those exact phrases appear in the source
   recommendation. Prefer naming the phenotype verbatim and letting the
   source CPIC text speak for itself.
4. NO NAMED ALTERNATIVES. If the source says "consider alternative",
   write "consider an alternative medication" — do NOT name candidate
   drugs. Naming alternatives risks suggesting a drug that is itself
   affected by the same gene.
5. NO ADDED MECHANISM OR DRUG-CLASS COMMENTARY. Do not add explanations
   the source does not contain (e.g. "tertiary amine antidepressant",
   "narrow therapeutic index", "CYP-mediated"). Paraphrase the source
   recommendation tightly; do not enrich it.
6. Write in plain, direct English. End each `recommendation` with the
   exact phrase: "Discuss with your physician before any medication
   change."

Output: JSON conforming to the provided schema. One card per (gene,
drug) finding.
"""


# Canonical JSON schema. Anthropic's structured-output mode requires
# `additionalProperties: false` on every object node; we make that the
# default and let GeminiProvider strip the keyword before calling
# Gemini, which doesn't accept it.
OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gene": {"type": "string"},
                    "diplotype": {"type": "string"},
                    "phenotype": {"type": "string"},
                    "drug": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "cited_pmids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "gene",
                    "diplotype",
                    "phenotype",
                    "drug",
                    "recommendation",
                    "cited_pmids",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cards"],
    "additionalProperties": False,
}


def _strip_additional_properties(schema: dict) -> dict:
    """Recursively drop `additionalProperties` from a JSON schema.

    Gemini's `response_schema` (OpenAPI 3.0 subset) rejects this
    keyword; everything else in our schema is supported by both
    providers as-is.
    """
    if not isinstance(schema, dict):
        return schema
    cleaned: dict = {}
    for k, v in schema.items():
        if k == "additionalProperties":
            continue
        if isinstance(v, dict):
            cleaned[k] = _strip_additional_properties(v)
        elif isinstance(v, list):
            cleaned[k] = [
                _strip_additional_properties(x) if isinstance(x, dict) else x
                for x in v
            ]
        else:
            cleaned[k] = v
    return cleaned


# ---------------------------------------------------------------------------
# Provider abstraction — one method, JSON-schema-constrained generation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response from any provider.

    Token counts are best-effort: providers that don't report cache
    metrics simply return zero for them.
    """

    text: str
    raw: Any  # provider-specific response object, for tests / inspection
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class Provider(ABC):
    """A model backend that can generate JSON-schema-constrained text."""

    name: str  # short identifier, e.g. "anthropic", "gemini"
    model: str  # concrete model id, e.g. "claude-haiku-4-5"

    @abstractmethod
    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int,
    ) -> ProviderResponse: ...


class AnthropicProvider(Provider):
    """Anthropic backend. Defaults to Claude Haiku 4.5.

    Tests can inject a fake `client` whose `messages.create(...)` returns
    a stand-in response object exposing `.content[0].text` and `.usage`.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        client: Any = None,
    ) -> None:
        self.model = model
        self._client = client or anthropic.Anthropic()

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int,
    ) -> ProviderResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={
                "format": {"type": "json_schema", "schema": schema}
            },
            messages=[{"role": "user", "content": user}],
        )
        text = next(b.text for b in response.content if b.type == "text")
        usage = response.usage
        return ProviderResponse(
            text=text,
            raw=response,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(
                getattr(usage, "cache_read_input_tokens", 0) or 0
            ),
            cache_creation_tokens=int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
        )


class GeminiProvider(Provider):
    """Google Gemini backend. Defaults to gemini-2.5-flash (free tier).

    Tests can inject a fake `client` whose
    `models.generate_content(...)` returns a stand-in response exposing
    `.text` and `.usage_metadata`.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        client: Any = None,
    ) -> None:
        self.model = model
        if client is None:
            from google import genai

            self._client = genai.Client()  # reads GEMINI_API_KEY from env
        else:
            self._client = client

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int,
    ) -> ProviderResponse:
        from google.genai import types

        gemini_schema = _strip_additional_properties(schema)
        config_kwargs: dict[str, Any] = dict(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=gemini_schema,
            max_output_tokens=max_tokens,
        )
        # `thinking_config` applies only to gemini-2.5-* models. 2.5-flash
        # has reasoning on by default and reasoning tokens count against
        # `max_output_tokens`, so a structured-JSON call can run out of
        # budget before emitting any response — we disable it. Earlier
        # models reject the field, so it must be omitted for them.
        if self.model.startswith("gemini-2.5"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0
            )
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text
        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            text=text,
            raw=response,
            input_tokens=int(
                getattr(usage, "prompt_token_count", 0) or 0
            ),
            output_tokens=int(
                getattr(usage, "candidates_token_count", 0) or 0
            ),
            cache_read_tokens=int(
                getattr(usage, "cached_content_token_count", 0) or 0
            ),
            cache_creation_tokens=0,
        )


# ---------------------------------------------------------------------------
# Drafter implementations
# ---------------------------------------------------------------------------


class Drafter(ABC):
    """Abstract Drafter — picks a backend and runs structured generation."""

    @abstractmethod
    def draft(self, bundle: Bundle[PGxFinding]) -> Draft: ...


DrafterMode = Literal["batch", "per_card"]


class LLMDrafter(Drafter):
    """Cloud Drafter, provider-agnostic.

    Refuses to run on `LOCAL_ONLY` bundles regardless of provider — the
    privacy story is enforced here, not in any single provider.

    Two generation modes:

    - `batch` (default): one provider call per Bundle. The whole bundle
      is serialized as JSON; the model emits all cards in one response.
      Cheaper in API-call count and good for small bundles, but output
      size grows linearly with bundle size and the model's view of
      cross-card context can leak inappropriate gene references.

    - `per_card`: one provider call per (PGxFinding, DrugRec) pair,
      fanned out concurrently via ThreadPoolExecutor. Each call only
      sees one finding + one drug — structurally eliminates cross-gene
      contamination. Scales to large bundles without hitting output
      `max_tokens`. Privacy side benefit: each API call sees only one
      gene, reducing the quasi-identifying multi-gene fingerprint of a
      single request.

    By default the JSON shipped to the provider is *redacted*: the
    `source_variants` field (rsids, chromosomal positions, genotype
    calls) is stripped from each finding. The LLM doesn't need raw
    variant data to write prose — the `diplotype` name alone suffices.
    Set `redact_provenance=False` to send the full bundle for debugging.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        max_tokens: int = 8192,
        redact_provenance: bool = True,
        mode: DrafterMode = "batch",
        max_workers: int = 8,
    ) -> None:
        self.provider = provider or AnthropicProvider()
        self.max_tokens = max_tokens
        self.redact_provenance = redact_provenance
        self.mode: DrafterMode = mode
        self.max_workers = max_workers
        # Aggregate usage across per_card workers is stored here so the
        # eval runner sees consistent tokens-and-latency regardless of
        # which mode produced the draft.
        self.last_response: ProviderResponse | None = None

    @property
    def model(self) -> str:
        return self.provider.model

    def _serialize_bundle(self, bundle: Bundle[PGxFinding]) -> str:
        """Render the bundle as JSON for the provider, with optional
        provenance redaction. The Verifier still sees the original
        Bundle object — only the prompt payload is affected.
        """
        items: list[dict[str, Any]] = []
        for finding in bundle.items:
            row = asdict(finding)
            if self.redact_provenance:
                # Drop raw variant calls — they identify a specific
                # genome at chromosomal positions, and the LLM never
                # needs them for prose synthesis.
                row["source_variants"] = []
            items.append(row)
        return json.dumps(items, sort_keys=True, default=str)

    def draft(self, bundle: Bundle[PGxFinding]) -> Draft:
        if bundle.privacy_tier == PrivacyTier.LOCAL_ONLY:
            raise PrivacyViolation(
                f"LLMDrafter ({self.provider.name}) refuses to run on "
                f"LOCAL_ONLY bundle (source={bundle.source!r}). Use "
                f"OllamaDrafter for personal genome data."
            )
        if self.mode == "per_card":
            return self._draft_per_card(bundle)
        return self._draft_batch(bundle)

    def _draft_batch(self, bundle: Bundle[PGxFinding]) -> Draft:
        bundle_json = self._serialize_bundle(bundle)
        resp = self.provider.generate(
            system=SYSTEM_PROMPT,
            user=f"Bundle (JSON):\n{bundle_json}",
            schema=OUTPUT_SCHEMA,
            max_tokens=self.max_tokens,
        )
        self.last_response = resp

        data = json.loads(resp.text)
        cards = tuple(
            DraftedCard(
                gene=c["gene"],
                diplotype=c["diplotype"],
                phenotype=c["phenotype"],
                drug=c["drug"],
                recommendation=c["recommendation"],
                cited_pmids=tuple(c["cited_pmids"]),
            )
            for c in data["cards"]
        )
        return Draft(cards=cards, raw_text=resp.text)

    def _draft_per_card(self, bundle: Bundle[PGxFinding]) -> Draft:
        """Fan out one provider call per (finding, drug) pair.

        A worker may return `(None, resp)` if the model declined to
        emit a card (typically for Unknown/Uncertain phenotype pairs
        where there's no actionable recommendation). Those slots are
        filtered out — token usage is still accounted for.
        """
        pairs = [
            (f, d) for f in bundle.items for d in f.affected_drugs
        ]
        if not pairs:
            self.last_response = None
            return Draft(cards=(), raw_text="")

        n_workers = min(self.max_workers, len(pairs))
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            future_results = [
                exe.submit(self._draft_one_pair, bundle, f, d)
                for f, d in pairs
            ]
            results = [fut.result() for fut in future_results]

        cards = tuple(r[0] for r in results if r[0] is not None)
        per_card_responses = [r[1] for r in results]

        # Aggregate ProviderResponse so the eval runner sees a single,
        # consistent picture regardless of mode. raw_text is a marker;
        # individual responses are not retained at this level (they
        # were temporary per-worker).
        self.last_response = ProviderResponse(
            text=f"<per_card x{len(pairs)}>",
            raw=None,
            input_tokens=sum(r.input_tokens for r in per_card_responses),
            output_tokens=sum(r.output_tokens for r in per_card_responses),
            cache_read_tokens=sum(
                r.cache_read_tokens for r in per_card_responses
            ),
            cache_creation_tokens=sum(
                r.cache_creation_tokens for r in per_card_responses
            ),
        )
        return Draft(cards=cards, raw_text=self.last_response.text)

    def _draft_one_pair(
        self,
        bundle: Bundle[PGxFinding],
        finding: PGxFinding,
        drug: DrugRec,
    ) -> tuple[DraftedCard | None, ProviderResponse]:
        """Single-pair worker. Builds a one-finding-one-drug mini-bundle,
        calls the provider, parses the single card. Used by per_card mode.

        The model occasionally returns `{"cards": []}` — typically for
        Unknown / Uncertain Susceptibility phenotypes where there's no
        actionable CPIC recommendation. The caller filters None out.
        """
        mini = Bundle(
            items=(
                PGxFinding(
                    gene=finding.gene,
                    diplotype=finding.diplotype,
                    source_variants=finding.source_variants,
                    phenotype=finding.phenotype,
                    phenotype_source=finding.phenotype_source,
                    affected_drugs=(drug,),
                    confidence=finding.confidence,
                ),
            ),
            privacy_tier=bundle.privacy_tier,
            source=bundle.source,
            schema_version=bundle.schema_version,
            metadata=bundle.metadata,
        )
        bundle_json = self._serialize_bundle(mini)
        resp = self.provider.generate(
            system=SYSTEM_PROMPT,
            user=f"Bundle (JSON):\n{bundle_json}",
            schema=OUTPUT_SCHEMA,
            max_tokens=self.max_tokens,
        )
        data = json.loads(resp.text)
        cards_data = data.get("cards") or []
        if not cards_data:
            return None, resp
        c = cards_data[0]
        card = DraftedCard(
            gene=c["gene"],
            diplotype=c["diplotype"],
            phenotype=c["phenotype"],
            drug=c["drug"],
            recommendation=c["recommendation"],
            cited_pmids=tuple(c["cited_pmids"]),
        )
        return card, resp


class TriagingDrafter(Drafter):
    """Drafter that routes each (finding, drug) pair through a Triage step.

    Triage decisions:
      - `template` cases go to a `TemplateDrafter` (no API call).
      - `llm` cases are bundled into a trimmed sub-Bundle and passed to
        a wrapped `LLMDrafter` in a single API call.
      - `skip` cases produce no card.

    Cost behavior: zero API calls when all cases are template-able or
    skipped. Otherwise one API call for whatever's left. The wrapped
    LLMDrafter's privacy refusal still applies — but only when there
    is actual LLM work to do, so a fully-Normal LOCAL_ONLY bundle can
    still be drafted by templates alone (privacy-preserving by design).
    """

    def __init__(
        self,
        llm_drafter: "LLMDrafter | None" = None,
        triage: "object | None" = None,
        template_drafter: "object | None" = None,
    ) -> None:
        # Imported here to avoid an import cycle: pgx_digest.triage
        # imports DraftedCard from this module.
        from pgx_digest.triage import TemplateDrafter, Triage

        self.llm_drafter = llm_drafter or LLMDrafter()
        self.triage = triage or Triage()
        self.template_drafter = template_drafter or TemplateDrafter()
        # Mirrors LLMDrafter's contract so the eval runner can read tokens.
        self.last_response: ProviderResponse | None = None
        # Audit trail — populated on each draft() call.
        self.last_decisions: tuple[tuple[str, str, str], ...] = ()

    def draft(self, bundle: Bundle[PGxFinding]) -> Draft:
        llm_pairs: list[tuple[PGxFinding, DrugRec]] = []
        template_pairs: list[tuple[PGxFinding, DrugRec]] = []
        decisions: list[tuple[str, str, str]] = []  # (gene, drug, route)

        for finding in bundle.items:
            for drug in finding.affected_drugs:
                decision = self.triage.classify(finding, drug)
                decisions.append(
                    (finding.gene, drug.drug, decision.route)
                )
                if decision.route == "llm":
                    llm_pairs.append((finding, drug))
                elif decision.route == "template":
                    template_pairs.append((finding, drug))
                # "skip" → no card emitted

        self.last_decisions = tuple(decisions)

        # Template cards: deterministic, no API.
        template_cards = tuple(
            self.template_drafter.draft_card(f, d)
            for f, d in template_pairs
        )

        # LLM cards: one API call on a trimmed bundle, or none if empty.
        llm_cards: tuple[DraftedCard, ...] = ()
        raw_text = ""
        if llm_pairs:
            llm_bundle = _trim_bundle_to_pairs(bundle, llm_pairs)
            llm_draft = self.llm_drafter.draft(llm_bundle)
            self.last_response = self.llm_drafter.last_response
            llm_cards = llm_draft.cards
            raw_text = llm_draft.raw_text
        else:
            self.last_response = None

        return Draft(cards=llm_cards + template_cards, raw_text=raw_text)


def _trim_bundle_to_pairs(
    bundle: Bundle[PGxFinding],
    pairs: list[tuple[PGxFinding, DrugRec]],
) -> Bundle[PGxFinding]:
    """Return a Bundle containing only the (finding, drug) pairs given.

    Identity (`id()`) is used to match, because two `DrugRec` instances
    with equal fields would compare equal under `==`, but we want to
    keep exactly the instances Triage classified.
    """
    kept_ids = {(id(f), id(d)) for f, d in pairs}
    new_findings: list[PGxFinding] = []
    for finding in bundle.items:
        kept_drugs = tuple(
            d
            for d in finding.affected_drugs
            if (id(finding), id(d)) in kept_ids
        )
        if not kept_drugs:
            continue
        new_findings.append(
            PGxFinding(
                gene=finding.gene,
                diplotype=finding.diplotype,
                source_variants=finding.source_variants,
                phenotype=finding.phenotype,
                phenotype_source=finding.phenotype_source,
                affected_drugs=kept_drugs,
                confidence=finding.confidence,
            )
        )
    return Bundle(
        items=tuple(new_findings),
        privacy_tier=bundle.privacy_tier,
        source=bundle.source,
        schema_version=bundle.schema_version,
        metadata=bundle.metadata,
    )


class OllamaDrafter(Drafter):
    """Local Drafter backed by Ollama. Not yet implemented.

    Reserved for the LOCAL_ONLY privacy path. Same output contract as
    LLMDrafter — `draft()` will return a `Draft` once wired up.
    """

    def __init__(self, model: str = "qwen2.5:7b-instruct") -> None:
        self.model = model

    def draft(self, bundle: Bundle[PGxFinding]) -> Draft:
        raise NotImplementedError(
            "OllamaDrafter is pending implementation. Use LLMDrafter "
            "for PUBLIC bundles."
        )


def select_drafter(bundle: Bundle[PGxFinding]) -> Drafter:
    """Default privacy-tier-aware Drafter selection."""
    if bundle.privacy_tier == PrivacyTier.LOCAL_ONLY:
        return OllamaDrafter()
    return LLMDrafter()
