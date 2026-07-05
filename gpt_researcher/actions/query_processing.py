import json_repair
import re

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion
from ..prompts import PromptFamily
from ..utils.enum import ReportType
from typing import Any, List, Dict
from ..config import Config
import logging

logger = logging.getLogger(__name__)

async def get_search_results(query: str, retriever: Any, query_domains: List[str] = None, researcher=None) -> List[Dict[str, Any]]:
    """
    Get web search results for a given query.

    Args:
        query: The search query
        retriever: The retriever instance
        query_domains: Optional list of domains to search
        researcher: The researcher instance (needed for MCP retrievers)

    Returns:
        A list of search results
    """
    # Check if this is an MCP retriever and pass the researcher instance
    if "mcpretriever" in retriever.__name__.lower():
        search_retriever = retriever(
            query, 
            query_domains=query_domains,
            researcher=researcher  # Pass researcher instance for MCP retrievers
        )
    else:
        search_retriever = retriever(query, query_domains=query_domains)
    
    return search_retriever.search()

async def generate_sub_queries(
    query: str,
    parent_query: str,
    report_type: str,
    context: List[Dict[str, Any]],
    cfg: Config,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> List[str]:
    """
    Generate sub-queries using the specified LLM model.

    Args:
        query: The original query
        parent_query: The parent query
        report_type: The type of report
        max_iterations: Maximum number of research iterations
        context: Search results context
        cfg: Configuration object
        cost_callback: Callback for cost calculation
        prompt_family: Family of prompts

    Returns:
        A list of sub-queries
    """
    gen_queries_prompt = prompt_family.generate_search_queries_prompt(
        query,
        parent_query,
        report_type,
        max_iterations=cfg.max_iterations or 3,
        context=context,
    )

    try:
        response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=[{"role": "user", "content": gen_queries_prompt}],
            llm_provider=cfg.strategic_llm_provider,
            max_tokens=None,
            llm_kwargs=cfg.llm_kwargs,
            reasoning_effort=ReasoningEfforts.Medium.value,
            cost_callback=cost_callback,
            **kwargs
        )
    except Exception as e:
        logger.warning(f"Error with strategic LLM: {e}. Retrying with max_tokens={cfg.strategic_token_limit}.")
        logger.warning(f"See https://github.com/assafelovic/gpt-researcher/issues/1022")
        try:
            response = await create_chat_completion(
                model=cfg.strategic_llm_model,
                messages=[{"role": "user", "content": gen_queries_prompt}],
                max_tokens=cfg.strategic_token_limit,
                llm_provider=cfg.strategic_llm_provider,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs
            )
            logger.warning(f"Retrying with max_tokens={cfg.strategic_token_limit} successful.")
        except Exception as e:
            logger.warning(f"Retrying with max_tokens={cfg.strategic_token_limit} failed.")
            logger.warning(f"Error with strategic LLM: {e}. Falling back to smart LLM.")
            response = await create_chat_completion(
                model=cfg.smart_llm_model,
                messages=[{"role": "user", "content": gen_queries_prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.smart_token_limit,
                llm_provider=cfg.smart_llm_provider,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs
            )

    return json_repair.loads(response)


async def decompose_template_to_sub_queries(
    query: str,
    template: str,
    search_results: List[Dict[str, Any]],
    cfg: Config,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> List[str]:
    """Decompose a report template into per-section sub-queries (DecomposedIR).

    Mirrors ``generate_sub_queries`` (LLM call + fallback) but uses the
    template-decomposition prompt instead of the generic search-query prompt.

    Args:
        query: The user's overall research task (supplies time/entity context).
        template: The normalized text outline of the report template.
        search_results: Initial search results (unused; kept for signature symmetry).
        cfg: Configuration object.
        cost_callback: Callback for cost calculation.
        prompt_family: Family of prompts.

    Returns:
        A flat list of sub-queries in the template's section order.
    """
    decomposition_prompt = prompt_family.generate_template_decomposition_prompt(
        template,
        query,
        context="",
    )

    try:
        response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=[{"role": "user", "content": decomposition_prompt}],
            llm_provider=cfg.strategic_llm_provider,
            max_tokens=None,
            llm_kwargs=cfg.llm_kwargs,
            reasoning_effort=ReasoningEfforts.Medium.value,
            cost_callback=cost_callback,
            **kwargs
        )
    except Exception as e:
        logger.warning(
            f"Error with strategic LLM during template decomposition: {e}. "
            f"Falling back to smart LLM."
        )
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[{"role": "user", "content": decomposition_prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.smart_token_limit,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )

    sub_queries = json_repair.loads(response)

    # Be defensive: normalize to a flat list of non-empty query strings.
    if isinstance(sub_queries, dict):
        flattened: List[str] = []
        for value in sub_queries.values():
            if isinstance(value, list):
                flattened.extend(str(v) for v in value)
            else:
                flattened.append(str(value))
        sub_queries = flattened
    if not isinstance(sub_queries, list):
        sub_queries = [str(sub_queries)]
    sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()]

    return sub_queries


def _normalize_heading(text: str) -> str:
    """Whitespace-collapsed, casefolded form used for lenient heading matching."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


async def decompose_template_to_sectioned_sub_queries(
    query: str,
    template: str,
    leaf_headings: List[str],
    search_results: List[Dict[str, Any]],
    cfg: Config,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> List[Dict[str, Any]]:
    """Decompose a report template into per-leaf-heading sub-queries, tagged by
    heading, for report_type "sub_template_isolated".

    Sibling of ``decompose_template_to_sub_queries`` (same LLM call + strategic
    -> smart fallback pattern) but the LLM is asked to tag its output by the
    given ``leaf_headings`` instead of returning one flat, untagged list.

    Args:
        query: The user's overall research task (supplies time/entity context).
        template: The normalized text outline of the report template.
        leaf_headings: The ordered list of leaf headings to decompose (from
            ``gpt_researcher.utils.template.get_leaf_nodes``).
        search_results: Initial search results (unused; kept for signature
            symmetry with ``decompose_template_to_sub_queries``).
        cfg: Configuration object.
        cost_callback: Callback for cost calculation.
        prompt_family: Family of prompts.

    Returns:
        Exactly ``len(leaf_headings)`` entries, always in ``leaf_headings``'
        order, each ``{"heading": <the given heading string>, "queries": [...]}``.
        Never raises: if the LLM response can't be matched to a given heading,
        that heading gets ``queries: []`` and a warning is logged.
    """
    decomposition_prompt = prompt_family.generate_template_decomposition_prompt_sectioned(
        template,
        query,
        leaf_headings,
        context="",
    )

    try:
        response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=[{"role": "user", "content": decomposition_prompt}],
            llm_provider=cfg.strategic_llm_provider,
            max_tokens=None,
            llm_kwargs=cfg.llm_kwargs,
            reasoning_effort=ReasoningEfforts.Medium.value,
            cost_callback=cost_callback,
            **kwargs
        )
    except Exception as e:
        logger.warning(
            f"Error with strategic LLM during sectioned template decomposition: {e}. "
            f"Falling back to smart LLM."
        )
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[{"role": "user", "content": decomposition_prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.smart_token_limit,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )

    parsed = json_repair.loads(response)
    if not isinstance(parsed, list):
        parsed = [parsed]

    # Defensively normalize each parsed entry to {"heading": str, "queries": [str, ...]}.
    parsed_entries: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading", "")).strip()
        raw_queries = item.get("queries", [])
        queries = (
            [str(q).strip() for q in raw_queries if str(q).strip()]
            if isinstance(raw_queries, list)
            else []
        )
        parsed_entries.append({"heading": heading, "queries": queries})

    # Match parsed entries back to leaf_headings: exact -> normalized -> positional -> empty.
    by_exact = {entry["heading"]: entry["queries"] for entry in parsed_entries}
    by_normalized = {
        _normalize_heading(entry["heading"]): entry["queries"] for entry in parsed_entries
    }
    positional_fallback = len(parsed_entries) == len(leaf_headings)

    result: List[Dict[str, Any]] = []
    for i, heading in enumerate(leaf_headings):
        if heading in by_exact:
            queries = by_exact[heading]
        elif _normalize_heading(heading) in by_normalized:
            queries = by_normalized[_normalize_heading(heading)]
        elif positional_fallback:
            queries = parsed_entries[i]["queries"]
        else:
            queries = []
            logger.warning(
                f"Could not match a decomposed sub-query group to leaf heading "
                f"'{heading}'; writing this leaf with no sub-queries."
            )
        result.append({"heading": heading, "queries": queries})

    return result


async def plan_research_outline(
    query: str,
    search_results: List[Dict[str, Any]],
    agent_role_prompt: str,
    cfg: Config,
    parent_query: str,
    report_type: str,
    cost_callback: callable = None,
    retriever_names: List[str] = None,
    template: str = None,
    **kwargs
) -> List[str]:
    """
    Plan the research outline by generating sub-queries.

    Args:
        query: Original query
        search_results: Initial search results
        agent_role_prompt: Agent role prompt
        cfg: Configuration object
        parent_query: Parent query
        report_type: Report type
        cost_callback: Callback for cost calculation
        retriever_names: Names of the retrievers being used

    Returns:
        A list of sub-queries
    """
    # Handle the case where retriever_names is not provided
    if retriever_names is None:
        retriever_names = []
    
    # For MCP retrievers, we may want to skip sub-query generation
    # Check if MCP is the only retriever or one of multiple retrievers
    if retriever_names and ("mcp" in retriever_names or "MCPRetriever" in retriever_names):
        mcp_only = (len(retriever_names) == 1 and 
                   ("mcp" in retriever_names or "MCPRetriever" in retriever_names))
        
        if mcp_only:
            # If MCP is the only retriever, skip sub-query generation
            logger.info("Using MCP retriever only - skipping sub-query generation")
            # Return the original query to prevent additional search iterations
            return [query]
        else:
            # If MCP is one of multiple retrievers, generate sub-queries for the others
            logger.info("Using MCP with other retrievers - generating sub-queries for non-MCP retrievers")

    # sub_template report type: decompose the template into per-section sub-queries
    if report_type == ReportType.SubTemplate.value and template:
        logger.info("sub_template report - decomposing template into sub-queries")
        return await decompose_template_to_sub_queries(
            query,
            template,
            search_results,
            cfg,
            cost_callback,
            **kwargs
        )

    # Generate sub-queries for research outline
    sub_queries = await generate_sub_queries(
        query,
        parent_query,
        report_type,
        search_results,
        cfg,
        cost_callback,
        **kwargs
    )

    return sub_queries
