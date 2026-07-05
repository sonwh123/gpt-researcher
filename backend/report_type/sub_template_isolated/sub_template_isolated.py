import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from gpt_researcher import GPTResearcher
from gpt_researcher.actions import choose_agent
from gpt_researcher.actions.query_processing import decompose_template_to_sectioned_sub_queries
from gpt_researcher.utils.enum import ReportType
from gpt_researcher.utils.llm import create_chat_completion
from gpt_researcher.utils.template import get_leaf_nodes, parse_template_outline


class SubTemplateIsolated:
    """Orchestrator for report_type "sub_template_isolated".

    Unlike "sub_template" (which pools all sub-query search results into one
    context and writes the whole report in a single LLM call), this mode
    decomposes the template once, then plans, searches, and writes each leaf
    (a Sub Section, or a bare Section with no Sub Sections) as an independent
    unit, and concatenates the leaves' own written content in template order.

    This is intentionally lighter than DetailedReport: it owns exactly one
    GPTResearcher instance, used only as a shared handle for config/retrievers/
    research_conductor/cost-tracking - conduct_research()/write_report() are
    never called on it, since this class fully owns planning/search/write for
    the isolated flow itself.
    """

    def __init__(
        self,
        query: str,
        template: str,
        report_source: str = "web",
        source_urls: List[str] = [],
        document_urls: List[str] = [],
        query_domains: List[str] = [],
        config_path: str = None,
        tone: Any = "",
        websocket: Optional[WebSocket] = None,
        headers: Optional[Dict] = None,
        complement_source_urls: bool = False,
        mcp_configs=None,
        mcp_strategy=None,
        max_search_results=None,
        max_concurrent_leaves: int = 3,
    ):
        self.query = query
        self.template = template
        self.websocket = websocket

        self.outline = parse_template_outline(template)
        if not self.outline:
            raise ValueError(
                "sub_template_isolated requires a template using the "
                "'Section N: ...' / '  Sub Section N.M: ...' convention "
                "(see gpt_researcher/utils/template.py); no sections were "
                "found in the provided template."
            )
        self.leaf_nodes = get_leaf_nodes(self.outline)

        # Generate a unique research ID for this report
        self.research_id = self._generate_research_id(query)

        gpt_researcher_params = {
            "query": query,
            "query_domains": query_domains,
            "report_type": ReportType.SubTemplateIsolated.value,
            "report_source": report_source,
            "source_urls": source_urls,
            "document_urls": document_urls,
            "config_path": config_path,
            "tone": tone,
            "websocket": websocket,
            "headers": headers or {},
            "complement_source_urls": complement_source_urls,
            "template": template,
        }
        if mcp_configs is not None:
            gpt_researcher_params["mcp_configs"] = mcp_configs
        if mcp_strategy is not None:
            gpt_researcher_params["mcp_strategy"] = mcp_strategy

        self.gpt_researcher = GPTResearcher(**gpt_researcher_params)

        if max_search_results is not None:
            self.gpt_researcher.cfg.max_search_results_per_query = int(max_search_results)

        self._leaf_semaphore = asyncio.Semaphore(max_concurrent_leaves)

    def _generate_research_id(self, query: str) -> str:
        """Generate a unique research ID from query and timestamp."""
        timestamp = str(int(time.time()))
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"sub_template_isolated_{timestamp}_{query_hash}"

    async def run(self) -> str:
        await self._choose_agent()
        leaf_query_map = await self._decompose_template()
        leaf_blocks = await self._process_all_leaves(leaf_query_map)
        return self._assemble_report(leaf_blocks)

    async def _choose_agent(self) -> None:
        gr = self.gpt_researcher
        if not (gr.agent and gr.role):
            gr.agent, gr.role = await choose_agent(
                query=gr.query,
                cfg=gr.cfg,
                parent_query=gr.parent_query,
                cost_callback=gr.add_costs,
                headers=gr.headers,
                prompt_family=gr.prompt_family,
                **gr.kwargs,
            )

    async def _decompose_template(self) -> Dict[str, List[str]]:
        gr = self.gpt_researcher
        leaf_headings = [leaf["heading"] for leaf in self.leaf_nodes]
        sectioned = await decompose_template_to_sectioned_sub_queries(
            query=self.query,
            template=self.template,
            leaf_headings=leaf_headings,
            search_results=[],
            cfg=gr.cfg,
            cost_callback=gr.add_costs,
            prompt_family=gr.prompt_family,
        )
        return {item["heading"]: item["queries"] for item in sectioned}

    async def _process_all_leaves(self, leaf_query_map: Dict[str, List[str]]) -> Dict[str, str]:
        async def _one(leaf):
            async with self._leaf_semaphore:
                body = await self._process_leaf(leaf, leaf_query_map.get(leaf["heading"], []))
                return leaf["heading"], body

        pairs = await asyncio.gather(*(_one(leaf) for leaf in self.leaf_nodes))
        return dict(pairs)

    async def _process_leaf(self, leaf: Dict, queries: List[str]) -> str:
        gr = self.gpt_researcher
        if queries:
            context = await gr.research_conductor._search_and_join(
                queries, scraped_data=[], query_domains=gr.query_domains
            )
        else:
            context = ""

        prompt = gr.prompt_family.generate_sub_template_leaf_prompt(
            question=self.query,
            context=context,
            heading=leaf["heading"],
            report_source=gr.report_source,
            report_format=gr.cfg.report_format,
            tone=gr.tone,
            language=gr.cfg.language,
        )

        messages = [
            {"role": "system", "content": gr.cfg.agent_role or gr.role},
            {"role": "user", "content": prompt},
        ]
        try:
            return await create_chat_completion(
                model=gr.cfg.smart_llm_model,
                messages=messages,
                temperature=0.35,
                llm_provider=gr.cfg.smart_llm_provider,
                max_tokens=gr.cfg.smart_token_limit,
                llm_kwargs=gr.cfg.llm_kwargs,
                cost_callback=gr.add_costs,
            )
        except Exception:
            return await create_chat_completion(
                model=gr.cfg.smart_llm_model,
                messages=[{"role": "user", "content": f"{gr.cfg.agent_role or gr.role}\n\n{prompt}"}],
                temperature=0.35,
                llm_provider=gr.cfg.smart_llm_provider,
                max_tokens=gr.cfg.smart_token_limit,
                llm_kwargs=gr.cfg.llm_kwargs,
                cost_callback=gr.add_costs,
            )

    def _assemble_report(self, leaf_blocks: Dict[str, str]) -> str:
        parts: List[str] = []
        for section in self.outline:
            if section["subsections"]:
                parts.append(f"## {section['heading']}")
                for sub in section["subsections"]:
                    body = leaf_blocks.get(sub["heading"], "")
                    parts.append(f"### {sub['heading']}\n\n{body}".rstrip())
            else:
                body = leaf_blocks.get(section["heading"], "")
                parts.append(f"## {section['heading']}\n\n{body}".rstrip())
        return "\n\n".join(parts)
