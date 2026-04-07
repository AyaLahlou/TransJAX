"""
Base Agent class with Claude API integration.

All specialized agents inherit from this base class.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from tenacity import retry, stop_after_attempt, wait_exponential

from transjax.agents.utils.tmux_runner import TmuxClaudeRunner

# Load environment variables
load_dotenv()

# Setup logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)
console = Console()


class BaseAgent:
    """
    Base class for all LLM agents.

    Provides common functionality:
    - Claude API integration
    - Conversation history management
    - Retry logic for API calls
    - Logging and cost tracking
    """

    def __init__(
        self,
        name: str,
        role: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 48000,
        use_tmux: bool = False,
        tmux_poll_interval: float = 2.0,
        tmux_timeout: float = 900.0,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name (e.g., "Orchestrator", "Static Analysis")
            role: Agent role description
            model: Claude model to use
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            use_tmux: If True, route all claude calls through a tmux session
                      running the ``claude`` CLI instead of the Anthropic SDK.
                      Recommended for HPC / long-running jobs.
            tmux_poll_interval: Seconds between output-file polls (tmux mode only).
            tmux_timeout: Max seconds to wait for a single claude call (tmux mode).
        """
        self.name = name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_tmux = use_tmux
        self.tmux_poll_interval = tmux_poll_interval
        self.tmux_timeout = tmux_timeout

        # Set by the pipeline/orchestrator before processing each module
        self.tmux_session_name: Optional[str] = None
        self.tmux_work_dir: Optional[Path] = None
        self._tmux_runner: Optional[TmuxClaudeRunner] = None

        # Initialize Anthropic client (used when use_tmux=False).
        # Authentication priority:
        #   1. CLAUDE_CODE_OAUTH_TOKEN — set automatically by `claude login`
        #      (Claude Pro / Max subscription, no per-token billing)
        #   2. ANTHROPIC_API_KEY — traditional pay-per-use API key
        oauth_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if use_tmux:
            # In tmux mode the claude CLI handles authentication itself.
            # We still initialise a lightweight client for fallback / cost tracking.
            self.client = None
            self._auth_method = "tmux_cli"
        elif oauth_token:
            self.client = anthropic.Anthropic(auth_token=oauth_token)
            self._auth_method = "subscription"
        elif api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            self._auth_method = "api_key"
        else:
            raise ValueError(
                "No Claude authentication found. Choose one of:\n"
                "  • Subscription login: run `claude login` (Claude Pro/Max)\n"
                "  • API key: set ANTHROPIC_API_KEY in your environment or .env file\n"
                "  • Tmux CLI mode: set use_tmux=True (uses `claude` CLI auth)"
            )

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        logger.info(
            f"[bold green]Initialized {self.name} Agent[/bold green] "
            f"[dim](auth: {self._auth_method})[/dim]",
            extra={"markup": True},
        )

    def set_tmux_session(self, session_name: str, work_dir: Path) -> None:
        """
        Attach this agent to a named tmux session for the current module.

        Called by PipelineRunner / OrchestratorAgent before processing each module
        when ``use_tmux=True``.

        Args:
            session_name: Tmux session name (e.g. ``transjax-clm_varcon``).
            work_dir:     Directory used for prompt/output temp files.
        """
        self.tmux_session_name = session_name
        self.tmux_work_dir = Path(work_dir)
        self._tmux_runner = TmuxClaudeRunner(
            session_name=session_name,
            work_dir=self.tmux_work_dir,
            poll_interval=self.tmux_poll_interval,
            timeout=self.tmux_timeout,
            verbose=False,
        )
        logger.info("%s attached to tmux session: %s", self.name, session_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=10, max=60)
    )
    def query_claude(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Query Claude with retry logic.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional, uses agent role if not provided)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Claude's response text

        Raises:
            anthropic.APIError: If API call fails after retries
        """
        # Use defaults if not overridden
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        sys_prompt = system_prompt if system_prompt is not None else self._get_default_system_prompt()

        # Log the query
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_interaction(f"PROMPT_{timestamp}", prompt, sys_prompt)

        # ── Tmux CLI path ──────────────────────────────────────────────────
        if self.use_tmux and self._tmux_runner is not None:
            console.print(f"[cyan]🖥  {self.name} → tmux session '{self.tmux_session_name}'[/cyan]")
            response_text = self._tmux_runner.run_query(
                prompt=prompt,
                system_prompt=sys_prompt,
            )
            self._log_interaction(f"RESPONSE_{timestamp}", response_text)
            # Token counts unavailable in tmux mode
            return response_text

        if self.use_tmux and self._tmux_runner is None:
            raise RuntimeError(
                f"{self.name}: use_tmux=True but no tmux session set. "
                "Call set_tmux_session() before querying."
            )

        try:
            # Make API call
            console.print(f"[cyan]🤖 {self.name} is thinking...[/cyan]")

            # Use streaming for large max_tokens to avoid timeout issues
            # Anthropic requires streaming for requests that may take >10 minutes
            if max_tok >= 10000:
                console.print(f"[dim]Using streaming mode for {max_tok} max_tokens[/dim]")

                response_text = ""
                input_tokens = 0
                output_tokens = 0

                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tok,
                    temperature=temp,
                    system=sys_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                ) as stream:
                    for text in stream.text_stream:
                        response_text += text

                    # Get final usage from stream
                    final_message = stream.get_final_message()
                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens

                # Update token counts
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                # Log the response
                self._log_interaction(f"RESPONSE_{timestamp}", response_text)

                # Log token usage
                logger.info(
                    f"{self.name}: Used {input_tokens} input + "
                    f"{output_tokens} output tokens"
                )

                return response_text
            else:
                # Non-streaming for smaller requests
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tok,
                    temperature=temp,
                    system=sys_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Extract response text
                response_text = response.content[0].text

                # Update token counts
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens

                # Log the response
                self._log_interaction(f"RESPONSE_{timestamp}", response_text)

                # Log token usage
                logger.info(
                    f"{self.name}: Used {response.usage.input_tokens} input + "
                    f"{response.usage.output_tokens} output tokens"
                )

                return response_text

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def multi_turn_conversation(
        self,
        initial_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Start a multi-turn conversation with Claude.

        Args:
            initial_prompt: Initial user prompt
            system_prompt: System prompt for the conversation

        Returns:
            Claude's response to initial prompt
        """
        # Clear conversation history
        self.conversation_history = []

        # Add initial message
        self.conversation_history.append({
            "role": "user",
            "content": initial_prompt
        })

        # Get response
        sys_prompt = system_prompt if system_prompt is not None else self._get_default_system_prompt()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=sys_prompt,
            messages=self.conversation_history
        )

        # Add response to history
        response_text = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })

        # Update token counts
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return response_text

    def continue_conversation(self, prompt: str) -> str:
        """
        Continue an existing conversation.

        Args:
            prompt: User's next message

        Returns:
            Claude's response
        """
        if not self.conversation_history:
            raise ValueError("No conversation started. Use multi_turn_conversation() first.")

        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })

        # Get response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self._get_default_system_prompt(),
            messages=self.conversation_history
        )

        # Add response to history
        response_text = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })

        # Update token counts
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return response_text

    def get_cost_estimate(self) -> Dict[str, float]:
        """
        Estimate cost based on token usage.

        Returns:
            Dictionary with cost breakdown.  When authenticated via a Claude
            subscription (``claude login``) costs are covered by the subscription
            and the USD figures will be 0.
        """
        if self._auth_method == "tmux_cli":
            # Token counts not available from claude CLI; cost is subscription-based.
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "note": "Running via claude CLI in tmux (token counts unavailable)",
            }

        if self._auth_method == "subscription":
            # Subscription users are not billed per-token.
            return {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "note": "Covered by Claude subscription (claude login)",
            }

        input_cost = (self.total_input_tokens / 1_000_000) * 3.00
        output_cost = (self.total_output_tokens / 1_000_000) * 15.00
        total_cost = input_cost + output_cost

        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
        }

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this agent."""
        return f"""You are an expert {self.name} agent specializing in converting Fortran code to JAX.

Your role: {self.role}

You have deep expertise in:
- Fortran 90/95 and modern Fortran
- Python and JAX (Google's numerical computing library)
- Functional programming and immutable data structures
- Scientific computing and numerical modeling

You follow these principles:
1. Preserve scientific accuracy - physics must match exactly
2. Use JAX best practices (pure functions, immutable state, JIT-compatible code)
3. Add comprehensive documentation and type hints
4. Follow the patterns established in existing JAX translations
5. Be precise and thorough in your analysis and code generation

You communicate clearly and provide detailed, actionable responses."""

    def _log_interaction(self, label: str, content: str, system_prompt: Optional[str] = None) -> None:
        """
        Log agent interactions to file.

        Args:
            label: Label for the log entry
            content: Content to log
            system_prompt: Optional system prompt to log
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name.lower().replace(' ', '_')}_{timestamp}.log"

        with open(log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{label}\n")
            f.write(f"{'='*80}\n")
            if system_prompt:
                f.write(f"\nSYSTEM PROMPT:\n{system_prompt}\n\n")
            f.write(f"{content}\n")

    def save_state(self, output_path: Path) -> None:
        """
        Save agent state to file.

        Args:
            output_path: Path to save state
        """
        import json

        state = {
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "cost_estimate": self.get_cost_estimate(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved {self.name} state to {output_path}")

    def reset(self) -> None:
        """Reset agent state (conversation history, token counts)."""
        self.conversation_history = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        logger.info(f"Reset {self.name} agent state")

