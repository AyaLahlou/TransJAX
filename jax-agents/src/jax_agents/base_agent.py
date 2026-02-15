"""
Base Agent class with Claude API integration.

All specialized agents inherit from this base class.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

import anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console
from rich.logging import RichHandler

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
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.0,
        max_tokens: int = 48000,
    ):
        """Initialize base agent with API client and tracking."""
        self.name = name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Tracking states
        self.conversation_history: List[Dict[str, str]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Logging setup
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        logger.info(f"[bold green]Initialized {self.name} Agent[/bold green]", extra={"markup": True})

    def _update_usage(self, usage: Any) -> None:
        """Helper to update token totals from API response usage objects."""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens

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
        """Query Claude with retry logic and optional streaming for large outputs."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        sys_prompt = system_prompt if system_prompt is not None else self._get_default_system_prompt()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_interaction(f"PROMPT_{timestamp}", prompt, sys_prompt)
        
        try:
            console.print(f"[cyan]🤖 {self.name} is thinking...[/cyan]")
            
            # Streaming path for high-token/long-running requests
            if max_tok >= 10000:
                console.print(f"[dim]Using streaming mode for {max_tok} max_tokens[/dim]")
                response_text = ""
                
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tok,
                    temperature=temp,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        response_text += text
                    
                    final_msg = stream.get_final_message()
                    self._update_usage(final_msg.usage)
            else:
                # Standard path
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tok,
                    temperature=temp,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
                self._update_usage(response.usage)
            
            self._log_interaction(f"RESPONSE_{timestamp}", response_text)
            logger.info(f"{self.name}: Interaction complete.")
            return response_text
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def multi_turn_conversation(
        self,
        initial_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Start a new multi-turn conversation history."""
        self.conversation_history = [{"role": "user", "content": initial_prompt}]
        return self._send_conversation_request(system_prompt)

    def continue_conversation(self, prompt: str) -> str:
        """Append to existing conversation history and get response."""
        if not self.conversation_history:
            raise ValueError("No conversation started. Use multi_turn_conversation() first.")
        
        self.conversation_history.append({"role": "user", "content": prompt})
        return self._send_conversation_request()

    def _send_conversation_request(self, system_prompt: Optional[str] = None) -> str:
        """Internal helper to handle conversation API calls."""
        sys_prompt = system_prompt if system_prompt is not None else self._get_default_system_prompt()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=sys_prompt,
            messages=self.conversation_history
        )
        
        response_text = response.content[0].text
        self.conversation_history.append({"role": "assistant", "content": response_text})
        self._update_usage(response.usage)
        return response_text

    def get_cost_estimate(self) -> Dict[str, float]:
        """Estimate cost based on current token usage for Claude Sonnet 4.5."""
        input_cost = (self.total_input_tokens / 1_000_000) * 3.00
        output_cost = (self.total_output_tokens / 1_000_000) * 15.00
        
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }

    def _get_default_system_prompt(self) -> str:
        """Returns the specialized system prompt for JAX translation."""
        return f"""You are an expert {self.name} agent specializing in converting Fortran CTSM code to JAX.

Your role: {self.role}

You have deep expertise in:
- Fortran 90/95 and modern Fortran
- Python and JAX (Google's numerical computing library)
- Functional programming and immutable data structures
- Scientific computing and Earth system modeling
- The CTSM (Community Terrestrial Systems Model) codebase

You follow these principles:
1. Preserve scientific accuracy - physics must match exactly
2. Use JAX best practices (pure functions, immutable state, JIT-compatible code)
3. Add comprehensive documentation and type hints
4. Follow the patterns established in the existing jax-ctsm codebase
5. Be precise and thorough in your analysis and code generation"""

    def _log_interaction(self, label: str, content: str, system_prompt: Optional[str] = None) -> None:
        """Log interactions to a timestamped file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name.lower().replace(' ', '_')}_{timestamp}.log"
        
        with log_file.open("a") as f:
            f.write(f"\n{'='*80}\n{label}\n{'='*80}\n")
            if system_prompt:
                f.write(f"\nSYSTEM PROMPT:\n{system_prompt}\n\n")
            f.write(f"{content}\n")

    def save_state(self, output_path: Path) -> None:
        """Save current agent metadata and cost state to JSON."""
        state = {
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "cost_estimate": self.get_cost_estimate(),
            "timestamp": datetime.now().isoformat(),
        }
        
        with output_path.open("w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved {self.name} state to {output_path}")

    def reset(self) -> None:
        """Reset conversation and token totals."""
        self.conversation_history = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        logger.info(f"Reset {self.name} agent state")