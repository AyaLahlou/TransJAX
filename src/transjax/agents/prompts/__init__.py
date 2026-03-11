"""Prompt templates for Fortran-to-JAX translation agents."""

from transjax.agents.prompts.repair_prompts import REPAIR_PROMPTS
from transjax.agents.prompts.translation_prompts import TRANSLATION_PROMPTS

__all__ = [
    "TRANSLATION_PROMPTS",
    "REPAIR_PROMPTS",
]

