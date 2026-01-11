#!/usr/bin/env python3
"""
Claim Extraction Module for KDS Hackathon 2026.

This module converts character backstories into atomic, machine-checkable claims.
These claims will later be verified against evidence retrieved from the novel.

This module is pure extraction — it does NOT reason about the novel.
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import os
from pathlib import Path

# Try to load `.env` if python-dotenv is installed; otherwise we'll
# attempt a manual .env read below when resolving the API key.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from openai import OpenAI


def _read_dotenv_manual() -> dict:
    """Manual loader for a simple .env file located at repo root.

    Returns a dict of key->value for lines like KEY=VALUE. Silent on errors.
    """
    env_path = Path(__file__).parent.parent / ".env"
    data = {}
    if not env_path.exists():
        return data
    try:
        text = env_path.read_text(encoding="utf-8")
    except Exception:
        return data

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip().strip("'\"")
    return data


def get_openai_api_key() -> Optional[str]:
    """Resolve the OpenAI API key from env or .env file.

    Preference: environment variable `OPENAI_API_KEY`. If missing,
    attempt a manual read of the project's `.env` file.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # Try manual .env parse
    env_map = _read_dotenv_manual()
    return env_map.get("OPENAI_API_KEY")


# Low temperature for deterministic, consistent output
LLM_TEMPERATURE = 0.1
LLM_MODEL = "gpt-4o-mini"  # Cost-effective, good at structured extraction


# =============================================================================
# Enums and Data Structures
# =============================================================================

class ClaimType(str, Enum):
    """Types of claims that can be extracted from a backstory."""
    BELIEF = "belief"        # worldview, ideology, assumptions
    BEHAVIOR = "behavior"    # habitual actions or tendencies
    BACKGROUND = "background"  # past facts (upbringing, history)
    MOTIVATION = "motivation"  # goals, drives, ambitions


class ClaimImportance(str, Enum):
    """Importance level of a claim."""
    CORE = "core"            # if violated, backstory becomes implausible
    SUPPORTING = "supporting"  # secondary detail that can wobble


@dataclass
class Claim:
    """A single atomic claim extracted from a backstory."""
    claim_id: str
    type: str
    text: str
    importance: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Claim':
        return cls(
            claim_id=data['claim_id'],
            type=data['type'],
            text=data['text'],
            importance=data['importance']
        )


# =============================================================================
# Validation
# =============================================================================

class ClaimValidationError(Exception):
    """Raised when claim validation fails."""
    pass


def validate_claim(claim_data: Dict[str, Any], index: int) -> Claim:
    """
    Validate a single claim dictionary and convert to Claim object.
    
    Args:
        claim_data: Dictionary containing claim fields
        index: Index of the claim for error messages
        
    Returns:
        Validated Claim object
        
    Raises:
        ClaimValidationError: If validation fails
    """
    # Check required fields exist
    required_fields = ['claim_id', 'type', 'text', 'importance']
    for field in required_fields:
        if field not in claim_data:
            raise ClaimValidationError(
                f"Claim {index}: Missing required field '{field}'"
            )
    
    # Validate type
    valid_types = {t.value for t in ClaimType}
    if claim_data['type'] not in valid_types:
        raise ClaimValidationError(
            f"Claim {index} ({claim_data.get('claim_id', '?')}): "
            f"Invalid type '{claim_data['type']}'. "
            f"Must be one of: {valid_types}"
        )
    
    # Validate importance
    valid_importance = {i.value for i in ClaimImportance}
    if claim_data['importance'] not in valid_importance:
        raise ClaimValidationError(
            f"Claim {index} ({claim_data.get('claim_id', '?')}): "
            f"Invalid importance '{claim_data['importance']}'. "
            f"Must be one of: {valid_importance}"
        )
    
    # Validate text is non-empty
    if not claim_data['text'] or not claim_data['text'].strip():
        raise ClaimValidationError(
            f"Claim {index} ({claim_data.get('claim_id', '?')}): "
            f"Text cannot be empty"
        )
    
    # Validate claim_id format (should be C1, C2, etc.)
    if not claim_data['claim_id'] or not isinstance(claim_data['claim_id'], str):
        raise ClaimValidationError(
            f"Claim {index}: Invalid claim_id '{claim_data.get('claim_id')}'"
        )
    
    return Claim.from_dict(claim_data)


def validate_claims(claims_data: List[Dict[str, Any]]) -> List[Claim]:
    """
    Validate a list of claim dictionaries.
    
    Args:
        claims_data: List of claim dictionaries from LLM
        
    Returns:
        List of validated Claim objects
        
    Raises:
        ClaimValidationError: If any validation fails
    """
    if not isinstance(claims_data, list):
        raise ClaimValidationError(
            f"Expected list of claims, got {type(claims_data).__name__}"
        )
    
    if len(claims_data) == 0:
        raise ClaimValidationError("No claims extracted from backstory")
    
    validated_claims = []
    for i, claim_data in enumerate(claims_data):
        if not isinstance(claim_data, dict):
            raise ClaimValidationError(
                f"Claim {i}: Expected dictionary, got {type(claim_data).__name__}"
            )
        validated_claims.append(validate_claim(claim_data, i))
    
    return validated_claims


# =============================================================================
# Prompt Template
# =============================================================================

EXTRACTION_PROMPT = """You are an information extraction system.

Task:
Given a character backstory, extract ALL atomic claims implied by the text.

Rules:
- Each claim must express ONE idea only.
- Claims must be objectively checkable against story events.
- Do NOT merge multiple ideas into one claim.
- Do NOT invent facts not present in the backstory.
- Avoid vague psychological descriptions.

For each claim:
- Assign a type: belief, behavior, background, or motivation
- Assign importance:
  - core → if violated, backstory becomes implausible
  - supporting → secondary detail

Output STRICT JSON only.
No explanations.
No extra text.

JSON schema:
[
  {
    "claim_id": "C1",
    "type": "...",
    "text": "...",
    "importance": "core | supporting"
  }
]

Backstory:
<<<BACKSTORY_TEXT>>>"""


# =============================================================================
# LLM Integration
# =============================================================================

def build_prompt(backstory: str) -> str:
    """Build the extraction prompt with the backstory inserted."""
    return EXTRACTION_PROMPT.replace("<<<BACKSTORY_TEXT>>>", backstory)


def parse_llm_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM response as JSON.
    
    Args:
        response_text: Raw text response from LLM
        
    Returns:
        Parsed JSON as list of dictionaries
        
    Raises:
        ClaimValidationError: If JSON parsing fails
    """
    # Clean the response - remove markdown code blocks if present
    cleaned = response_text.strip()
    
    # Remove ```json ... ``` wrapper if present
    if cleaned.startswith("```"):
        # Find the end of the opening code fence
        first_newline = cleaned.find('\n')
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    
    # Try to extract JSON array if there's extra text
    json_match = re.search(r'\[[\s\S]*\]', cleaned)
    if json_match:
        cleaned = json_match.group(0)
    
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        raise ClaimValidationError(
            f"Failed to parse LLM response as JSON: {e}\n"
            f"Response was: {response_text[:500]}..."
        )


class ClaimExtractor:
    """
    Extracts atomic claims from character backstories using an LLM.
    
    This class handles:
    - Building the extraction prompt
    - Calling the LLM
    - Parsing and validating the response
    """
    
    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        api_key: Optional[str] = None
    ):
        """
        Initialize the claim extractor.
        
        Args:
            model: OpenAI model to use
            temperature: LLM temperature (low for determinism)
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model = model
        self.temperature = temperature

        # Resolve API key: prefer explicit arg, then environment, then .env
        if api_key is None:
            api_key = get_openai_api_key()

        if not api_key:
            raise RuntimeError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or add it to .env"
            )

        self.client = OpenAI(api_key=api_key)
    
    def extract(self, backstory: str) -> List[Claim]:
        """
        Extract claims from a backstory.
        
        Args:
            backstory: The character backstory text
            
        Returns:
            List of validated Claim objects
            
        Raises:
            ClaimValidationError: If extraction or validation fails
        """
        if not backstory or not backstory.strip():
            raise ClaimValidationError("Backstory cannot be empty")
        
        # Build prompt
        prompt = build_prompt(backstory)
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise information extraction system. Output only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}  # Enforce JSON output
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        # Handle JSON object wrapper (OpenAI json_object mode wraps in object)
        try:
            parsed = json.loads(response_text)
            # If it's wrapped in an object, extract the claims array
            if isinstance(parsed, dict):
                # Look for the claims array in common keys
                for key in ['claims', 'data', 'results', 'items']:
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                else:
                    # If no known key, check if there's any list value
                    for value in parsed.values():
                        if isinstance(value, list):
                            parsed = value
                            break
            claims_data = parsed
        except json.JSONDecodeError:
            # Fall back to our parser
            claims_data = parse_llm_response(response_text)
        
        # Validate and convert
        claims = validate_claims(claims_data)
        
        return claims
    
    def extract_to_dict(self, backstory: str) -> List[Dict[str, Any]]:
        """
        Extract claims and return as list of dictionaries.
        
        Args:
            backstory: The character backstory text
            
        Returns:
            List of claim dictionaries
        """
        claims = self.extract(backstory)
        return [claim.to_dict() for claim in claims]


# =============================================================================
# Utility Functions
# =============================================================================

def extract_claims_from_backstory(
    backstory: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract claims from a backstory.
    
    Args:
        backstory: The character backstory text
        model: OpenAI model to use
        temperature: LLM temperature
        
    Returns:
        List of claim dictionaries
    """
    extractor = ClaimExtractor(model=model, temperature=temperature)
    return extractor.extract_to_dict(backstory)


def get_core_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to only core claims."""
    return [c for c in claims if c['importance'] == 'core']


def get_supporting_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to only supporting claims."""
    return [c for c in claims if c['importance'] == 'supporting']


def claims_to_json(claims: List[Dict[str, Any]], indent: int = 2) -> str:
    """Convert claims list to formatted JSON string."""
    return json.dumps(claims, indent=indent)


# =============================================================================
# Main / Demo
# =============================================================================

def main():
    """Demo the claim extraction module."""
    # Example backstory for testing
    sample_backstory = """
    Thalcave's people faded as colonists advanced; his father, last of the tribal 
    guides, knew the pampas geography and animal ways, while his mother died giving 
    birth. Boyhood was spent roaming the plains with his father, learning to track, 
    tame horses and steer by the stars.
    """
    
    print("=" * 60)
    print("CLAIM EXTRACTION MODULE - DEMO")
    print("=" * 60)
    print("\nInput Backstory:")
    print("-" * 40)
    print(sample_backstory.strip())
    print("-" * 40)
    
    try:
        extractor = ClaimExtractor()
        claims = extractor.extract_to_dict(sample_backstory)
        
        print(f"\nExtracted {len(claims)} claims:\n")
        print(claims_to_json(claims))
        
        # Summary
        core_count = len(get_core_claims(claims))
        supporting_count = len(get_supporting_claims(claims))
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total claims: {len(claims)}")
        print(f"Core claims: {core_count}")
        print(f"Supporting claims: {supporting_count}")
        
        # Breakdown by type
        type_counts = {}
        for claim in claims:
            t = claim['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print("\nBy type:")
        for t, count in sorted(type_counts.items()):
            print(f"  {t}: {count}")
        
    except ClaimValidationError as e:
        print(f"\n❌ Validation Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
