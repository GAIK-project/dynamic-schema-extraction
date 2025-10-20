"""
Dynamic Schema Extraction with Structured Outputs
Using OpenAI's .parse() with dynamically created Pydantic schemas

Benefits of this approach:
1. Type-safe and guaranteed structure (enforced by the API)
2. Cost-effective (fewer tokens, no code generation)
3. Secure (no eval/exec needed)
4. Simple and maintainable
5. Reliable results with automatic retries
"""

import re
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

# Load environment variables
load_dotenv()

client = OpenAI()

# ============================================================================
# STEP 1: Fixed schema for parsing user's extraction requirements
# ============================================================================


class FieldSpec(BaseModel):
    """Specification for a single field to extract"""

    field_name: str = Field(description="Snake_case field name (e.g., 'project_title')")
    field_type: Literal["str", "int", "float", "bool", "list[str]"] = Field(
        description="Python type for this field"
    )
    description: str = Field(description="What this field represents")
    required: bool = Field(default=True, description="Whether this field is required")


class ExtractionRequirements(BaseModel):
    """Parsed extraction requirements from user input"""

    use_case_name: str = Field(description="Name for this extraction use case")
    fields: list[FieldSpec] = Field(description="List of fields to extract")


# ============================================================================
# STEP 2: Parse user's natural language description into field specs
# ============================================================================


def parse_user_requirements(user_description: str) -> ExtractionRequirements:
    """
    Parse user's natural language into structured field specifications.
    Using structured outputs ensures the response matches our schema.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at analyzing extraction requirements.
                Parse the user's description and identify all fields they want to extract.
                Convert field names to snake_case and choose appropriate types.""",
            },
            {
                "role": "user",
                "content": f"""Parse this extraction requirement:
                
{user_description}

Identify all fields to extract, their types, and descriptions.""",
            },
        ],
        response_format=ExtractionRequirements,
    )

    return response.choices[0].message.parsed


# ============================================================================
# STEP 3: Create dynamic Pydantic model from field specs
# ============================================================================


def sanitize_model_name(name: str) -> str:
    """
    Sanitize model name to match OpenAI's requirements.
    Only alphanumeric, underscores, and hyphens are allowed.
    """
    # Replace spaces and other characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def create_extraction_model(requirements: ExtractionRequirements) -> type[BaseModel]:
    """
    Create a Pydantic model dynamically from field specifications.
    This is type-safe and doesn't require code generation or eval().
    """

    # Map string type names to actual Python types
    type_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list[str]": list[str],
    }

    # Build field definitions for create_model()
    field_definitions = {}

    for field_spec in requirements.fields:
        python_type = type_mapping[field_spec.field_type]

        if field_spec.required:
            # Required field
            field_definitions[field_spec.field_name] = (
                python_type,
                Field(description=field_spec.description),
            )
        else:
            # Optional field
            field_definitions[field_spec.field_name] = (
                python_type | None,
                Field(default=None, description=field_spec.description),
            )

    # Sanitize the model name for OpenAI compatibility
    model_name = sanitize_model_name(requirements.use_case_name) + "_Extraction"

    # Create the model dynamically using Pydantic's built-in method
    DynamicModel = create_model(
        model_name,
        __doc__=f"Extraction model for {requirements.use_case_name}",
        **field_definitions,
    )

    return DynamicModel


# ============================================================================
# STEP 4: Extract data using the dynamic schema with structured outputs
# ============================================================================


def extract_from_document(
    document_text: str, extraction_model: type[BaseModel]
) -> BaseModel:
    """
    Extract structured data from document using .parse().
    The schema is enforced by OpenAI's structured outputs API.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "Extract the requested information from the document.",
            },
            {"role": "user", "content": document_text},
        ],
        response_format=extraction_model,
    )

    return response.choices[0].message.parsed


# ============================================================================
# COMPLETE WORKFLOW
# ============================================================================


def dynamic_extraction_workflow(
    user_description: str, documents: list[str]
) -> list[dict]:
    """
    Complete workflow from natural language description to structured extraction.

    Process:
    1. Parse user requirements into field specifications
    2. Create dynamic Pydantic schema from specifications
    3. Extract data using structured outputs (guaranteed format)

    Advantages:
    - Reliable: API enforces schema compliance
    - Efficient: Minimal API calls needed
    - Safe: No code execution or eval()
    - Type-safe: Full Pydantic validation
    """

    print("Step 1: Parsing user requirements...")
    requirements = parse_user_requirements(user_description)
    print(f"✓ Identified {len(requirements.fields)} fields to extract")
    print(f"  Fields: {[f.field_name for f in requirements.fields]}")

    print("\nStep 2: Creating dynamic Pydantic schema...")
    ExtractionModel = create_extraction_model(requirements)
    print(f"✓ Created schema: {ExtractionModel.__name__}")
    print(f"  Schema: {ExtractionModel.model_json_schema()}")

    print("\nStep 3: Extracting from documents...")
    results = []
    for i, doc in enumerate(documents):
        print(f"  Processing document {i + 1}/{len(documents)}...")
        extracted = extract_from_document(doc, ExtractionModel)
        results.append(extracted.model_dump())

    print(f"✓ Extracted data from {len(documents)} documents")

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # User's natural language input
    user_input = """
    Extract the following information from project reports:
    - Project title and acronym
    - Lead institution name
    - Total funding amount in euros (as a number)
    - List of partner countries
    - Project status (ongoing or completed)
    - Project start date
    """

    # Example documents
    sample_documents = [
        """
        Project Report
        
        Title: Advanced AI Research Initiative
        Acronym: AIRI
        Lead Institution: University of Helsinki
        Total Funding: 2500000 euros
        Partners: Finland, Sweden, Norway, Denmark
        Status: Ongoing
        Start Date: 2024-01-15
        """,
        """
        Project Summary
        
        Project Name: Green Energy Solutions
        Short Name: GES
        Lead: Technical University of Munich
        Budget: 1800000 EUR
        Participating Countries: Germany, Austria, Switzerland
        Current Status: Completed
        Started: 2023-06-01
        """,
    ]

    # Run the complete workflow
    results = dynamic_extraction_workflow(user_input, sample_documents)

    # Print results
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\nDocument {i}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
