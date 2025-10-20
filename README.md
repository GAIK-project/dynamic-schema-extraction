# Dynamic Schema Extraction with Structured Outputs

A simple approach to building dynamic data extraction schemas using OpenAI's structured outputs and Pydantic.

## The Problem This Solves

Traditional knowledge extraction tools generate Python code dynamically using LLMs (requiring `eval()`), build complex multi-stage pipelines, and rely on prompting without guarantees. This approach is:

- **Simpler** - Uses Pydantic's `create_model()` instead of code generation
- **Safer** - No `eval()` or `exec()` needed
- **More Reliable** - OpenAI's `.parse()` enforces schema compliance
- **Cost-Effective** - Fewer API calls and tokens

## How It Works

1. **User describes** what they want to extract in natural language
2. **LLM parses** the description into structured field specifications
3. **Pydantic creates** a dynamic schema from those specifications
4. **Structured extraction** runs with guaranteed output format

```python
# 1. User input
"Extract project title, budget in euros, and partner countries"

# 2. System creates schema dynamically
ProjectExtraction = create_model(
    "ProjectExtraction",
    title=(str, Field(description="Project title")),
    budget=(float, Field(description="Budget in euros")),
    partners=(list[str], Field(description="Partner countries"))
)

# 3. Extract with guaranteed structure
response = client.beta.chat.completions.parse(
    messages=[{"role": "user", "content": document}],
    response_format=ProjectExtraction
)
```

## Why Structured Outputs?

### Traditional Approach (Unreliable)

```python
response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": "Extract name, age, email as JSON"
    }]
)
# Hope it returns valid JSON
# Manually parse and validate
# Handle type errors
```

### Structured Outputs (Guaranteed)

```python
class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    email: str = Field(description="Email address")

response = client.beta.chat.completions.parse(
    response_format=Person  # API enforces this!
)
result = response.choices[0].message.parsed  # Already validated ✓
```

**Benefits:**

- ✅ **Guaranteed structure** - API enforces exact schema
- ✅ **Type safety** - No parsing errors or invalid data
- ✅ **Cost effective** - Fewer tokens than traditional prompting
- ✅ **Automatic retries** - API handles parsing failures
- ✅ **Field descriptions** guide extraction behavior
- ✅ **Minimal prompting** - Schema replaces most prompt engineering

## Field Descriptions = Extraction Instructions

```python
class Extract(BaseModel):
    title: str = Field(
        description="Project title in title case"
    )
    budget: float = Field(
        description="Budget in USD, convert from any currency mentioned"
    )
    tags: list[str] = Field(
        description="Relevant topic tags, maximum 5, lowercase"
    )
    status: Literal["ongoing", "completed"] = Field(
        description="Current project status"
    )
```

The `description` field tells the LLM **how** to extract and format each field.

**No complex prompts needed!** Just pass the document - the schema handles all instructions:

```python
# That's it - schema does the rest!
response = client.beta.chat.completions.parse(
    messages=[{"role": "user", "content": document}],
    response_format=Extract
)
```

## Example Output

```bash
Step 1: Parsing user requirements...
✓ Identified 7 fields to extract
  Fields: ['project_title', 'acronym', 'lead_institution', ...]

Step 2: Creating dynamic Pydantic schema...
✓ Created schema: Project_Reports_Extraction

Step 3: Extracting from documents...
  Processing document 1/2...
  Processing document 2/2...
✓ Extracted data from 2 documents

================================================================================
EXTRACTION RESULTS
================================================================================

Document 1:
  project_title: Advanced AI Research Initiative
  acronym: AIRI
  lead_institution: University of Helsinki
  total_funding_amount: 2500000.0
  partner_countries: ['Finland', 'Sweden', 'Norway', 'Denmark']
  project_status: Ongoing
  project_start_date: 2024-01-15

Document 2:
  project_title: Green Energy Solutions
  acronym: GES
  lead_institution: Technical University of Munich
  total_funding_amount: 1800000.0
  partner_countries: ['Germany', 'Austria', 'Switzerland']
  project_status: Completed
  project_start_date: 2023-06-01
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/GAIK-project/dynamic-schema-extraction
cd dynamic-schema-extraction

# Install dependencies
uv add openai pydantic python-dotenv

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run the example
uv run main.py
```

## Usage

```python
from main import dynamic_extraction_workflow

# Describe what you want to extract
description = """
Extract from invoices:
- Invoice number
- Date
- Total amount in USD
- Vendor name
"""

# Your documents
documents = [doc1_text, doc2_text, doc3_text]

# Run extraction
results = dynamic_extraction_workflow(description, documents)
```

## Advanced: Custom Field Descriptions

Want more control? Define field specifications manually with precise extraction instructions:

```python
from main import FieldSpec, ExtractionRequirements, create_extraction_model, extract_from_document

# Define exact field specs with custom descriptions
custom_fields = [
    FieldSpec(
        field_name="invoice_number",
        field_type="str",
        description="Extract invoice ID. Look for 'Invoice #', 'INV-', or similar patterns",
        required=True
    ),
    FieldSpec(
        field_name="amount",
        field_type="float",
        description="Total in USD. Convert EUR (×1.1) and GBP (×1.25) if needed",
        required=True
    ),
]

# Create schema from custom specs
requirements = ExtractionRequirements(use_case_name="Invoice", fields=custom_fields)
ExtractionModel = create_extraction_model(requirements)

# Use for extraction
result = extract_from_document(document, ExtractionModel)
```

**Benefits:** Save and reuse schemas, fine-tune extraction logic, share specs across teams.

## Resources

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic Dynamic Models](https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation)

## License

MIT
