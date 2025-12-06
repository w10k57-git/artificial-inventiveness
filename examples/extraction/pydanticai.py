import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "problem_desc.txt")

# This example uses the Groq LLM; replace with your desired model,
# ensuring it's available in your environment. First, make sure you have
# a valid API key set in your environment variables. You can get it from
# https://console.groq.com/keys

MODEL = "groq:llama-3.3-70b-versatile"


class TechnicalContradiction(BaseModel):
    """
    Technical Contradiction extraction model
    """

    action: str = Field(description="The action that causes the contradiction")
    positive_effect: str = Field(description="The positive effect of the action")
    negative_effect: str = Field(description="The negative effect of the action")


def extract_tc(text: str, model: str) -> TechnicalContradiction | str:
    """
    Extract a Technical Contradiction from the text.
    """
    agent = Agent(
        model=model,
        output_type=[TechnicalContradiction, str],
        system_prompt=(
            "Extract a Technical Contradiction from the user message, "
            "if you can't extract it, ask the user to clarify the problem."
            "You must respond with either a TechnicalContradiction object or a string."
        ),
    )
    return agent.run_sync(text).output


def print_result(result: TechnicalContradiction | str) -> None:
    if isinstance(result, TechnicalContradiction):
        print("\n✓ Technical Contradiction Extracted:\n")
        print(f"Action:          {result.action}")
        print(f"Positive Effect: {result.positive_effect}")
        print(f"Negative Effect: {result.negative_effect}")
    elif isinstance(result, str):
        print("\n✗ Could not extract Technical Contradiction:\n")
        print(f"{result}")


def process_text(title: str, text: str) -> None:
    """
    Process text and display the extraction result.
    """
    print("=" * 80)
    print(title)
    print("=" * 80)

    try:
        result = extract_tc(text, MODEL)
        print_result(result)
    except ValidationError as e:
        print(f"Validation error: {e}")


def main() -> None:
    # Example 1: Process the problem description file
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    process_text("EXAMPLE 1: Processing Technical Contradiction from File", data)

    # Example 2: Process generic text without a contradiction
    print()
    generic_text = "I don't understand how to set up my new smartphone."
    process_text("EXAMPLE 2: Processing Generic Text", generic_text)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
