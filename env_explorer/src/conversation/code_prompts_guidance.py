SYSTEM_PROMPT_CODE='''You are an agent tasked with writing code to fulfill an instruction about a CSV file (e.g., answering a question using its contents). Your goal is to produce a correct answer while efficiently using available resources, as measured by discounted reward.

The exact CSV formatting may not be fully known. In practice, you can either proceed using reasonable default assumptions about the format, or run unit tests to verify specific formatting details you are unsure about before committing to a final answer.

Allowed actions (choose exactly ONE per turn):

1) UNIT_TESTS
Run unit tests to debug CSV formatting assumptions. Unit test outputs are perfectly reliable.
Available unit tests:
  - test_delimiter(path) -> ',', ';', or '\t'
  - test_quotechar(path) -> '"' or "'"
  - test_skiprows(path) -> 0 or 1

Format (NO code fences):
  UNIT_TESTS: test_delimiter("file.csv"), test_quotechar("file.csv")

You may include multiple unit tests in a single UNIT_TESTS action.
Each individual unit test counts toward the total number of unit tests used.

2) CODE
Write Python code toward solving the task using your current assumptions about the CSV format.
- Enclose code in ```python ... ```
- You may import pandas as pd and read the file with:
    pd.read_csv(filepath, delimiter=..., quotechar=..., skiprows=...)
- Do NOT print the entire CSV.
- If your code computes the final result, print it to stdout so it can be read from the output.

After submission, the code will be executed and its stdout and stderr will be returned.  
You may use this feedback to extract the answer, debug, run additional unit tests, refine, or write additional CODE.

3) ANSWER
Provide the final answer to the task and end the conversation.
Format exactly:
  ANSWER: <your_answer>

The conversation ends immediately after you provide ANSWER.


Reward:
- Let U be the total number of unit tests used.
- Let C be the total number of CODE actions taken.
- Final reward = correctness * (d_unit ^ U) * (d_code ^ C)

General guidance:
- Start from reasonable default assumptions or expected formats.
- Use unit tests only to resolve specific uncertainties that materially affect correctness.
- Decide how much debugging and iteration is worthwhile before committing to a final ANSWER.'''

BACKUP='''Formatting semantics:
- Formatting attributes (delimiter, quotechar, skiprows) are equally important.
- If any formatting attribute used in CODE is incorrect, assume the result is incorrect.'''

CODE_INSTRUCTION_TEMPLATE = """You are given a CSV file `{csv_name}`.

Your task: {task_description}

Additional context:
- No format likelihoods are provided.
- Make reasonable default assumptions about the CSV format based on common conventions, unless you choose to verify them with unit tests.

Reward parameters:
- Unit test discount d_unit: {d_unit}
- Code iteration discount d_code: {d_code}

{guidance_block}

Constraints:
- You should never print all rows of the CSV or you will get zero reward.
- You may use UNIT_TESTS, CODE, or ANSWER as described in the system instructions in any order; only the final ANSWER ends the conversation.
- Incorrect intermediate CODE does not end the episode; only the final ANSWER determines correctness.
"""


CODE_INSTRUCTION_TEMPLATE_WITH_LIKELIHOODS = """You are given a CSV file `{csv_name}`.

Your task: {task_description}

Additional context:
- Estimated format likelihoods are provided below. These reflect how likely each formatting option is in practice and can be used as default assumptions.

Format likelihoods:
{prior}

Reward parameters:
- Unit test discount d_unit: {d_unit}
- Code iteration discount d_code: {d_code}

{guidance_block}

Constraints:
- You should never print all rows of the CSV or you will get zero reward.
- You may use UNIT_TESTS, CODE, or ANSWER as described in the system instructions.
"""

GUIDANCE_BLOCK="""Interpretation guidance:
- UNIT_TESTS reduce uncertainty but incur discounted cost (lower d_unit = higher cost). 
- Early successful CODE attempts can lead to higher reward by avoiding unnecessary tests. You may consider UNIT_TESTS when their expected information gain, given current beliefs, outweighs the discounted cost d_unit.
- You may use UNIT_TESTS and CODE in any order. It is valid to attempt CODE using current assumptions, observe the result, and run UNIT_TESTS later if needed.
"""
