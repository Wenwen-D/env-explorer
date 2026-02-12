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
  UNIT_TESTS: test_delimiter("file.csv"), test_quotechar("file.csv"), test_skiprows("file.csv")

You may include one or multiple unit tests in a single UNIT_TESTS action.
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
- IMPORTANT: Discount factors represent cost multiplicatively.
- A smaller discount factor means a MORE expensive action (it reduces reward more).
- For example, if d_code = d_unit^k, then one CODE attempt costs about as much as k UNIT_TESTS.

General guidance:
- Start from reasonable default beliefs about the CSV format based on common conventions or provided likelihoods.
- Both UNIT_TESTS and CODE are costly actions; neither should be treated as free.
- Use UNIT_TESTS to reduce uncertainty when the expected benefit outweighs their cost.
- Use CODE to make progress toward solving the task, but recognize that failed or repeated CODE attempts are also costly.
- Decide when it is better to verify assumptions with UNIT_TESTS versus attempting CODE earlier, taking into account
  your confidence and the relative cost of these actions.
- Decide rationally how much debugging and iteration is worthwhile before committing to a final ANSWER.'''


CODE_INSTRUCTION_TEMPLATE = """You are given a CSV file `{csv_name}`.

Your task: {task_description}

Additional context:
- No format likelihoods are provided.
- Make reasonable default assumptions about the CSV format based on common conventions, unless you choose to verify them with unit tests.

Reward parameters:
- Unit test discount d_unit: {d_unit}
- Code iteration discount d_code: {d_code}
{rho_info}

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
{rho_info}

{guidance_block}

Constraints:
- You should never print all rows of the CSV or you will get zero reward.
- You may use UNIT_TESTS, CODE, or ANSWER as described in the system instructions.
"""


GUIDANCE_BLOCK="""Interpretation guidance:
- UNIT_TESTS reduce uncertainty about individual formatting attributes but incur discounted cost (lower d_unit = higher cost).
- CODE attempts test a full formatting configuration and potentially reveals the answer in stdout, and also incur discounted cost (d_code).
- IMPORTANT: Discount factors represent cost multiplicatively.
- A smaller discount factor means a MORE expensive action (it reduces reward more).
- Neither UNIT_TESTS nor CODE should be overused; both trade off information, risk, and cost.
- Early successful CODE attempts can yield high reward, but failed or repeated CODE attempts can be costly,
  especially when CODE is expensive relative to UNIT_TESTS (e.g., when d_code = d_unit^3, one CODE attempt costs about as much as three UNIT_TESTS).
- When deciding between UNIT_TESTS and attempting CODE early, consider:
  (1) your current estimated probability that the full formatting configuration is correct,
  (2) how much uncertainty remains,
  (3) the relative cost of UNIT_TESTS versus CODE in the current setting.
- You may use UNIT_TESTS and CODE in any order. It is valid to attempt CODE early,
  or to verify assumptions first, depending on which yields higher expected reward."""


TASK_TYPES = [
    ("Find the `{id_key}` of the record with the maximum `{number_key}`.", "max"),
    ("Compute the average of `{number_key}` (excluding None).", "mean"),
    ("Find the minimum of `{number_key}` (excluding None).", "min"),
]



HELPER_ENCODING="""def detect_encoding(path):
    with open(path, "rb") as f:
        head = f.read(4)
    if head.startswith(b'\\xff\\xfe') or head.startswith(b'\\xfe\\xff'):
        return "utf-16"
    if head.startswith(b'\\xef\\xbb\\xbf'):
        return "utf-8"
    return "utf-8"
"""

HELPER_SKIPROWS="""def detect_skiprows(path):
    with open(path, "r") as f:
        first_line = f.readline()
        second_line = f.readline()
    # Simple heuristic: if the first line starts with "#" or if the first line has letters but the second line has digits
    count = 0
    if first_line.strip().startswith("#"):
        count += 1
    if second_line.strip().startswith("#"):
        count += 1
    return count
"""

HELPER_QUOTECHAR="""import re

def detect_quotechar(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(200)
    if re.search(r'"[^"\\n]*"', text):
        return '"'
    if re.search(r"'[^'\\n]*'", text):
        return "'"
    return None
"""

HELPER_DELIMITER="""def detect_delimiter(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [f.readline() for _ in range(3)]
    text = "".join(lines)
    counts = {
        ",": text.count(","),
        ";": text.count(";"),
        "\\t": text.count("\\t"),
        "|": text.count("|"),
    }
    return max(counts, key=counts.get)
"""
