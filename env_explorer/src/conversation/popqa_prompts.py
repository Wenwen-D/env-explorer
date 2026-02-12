POPQA_DIRECT='''Answer the question with only the short factual answer. Do not include explanations or extra text.'''
POPQA_DIRECT_IMPROVED = '''You are tasked with answering a factual question accurately. Consider what you know about this topic carefully before responding.        
                                                                                                                                                                      
  Question: {question}                                                                                                                                                
                                                                                                                                                                      
  Provide only the short factual answer without explanations or extra text.                                                                                           
                                                                                                                                                                      
  Respond with:                                                                                                                                                       
  ANSWER: <short factual answer>''' 
POPQA_DIRECT_WITH_RET=(
    "Answer the question with only the short factual answer. "
    "A related context is provided for reference, which may or may not be relevant—refer to it only if it is helpful. "
    "Do not include explanations or extra text."
)
POPQA_DIRECT_WITH_RET_USER=(
    "Context: {context}\n"
    "Question: {question}"
)


POPQA_ASK_CONFIDENCE_SYS = (
    "You are estimating your confidence in answering a factual question correctly. "
    "Respond with a single decimal number between 0 and 1, representing your confidence level. "
    "Do not include any words, explanations, or symbols — only the numeric value."
)
POPQA_ASK_CONFIDENCE_USER = (
    "Question: {question}\n"
    "How confident are you that you can answer this question correctly? "
    "Provide only your confidence score between 0 and 1."
)

POPQA_ASK_CONFIDENCE_SYS_RET = (
    "You are estimating your confidence in being able to answer a factual question correctly. "
    "A related context is provided, which may or may not include the correct answer. "
    "Your confidence should reflect the likelihood that you can produce the correct answer — "
    "either because the answer is present in the context or because you already know it. "
    "Respond with a single decimal number between 0 and 1, representing this confidence level. "
    "Do not include any words, explanations, or symbols — only the numeric value."
)
POPQA_ASK_CONFIDENCE_USER_RET = (
    "Question: {question}\n"
    "Context: {context}\n"
    "How confident are you that you can correctly answer this question — "
    "either because the answer is contained in the context or because you already know it? "
    "Respond with only a single confidence score between 0 and 1."
)

POPQA_MULTI_SYS = (
    "You are a rational agent tasked with answering factual questions under uncertainty. "
    "At each step, you can either directly answer the question or retrieve additional context before answering. "
    "\n\n"
    "Available actions:\n"
    "- RETRIEVE: request a related context to consult before deciding your final answer. This consumes one timestep.\n"
    "- ANSWER: <your short factual answer> — provide your final answer and end the interaction.\n\n"
    "Each question comes with your estimated probabilities of answering correctly:\n"
    "- p_no_context: probability you can answer correctly without retrieving.\n"
    "- p_with_context: probability you can answer correctly after retrieval.\n\n"
    "Your goal is to maximize expected discounted reward:\n"
    "Reward = r^t * correctness, where t is the timestep when you issue ANSWER and correctness ∈ {{0,1}}.\n\n"

    "Retrieval information:\n"
    "If you choose RETRIEVE, your expected answer accuracy after retrieval is {retrieval_accuracy:.2f}.\n"
    "If you choose not to retrieve, you must rely only on your current knowledge.\n\n"

    "Be deliberate — retrieving may improve accuracy but reduces reward due to time discounting. "
    "Balance speed and correctness carefully.\n\n"
    "Always respond with exactly one action token per step, using the format: \n"
    "RETRIEVE or ANSWER: <short factual answer>."
)
POPQA_MULTI_T1 = (
    "--- NEW QUESTION ---\n"
    "TIMESTEP: t=0\n\n"
    "Question: {question}\n"
    "Parameters:\n"
    "- Discount factor (r): {r}\n"
    "- Success probability without retrieval (p_no_context): {p_no_context}\n"
    # "- Success probability with retrieval (p_with_context): {p_with_context}\n\n"
    "Choose your action:\n"
    "RETRIEVE or ANSWER: <short factual answer>."
)
POPQA_MULTI_T2 = (
    "TIMESTEP: t=1\n"
    "You have retrieved the following context:\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Now decide whether to answer:\n"
    "Respond with:\n"
    "ANSWER: <short factual answer>"
)


# POPQA_MULTI_SYS_NOPRIOR = (
#     "You are a rational agent tasked with answering factual questions under uncertainty. "
#     "At each step, you can either directly answer the question or retrieve additional context before answering. "
#     "\n\n"
#     "Available actions:\n"
#     "- RETRIEVE: request a related context to consult before deciding your final answer. This consumes one timestep.\n"
#     "- ANSWER: <your short factual answer> — provide your final answer and end the interaction.\n\n"
#     "Your goal is to maximize expected discounted reward:\n"
#     "Reward = r^t * correctness, where t is the timestep when you issue ANSWER and correctness ∈ {0,1}.\n\n"
#     "Be deliberate — retrieving may improve accuracy but reduces reward due to time discounting. "
#     "Balance speed and correctness carefully.\n\n"
#     "Always respond with exactly one action token per step, using the format: \n"
#     "RETRIEVE or ANSWER: <short factual answer>."
# )

# POPQA_MULTI_T1_NOPRIOR = (
#     "--- NEW QUESTION ---\n"
#     "TIMESTEP: t=0\n\n"
#     "Question: {question}\n"
#     "Parameters:\n"
#     "- Discount factor (r): {r}\n\n"
#     "Choose your action:\n"
#     "RETRIEVE or ANSWER: <short factual answer>."
# )

POPQA_MULTI_SYS_NOPRIOR = (
    "You are a rational agent tasked with answering factual questions under uncertainty. "
    "At each step, you can either directly answer the question or retrieve additional context before answering. "
    "\n\n"
    "Available actions:\n"
    "- RETRIEVE: request a related context to consult before deciding your final answer. This consumes one timestep.\n"
    "- ANSWER: <your short factual answer> - provide your final answer and end the interaction.\n\n"
    "Your goal is to maximize expected discounted reward:\n"
    "Reward = r^t * correctness, where t is the timestep when you issue ANSWER and correctness ∈ {{0,1}}.\n\n"

    "Retrieval information:\n"
    "If you choose RETRIEVE, your expected answer accuracy after retrieval is {retrieval_accuracy:.2f}.\n"
    "If you choose not to retrieve, you must rely only on your current knowledge.\n\n"

    "Be deliberate — retrieving may improve accuracy but reduces reward due to time discounting. "
    "Balance speed and correctness carefully.\n\n"
    "Always respond with exactly one action token per step, using the format: \n"
    "RETRIEVE or ANSWER: <short factual answer>."
)

POPQA_MULTI_T1_NOPRIOR = (
    "--- NEW QUESTION ---\n"
    "TIMESTEP: t=0\n\n"
    "Question: {question}\n"
    "Parameters:\n"
    "- Discount factor (r): {r}\n\n"
    "Choose your action:\n"
    "RETRIEVE or ANSWER: <short factual answer>."
)

# POPQA_MULTI_T2_NOPRIOR = (
#     "TIMESTEP: t=1\n"
#     "You have retrieved the following context:\n"
#     "{context}\n\n"
#     "Question: {question}\n"
#     "Now decide whether to answer:\n"
#     "Respond with:\n"
#     "ANSWER: <short factual answer>"
# )

POPQA_MULTI_T2_NOPRIOR = (
    "TIMESTEP: t=1\n"
    "You have retrieved the following context:\n"
    "{context}\n\n"
    "Now answer the question:\n"
    "Respond with:\n"
    "ANSWER: <short factual answer>"
)