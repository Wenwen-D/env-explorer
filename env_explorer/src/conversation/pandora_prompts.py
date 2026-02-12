BANDIT_SYSTEM_PROMPT='''You are a rational agent tasked with solving sequential decision-making problems under uncertainty. You are given a set of options (bags) with prior probabilities of containing a prize with value 1. You can either VERIFY an option to get information (YES/NO) or GUESS an option to end the game and collect the reward. 

- Each VERIFY action consumes one timestep. 
- The reward for a correct GUESS is discounted by a factor r^t, where t is the timestep when you GUESS. 
- You must balance information gathering (VERIFY) with timely exploitation (GUESS) to maximize expected discounted reward. 

Always respond with exactly one action token per step, using the format: 
VERIFY <Option> or GUESS <Option>.'''

# BANDIT_SYSTEM_PROMPT='''Think about the problem step by step inside <think>...</think>.'''
BANDIT_SYSTEM_PROMPT_NO_EXPLAIN='''\nDo not provide any explanations, comments, or probabilities in your response—only the action token.'''

INSTRUCTION_TEMPLATE_t1=''''''

BANDIT_SYSTEM_PROMPT_t2='''Think about the problem step by step.'''

INSTRUCTION_TEMPLATE_t2='''--- NEW GAME ---
TIMESTEP: t=0

PROBLEM PARAMETERS:
- Bag Labels: {labels_str}
- Prior Probabilities: {priors_str}
- Discount Factor r: {r}

Choose your action.'''