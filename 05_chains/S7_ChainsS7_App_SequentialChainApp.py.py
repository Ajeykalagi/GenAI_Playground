from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables import RunnableLambda
# First chain: Receives one user input 'n', generates two outputs 'square' and 'cube'
def first_chain_func(input_data):
    n = input_data["n"]
    square = n ** 2
    cube = n ** 3
    return {"square": square, "cube": cube}
first_chain = RunnableLambda(first_chain_func)
# Merge chain: Merges the output of the first chain with the second user input 'm'
def merge_func(previous_output, input_data):
    # previous_output contains 'square' and 'cube'
    # input_data contains 'm'
    merged_data = {
        "square": previous_output["square"],
        "cube": previous_output["cube"],
        "m": input_data["m"]
    }
    return merged_data
# Wrap the merge function in RunnableLambda and set 'multiple_inputs' to True
merge_chain = RunnableLambda(merge_func, multiple_inputs=True)
# Second chain: Receives 'square', 'cube', and 'm', outputs the sum
def second_chain_func(merged_data):
    square = merged_data["square"]
    cube = merged_data["cube"]
    m = merged_data["m"]
    result = square + cube + m
    return result
second_chain = RunnableLambda(second_chain_func)
# Build the RunnableSequence
sequence = RunnableSequence(
    first=first_chain,
    middle=[merge_chain],
    last=second_chain
)
# Prepare the user inputs
user_inputs = {"n": 2, "m": 5}
# Run the sequence
result = sequence.invoke(user_inputs)
print(f"The result is: {result}")