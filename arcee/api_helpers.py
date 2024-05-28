import numpy as np

def _chat_ml_messages_to_qa_pair(messages: np.ndarray) -> (str, str):
    """
    Helper function to convert a ChatML messages field into a QA pair.

    Given a ChatML messages field, like:

    [
        {
            "content": "Does this feature apply to all sections of the theme or just specific ones as listed in the text material?",
            "role": "user"
        },
        {
            "content": "This feature only applies to Collection pages and Featured Collections sections of the section-based themes listed in the text material.",
            "role": "assistant"
        },
        {
            "content": "Can you guide me through the process of enabling the secondary image hover feature on my Collection pages and Featured Collections sections?",
            "role": "user"
        },
        {
            "content": "Sure, here are the steps to enable the secondary image hover [snip ..].",
            "role": "assistant"
        },
        etc ..
    ]

    Extract the _first_ user message and the _first_ assistant message and return them as a QA pair.

    Args:
        messages (np.ndarray): Array of messages.

    Returns:
        (str, str): A tuple of a question and an answer.

    """

    # The first role should be a user role
    if messages[0]["role"] != "user":
        raise Exception("First message must be a user message")

    # Get the question
    question = messages[0]["content"]

    # The second message role should be an assistant role
    if messages[1]["role"] != "assistant":
        raise Exception("Second message must be an assistant message")

    # Get the answer
    answer = messages[1]["content"]

    return (question, answer)
