import copy

from oagents.models import MessageRole, Model


def prepare_response(original_task: str, inner_messages, reformulation_model: Model, agent_answer=None) -> str:
    messages = [
        {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": f"""Earlier you were asked the following:
                    {original_task}
                    Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
                }
            ],
        }
    ]

    try:
        for message in inner_messages:
            if not message.get("content"):
                continue
            message = copy.deepcopy(message)
            message["role"] = MessageRole.USER
            messages.append(message)
    except Exception:
        messages += [{"role": MessageRole.ASSISTANT, "content": str(inner_messages)}]

    # The agent's own conclusion anchors the reformulation: the reformulator's job
    # is to FORMAT that answer, not to re-answer the task from the transcript.
    if agent_answer is not None and str(agent_answer).strip():
        messages.append(
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The team concluded with this proposed FINAL ANSWER:\n"
                            f"{str(agent_answer)}\n"
                            "Unless the transcript above clearly contradicts it, reformat THIS answer "
                            "according to the formatting rules that follow instead of deriving a new one."
                        ),
                    }
                ],
            }
        )

    # ask for the final answer
    messages.append(
        {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": f"""
                Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

                {original_task}

                FINAL ANSWER FORMAT: Your response must strictly follow these formatting rules:
                - For NUMBERS: Use digits only (not words), omit commas and units (no $, USD, %, etc.) unless specifically requested
                - For TEXT: Omit articles and abbreviations unless specified, exclude final punctuation (.!?)
                - For LISTS: Provide comma-separated values following the above number/text rules
                - Follow ALL formatting instructions in the original question (alphabetization, sequencing, decimal places, etc.)
                - Please carefully understand the requirements of the original task and ensure that the final output meets the specific units given in the question (/Angstrom,/thousand hours, etc.)
                - If you cannot determine an answer, respond only with: "Unable to determine"
                - Your entire response should consist of ONLY the requested information in the EXACT format specified - nothing more, nothing less.
                """,
                }
            ],
        }
    )
    response = reformulation_model(messages).content

    final_answer = response.split("FINAL ANSWER: ")[-1].strip()
    print("> Reformulated answer: ", final_answer)

    return final_answer
