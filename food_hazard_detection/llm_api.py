from mistralai import Mistral


def call_mistral_for_synthetic_data_generation(
        mistral_client: Mistral, prompt: str, hazard: str, product: str, examples: list) -> str:
    """
    Generates synthetic data using the Mistral language model by formatting a prompt
    with hazard, product, and example data, then invoking the model.

    Args:
        mistral_client (Mistral): An instance of the Mistral client used to interact with the model.
        prompt (str): The base prompt containing placeholders for data injection.
        hazard (str): The hazard type to insert into the prompt.
        product (str): The product type to insert into the prompt.
        examples (list): A list of example data entries to include in the prompt.

    Returns:
        str: One synthetic data point similar to the examples generated by the Mistral model.
    """
    message = prompt.replace("XXDATASET_EXAMPLESXX", str(examples))
    message = message.replace("XXHAZARDXX", hazard)
    message = message.replace("XXPRODUCTXX", product)

    chat_response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": message,
            },
        ]
    )
    print(f"Example hazard: {hazard}, product: {product} processed...")
    return chat_response.choices[0].message.content
