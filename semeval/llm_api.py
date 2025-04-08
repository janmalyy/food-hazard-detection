from mistralai import Mistral


def call_mistral(mistral_client: Mistral, prompt: str, hazard: str, product: str, examples: list) -> str:
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
