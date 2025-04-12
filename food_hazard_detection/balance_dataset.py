import csv
import re
from typing import List, Dict, Any

import pandas as pd
from mistralai import Mistral

from food_hazard_detection import settings
from food_hazard_detection.settings import MISTRAL_API_KEY
from food_hazard_detection.llm_api import call_mistral_for_synthetic_data_generation


def generate_prompt_triplets_by_hazard(rare_hazards: List[str], dataset: pd.DataFrame) -> List[List[Any]]:
    """
    Generate example triplets for prompt generation using rare hazard categories.

    Each hazard is paired with product categories that appear at least three times
    with that hazard. Then, groups of three examples are sampled for each (hazard, product) pair.

    Args:
        rare_hazards (List[str]): List of hazard categories considered low-frequency.
        dataset (pd.DataFrame): DataFrame containing 'hazard_category' and 'product_category' columns.

    Returns:
        List[List[Any]]: A list of [hazard, product_category, example_triplets] entries,
                         where each example_triplet is a list of three CSV-formatted strings.
    """
    hazard_to_valid_products: Dict[str, List[str]] = {}

    for hazard in rare_hazards:
        grouped = (
            dataset[dataset["hazard_category"] == hazard]
            .groupby("product_category")
            .size()
            .reset_index(name="count")
        )
        valid_products = grouped[grouped["count"] >= 3]["product_category"].unique().tolist()
        hazard_to_valid_products[hazard] = valid_products

    return _generate_triplets(hazard_to_valid_products, dataset, by="hazard")


def generate_prompt_triplets_by_product(rare_products: List[str], dataset: pd.DataFrame) -> List[List[Any]]:
    """
    Generate example triplets for prompt generation using rare product categories.

    Each product is paired with hazard categories that appear at least three times
    with that product. Then, groups of three examples are sampled for each (hazard, product) pair.

    Args:
        rare_products (List[str]): List of product categories considered low-frequency.
        dataset (pd.DataFrame): DataFrame containing 'hazard_category' and 'product_category' columns.

    Returns:
        List[List[Any]]: A list of [hazard, product_category, example_triplets] entries,
                         where each example_triplet is a list of three CSV-formatted strings.
    """
    product_to_valid_hazards: Dict[str, List[str]] = {}

    for product in rare_products:
        grouped = (
            dataset[dataset["product_category"] == product]
            .groupby("hazard_category")
            .size()
            .reset_index(name="count")
        )
        valid_hazards = grouped[grouped["count"] >= 3]["hazard_category"].unique().tolist()
        product_to_valid_hazards[product] = valid_hazards

    return _generate_triplets(product_to_valid_hazards, dataset, by="product")


def _generate_triplets(category_map: Dict[str, List[str]], dataset: pd.DataFrame, by: str) -> List[List[Any]]:
    """
    Internal helper to generate triplets for combinations of product and hazard categories.

    Args:
        category_map (Dict[str, List[str]]): A mapping of primary category (hazard or product)
        to related categories (hazard or product pairwise).
        dataset (pd.DataFrame): The original dataset.
        by (str): Whether mapping is by "hazard" or "product".

    Returns:
        List[List[Any]]: A list of triplet examples with format [hazard, product, [example1, example2, example3]].
    """
    triplet_list: List[List[Any]] = []
    for primary, related_list in category_map.items():
        for related in related_list:
            if by == "hazard":
                df_subset = dataset[
                    (dataset["product_category"] == related)
                    & (dataset["hazard_category"] == primary)
                    ]
                hazard = primary
                product = related
            elif by == "product":
                df_subset = dataset[
                    (dataset["product_category"] == primary)
                    & (dataset["hazard_category"] == related)
                    ]
                hazard = related
                product = primary
            else:
                raise ValueError("Parameter 'by' must be either 'hazard' or 'product'.")

            x = 0
            for i in range(len(df_subset) // 3):
                triplet = [df_subset.iloc[[x + i]].to_csv(index=False, header=False, sep=settings.SEPARATOR).strip()
                           for i in range(3)
                           ]
                triplet_list.append([hazard, product, triplet])
                x += 3
            # if you want 5 times more data, uncomment this and comment the above lines starting x = 0
            # the datasets called big are created this way
            # for j in range(5):
            #     x = 0
            #     df_subset = df_subset.sample(frac=1).reset_index(drop=True)
            #     for i in range(len(df_subset) // 3):
            #         triplet = [df_subset.iloc[[x + i]].to_csv(index=False, header=False, quoting=csv.QUOTE_ALL).strip()
            #                    for i in range(3)
            #                    ]
            #         triplet_list.append([hazard, product, triplet])
            #         x += 3

    return triplet_list


def process_llm_output(text: str) -> str:
    text = text.removeprefix("```")
    text = text.removeprefix("'")
    text = text.removesuffix("```")
    text = text.removesuffix("'")
    text = re.sub("\"", "", text)
    return text


def generate_synthetic_data(output_path: str, prompt_path: str, combinations: List[List[str]],
                            client_name: str = "mistral") -> None:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    if client_name == "mistral":
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        with open(output_path, "a", encoding="utf-8") as f:
            for combination in combinations:
                output = call_mistral_for_synthetic_data_generation(mistral_client, prompt, *combination).strip()
                output = process_llm_output(output)
                f.write(output)
                f.write("\n")
    else:
        raise NotImplementedError("Other LLM APIs are not implemented yet.")


def repare_csv(file_path: str) -> None:
    """
    Caution: it does change the csv file inplace.
    Wraps the text field in row in double quotes to help recognize the csv parser the columns properly.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # one line in csv should look similar to this:
        # 2021, 3, 15, uk, title, text, hazard-category, product-category, hazard, product
        line = re.sub(r"^(\d{4},\d{1,2},\d{1,2},\w{2},.*?,)(.*)(,.*,.*,.*,.*)$", r'\1"\2"\3', line)
        lines[i] = line

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
