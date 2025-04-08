import pandas as pd

from balance_dataset import (generate_prompt_triplets_by_hazard, generate_prompt_triplets_by_product,
                             generate_synthetic_data)

import settings
from settings import FILES_DIR


if __name__ == '__main__':
    # change end of url to: incidents_valid.csv, incidents_test.csv if needed
    # trainset = pd.read_csv("https://raw.githubusercontent.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/refs/heads/main/data/incidents_train.csv", index_col=0)
    # trainset.to_csv("trainset.csv")
    trainset = pd.read_csv(FILES_DIR / "datasets/trainset.csv")
    trainset = trainset.rename(columns={"hazard-category": "hazard_category", "product-category": "product_category"})

    rare_hazard_categories = ["migration", "food additives and flavourings",
                              "organoleptic aspects", "packaging defect"]
    rare_product_categories = ["sugars and syrups", "feed materials", "food contact materials",
                               "honey and royal jelly", "food additives and flavourings", "fats and oils",
                               "pet feed", "other food product / mixed", "alcoholic beverages"]

    # combinations = generate_prompt_triplets_by_hazard(rare_hazard_categories, trainset)
    # generate_synthetic_data(FILES_DIR / "datasets/synthetic_data_hazard.csv", FILES_DIR / "prompts/prompt.md", combinations)

    # combinations = generate_prompt_triplets_by_product(rare_product_categories, trainset)
    # generate_synthetic_data(FILES_DIR / "synthetic_data_product.csv", FILES_DIR / "prompts/prompt.md", combinations)

    # for i in range(10):
    #     x = random.randint(0, len(trainset))
    #     print(trainset["title"][i])
    #     print()
    #     print("XXX")
    #     print()
    #
    # X_train = trainset[["year", "month", "day", "country", "title", "text"]]
    # y_train = trainset[["hazard_category"]]
    # with open("synthetic_data_hazard.csv", "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    # with open("synthetic_data_hazard.csv", "w", encoding="utf-8") as f:
    #     for line in lines:
    #         line = re.sub("\"", "", line)
    #     f.writelines(lines)

    synthetic_hazard = pd.read_csv(FILES_DIR / "datasets/synthetic_data_hazard.csv", sep=settings.SEPARATOR, engine='python')
    synthetic_product = pd.read_csv(FILES_DIR / "datasets/synthetic_data_product.csv", sep=settings.SEPARATOR, engine='python')
    print(synthetic_hazard.info())
    print(synthetic_hazard[synthetic_hazard["product"].isnull()])

    print(synthetic_product.info())
