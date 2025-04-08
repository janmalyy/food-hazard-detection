# Instruction

Generate a synthetic data example similar to the given examples.
Ensure that:
1. The language is natural and coherent, using varied, human-like phrasing.
2. The numerical fields are plausible and consistent with typical real-world distributions.
3. Examples do **NOT** repeat or use formulaic language.

## Dataset Description

- The data features include `year,` `month,` `day,` `country,` and `title.`
- Additionally, we provide the full text of the recall in the column `text.`
- The task is to predict the labels `product-category` and `hazard-category`.
- The class distribution is heavily imbalanced.
- Possible values in `hazard-category` are:


        ['biological' 'foreign bodies' 'chemical' 'fraud' 'organoleptic aspects'
         'allergens' 'packaging defect' 'other hazard'
         'food additives and flavourings' 'migration']
  - Possible values in `product-category` are:
  

      ['meat, egg and dairy products' 'prepared dishes and snacks'
       'cereals and bakery products' 'confectionery' 'ices and desserts'
       'alcoholic beverages' 'fruits and vegetables'
       'other food product / mixed'
       'cocoa and cocoa preparations, coffee and tea'
       'nuts, nut products and seeds' 'seafood'
       'soups, broths, sauces and condiments' 'fats and oils'
       'non-alcoholic beverages' 'food contact materials'
       'dietetic foods, food supplements, fortified foods' 'herbs and spices'
       'food additives and flavourings' 'sugars and syrups'
       'honey and royal jelly' 'feed materials' 'pet feed']

## Dataset Column Names
`year`, `month`, `day`, `country`, `title`, `text`, `hazard-category`, `product-category`, `hazard`, `product`

## Dataset Examples
XXDATASET_EXAMPLESXX

## Return
- The generated data in the same format as given examples, i.e. in csv format.
- Do **NOT** return anything else, only the synthetic generated data.
- The generated data must have the same `hazard-category` and `product-category`, i.e. XXHAZARDXX and XXPRODUCTXX.
- The exact `hazard` and `product` values can differ.
