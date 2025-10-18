import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import timeit

# Function to generate potential itemsets
def generate_itemsets(items):
    subsets = [[]]
    for item in items:
        new_combinations = [subset + [item] for subset in subsets]
        subsets.extend(new_combinations)
    return subsets[1:]

# Brute Force Approach for Association Rule Mining
def brute_force_association(dataframe, min_support, min_confidence):
    transactions = dataframe["Items"].str.split(", ")
    num_transactions = len(transactions)
    
    item_list = [item for sublist in transactions for item in sublist]
    item_counts = pd.Series(item_list).value_counts()
    
    # Filtering based on minimum support
    freq_items = item_counts[item_counts >= (min_support / 100) * num_transactions].reset_index()
    freq_items.columns = ["Itemsets", "Frequency"]
    freq_items["Support"] = (freq_items["Frequency"] / num_transactions) * 100
    
    # Generate possible itemsets
    potential_itemsets = generate_itemsets(freq_items["Itemsets"])
    valid_itemsets = [iset for iset in potential_itemsets if len(iset) > 1]
    
    # Count occurrences of each itemset
    itemset_occurrences = {}
    for i_set in valid_itemsets:
        count = sum(1 for transaction in transactions if set(i_set).issubset(set(transaction)))
        itemset_occurrences[tuple(i_set)] = count
    
    itemset_df = pd.DataFrame(list(itemset_occurrences.items()), columns=["Itemsets", "Frequency"])
    itemset_df["Support"] = (itemset_df["Frequency"] / num_transactions) * 100
    
    # Combine frequent single items with frequent itemsets
    all_frequent_items = pd.concat([freq_items, itemset_df[itemset_df["Support"] >= min_support]], ignore_index=True)
    
    # Generate association rules based on confidence threshold
    item_support_map = dict(zip(all_frequent_items["Itemsets"], all_frequent_items["Support"]))
    association_dict = {}
    support_values = []
    
    for antecedent in item_support_map:
        for consequent in item_support_map:
            if set(consequent).issubset(set(antecedent)) and antecedent != consequent:
                conf_value = (item_support_map[antecedent] / item_support_map[consequent]) * 100
                association_dict[(antecedent, consequent)] = conf_value
                support_values.append(item_support_map[antecedent])
    
    rule_df = pd.DataFrame(association_dict.items(), columns=["Rules", "Confidence"])
    rule_df["Support"] = support_values
    rule_df = rule_df[rule_df["Confidence"] >= min_confidence].reset_index(drop=True)
    
    return rule_df, all_frequent_items

# Apriori Algorithm

def apriori_algorithm(dataframe, min_support, min_confidence):
    transaction_matrix = dataframe["Items"].str.get_dummies(sep=", ").astype(bool)
    frequent_itemsets = apriori(transaction_matrix, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    frequent_itemsets.rename(columns={"itemsets": "Itemsets", "support": "Support"}, inplace=True)
    frequent_itemsets["Support"] *= 100
    return rules, frequent_itemsets

# FP-Growth Algorithm
def fpgrowth_algorithm(dataframe, min_support, min_confidence):
    transaction_matrix = dataframe["Items"].str.get_dummies(sep=", ").astype(bool)
    frequent_itemsets = fpgrowth(transaction_matrix, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    frequent_itemsets.rename(columns={"itemsets": "Itemsets", "support": "Support"}, inplace=True)
    frequent_itemsets["Support"] *= 100
    return rules, frequent_itemsets

# Display results for Brute Force Approach
def display_manual_rules(rule_df):
    for idx, row in rule_df.iterrows():
        print(f"Rule_{idx + 1}: {row['Rules'][0]} --> {row['Rules'][1]} Support: {row['Support']} Confidence: {row['Confidence']}\n")

# Display results for Apriori and FP-Growth Approaches
def display_algorithmic_rules(rule_df):
    for idx, row in rule_df.iterrows():
        print(f"Rule_{idx + 1}: {tuple(row['antecedents'])} --> {tuple(row['consequents'])} Support: {row['support'] * 100} Confidence: {row['confidence'] * 100}\n")

# Company Selection
companies = {1: "amazon", 2: "costco", 3: "ikea", 4: "target", 5: "7-11", 6: "walmart", 7: "Quit"}

while True:
    print("\nWelcome to Market Basket Analysis Tool\n")
    for num, name in companies.items():
        print(f"{num}: {name}")
    
    choice = int(input("\nSelect a company from the above list: "))
    if choice == 7:
        print("Exiting program...")
        break
    
    print(f"\nSelected: {companies.get(choice, 'Invalid Choice')}")
    min_sup = float(input("Enter minimum support (1-100%): "))
    min_conf = float(input("Enter minimum confidence (1-100%): "))
    
    if choice in companies:
        data_url = f"https://raw.githubusercontent.com/Pikaboo69/Datasets/refs/heads/main/{companies[choice]}.csv"
        transaction_data = pd.read_csv(data_url)
        
        # Brute Force Execution
        bf_rules, bf_freq_items = brute_force_association(transaction_data, min_sup, min_conf)
        print("\nBrute Force Frequent Itemsets:\n", bf_freq_items)
        print("\nBrute Force Association Rules:")
        display_manual_rules(bf_rules)
        
        # Apriori Execution
        apriori_rules, apriori_items = apriori_algorithm(transaction_data, min_sup / 100, min_conf / 100)
        print("\nApriori Frequent Itemsets:\n", apriori_items)
        print("\nApriori Association Rules:")
        display_algorithmic_rules(apriori_rules)
        
        # FP-Growth Execution
        fp_rules, fp_items = fpgrowth_algorithm(transaction_data, min_sup / 100, min_conf / 100)
        print("\nFP-Growth Frequent Itemsets:\n", fp_items)
        print("\nFP-Growth Association Rules:")
        display_algorithmic_rules(fp_rules)
        
        # Execution Time Comparison
        print(f"Execution Time (Brute Force): {timeit.timeit(lambda: brute_force_association(transaction_data, min_sup, min_conf), number=1)} sec")
        print(f"Execution Time (Apriori): {timeit.timeit(lambda: apriori_algorithm(transaction_data, min_sup / 100, min_conf / 100), number=1)} sec")
        print(f"Execution Time (FP-Growth): {timeit.timeit(lambda: fpgrowth_algorithm(transaction_data, min_sup / 100, min_conf / 100), number=1)} sec")
    else:
        print("Invalid choice, please try again.")
