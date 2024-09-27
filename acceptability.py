import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None, result=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.result = result

#to keep track of the node count
Node.node_count = 0

def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(data, feature, threshold, target):
    total_entropy = entropy(data[target])

    left_mask = data[feature] <= threshold
    right_mask = ~left_mask

    left_entropy = entropy(data[left_mask][target])
    right_entropy = entropy(data[right_mask][target])

    left_weight = len(data[left_mask]) / len(data)
    right_weight = len(data[right_mask]) / len(data)

    gain = total_entropy - (left_weight * left_entropy + right_weight * right_entropy)
    return gain

def find_best_split(data, features, target):
    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature in features:
        unique_values = data[feature].unique()
        for value in unique_values:
            gain = information_gain(data, feature, value, target)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = value

    return best_feature, best_threshold


def build_decision_tree(data, features, target):
    Node.node_count += 1  # Increment node count when a new node is created

    if len(np.unique(data[target])) == 1:
        return Node(result=data[target].iloc[0])

    if len(features) == 0:
        majority_label = data[target].mode().iloc[0]
        return Node(result=majority_label)

    best_feature, best_threshold = find_best_split(data, features, target)

    if best_feature is None:
        majority_label = data[target].mode().iloc[0]
        return Node(result=majority_label)

    left_mask = data[best_feature] <= best_threshold
    right_mask = ~left_mask

    left_subtree = build_decision_tree(data[left_mask], features, target)
    right_subtree = build_decision_tree(data[right_mask], features, target)

    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)


def predict(tree, example):
    if tree.result is not None:
        return tree.result

    if example[tree.feature] <= tree.threshold:
        return predict(tree.left, example)
    else:
        return predict(tree.right, example)


train_data = pd.read_excel("trainDATA.xlsx")

# Separate features and target variable
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Build the decision tree
features = X_train.columns.tolist()
target = y_train.name
decision_tree = build_decision_tree(train_data, features, target)

# Display the tree structure and the number of nodes
def display_tree_structure(node, depth=0):
    if node.result is not None:
        print(f"{'  ' * depth}Result: {node.result}")
    else:
        print(f"{'  ' * depth}{node.feature} <= {node.threshold}")
        display_tree_structure(node.left, depth + 1)
        display_tree_structure(node.right, depth + 1)

print("Decision Tree Structure:")
display_tree_structure(decision_tree)
print(f"Number of Nodes: {Node.node_count}")

test_data = pd.read_excel("testDATA.xlsx")

test_predictions = [predict(decision_tree, example) for _, example in test_data.iterrows()]

result_df = pd.concat([test_data, pd.DataFrame({'Predicted_Acceptability': test_predictions})], axis=1)
result_df.to_excel("results.xlsx", index=False)
