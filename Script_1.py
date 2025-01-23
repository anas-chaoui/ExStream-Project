import pandas as pd
import numpy as np
import math
from collections import Counter
from collections import defaultdict
import numpy as np
from kneed import KneeLocator
import networkx as nx
import contextlib
import io
from scipy.stats import entropy
import os
from tqdm import tqdm  # Import tqdm for progress bar




# ----------------- Functions -----------------

def class_entropy(normal_values, abnormal_values):
    """
    This function computes the class entropy of the normal and abnormal values
    """
    #We compute the number of normal and abnormal values
    n_normal = len(normal_values)
    n_abnormal = len(abnormal_values)
    
    #We compute the total number of values
    n_total = n_normal + n_abnormal
    
    #we compute the proportion of normal and abnormal values

    p_normal = n_normal/n_total
    p_abnormal = n_abnormal/n_total

    #We compute the class entropy
    class_entropy = -p_normal * np.log(p_normal) - p_abnormal * np.log(p_abnormal)

    return class_entropy

# Since we will have tuples with (count_for_normal, count_for_abnormal) for mixed values
# we need to implement a function to sum the values of the tuples and the values of the list

def sum_values_and_tuples(arr):
    total_sum = 0
    for item in arr:
        if isinstance(item, tuple):
            total_sum += sum(item)
        else:
            total_sum += item
    return total_sum


# We implement the initial function of Segmentation entropy

def segmentation_entropy(list_of_segments, total_count):
    """
    This function computes the segmentation entropy of a list of segments
    """

    # Reminder, a segment is a dictionary with keys "type", "values" and "counts"
    segmentation_entropy = 0
    for segment in list_of_segments:
        segment_size = sum_values_and_tuples(segment["counts"])
        p_i = segment_size / total_count
        segmentation_entropy -= p_i * math.log(p_i)
        if segment["type"] == "M":
            # If the segment is mixed, we need to compute the entropy of the mixed segment
            worst_case_entropy = worst_case_ordering_and_entropy(segment)
            segmentation_entropy += worst_case_entropy
    
    return segmentation_entropy

#We implement a function to give us the worst case entropy of a list of segments

def worst_case_ordering_and_entropy(segment):
    values = segment['values']
    counts = segment['counts']
    
    # Generate the worst-case sequence by alternating between Ns and As
    total_counts = [sum(count) for count in counts]
    total_count = sum(total_counts)
    result = []
    
    last_type = None
    while any(total_counts):
        for i in range(len(values)):
            if total_counts[i] > 0:
                if counts[i][0] > 0 and (last_type is None or last_type == 'A'):
                    result.append('N')
                    counts[i] = (counts[i][0] - 1, counts[i][1])
                    total_counts[i] -= 1
                    last_type = 'N'
                    break
                elif counts[i][1] > 0 and (last_type is None or last_type == 'N'):
                    result.append('A')
                    counts[i] = (counts[i][0], counts[i][1] - 1)
                    total_counts[i] -= 1
                    last_type = 'A'
                    break
        else:
            # If we can't find a different type, just add any available type
            for i in range(len(values)):
                if total_counts[i] > 0:
                    if counts[i][0] > 0:
                        result.append('N')
                        counts[i] = (counts[i][0] - 1, counts[i][1])
                        total_counts[i] -= 1
                        last_type = 'N'
                    elif counts[i][1] > 0:
                        result.append('A')
                        counts[i] = (counts[i][0], counts[i][1] - 1)
                        total_counts[i] -= 1
                        last_type = 'A'
                    break
    
    #print("the worst case sequence is: ", result)
    # Compute the segmentation entropy
    total_points = len(result)
    entropy = 0
    current_segment_length = 1
    
    for i in range(1, total_points):
        if result[i] != result[i - 1]:
            p = current_segment_length / total_points
            entropy -= p * math.log(p)
            current_segment_length = 1
        else:
            current_segment_length += 1
    
    # Add the last segment
    p = current_segment_length / total_points
    entropy -= p * math.log(p)
    
    #print("the worst case entropy is: ", entropy)
    
    return entropy


    
def compute_sequential_entropy(normal_seq, abnormal_seq):
    """
    Compute sequential entropy for merged normal and abnormal sequences.
    
    Args:
        normal_seq (list): List of normal class values.
        abnormal_seq (list): List of abnormal class values.
        
    Returns:
        dict: {
            "segments": List of segments (type, values),
            "segmentation_entropy": Segmentation entropy,
            "worst_case_entropy": Worst-case entropy for mixed segments,
        }
    """

    # Count occurrences of each value (will be useful for worst-case entropy)
    normal_counter = Counter(normal_seq)
    abnormal_counter = Counter(abnormal_seq)

    total_count = sum_values_and_tuples(normal_counter.values()) + sum_values_and_tuples(abnormal_counter.values())

    # Convert counters to sets of keys
    normal_set = set(normal_counter.keys())
    abnormal_set = set(abnormal_counter.keys())

    # Perform set operations
    mixed_values = normal_set & abnormal_set  # Intersection: values in both N and A
    normal_only = normal_set - mixed_values  # Values only in N
    abnormal_only = abnormal_set - mixed_values  # Values only in A

    # Create a merged list with labels and counts
    merged = [(val, 'N', normal_counter[val]) for val in normal_only] + \
            [(val, 'A', abnormal_counter[val]) for val in abnormal_only] + \
            [(val, 'M', (normal_counter[val], abnormal_counter[val])) for val in mixed_values]

    # Sort by value
    merged.sort(key=lambda x: x[0])


    #print("the sorted merged list of values and labels is: ", merged)

    # Step 2: Create segments (normal, abnormal, and mixed)
    segments = []
    current_segment = {"type": None, "values": [], "counts": []}
    
    for value, label, count in merged:
        if current_segment["type"] is None:
            current_segment["type"] = label
            current_segment["values"].append(value)
            current_segment["counts"].append(count)
        elif current_segment["type"] == label:
            current_segment["values"].append(value)
            current_segment["counts"].append(count)
        else:
            # Append the current segment
            segments.append(current_segment)
            # Start a new segment
            current_segment = {"type": label, "values": [value], "counts": [count]}

    # Append the last segment
    if current_segment["values"]:
        segments.append(current_segment)

    # Segments is a list of dictionaries with keys "type" and "values"

    #print("the segments are: ", segments)

    # Step 3: Compute segmentation entropy
    #total_points = len(merged)

    seg_entropy = segmentation_entropy(segments, total_count)


    #with the list of mixed segments, we can compute the worst case entropy

    return seg_entropy

#let's compute the reward

def compute_reward(normal_seq, abnormal_seq):
    """
    Compute the reward for a given pair of normal and abnormal sequences.
    
    Args:
        normal_seq (list): List of normal class values.
        abnormal_seq (list): List of abnormal class values.
        
    Returns:
        float: Reward value.
    """
    # Compute sequential entropy
    seq_entropy = compute_sequential_entropy(normal_seq, abnormal_seq)
    #print("the sequential entropy is: ", seq_entropy)
    
    # Compute class entropy
    cl_entropy = class_entropy(normal_seq, abnormal_seq)
    #print("the class entropy is: ", cl_entropy)
    
    # Compute reward
    if seq_entropy == 0:
        return 0

    reward = cl_entropy/seq_entropy
    
    return reward

def compute_rewards(trace, labels, trace_id):
    """
    Compute the reward for each feature of each anomaly in a given trace.
    """
    rewards = {}

    # ✅ Step 1: Ensure first column is removed if it is unnamed
    if trace.columns[0].startswith("Unnamed"):
        trace = trace.iloc[:, 1:]  # Remove first column
    if labels.columns[0].startswith("Unnamed"):
        labels = labels.iloc[:, 1:]  # Remove first column

    # ✅ Step 2: Filter labels based on the provided trace_id
    filtered_labels = labels[labels["trace_id"] == trace_id]  

    print(f"\n📌 Computing rewards for trace: {trace_id}")
    print(f"📝 Filtered labels:\n{filtered_labels}")

    # ✅ Step 3: Iterate over anomalies in the current trace
    for _, row in filtered_labels.iterrows():
        ano_id = row["ano_id"]

        # ✅ Convert indices to integers
        ref_start, ref_end = int(row["ref_start"]), int(row["ref_end"])
        ano_start, ano_end = int(row["ano_start"]), int(row["ano_end"])

        max_index = len(trace) - 1  # Last valid index

        # ✅ Ensure indices are within valid range
        if ref_start > max_index or ref_end > max_index or ano_start > max_index or ano_end > max_index:
            print(f"⚠️ Error: Indices out of bounds for trace {trace_id}, ano_id {ano_id}. Skipping this anomaly.")
            continue  # Skip anomaly

        if ref_start < 0 or ref_end < 0 or ano_start < 0 or ano_end < 0:
            print(f"⚠️ Error: Negative indices detected for trace {trace_id}, ano_id {ano_id}. Skipping this anomaly.")
            continue  # Skip anomaly

        # ✅ Step 4: Extract normal and anomalous periods
        normal_period = trace.iloc[ref_start:ref_end, :]
        abnormal_period = trace.iloc[ano_start:ano_end, :]

        if abnormal_period.empty:
            print(f"⚠️ Warning: Anomalous period for (trace_id, ano_id) = ({trace_id}, {ano_id}) is empty! Skipping...")
            continue  # Skip anomaly

        print(f"\n🔍 Vérification des valeurs moyennes pour (trace_id, ano_id) = ({trace_id}, {ano_id})")
        print(f"   - Moyenne période normale: {normal_period.mean()}")
        print(f"   - Moyenne période anormale: {abnormal_period.mean()}")
        print(f"   - Différences absolues: {np.abs(normal_period.mean() - abnormal_period.mean())}")

        # Convert to NumPy arrays
        normal_values = normal_period.values
        abnormal_values = abnormal_period.values

        # ✅ Step 5: Compute rewards for each feature
        for i in range(normal_values.shape[1]):  # Loop through features
            normal_feature = normal_values[:, i]
            abnormal_feature = abnormal_values[:, i]
            rewards[(trace_id, ano_id, i)] = compute_reward(normal_feature, abnormal_feature)

    return rewards

def filter_and_sort_rewards(rewards):
    """
    Creates a sorted dictionary of rewards by (trace_id, ano_id).

    Args:
        rewards (dict): Dictionary with (trace_id, ano_id, feature) as keys and rewards as values.

    Returns:
        dict: Dictionary with (trace_id, ano_id) as keys and a sorted list of features and their rewards.
    """
    filtered_rewards = defaultdict(list)
    for (trace_id, ano_id, feature), reward in rewards.items():
        filtered_rewards[(trace_id, ano_id)].append((feature, reward))
    for key in filtered_rewards:
        filtered_rewards[key] = sorted(filtered_rewards[key], key=lambda x: x[1], reverse=True)
    return filtered_rewards

def find_all_elbows_threshold(reward_values, min_relative_drop=0.05):
    """
    Trouve **tous** les points de coude (elbows) et sélectionne le plus pertinent.

    Args:
        reward_values (list): Liste des valeurs de récompense triées **dans l'ordre décroissant**.
        min_relative_drop (float): Seuil minimal de variation relative pour considérer un coude.

    Returns:
        float: Le seuil optimal sélectionné.
    """
    if len(reward_values) < 2:
        return 0  # Cas trivial

    # Trouver **tous** les coudes possibles
    kneedle = KneeLocator(range(len(reward_values)), reward_values, curve="convex", direction="decreasing", online=True)
    all_knees = list(sorted(kneedle.all_knees))  # Convertir en liste triée

    if not all_knees:
        return 0  # Aucun coude détecté

    # Calculer les variations relatives entre les points détectés
    elbow_candidates = []
    for idx in all_knees:
        if idx == 0 or idx >= len(reward_values) - 1:
            continue  # Ignorer le premier et dernier point (cas bord)

        drop = abs(reward_values[idx] - reward_values[idx + 1]) / reward_values[idx]
        if drop >= min_relative_drop:  # Seuil de variation minimale pour considérer un coude
            elbow_candidates.append((idx, reward_values[idx], drop))

    if not elbow_candidates:
        return reward_values[all_knees[-1]]  # Prendre le dernier coude détecté si aucun ne respecte le seuil

    # Sélectionner le coude avec la **plus forte chute relative**
    best_elbow = max(elbow_candidates, key=lambda x: x[2])
    return best_elbow[1]  # Retourne la valeur seuil optimale

def apply_reward_filtering(filtered_rewards, max_features=20):
    """
    Applies Reward Leap Filtering using the Elbow Method and an optional fixed threshold.

    Args:
        filtered_rewards (dict): Dictionary of sorted rewards by (trace_id, ano_id).
        max_features (int): Maximum number of features allowed. A fixed threshold is applied if exceeded.

    Returns:
        dict: Final filtered rewards.
        dict: Number of features kept per anomaly.
        dict: Thresholds used per anomaly.
    """
    final_filtered_rewards = {}
    features_per_anomaly = defaultdict(int)
    thresholds_per_anomaly = defaultdict(list)

    for (trace_id, ano_id), features in filtered_rewards.items():
        selected_features = []
        prev_reward = None
        
        # Extract reward values for threshold calculation
        reward_values = [reward for _, reward in features]
        threshold = find_all_elbows_threshold(reward_values)
        thresholds_per_anomaly[ano_id].append(threshold)

        # Apply reward filtering
        for feature, reward in features:
            if prev_reward is not None and abs(prev_reward - reward) > threshold:
                break  # Stop when the threshold is exceeded
            selected_features.append((feature, reward))
            prev_reward = reward

        # If more than max_features are kept, apply a fixed threshold
        if len(selected_features) > max_features:
            print(f"⚠️ More than {max_features} features retained for (trace_id, ano_id)=({trace_id}, {ano_id})! Applying fixed threshold = 0.1")
            selected_features = []
            prev_reward = None
            threshold = 0.1  # Fixed threshold
            thresholds_per_anomaly[ano_id] = threshold  # Update the final threshold used

            for feature, reward in features:
                if prev_reward is not None and abs(prev_reward - reward) > threshold:
                    break  # Stop when the fixed threshold is exceeded
                selected_features.append((feature, reward))
                prev_reward = reward

        final_filtered_rewards[(trace_id, ano_id)] = selected_features
        features_per_anomaly[ano_id] += len(selected_features)

        print(f"For (trace_id, ano_id)=({trace_id}, {ano_id}), {len(selected_features)} features retained with a threshold of {threshold:.4f}.")
    
    return final_filtered_rewards, features_per_anomaly, thresholds_per_anomaly

def summarize_feature_filtering(features_per_anomaly, thresholds_per_anomaly):
    """
    Summarizes the number of features retained and the thresholds used for each anomaly.

    Args:
        features_per_anomaly (dict): Dictionary with the number of features retained per anomaly.
        thresholds_per_anomaly (dict): Dictionary with the thresholds used per anomaly.

    Returns:
        None
    """
    print("\n📊 Summary of retained features per anomaly and thresholds used:")
    for ano_id, count in features_per_anomaly.items():
        avg_threshold = np.mean(thresholds_per_anomaly[ano_id]) if thresholds_per_anomaly[ano_id] else 0
        print(f"  Anomaly {ano_id}: {count} features retained | Final threshold: {avg_threshold:.4f}")

def display_final_features(final_filtered_rewards):
    """
    Displays the final retained features for each (trace_id, ano_id).

    Args:
        final_filtered_rewards (dict): Dictionary of final retained features and their rewards.

    Returns:
        None
    """
    print("\n📋 Final retained features:")
    for (trace_id, ano_id), features in final_filtered_rewards.items():
        print(f"\n(trace_id, ano_id)=({trace_id}, {ano_id}):")
        for feature, reward in features:
            print(f"  Feature: {feature}, Reward: {reward}")

def extract_features_per_anomaly(final_filtered_rewards):
    """
    Extracts the features retained for each anomaly and organizes them by (trace_id, ano_id).
    
    Args:
        final_filtered_rewards (dict): Dictionary of features retained with rewards for each anomaly.
        
    Returns:
        dict: Features retained per anomaly in a structured format.
    """
    features_per_anomaly = {}
    for (trace_id, ano_id), features in final_filtered_rewards.items():
        feature_names = [feat for feat, _ in features]
        features_per_anomaly[(trace_id, ano_id)] = feature_names
    print("📊 Selected features before clustering:", features_per_anomaly)
    return features_per_anomaly

def compute_correlation_matrices(features_per_anomaly, correlation_matrices):
    """
    Computes the correlation matrix for each anomaly based on selected features.
    
    Args:
        features_per_anomaly (dict): Selected features per anomaly.
        correlation_matrices (dict): Dictionary to store correlation matrices.
        
    Returns:
        dict: Updated dictionary containing correlation matrices for each anomaly.
    """
    for (trace_id, ano_id), feature_indices in features_per_anomaly.items():
        trace_df = pd.read_csv(f"{trace_id}.csv")  # Load data
        
        # Validate feature indices or names
        if all(isinstance(idx, int) for idx in feature_indices):
            feature_names = [trace_df.columns[idx] for idx in feature_indices if idx < len(trace_df.columns)]
        else:
            feature_names = feature_indices  # If already names, no conversion needed
        
        # Check for missing features
        missing_features = [f for f in feature_names if f not in trace_df.columns]
        if missing_features:
            print(f"⚠️ Missing features in {trace_id}.csv: {missing_features}")
            continue  # Skip to the next anomaly
        
        # Filter and standardize data
        filtered_df = trace_df[feature_names]
        filtered_df = (filtered_df - filtered_df.mean()) / filtered_df.std()

        # Compute correlation matrix
        corr_matrix = filtered_df.corr()
        correlation_matrices[(trace_id, ano_id)] = corr_matrix
        print(f"📌 Correlation matrix computed for (trace_id, ano_id) = ({trace_id}, {ano_id})")
        print(corr_matrix)
    return correlation_matrices

def build_correlation_graphs(correlation_matrices, correlation_threshold=0.85):
    """
    Constructs correlation graphs for features exceeding the specified correlation threshold.
    
    Args:
        correlation_matrices (dict): Correlation matrices for each anomaly.
        correlation_threshold (float): Threshold to consider two features as correlated.
        
    Returns:
        dict: Graphs of correlated features for each anomaly.
    """
    correlation_graphs = {}
    for (trace_id, ano_id), corr_matrix in correlation_matrices.items():
        G = nx.Graph()  # Initialize a graph
        features = corr_matrix.columns
        G.add_nodes_from(features)  # Add features as nodes

        # Add edges for correlated features
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    G.add_edge(features[i], features[j])

        correlation_graphs[(trace_id, ano_id)] = G
        print(f"📊 Graph constructed for (trace_id, ano_id) = ({trace_id}, {ano_id}) with {len(G.nodes)} nodes.")
    return correlation_graphs

def detect_feature_clusters(correlation_graphs):
    """
    Detects clusters of correlated features for each anomaly.
    
    Args:
        correlation_graphs (dict): Graphs of correlated features for each anomaly.
        
    Returns:
        dict: Clusters of correlated features for each anomaly.
    """
    feature_clusters = {}
    for (trace_id, ano_id), G in correlation_graphs.items():
        clusters = list(nx.connected_components(G))  # Identify connected components (clusters)
        feature_clusters[(trace_id, ano_id)] = clusters
        print(f"📌 {len(clusters)} clusters detected for (trace_id, ano_id) = ({trace_id}, {ano_id})")
    return feature_clusters

def select_best_features(feature_clusters, selected_features):
    """
    Selects the best feature from each cluster based on the highest reward.
    
    Args:
        feature_clusters (dict): Clusters of correlated features.
        selected_features (dict): Features and their rewards for each anomaly.
        
    Returns:
        dict: Final retained features after removing redundancies.
    """
    final_features = {}
    for (trace_id, ano_id), clusters in feature_clusters.items():
        retained_features = []
        feature_rewards = dict(selected_features.get((trace_id, ano_id), []))
        
        for cluster in clusters:
            # Choose the feature with the highest reward in the cluster
            best_feature = max(cluster, key=lambda f: feature_rewards.get(f, 0))
            retained_features.append(best_feature)  # Stocker uniquement le nom de la feature
        
        final_features[(trace_id, ano_id)] = retained_features
        print(f"✅ {len(retained_features)} features retained after clustering for (trace_id, ano_id) = ({trace_id}, {ano_id})")
    
    return final_features


def display_final_summary(final_features):
    """
    Displays the final retained features for each anomaly after clustering.
    
    Args:
        final_features (dict): Final retained features after redundancy removal.
        
    Returns:
        None
    """
    print("\n📊 Final summary of retained features after clustering:")
    for (trace_id, ano_id), features in final_features.items():
        print(f"\n(trace_id, ano_id) = ({trace_id}, {ano_id}) :")
        for feature in features:  # Correction ici : on boucle simplement sur les features
            print(f"  - Feature: {feature}")

def organize_anomalies_features(final_features):
    """
    Organizes the final features for each anomaly into a dictionary.
    
    Args:
        final_features (dict): Final retained features after redundancy removal.
        
    Returns:
        dict: Dictionary mapping each anomaly to its final features.
    """
    anomalies_features = {}
    for (trace_id, ano_id), features in final_features.items():
        anomalies_features[(trace_id, ano_id)] = [feat for feat in features]  # On s'assure qu'on ne garde que les noms
    print(anomalies_features)
    return anomalies_features

def select_best_features(feature_clusters, final_filtered_rewards):
    """
    Sélectionne la meilleure caractéristique par cluster.
    """
    final_features = {}

    for (trace_id, ano_id), clusters in feature_clusters.items():
        retained_features = []
        feature_rewards = dict(final_filtered_rewards.get((trace_id, ano_id), []))

        for cluster in clusters:
            # Sélectionner **uniquement** la meilleure feature par cluster
            best_feature = max(cluster, key=lambda f: feature_rewards.get(f, 0), default=None)
            if best_feature:
                retained_features.append(best_feature)

        final_features[(trace_id, ano_id)] = retained_features

    return final_features

def process_full_pipeline(trace, labels, trace_id, sample_fraction=0.8):
    """
    Process features for anomalies in a sampled trace, compute rewards, apply filtering,
    compute correlation, detect clusters, and select final features after clustering.
    
    This version performs sampling at 80% inside each anomaly's reference and anomalous period.
    
    Args:
        trace (pd.DataFrame): The full trace dataset.
        labels (pd.DataFrame): The anomaly labels for the dataset.
        trace_id (str): The ID of the trace being processed.
        sample_fraction (float): Fraction of each anomaly's segment to keep (default is 0.8).
    
    Returns:
        dict: Final retained features for each (trace_id, ano_id) after clustering.
    """
    print(f"\n🚀 STARTING PROCESS FOR TRACE: {trace_id}")

    # ✅ Étape 1: Filtrer les labels de la bonne trace
    filtered_labels = labels[labels["trace_id"] == trace_id].copy()
    print(f"\n📝 Filtered Labels for Trace {trace_id}:")
    print(filtered_labels)

    if filtered_labels.empty:
        print(f"❌ No anomalies found for trace {trace_id}. Skipping processing.")
        return {}

    # ✅ Étape 2: Échantillonnage à 80% des périodes de référence et anormales
    updated_labels = []
    for i, row in filtered_labels.iterrows():
        ano_id = row["ano_id"]
        ref_start, ref_end = int(row["ref_start"]), int(row["ref_end"])
        ano_start, ano_end = int(row["ano_start"]), int(row["ano_end"])

        # Sample 80% of the normal period
        normal_indices = np.linspace(ref_start, ref_end-1, num=(ref_end - ref_start), dtype=int)
        sampled_normal_indices = np.random.choice(normal_indices, size=int(len(normal_indices) * sample_fraction), replace=False)
        sampled_normal_indices.sort()
        
        # Update ref_end_post_sampling to last sampled index
        ref_end_post_sampling = sampled_normal_indices[-1]

        # Sample 80% of the anomalous period
        abnormal_indices = np.linspace(ano_start, ano_end-1, num=(ano_end - ano_start), dtype=int)
        sampled_abnormal_indices = np.random.choice(abnormal_indices, size=int(len(abnormal_indices) * sample_fraction), replace=False)
        sampled_abnormal_indices.sort()
        
        # Update ano_start and ano_end
        ano_start_post_sampling = ref_end_post_sampling  # Anomaly starts right after last normal index
        ano_end_post_sampling = sampled_abnormal_indices[-1]

        updated_labels.append({
            "trace_id": trace_id,
            "ano_id": ano_id,
            "ref_start": ref_start,  # ref_start remains unchanged
            "ref_end": ref_end_post_sampling,
            "ano_start": ano_start_post_sampling,
            "ano_end": ano_end_post_sampling,
            "ano_type": row["ano_type"]
        })

    updated_labels = pd.DataFrame(updated_labels)
    print(f"\n🔄 Updated Labels After Sampling:")
    print(updated_labels)

    # ✅ Étape 3: Calculer les récompenses pour chaque caractéristique dans chaque anomalie
    rewards = compute_rewards(trace, updated_labels, trace_id)
    print(f"\n🏆 Rewards Computed:")
    print(rewards)

    # ✅ Étape 4: Filtrer et trier les récompenses
    filtered_rewards = filter_and_sort_rewards(rewards)
    print(f"\n🔍 Filtered and Sorted Rewards:")
    print(filtered_rewards)

    # ✅ Étape 5: Appliquer le filtrage des récompenses avec la méthode du coude (Elbow Method)
    final_filtered_rewards, features_per_anomaly, _ = apply_reward_filtering(filtered_rewards)
    print(f"\n✅ Final Filtered Rewards:")
    print(final_filtered_rewards)

    # ✅ Étape 6: Extraire les caractéristiques par anomalie
    features_per_anomaly = extract_features_per_anomaly(final_filtered_rewards)
    print(f"\n📌 Features Per Anomaly Before Clustering:")
    print(features_per_anomaly)

    # ✅ Étape 7: Calculer les matrices de corrélation
    correlation_matrices = compute_correlation_matrices(features_per_anomaly, {})
    print(f"\n📊 Correlation Matrices:")
    print(correlation_matrices)

    # ✅ Étape 8: Construire les graphes de corrélation
    correlation_graphs = build_correlation_graphs(correlation_matrices)
    print(f"\n📊 Correlation Graphs:")
    print(correlation_graphs)

    # ✅ Étape 9: Détecter les clusters de caractéristiques
    feature_clusters = detect_feature_clusters(correlation_graphs)
    print(f"\n📌 Feature Clusters:")
    print(feature_clusters)

    # ✅ Étape 10: Sélectionner les meilleures caractéristiques par cluster
    final_features = select_best_features(feature_clusters, final_filtered_rewards)
    print(f"\n✅ Final Features Selected:")
    print(final_features)

    # ✅ Étape 11: Organiser les caractéristiques finales par anomalie
    anomalies_features = organize_anomalies_features(final_features)
    print(f"\n📊 Anomalies Features (Final Output):")
    print(anomalies_features)

    print(f"\n🎯 PROCESS COMPLETED FOR TRACE: {trace_id}\n")

    return anomalies_features

def compute_feature_consistency_entropy_silent(trace, labels, trace_id, num_samples=5, sample_fraction=0.8):
    all_sampled_features = defaultdict(list)
    
    for i in range(num_samples):
        print(f"Executing run {i+1}/{num_samples}") 
        
        with contextlib.redirect_stdout(io.StringIO()):  # Suppress all other prints
            final_features = process_full_pipeline(trace, labels, trace_id, sample_fraction=sample_fraction)

        # Store the features selected for each anomaly in this run
        for (trace_id, ano_id), features in final_features.items():
            feature_set = set(features) 
            all_sampled_features[(trace_id, ano_id)].append(feature_set)

    # Compute entropy for feature consistency
    feature_entropies = {}
    for (trace_id, ano_id), feature_lists in all_sampled_features.items():
        feature_counts = defaultdict(int)
        total_runs = len(feature_lists)

        # Count occurrences of each feature across all runs
        for feature_set in feature_lists:
            for feature in feature_set:
                feature_counts[feature] += 1

        # Compute entropy
        probabilities = np.array(list(feature_counts.values())) / total_runs
        feature_entropy = entropy(probabilities, base=2)  # Compute entropy in base 2
        feature_entropies[(trace_id, ano_id)] = feature_entropy

    return feature_entropies

def generate_explanation_metrics(trace_id, labels, num_samples=5, sample_fraction=0.8):
    """
    Compute explanations and instability metrics for a given trace.

    Args:
        trace_id (str): The trace identifier.
        labels (pd.DataFrame): The anomaly labels DataFrame.
        num_samples (int): Number of sampling runs for entropy computation.
        sample_fraction (float): Fraction of the dataset sampled per run.

    Returns:
        list: List of dictionaries containing explanation metrics.
    """
    trace_path = f"{trace_id}.csv"
    if not os.path.exists(trace_path):
        print(f"❌ Warning: Trace file {trace_path} not found. Skipping.")
        return []

    trace = pd.read_csv(trace_path)  # Charger les données de la trace

    # ✅ Calcul de l'entropie d'instabilité pour chaque anomalie
    entropy_values = compute_feature_consistency_entropy_silent(
        trace, labels, trace_id, num_samples=num_samples, sample_fraction=sample_fraction
    )

    results = []
    for (trace_id, ano_id), entropy_value in tqdm(entropy_values.items(), desc=f"Processing {trace_id}", leave=False):
        
        # ✅ Exécuter process_full_pipeline pour obtenir les caractéristiques finales
        with contextlib.redirect_stdout(io.StringIO()):  
            final_features = process_full_pipeline(trace, labels, trace_id, sample_fraction=sample_fraction)
        
        # Récupérer les caractéristiques finales utilisées pour l'explication
        selected_features = final_features.get((trace_id, ano_id), [])

        # ✅ Convertir les noms de caractéristiques en indices
        feature_indices = [trace.columns.get_loc(feature) for feature in selected_features if feature in trace.columns]

        # ✅ Stocker les résultats
        results.append({
            "trace_id": trace_id,
            "ano_id": ano_id,
            "exp_size": len(selected_features),
            "exp_instability": entropy_value,
            "explanation": ', '.join(map(str, feature_indices))  # Stocker les indices des features sous forme de chaîne
        })

    return results

# ----------------- Main -----------------

def main():
        
    # ✅ Charger les labels
    labels = pd.read_csv('labels.csv')

    # ✅ Extraire les IDs uniques des traces
    trace_ids = labels["trace_id"].unique()

    # ✅ Générer les explications et les métriques pour toutes les traces
    all_results = []
    for trace_id in tqdm(trace_ids, desc="Processing all traces"):
        all_results.extend(generate_explanation_metrics(trace_id, labels))

    # ✅ Convertir en DataFrame et sauvegarder les résultats
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("folder_1_results.csv", index=False)

    print("\n✅ Processing complete! Results saved in folder_1_results.csv")

if __name__ == "__main__":
    main()