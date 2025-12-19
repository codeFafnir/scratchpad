"""
# Assignment 1: Book Recommendation System
This assignment is divided into three tasks:
1. Task 1: Read Prediction using Jaccard Similarity
2. Task 2: Hybrid Recommender System
3. Task 3: Logistic Regression on Hybrid Features
"""

# =============================================================================
# IMPORTS
# =============================================================================
import gzip
from collections import defaultdict
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# =============================================================================
# DATA LOADING
# =============================================================================
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u, b, r = l.strip().split(',')
        r = int(r)
        yield u, b, r

# Load all ratings
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# Split into train and validation
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]

# Build user and item dictionaries
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u, b, r in ratingsTrain:
    ratingsPerUser[u].append((b, r))
    ratingsPerItem[b].append((u, r))

# Calculate book popularity
bookCount = defaultdict(int)
totalRead = 0
for user, book, _ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort(reverse=True)


# =============================================================================
# TASK 1: READ PREDICTION USING JACCARD SIMILARITY
# =============================================================================
# This task implements a Jaccard similarity-based approach for predicting 
# whether a user will read a book.

def generateValidation(allRatings, ratingsValid):
    """
    Generate validation sets with positive and negative samples.
    
    Returns:
        readValid: set of (u, b) pairs in the validation set
        notRead: set of (u, b') negative pairs for each validation user
    """
    readValid = set()
    for u, b, r in ratingsValid:
        readValid.add((u, b))
    
    booksPerUser = defaultdict(set)
    for u, b, r in allRatings:
        booksPerUser[u].add(b)
    
    allBooks = set(b for u, b, r in allRatings)
    
    notRead = set()
    random.seed(42)
    usedNegatives = defaultdict(set)
    
    for u, b, r in ratingsValid:
        readByUser = booksPerUser[u]
        unreadBooks = allBooks - readByUser - usedNegatives[u]
        
        if len(unreadBooks) > 0:
            negBook = random.choice(list(unreadBooks))
            notRead.add((u, negBook))
            usedNegatives[u].add(negBook)
        else:
            unreadBooks = allBooks - readByUser
            if len(unreadBooks) > 0:
                negBook = random.choice(list(unreadBooks))
                notRead.add((u, negBook))
    
    return readValid, notRead


def jaccardThresh(u, b, ratingsPerItem, ratingsPerUser, jaccard_threshold=0.013, popularity_threshold=40):
    """
    Predict if user will read book based on Jaccard similarity.
    
    Args:
        u: user ID
        b: book ID
        ratingsPerItem: dict mapping book -> list of (user, rating) pairs
        ratingsPerUser: dict mapping user -> list of (book, rating) pairs
        jaccard_threshold: threshold for Jaccard similarity
        popularity_threshold: threshold for book popularity
    
    Returns:
        1 if predict "will read", 0 otherwise
    """
    if b in ratingsPerItem and len(ratingsPerItem[b]) > popularity_threshold:
        return 1
    
    if u not in ratingsPerUser or b not in ratingsPerItem:
        return 0
    
    usersB = set(user for user, _ in ratingsPerItem[b])
    userBooks = [book for book, _ in ratingsPerUser[u]]
    
    maxSim = 0.0
    for book_prime in userBooks:
        if book_prime == b or book_prime not in ratingsPerItem:
            continue
        
        usersBPrime = set(user for user, _ in ratingsPerItem[book_prime])
        intersection_size = len(usersB.intersection(usersBPrime))
        union_size = len(usersB.union(usersBPrime))
        
        if union_size > 0:
            similarity = intersection_size / union_size
            maxSim = max(maxSim, similarity)
    
    if maxSim > jaccard_threshold or len(ratingsPerItem[b]) > popularity_threshold:
        return 1
    return 0


def evaluateJaccard(readValid, notRead, ratingsPerItem, ratingsPerUser):
    """Evaluate Jaccard strategy with detailed metrics."""
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for u, b in readValid:
        prediction = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser)
        if prediction == 1:
            true_positives += 1
        else:
            false_negatives += 1
    
    for u, b in notRead:
        prediction = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser)
        if prediction == 0:
            true_negatives += 1
        else:
            false_positives += 1
    
    total = len(readValid) + len(notRead)
    accuracy = (true_positives + true_negatives) / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }


def optimizeJaccardThresholds(readValid, notRead, ratingsPerItem, ratingsPerUser,
                               jaccard_range=None, popularity_range=None):
    """
    Find optimal Jaccard and popularity thresholds.
    
    Returns:
        Dictionary with best combinations by F1 and accuracy
    """
    if jaccard_range is None:
        jaccard_range = [0.005, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013,
                        0.014, 0.015, 0.018, 0.020, 0.025]
    if popularity_range is None:
        popularity_range = [20, 25, 28, 30, 32, 35, 37, 40, 42, 45, 50, 55, 60]
    
    results = []
    
    for jaccard_thresh in jaccard_range:
        for pop_thresh in popularity_range:
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            for u, b in readValid:
                prediction = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser,
                                          jaccard_thresh, pop_thresh)
                if prediction == 1:
                    true_positives += 1
                else:
                    false_negatives += 1
            
            for u, b in notRead:
                prediction = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser,
                                          jaccard_thresh, pop_thresh)
                if prediction == 0:
                    true_negatives += 1
                else:
                    false_positives += 1
            
            total = len(readValid) + len(notRead)
            accuracy = (true_positives + true_negatives) / total
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'jaccard_threshold': jaccard_thresh,
                'popularity_threshold': pop_thresh,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })
    
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    results_by_accuracy = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    return {
        'all_results': results,
        'top_by_f1': results[:3],
        'top_by_accuracy': results_by_accuracy[:3]
    }


def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    """Write predictions to file."""
    predictions = open("predictions_Read.csv", 'w', encoding='utf-8')
    for line in open("pairs_Read.csv", encoding='utf-8'):
        if line.startswith("userID"):
            predictions.write(line)
            continue
        u, b = line.strip().split(',')
        pred = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser)
        predictions.write(u + ',' + b + ',' + str(pred) + '\n')
    predictions.close()


# =============================================================================
# TASK 2: HYBRID RECOMMENDER SYSTEM
# =============================================================================
# This task implements a hybrid recommender system that combines multiple 
# similarity metrics for improved predictions.

def generateBalancedNegativeSamples(ratingsData, ratingsPerItem, ratingsPerUser, 
                                     num_negatives=None, popularity_distribution='balanced'):
    """
    Generate negative samples with configurable popularity distribution.
    
    Args:
        ratingsData: List of (user, book, rating) tuples
        ratingsPerItem: Dict mapping book -> list of (user, rating)
        ratingsPerUser: Dict mapping user -> list of (book, rating)
        num_negatives: Number of negative samples to generate
        popularity_distribution: 'balanced' (40/40/20) or 'unpopular' (0/20/80)
    """
    all_books = list(ratingsPerItem.keys())
    book_popularity = {b: len(ratingsPerItem[b]) for b in all_books}
    
    books_by_popularity = sorted(all_books, key=lambda b: book_popularity[b], reverse=True)
    
    n_books = len(all_books)
    popular_books = set(books_by_popularity[:n_books//3])
    medium_books = set(books_by_popularity[n_books//3:2*n_books//3])
    unpopular_books = set(books_by_popularity[2*n_books//3:])
    
    if num_negatives is None:
        num_negatives = len(ratingsData)
    
    if popularity_distribution == 'unpopular':
        n_popular = 0
        n_medium = int(num_negatives * 0.2)
        n_unpopular = num_negatives - n_popular - n_medium
    else:
        n_popular = int(num_negatives * 0.4)
        n_medium = int(num_negatives * 0.4)
        n_unpopular = num_negatives - n_popular - n_medium
    
    negative_samples = set()
    random.seed(42)
    all_users = list(set(u for u, b, r in ratingsData))
    
    def sample_negatives_from_bucket(bucket, n_samples):
        samples = set()
        attempts = 0
        max_attempts = n_samples * 50
        
        while len(samples) < n_samples and attempts < max_attempts:
            u = random.choice(all_users)
            
            if u in ratingsPerUser:
                read_books = set(book for book, rating in ratingsPerUser[u])
            else:
                read_books = set()
            
            unread_in_bucket = bucket - read_books
            
            if len(unread_in_bucket) > 0:
                b = random.choice(list(unread_in_bucket))
                samples.add((u, b))
            
            attempts += 1
        
        return samples
    
    if n_popular > 0:
        negative_samples.update(sample_negatives_from_bucket(popular_books, n_popular))
    if n_medium > 0:
        negative_samples.update(sample_negatives_from_bucket(medium_books, n_medium))
    if n_unpopular > 0:
        negative_samples.update(sample_negatives_from_bucket(unpopular_books, n_unpopular))
    
    return negative_samples


class ImprovedHybridRecommender:
    """Hybrid recommender with cold start handling and multiple similarity metrics."""
    
    def __init__(self, ratingsPerItem, ratingsPerUser):
        self.ratingsPerItem = ratingsPerItem
        self.ratingsPerUser = ratingsPerUser
        
        self.book_popularity = {b: len(ratingsPerItem[b]) for b in ratingsPerItem}
        self.user_activity = {u: len(ratingsPerUser[u]) for u in ratingsPerUser}
        
        self.book_cooccurrence = self._build_cooccurrence_matrix()
        
        self.weights = {
            'jaccard': 0.5,
            'cosine': 0.3,
            'popularity': 2.0,
            'coverage': 0.5,
            'cooccurrence': 3.5,
            'adamic_adar': 3.0,
            'preferential': 1.0
        }
        
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        self.threshold = 0.15
    
    def _build_cooccurrence_matrix(self):
        """Build book co-occurrence matrix."""
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for u in self.ratingsPerUser:
            books = [b for b, r in self.ratingsPerUser[u]]
            for i, b1 in enumerate(books):
                for b2 in books[i+1:]:
                    cooccurrence[b1][b2] += 1
                    cooccurrence[b2][b1] += 1
        
        return cooccurrence
    
    def jaccard_similarity_score(self, u, b):
        """Enhanced Jaccard with top-K averaging."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        similarities = []
        for book_prime in userBooks:
            if book_prime == b or book_prime not in self.ratingsPerItem:
                continue
            
            usersBPrime = set(user for user, _ in self.ratingsPerItem[book_prime])
            intersection = len(usersB & usersBPrime)
            union = len(usersB | usersBPrime)
            
            if union > 0:
                similarities.append(intersection / union)
        
        if not similarities:
            return 0.0
        
        similarities.sort(reverse=True)
        top_k = min(10, len(similarities))
        return np.mean(similarities[:top_k])
    
    def cosine_similarity_score(self, u, b):
        """Cosine similarity."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        similarities = []
        for book_prime in userBooks:
            if book_prime == b or book_prime not in self.ratingsPerItem:
                continue
            
            usersBPrime = set(user for user, _ in self.ratingsPerItem[book_prime])
            intersection = len(usersB & usersBPrime)
            
            denom = np.sqrt(len(usersB) * len(usersBPrime))
            if denom > 0:
                similarities.append(intersection / denom)
        
        if not similarities:
            return 0.0
        
        similarities.sort(reverse=True)
        return np.mean(similarities[:10])
    
    def popularity_score(self, u, b):
        """Normalized popularity with cold start boost."""
        pop = self.book_popularity.get(b, 0)
        max_pop = max(self.book_popularity.values())
        base_score = pop / max_pop if max_pop > 0 else 0
        
        user_act = self.user_activity.get(u, 0)
        if user_act < 10 and pop > 50:
            base_score += 0.3
        
        return min(base_score, 1.0)
    
    def user_coverage_score(self, u, b):
        """Network overlap."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        user_network = set()
        for book_prime in userBooks:
            if book_prime in self.ratingsPerItem:
                user_network.update(user for user, _ in self.ratingsPerItem[book_prime])
        
        if len(usersB) == 0:
            return 0.0
        
        overlap = len(usersB & user_network)
        return overlap / len(usersB)
    
    def cooccurrence_score(self, u, b):
        """Enhanced co-occurrence using precomputed matrix."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        if b not in self.book_cooccurrence:
            return 0.0
        
        total_cooccur = 0
        for book_prime in userBooks:
            if book_prime in self.book_cooccurrence[b]:
                total_cooccur += self.book_cooccurrence[b][book_prime]
        
        user_act = len(userBooks)
        book_pop = self.book_popularity.get(b, 1)
        
        normalized = total_cooccur / (user_act * np.log1p(book_pop))
        return min(normalized, 1.0)
    
    def adamic_adar_score(self, u, b):
        """Adamic-Adar with better normalization."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        aa_scores = []
        for book_prime in userBooks:
            if book_prime == b or book_prime not in self.ratingsPerItem:
                continue
            
            usersBPrime = set(user for user, _ in self.ratingsPerItem[book_prime])
            common_users = usersB & usersBPrime
            
            aa_sum = 0
            for common_user in common_users:
                degree = len(self.ratingsPerUser.get(common_user, []))
                if degree > 1:
                    aa_sum += 1.0 / np.log(degree)
            
            if len(common_users) > 0:
                aa_scores.append(aa_sum)
        
        if not aa_scores:
            return 0.0
        
        aa_scores.sort(reverse=True)
        top_10 = aa_scores[:10]
        normalized = np.mean(top_10) / 10.0
        return min(normalized, 1.0)
    
    def preferential_attachment_score(self, u, b):
        """Preferential attachment."""
        user_act = self.user_activity.get(u, 0)
        book_pop = self.book_popularity.get(b, 0)
        
        max_act = max(self.user_activity.values()) if self.user_activity else 1
        max_pop = max(self.book_popularity.values()) if self.book_popularity else 1
        
        norm_act = user_act / max_act
        norm_pop = book_pop / max_pop
        
        return norm_act * norm_pop
    
    def second_order_similarity(self, u, b):
        """Second-order similarity for no-overlap cases."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        
        similar_users = set()
        for book_prime in userBooks:
            if book_prime in self.ratingsPerItem:
                similar_users.update(user for user, _ in self.ratingsPerItem[book_prime])
        
        overlap = len(similar_users & usersB)
        
        if len(similar_users) == 0:
            return 0.0
        
        return overlap / len(similar_users)
    
    def compute_all_scores(self, u, b):
        """Compute all scores."""
        return {
            'jaccard': self.jaccard_similarity_score(u, b),
            'cosine': self.cosine_similarity_score(u, b),
            'popularity': self.popularity_score(u, b),
            'coverage': self.user_coverage_score(u, b),
            'cooccurrence': self.cooccurrence_score(u, b),
            'adamic_adar': self.adamic_adar_score(u, b),
            'preferential': self.preferential_attachment_score(u, b),
            'second_order': self.second_order_similarity(u, b)
        }
    
    def predict_single(self, u, b):
        """Predict with cold start handling."""
        scores = self.compute_all_scores(u, b)
        
        if 'second_order' not in self.weights:
            self.weights['second_order'] = 0.8
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        final_score = sum(scores[k] * self.weights.get(k, 0) for k in scores.keys())
        
        user_act = self.user_activity.get(u, 0)
        book_pop = self.book_popularity.get(b, 0)
        
        if user_act < 5:
            if book_pop > 100:
                final_score += 0.15
            elif book_pop > 50:
                final_score += 0.10
        elif user_act < 10:
            if book_pop > 100:
                final_score += 0.08
        
        similarity_scores = [scores['jaccard'], scores['cosine'], scores['coverage'], scores['cooccurrence']]
        if max(similarity_scores) < 0.05 and book_pop > 80:
            final_score += 0.12
        
        return 1 if final_score > self.threshold else 0, final_score
    
    def predict_batch(self, user_book_pairs, show_progress=False):
        """Batch prediction."""
        predictions = []
        probabilities = []
        
        for i, (u, b) in enumerate(user_book_pairs):
            if show_progress and (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1}/{len(user_book_pairs)}")
            
            pred, prob = self.predict_single(u, b)
            predictions.append(pred)
            probabilities.append(prob)
        
        return np.array(predictions), np.array(probabilities)


def evaluateHybridModel(model, validPositive, validNegative):
    """Evaluate hybrid model."""
    valid_pairs = list(validPositive) + list(validNegative)
    valid_labels = np.array([1] * len(validPositive) + [0] * len(validNegative))
    
    predictions, probabilities = model.predict_batch(valid_pairs)
    
    tp = np.sum((predictions == 1) & (valid_labels == 1))
    fp = np.sum((predictions == 1) & (valid_labels == 0))
    tn = np.sum((predictions == 0) & (valid_labels == 0))
    fn = np.sum((predictions == 0) & (valid_labels == 1))
    
    accuracy = (tp + tn) / len(valid_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(valid_labels, probabilities)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': predictions,
        'probabilities': probabilities
    }


def runHybridPipeline(ratingsTrain, ratingsValid, ratingsPerItem, ratingsPerUser):
    """Run hybrid pipeline."""
    _ = set((u, b) for u, b, r in ratingsTrain)  # trainPositive (unused but kept for reference)
    validPositive = set((u, b) for u, b, r in ratingsValid)
    
    validNegative = generateBalancedNegativeSamples(
        ratingsValid, ratingsPerItem, ratingsPerUser,
        num_negatives=len(validPositive),
        popularity_distribution='balanced'
    )
    
    model = ImprovedHybridRecommender(ratingsPerItem, ratingsPerUser)
    results = evaluateHybridModel(model, validPositive, validNegative)
    
    return model, results


def optimizeThresholdForRecall(model, validPositive, validNegative, target_recall=0.7):
    """Find threshold that achieves target recall while maximizing F1."""
    valid_pairs = list(validPositive) + list(validNegative)
    valid_labels = np.array([1] * len(validPositive) + [0] * len(validNegative))
    
    _, probabilities = model.predict_batch(valid_pairs, show_progress=False)
    
    best_threshold = 0.5
    best_f1 = 0
    
    for thresh in np.arange(0.05, 0.4, 0.02):
        preds = (probabilities > thresh).astype(int)
        
        tp = np.sum((preds == 1) & (valid_labels == 1))
        fp = np.sum((preds == 1) & (valid_labels == 0))
        fn = np.sum((preds == 0) & (valid_labels == 1))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if recall >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    model.threshold = best_threshold
    return best_threshold


# =============================================================================
# TASK 3: LOGISTIC REGRESSION ON HYBRID FEATURES
# =============================================================================
# This task implements logistic regression trained on hybrid similarity features
# for improved prediction performance.

class HybridLogisticRecommender:
    """Logistic Regression trained on hybrid similarity features."""
    
    def __init__(self, ratingsPerItem, ratingsPerUser):
        self.ratingsPerItem = ratingsPerItem
        self.ratingsPerUser = ratingsPerUser
        
        self.book_popularity = {b: len(ratingsPerItem[b]) for b in ratingsPerItem}
        self.user_activity = {u: len(ratingsPerUser[u]) for u in ratingsPerUser}
        
        self.book_cooccurrence = self._build_cooccurrence_matrix()
        
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'jaccard', 'cosine', 'popularity', 'coverage',
            'cooccurrence', 'adamic_adar', 'preferential'
        ]
    
    def _build_cooccurrence_matrix(self):
        """Build book co-occurrence matrix."""
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for u in self.ratingsPerUser:
            books = [b for b, r in self.ratingsPerUser[u]]
            for i, b1 in enumerate(books):
                for b2 in books[i+1:]:
                    cooccurrence[b1][b2] += 1
                    cooccurrence[b2][b1] += 1
        
        return cooccurrence
    
    def jaccard_similarity_score(self, u, b):
        """Jaccard similarity with top-K averaging."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        similarities = []
        for book_prime in userBooks:
            if book_prime == b or book_prime not in self.ratingsPerItem:
                continue
            
            usersBPrime = set(user for user, _ in self.ratingsPerItem[book_prime])
            intersection = len(usersB & usersBPrime)
            union = len(usersB | usersBPrime)
            
            if union > 0:
                similarities.append(intersection / union)
        
        if not similarities:
            return 0.0
        
        similarities.sort(reverse=True)
        top_k = min(10, len(similarities))
        return np.mean(similarities[:top_k])
    
    def cosine_similarity_score(self, u, b):
        """Cosine similarity."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        similarities = []
        for book_prime in userBooks:
            if book_prime == b or book_prime not in self.ratingsPerItem:
                continue
            
            usersBPrime = set(user for user, _ in self.ratingsPerItem[book_prime])
            intersection = len(usersB & usersBPrime)
            
            denom = np.sqrt(len(usersB) * len(usersBPrime))
            if denom > 0:
                similarities.append(intersection / denom)
        
        if not similarities:
            return 0.0
        
        similarities.sort(reverse=True)
        return np.mean(similarities[:10])
    
    def popularity_score(self, u, b):  # noqa: ARG002 - u unused but kept for API consistency
        """Normalized popularity."""
        pop = self.book_popularity.get(b, 0)
        max_pop = max(self.book_popularity.values()) if self.book_popularity else 1
        return pop / max_pop
    
    def user_coverage_score(self, u, b):
        """Network overlap."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        user_network = set()
        for book_prime in userBooks:
            if book_prime in self.ratingsPerItem:
                user_network.update(user for user, _ in self.ratingsPerItem[book_prime])
        
        if len(usersB) == 0:
            return 0.0
        
        overlap = len(usersB & user_network)
        return overlap / len(usersB)
    
    def cooccurrence_score(self, u, b):
        """Co-occurrence score."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        if b not in self.book_cooccurrence:
            return 0.0
        
        total_cooccur = 0
        for book_prime in userBooks:
            if book_prime in self.book_cooccurrence[b]:
                total_cooccur += self.book_cooccurrence[b][book_prime]
        
        user_act = len(userBooks)
        book_pop = self.book_popularity.get(b, 1)
        
        normalized = total_cooccur / (user_act * np.log1p(book_pop))
        return min(normalized, 10.0)
    
    def adamic_adar_score(self, u, b):
        """Adamic-Adar index."""
        if u not in self.ratingsPerUser or b not in self.ratingsPerItem:
            return 0.0
        
        usersB = set(user for user, _ in self.ratingsPerItem[b])
        userBooks = [book for book, _ in self.ratingsPerUser[u]]
        
        aa_scores = []
        for book_prime in userBooks:
            if book_prime == b or book_prime not in self.ratingsPerItem:
                continue
            
            usersBPrime = set(user for user, _ in self.ratingsPerItem[book_prime])
            common_users = usersB & usersBPrime
            
            aa_sum = 0
            for common_user in common_users:
                degree = len(self.ratingsPerUser.get(common_user, []))
                if degree > 1:
                    aa_sum += 1.0 / np.log(degree)
            
            if len(common_users) > 0:
                aa_scores.append(aa_sum)
        
        if not aa_scores:
            return 0.0
        
        aa_scores.sort(reverse=True)
        return np.mean(aa_scores[:10])
    
    def preferential_attachment_score(self, u, b):
        """Preferential attachment."""
        user_act = self.user_activity.get(u, 0)
        book_pop = self.book_popularity.get(b, 0)
        
        max_act = max(self.user_activity.values()) if self.user_activity else 1
        max_pop = max(self.book_popularity.values()) if self.book_popularity else 1
        
        norm_act = user_act / max_act
        norm_pop = book_pop / max_pop
        
        return norm_act * norm_pop
    
    def extract_features(self, u, b):
        """Extract feature vector for (user, book) pair."""
        features = [
            self.jaccard_similarity_score(u, b),
            self.cosine_similarity_score(u, b),
            self.popularity_score(u, b),
            self.user_coverage_score(u, b),
            self.cooccurrence_score(u, b),
            self.adamic_adar_score(u, b),
            self.preferential_attachment_score(u, b)
        ]
        return np.array(features)
    
    def prepare_training_data(self, trainPositive, trainNegative, subsample=None):
        """Prepare feature matrix and labels for training."""
        train_pairs = list(trainPositive) + list(trainNegative)
        train_labels = [1] * len(trainPositive) + [0] * len(trainNegative)
        
        if subsample and len(train_pairs) > subsample:
            indices = random.sample(range(len(train_pairs)), subsample)
            train_pairs = [train_pairs[i] for i in indices]
            train_labels = [train_labels[i] for i in indices]
        
        X = []
        y = []
        
        for i, (u, b) in enumerate(train_pairs):
            features = self.extract_features(u, b)
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue
            
            X.append(features)
            y.append(train_labels[i])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, C=1.0, class_weight='balanced'):
        """Train logistic regression model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        self.model.fit(X_train_scaled, y_train)
    
    def predict_proba(self, u, b):
        """Predict probability that user will read book."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        features = self.extract_features(u, b).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        proba = self.model.predict_proba(features_scaled)[0, 1]
        return proba
    
    def predict(self, u, b, threshold=0.5):
        """Predict binary outcome (0 or 1)."""
        proba = self.predict_proba(u, b)
        return 1 if proba >= threshold else 0
    
    def predict_batch(self, user_book_pairs, threshold=0.5, show_progress=False):  # noqa: ARG002
        """Predict for multiple pairs."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = []
        for u, b in user_book_pairs:
            features = self.extract_features(u, b)
            X.append(features)
        
        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities


def evaluateLogisticModel(model, validPositive, validNegative, threshold=0.5):
    """Comprehensive evaluation of logistic regression model."""
    valid_pairs = list(validPositive) + list(validNegative)
    valid_labels = np.array([1] * len(validPositive) + [0] * len(validNegative))
    
    predictions, probabilities = model.predict_batch(valid_pairs, threshold=threshold)
    
    tp = np.sum((predictions == 1) & (valid_labels == 1))
    fp = np.sum((predictions == 1) & (valid_labels == 0))
    tn = np.sum((predictions == 0) & (valid_labels == 0))
    fn = np.sum((predictions == 0) & (valid_labels == 1))
    
    accuracy = (tp + tn) / len(valid_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(valid_labels, probabilities)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': valid_labels
    }


def optimizeLogisticThreshold(model, validPositive, validNegative):
    """Find optimal threshold for logistic regression."""
    valid_pairs = list(validPositive) + list(validNegative)
    valid_labels = np.array([1] * len(validPositive) + [0] * len(validNegative))
    
    _, probabilities = model.predict_batch(valid_pairs, show_progress=False)
    
    best_f1 = 0
    best_threshold = 0.5
    results = []
    
    for thresh in np.arange(0.000001, 0.0001, 0.000005):
        preds = (probabilities >= thresh).astype(int)
        
        tp = np.sum((preds == 1) & (valid_labels == 1))
        fp = np.sum((preds == 1) & (valid_labels == 0))
        fn = np.sum((preds == 0) & (valid_labels == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        results.append({'threshold': thresh, 'f1': f1})
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, results


def plotROCCurve(results, save_path='roc_curve.png'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {results['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - Logistic Regression on Hybrid Features', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def runLogisticRegressionPipeline(ratingsTrain, ratingsValid, ratingsPerItem, ratingsPerUser,
                                   train_subsample=50000, C=1.0):
    """Complete pipeline: feature extraction -> training -> evaluation."""
    trainPositive = set((u, b) for u, b, r in ratingsTrain)
    validPositive = set((u, b) for u, b, r in ratingsValid)
    
    trainNegative = generateBalancedNegativeSamples(
        ratingsTrain, ratingsPerItem, ratingsPerUser,
        num_negatives=int(len(trainPositive) * 0.5),
        popularity_distribution='unpopular'
    )
    
    validNegative = generateBalancedNegativeSamples(
        ratingsValid, ratingsPerItem, ratingsPerUser,
        num_negatives=len(validPositive),
        popularity_distribution='balanced'
    )
    
    model = HybridLogisticRecommender(ratingsPerItem, ratingsPerUser)
    
    X_train, y_train = model.prepare_training_data(
        trainPositive, trainNegative,
        subsample=train_subsample
    )
    
    model.train(X_train, y_train, C=C, class_weight='balanced')
    
    # Initial evaluation
    _ = evaluateLogisticModel(model, validPositive, validNegative, threshold=0.00001)
    
    best_threshold, _ = optimizeLogisticThreshold(model, validPositive, validNegative)
    
    final_results = evaluateLogisticModel(model, validPositive, validNegative, threshold=best_threshold)
    
    try:
        plotROCCurve(final_results)
    except Exception:
        pass
    
    return model, final_results, best_threshold


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("TASK 1: READ PREDICTION USING JACCARD SIMILARITY")
    print("=" * 80)
    
    # Generate validation data
    readValid, notRead = generateValidation(allRatings, ratingsValid)
    
    # Evaluate with default thresholds
    metrics = evaluateJaccard(readValid, notRead, ratingsPerItem, ratingsPerUser)
    print("\nDefault Thresholds Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    # Optimize thresholds
    best_results = optimizeJaccardThresholds(readValid, notRead, ratingsPerItem, ratingsPerUser)
    best_combo = best_results['top_by_f1'][0]
    print("\nOptimized Thresholds:")
    print(f"  Best Jaccard Threshold: {best_combo['jaccard_threshold']}")
    print(f"  Best Popularity Threshold: {best_combo['popularity_threshold']}")
    print(f"  Best F1 Score: {best_combo['f1_score']:.4f}")
    
    print("\n" + "=" * 80)
    print("TASK 2: HYBRID RECOMMENDER SYSTEM")
    print("=" * 80)
    
    # Run hybrid pipeline
    hybrid_model, hybrid_results = runHybridPipeline(
        ratingsTrain, ratingsValid, ratingsPerItem, ratingsPerUser
    )
    
    print("\nHybrid Model Results:")
    print(f"  Accuracy:  {hybrid_results['accuracy']:.4f}")
    print(f"  Precision: {hybrid_results['precision']:.4f}")
    print(f"  Recall:    {hybrid_results['recall']:.4f}")
    print(f"  F1 Score:  {hybrid_results['f1']:.4f}")
    print(f"  AUC:       {hybrid_results['auc']:.4f}")
    
    # Optimize threshold
    validPositive = set((u, b) for u, b, r in ratingsValid)
    validNegative = generateBalancedNegativeSamples(
        ratingsValid, ratingsPerItem, ratingsPerUser,
        num_negatives=len(validPositive),
        popularity_distribution='balanced'
    )
    best_threshold = optimizeThresholdForRecall(hybrid_model, validPositive, validNegative)
    
    # Re-evaluate with optimized threshold
    final_hybrid_results = evaluateHybridModel(hybrid_model, validPositive, validNegative)
    print("\nOptimized Hybrid Model Results:")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Accuracy:  {final_hybrid_results['accuracy']:.4f}")
    print(f"  F1 Score:  {final_hybrid_results['f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("TASK 3: LOGISTIC REGRESSION ON HYBRID FEATURES")
    print("=" * 80)
    
    # Run logistic regression pipeline
    lr_model, lr_results, lr_threshold = runLogisticRegressionPipeline(
        ratingsTrain,
        ratingsValid,
        ratingsPerItem,
        ratingsPerUser,
        train_subsample=50000,
        C=1.0
    )
    
    print("\nLogistic Regression Results:")
    print(f"  Best Threshold: {lr_threshold:.6f}")
    print(f"  Accuracy:  {lr_results['accuracy']:.4f}")
    print(f"  Precision: {lr_results['precision']:.4f}")
    print(f"  Recall:    {lr_results['recall']:.4f}")
    print(f"  F1 Score:  {lr_results['f1']:.4f}")
    print(f"  AUC:       {lr_results['auc']:.4f}")
