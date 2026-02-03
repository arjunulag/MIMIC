"""
Three Agent Word Guessing Game (Improved Version)

Agent 1 (Knower): Knows the secret word
Agent 2 (Imposter): Doesn't know the word, tries to figure it out
Agent 3 (Guesser): Knows the word, tries to identify who actually knows it

Rules:
- Agent 1 wins by being correctly identified as the knower
- Agent 2 wins by either being misidentified as the knower OR figuring out the word
- Agent 3 wins by correctly identifying Agent 1 as the knower
"""

import random
from abc import ABC, abstractmethod
from glove_loader import GloVeLoader
import numpy as np


class Agent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.clue_history = []
    
    @abstractmethod
    def give_clue(self, context: dict) -> str:
        """Generate a clue based on available information."""
        pass
    
    def receive_info(self, info: str):
        """Receive information from the game."""
        self.clue_history.append(info)


class KnowerAgent(Agent):
    """
    Agent 1: Knows the secret word and its similar words.
    
    IMPROVED STRATEGY:
    - Give HIGH similarity words first to prove knowledge
    - Avoid words that might be in the imposter's hint list
    - Build a coherent semantic cluster around the secret word
    """
    
    def __init__(self, name: str = "Agent 1 (Knower)", glove_loader: GloVeLoader = None):
        super().__init__(name)
        self.secret_word = None
        self.similar_words = []
        self.used_clues = set()
        self.glove = glove_loader
        self.all_similar = []  # Full list of similar words
    
    def set_knowledge(self, word: str, similar_words: list):
        """Set the secret word and similar words."""
        self.secret_word = word
        self.similar_words = similar_words
        self.used_clues = set()
        
        # Get a broader set of very similar words (high similarity)
        if self.glove:
            self.all_similar = self.glove.find_similar_words(word, n=30)
    
    def give_clue(self, context: dict) -> str:
        """
        Give a clue that strongly hints at the secret word.
        Strategy: Use the MOST similar words to prove knowledge.
        """
        # First, try to use words with HIGH similarity (not in the shared hint list)
        hint_words = set(w for w, s in self.similar_words)
        
        # Get best clues - high similarity words not already used or in shared hints
        best_clues = [(w, s) for w, s in self.all_similar 
                      if w not in self.used_clues and w not in hint_words]
        
        if best_clues:
            # Pick the most similar word available
            clue_word, similarity = best_clues[0]
            self.used_clues.add(clue_word)
            return f"My clue: '{clue_word}'"
        
        # Fallback to similar words list
        available = [w for w, s in self.similar_words if w not in self.used_clues]
        if available:
            # Pick highest similarity one
            clue_word = available[0]
            self.used_clues.add(clue_word)
            return f"My clue: '{clue_word}'"
        
        # Last resort: describe the word conceptually
        return f"My clue: This is something you might find in nature."
    
    def get_remaining_clues(self) -> list:
        """Get list of remaining clue words."""
        return [w for w, s in self.similar_words if w not in self.used_clues]


class ImposterAgent(Agent):
    """
    Agent 2: Doesn't know the word, tries to figure it out and blend in.
    
    IMPROVED STRATEGY:
    - Track ALL clues from Agent 1 carefully
    - Find words that are similar to MULTIPLE clue words (intersection)
    - Weight by similarity scores
    - Use hint words strategically
    """
    
    def __init__(self, name: str = "Agent 2 (Imposter)", glove_loader: GloVeLoader = None):
        super().__init__(name)
        self.similar_words_given = []  # Words given as hints
        self.guessed_word = None
        self.glove = glove_loader
        self.confidence = 0.0
        self.candidate_words = {}
        self.clues_given = []
        self.observed_clues = []  # Track clues from other agents
    
    def set_hints(self, similar_words: list):
        """
        Receive the similar words as hints (but not the actual word).
        """
        self.similar_words_given = similar_words
        self.candidate_words = {}
        self.guessed_word = None
        self.confidence = 0.0
        self.observed_clues = []
        self.clues_given = []
    
    def extract_clue_word(self, clue_text: str) -> str:
        """Extract the actual clue word from clue text."""
        # Format is usually "Agent 1 said: My clue: 'word'"
        if "'" in clue_text:
            parts = clue_text.split("'")
            if len(parts) >= 2:
                return parts[1].lower()
        return None
    
    def analyze_clues(self):
        """
        Try to deduce the secret word from observed clues.
        IMPROVED: Find words that are neighbors of ALL observed clue words.
        """
        if self.glove is None:
            return
        
        # Extract clue words from history
        clue_words = []
        for clue in self.clue_history:
            word = self.extract_clue_word(clue)
            if word and self.glove.word_in_vocab(word):
                clue_words.append(word)
        
        # Also add the hint words we were given (they're similar to the target)
        hint_words = [w for w, s in self.similar_words_given]
        
        if not clue_words:
            # No clues yet, use hint words to guess
            if hint_words and self.glove:
                # Find common neighbors of hint words
                self._find_common_neighbors(hint_words[:5])
            return
        
        # Find words that are similar to multiple clue words
        # This is the key insight: the secret word should be similar to ALL clues
        self._find_common_neighbors(clue_words + hint_words[:3])
    
    def _find_common_neighbors(self, words: list):
        """Find words that are semantically close to ALL given words."""
        if not words or not self.glove:
            return
        
        # For each word, get its neighbors
        neighbor_scores = {}
        
        for word in words:
            try:
                neighbors = self.glove.find_similar_words(word, n=100)
                for neighbor, score in neighbors:
                    if neighbor not in words:
                        if neighbor not in neighbor_scores:
                            neighbor_scores[neighbor] = {'total': 0, 'count': 0, 'min': 1.0}
                        neighbor_scores[neighbor]['total'] += score
                        neighbor_scores[neighbor]['count'] += 1
                        neighbor_scores[neighbor]['min'] = min(neighbor_scores[neighbor]['min'], score)
            except KeyError:
                continue
        
        # Score by: (average similarity) * (count / total words) * min_similarity
        # This favors words that are consistently similar to ALL clue words
        final_scores = {}
        num_words = len(words)
        for word, data in neighbor_scores.items():
            if data['count'] >= max(2, num_words // 2):  # Must be neighbor of at least half
                avg_sim = data['total'] / data['count']
                coverage = data['count'] / num_words
                final_scores[word] = avg_sim * coverage * (1 + data['min'])
        
        if final_scores:
            sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            self.candidate_words = dict(sorted_candidates[:20])
            
            if sorted_candidates:
                self.guessed_word = sorted_candidates[0][0]
                # Confidence based on how much better top guess is than second
                if len(sorted_candidates) > 1:
                    self.confidence = sorted_candidates[0][1] / (sorted_candidates[0][1] + sorted_candidates[1][1])
                else:
                    self.confidence = 0.8
    
    def give_clue(self, context: dict) -> str:
        """
        Give a clue while trying to blend in.
        IMPROVED: Echo style of Agent 1's clues if possible.
        """
        self.analyze_clues()
        
        # If we have a good guess, use words similar to our guess
        if self.guessed_word and self.confidence > 0.5 and self.glove:
            try:
                similar_to_guess = self.glove.find_similar_words(self.guessed_word, n=10)
                for word, score in similar_to_guess:
                    if word not in self.clues_given and word != self.guessed_word:
                        self.clues_given.append(word)
                        return f"My clue: '{word}'"
            except KeyError:
                pass
        
        # Use one of the hint words we were given
        available_hints = [w for w, s in self.similar_words_given 
                         if w not in self.clues_given]
        
        if available_hints:
            # Pick hint word with highest similarity (safest bet)
            hint = available_hints[0]
            self.clues_given.append(hint)
            return f"My clue: '{hint}'"
        
        # Give a vague but plausible clue
        vague_clues = [
            "It's something everyone knows about.",
            "Think about the bigger picture.",
            "Consider what connects these ideas.",
            "It's more common than you might think."
        ]
        return random.choice(vague_clues)
    
    def make_guess(self) -> tuple:
        """
        Make a guess about the secret word.
        Returns: (guessed_word, confidence)
        """
        self.analyze_clues()
        return self.guessed_word, self.confidence
    
    def get_top_candidates(self, n: int = 5) -> list:
        """Get top N candidate words."""
        return sorted(self.candidate_words.items(), key=lambda x: x[1], reverse=True)[:n]


class GuesserAgent(Agent):
    """
    Agent 3: Knows the word, tries to identify who actually knows it.
    
    IMPROVED STRATEGY:
    - Measure semantic COHERENCE of each agent's clues
    - Check if clues form a tight cluster around the secret word
    - Penalize vague or off-topic clues
    - Track clue quality over time
    """
    
    def __init__(self, name: str = "Agent 3 (Guesser)", glove_loader: GloVeLoader = None):
        super().__init__(name)
        self.secret_word = None
        self.similar_words = []
        self.glove = glove_loader
        self.agent_scores = {"Agent 1": 0.0, "Agent 2": 0.0}
        self.agent_clues = {"Agent 1": [], "Agent 2": []}
        self.clue_similarities = {"Agent 1": [], "Agent 2": []}
    
    def set_knowledge(self, word: str, similar_words: list):
        """Set the secret word and similar words."""
        self.secret_word = word
        self.similar_words = [w for w, s in similar_words]
        self.similar_words_with_scores = similar_words
        self.agent_scores = {"Agent 1": 0.0, "Agent 2": 0.0}
        self.agent_clues = {"Agent 1": [], "Agent 2": []}
        self.clue_similarities = {"Agent 1": [], "Agent 2": []}
    
    def extract_clue_word(self, clue_text: str) -> str:
        """Extract the actual clue word from clue text."""
        if "'" in clue_text:
            parts = clue_text.split("'")
            if len(parts) >= 2:
                return parts[1].lower()
        return None
    
    def record_clue(self, agent_name: str, clue: str):
        """Record a clue from an agent for analysis."""
        clue_word = self.extract_clue_word(clue)
        
        if not clue_word:
            # Vague clue - slight penalty
            self.agent_scores[agent_name] -= 0.5
            return
        
        self.agent_clues[agent_name].append(clue_word)
        
        if not self.glove:
            return
        
        # Calculate similarity to secret word
        try:
            secret_vec = self.glove.get_vector(self.secret_word)
            clue_vec = self.glove.get_vector(clue_word)
            similarity = self.glove.cosine_similarity(clue_vec, secret_vec)
            
            self.clue_similarities[agent_name].append(similarity)
            
            # Score based on similarity
            # High similarity = strong evidence of knowledge
            if similarity > 0.6:
                self.agent_scores[agent_name] += 3.0
            elif similarity > 0.4:
                self.agent_scores[agent_name] += 1.5
            elif similarity > 0.2:
                self.agent_scores[agent_name] += 0.5
            else:
                # Low similarity - suspicious!
                self.agent_scores[agent_name] -= 1.0
                
        except KeyError:
            # Word not in vocabulary - slightly suspicious
            self.agent_scores[agent_name] -= 0.3
    
    def calculate_coherence(self, agent_name: str) -> float:
        """
        Calculate how coherent an agent's clues are with each other.
        A knower's clues should form a tight semantic cluster.
        """
        clues = self.agent_clues[agent_name]
        if len(clues) < 2 or not self.glove:
            return 0.0
        
        # Calculate pairwise similarities between clues
        similarities = []
        for i, clue1 in enumerate(clues):
            for clue2 in clues[i+1:]:
                try:
                    vec1 = self.glove.get_vector(clue1)
                    vec2 = self.glove.get_vector(clue2)
                    sim = self.glove.cosine_similarity(vec1, vec2)
                    similarities.append(sim)
                except KeyError:
                    continue
        
        if similarities:
            return np.mean(similarities)
        return 0.0
    
    def give_clue(self, context: dict) -> str:
        """Ask a probing question to test the agents."""
        questions = [
            "Can you give me another word that relates to this?",
            "What category does this word belong to?",
            "Is this word more abstract or concrete?",
            "Give me an example of when you'd use this word.",
            "What's the opposite of this concept?",
        ]
        return random.choice(questions)
    
    def make_guess(self) -> str:
        """
        Guess which agent actually knows the word.
        IMPROVED: Use coherence + similarity scores.
        """
        # Add coherence bonus
        for agent in ["Agent 1", "Agent 2"]:
            coherence = self.calculate_coherence(agent)
            self.agent_scores[agent] += coherence * 2
            
            # Bonus for consistent high similarity
            if self.clue_similarities[agent]:
                avg_sim = np.mean(self.clue_similarities[agent])
                std_sim = np.std(self.clue_similarities[agent]) if len(self.clue_similarities[agent]) > 1 else 0
                # High average + low variance = consistent knowledge
                consistency_bonus = avg_sim * (1 - std_sim)
                self.agent_scores[agent] += consistency_bonus * 2
        
        if self.agent_scores["Agent 1"] > self.agent_scores["Agent 2"]:
            return "Agent 1"
        elif self.agent_scores["Agent 2"] > self.agent_scores["Agent 1"]:
            return "Agent 2"
        else:
            return random.choice(["Agent 1", "Agent 2"])
    
    def get_confidence(self) -> dict:
        """Get confidence scores for each agent."""
        total = abs(self.agent_scores["Agent 1"]) + abs(self.agent_scores["Agent 2"])
        if total == 0:
            return {"Agent 1": 0.5, "Agent 2": 0.5}
        
        # Normalize to positive values first
        min_score = min(self.agent_scores["Agent 1"], self.agent_scores["Agent 2"])
        adj_1 = self.agent_scores["Agent 1"] - min_score + 0.1
        adj_2 = self.agent_scores["Agent 2"] - min_score + 0.1
        total_adj = adj_1 + adj_2
        
        return {
            "Agent 1": adj_1 / total_adj,
            "Agent 2": adj_2 / total_adj
        }
    
    def get_analysis(self) -> dict:
        """Get detailed analysis of both agents."""
        return {
            "Agent 1": {
                "clues": self.agent_clues["Agent 1"],
                "similarities": self.clue_similarities["Agent 1"],
                "avg_similarity": np.mean(self.clue_similarities["Agent 1"]) if self.clue_similarities["Agent 1"] else 0,
                "coherence": self.calculate_coherence("Agent 1"),
                "score": self.agent_scores["Agent 1"]
            },
            "Agent 2": {
                "clues": self.agent_clues["Agent 2"],
                "similarities": self.clue_similarities["Agent 2"],
                "avg_similarity": np.mean(self.clue_similarities["Agent 2"]) if self.clue_similarities["Agent 2"] else 0,
                "coherence": self.calculate_coherence("Agent 2"),
                "score": self.agent_scores["Agent 2"]
            }
        }


class WordGame:
    """Main game controller."""
    
    def __init__(self, glove_loader: GloVeLoader):
        self.glove = glove_loader
        self.knower = KnowerAgent(glove_loader=glove_loader)
        self.imposter = ImposterAgent(glove_loader=glove_loader)
        self.guesser = GuesserAgent(glove_loader=glove_loader)
        
        self.secret_word = None
        self.similar_words = []
        self.rounds_played = 0
    
    def setup_game(self, word: str = None, num_similar: int = 10, 
                   target_similarity: float = 0.5, tolerance: float = 0.15):
        """
        Set up a new game with a secret word.
        
        Args:
            word: The secret word (random if None)
            num_similar: Number of similar words to provide
            target_similarity: How similar the hint words should be (lower = harder for imposter)
            tolerance: Tolerance for similarity range
        """
        # Select a word if not provided
        if word is None:
            common_words = ['king', 'city', 'happy', 'water', 'computer', 
                          'music', 'friend', 'science', 'money', 'time',
                          'banana', 'mountain', 'ocean', 'teacher', 'hospital']
            word = random.choice(common_words)
        
        self.secret_word = word.lower()
        
        # Get similar words at target distance (these are hints for imposter)
        self.similar_words = self.glove.find_words_at_distance(
            self.secret_word,
            n=num_similar,
            target_similarity=target_similarity,
            tolerance=tolerance
        )
        
        # If not enough words at target distance, get most similar
        if len(self.similar_words) < num_similar:
            self.similar_words = self.glove.find_similar_words(
                self.secret_word, n=num_similar
            )
        
        # Setup agents
        self.knower.set_knowledge(self.secret_word, self.similar_words)
        self.imposter.set_hints(self.similar_words)  # Gets hints but not the word
        self.guesser.set_knowledge(self.secret_word, self.similar_words)
        
        self.rounds_played = 0
        
        print(f"\n{'='*60}")
        print("GAME SETUP COMPLETE")
        print(f"{'='*60}")
        print(f"Secret Word: {self.secret_word}")
        print(f"\nHint words given to imposter (similar at distance ~{target_similarity}):")
        for word, score in self.similar_words:
            print(f"  - {word:20s} (similarity: {score:.4f})")
        print(f"{'='*60}\n")
    
    def play_round(self):
        """Play one round of the game."""
        self.rounds_played += 1
        print(f"\n--- ROUND {self.rounds_played} ---\n")
        
        # Knower gives clue first
        knower_clue = self.knower.give_clue({})
        print(f"Agent 1 (Knower): {knower_clue}")
        self.guesser.record_clue("Agent 1", knower_clue)
        self.imposter.receive_info(f"Agent 1 said: {knower_clue}")
        
        # Imposter gives clue
        imposter_clue = self.imposter.give_clue({})
        print(f"Agent 2 (Imposter): {imposter_clue}")
        self.guesser.record_clue("Agent 2", imposter_clue)
        
        # Guesser asks a question
        question = self.guesser.give_clue({})
        print(f"\nAgent 3 (Guesser): {question}")
        
        # Check if imposter has figured out the word
        imposter_guess, confidence = self.imposter.make_guess()
        if imposter_guess:
            print(f"\n[Imposter internally guesses: '{imposter_guess}' "
                  f"(confidence: {confidence:.2f})]")
            top_candidates = self.imposter.get_top_candidates(3)
            if top_candidates:
                print(f"[Top candidates: {', '.join([f'{w}({s:.2f})' for w,s in top_candidates])}]")
    
    def evaluate_game(self) -> dict:
        """
        Evaluate the game and determine winners.
        """
        print(f"\n{'='*60}")
        print("GAME EVALUATION")
        print(f"{'='*60}")
        
        # Get detailed analysis
        analysis = self.guesser.get_analysis()
        
        print(f"\nAgent Analysis:")
        for agent in ["Agent 1", "Agent 2"]:
            data = analysis[agent]
            print(f"\n  {agent}:")
            print(f"    Clues given: {data['clues']}")
            print(f"    Avg similarity to secret: {data['avg_similarity']:.4f}")
            print(f"    Clue coherence: {data['coherence']:.4f}")
            print(f"    Total score: {data['score']:.2f}")
        
        # Guesser makes their choice
        guesser_choice = self.guesser.make_guess()
        confidence = self.guesser.get_confidence()
        
        print(f"\nAgent 3's Final Assessment:")
        print(f"  Agent 1 confidence: {confidence['Agent 1']:.2%}")
        print(f"  Agent 2 confidence: {confidence['Agent 2']:.2%}")
        print(f"\nAgent 3 guesses: {guesser_choice} knows the word!")
        
        # Check imposter's final guess
        imposter_guess, imp_conf = self.imposter.make_guess()
        imposter_correct = imposter_guess and imposter_guess.lower() == self.secret_word
        
        print(f"\nImposter's final guess: '{imposter_guess}' (confidence: {imp_conf:.2f})")
        print(f"Imposter guessed correctly: {imposter_correct}")
        
        # Determine winners
        results = {
            "secret_word": self.secret_word,
            "guesser_choice": guesser_choice,
            "guesser_correct": guesser_choice == "Agent 1",
            "imposter_guess": imposter_guess,
            "imposter_correct": imposter_correct,
            "knower_wins": guesser_choice == "Agent 1",
            "imposter_wins": guesser_choice == "Agent 2" or imposter_correct,
            "guesser_wins": guesser_choice == "Agent 1"
        }
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"The secret word was: '{self.secret_word}'")
        print(f"\nWinners:")
        if results["knower_wins"]:
            print("  ✓ Agent 1 (Knower) WINS - Successfully identified!")
        if results["imposter_wins"]:
            if results["imposter_correct"]:
                print("  ✓ Agent 2 (Imposter) WINS - Figured out the word!")
            else:
                print("  ✓ Agent 2 (Imposter) WINS - Fooled the guesser!")
        if results["guesser_wins"]:
            print("  ✓ Agent 3 (Guesser) WINS - Correctly identified the knower!")
        
        if not results["knower_wins"] and not results["imposter_wins"]:
            print("  No clear winner this round.")
        
        print(f"{'='*60}\n")
        
        return results


def main():
    """Main entry point for the game."""
    import os
    
    print("="*60)
    print("THREE AGENT WORD GUESSING GAME (IMPROVED)")
    print("="*60)
    
    # Initialize GloVe loader
    loader = GloVeLoader()
    cache_path = 'glove_cache.pkl'
    
    if os.path.exists(cache_path):
        loader.load_cache(cache_path)
    else:
        # Look for GloVe files
        glove_files = ['dolma_300_2024_1.2M.100_combined.txt',
                      'glove.6B.100d.txt', 'glove.6B.50d.txt', 
                      'glove.6B.200d.txt', 'glove.6B.300d.txt']
        glove_path = None
        for gf in glove_files:
            if os.path.exists(gf):
                glove_path = gf
                break
        
        if glove_path is None:
            print("\nERROR: GloVe vectors not found!")
            print("Please download from: https://nlp.stanford.edu/projects/glove/")
            print("Place the .txt file in this directory.")
            return
        
        loader.load_glove(glove_path)
        loader.save_cache(cache_path)
    
    # Create and run game
    game = WordGame(loader)
    
    # Setup with a specific word (or None for random)
    test_word = input("\nEnter a secret word (or press Enter for random): ").strip()
    if not test_word:
        test_word = None
    elif not loader.word_in_vocab(test_word):
        print(f"Word '{test_word}' not in vocabulary. Using random word.")
        test_word = None
    
    game.setup_game(word=test_word, target_similarity=0.45, tolerance=0.1)
    
    # Play multiple rounds
    num_rounds = 3
    print(f"\nPlaying {num_rounds} rounds...\n")
    
    for _ in range(num_rounds):
        game.play_round()
    
    # Final evaluation
    results = game.evaluate_game()
    
    return results


if __name__ == "__main__":
    main()
