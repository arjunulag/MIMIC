# MIMIC

**Multi-agent Impostor Model for Intelligent Coordination**

A reinforcement learning framework for social deduction word games, featuring strategic player and impostor agents that learn to deceive, detect, and coordinate.

---

## Overview

MIMIC explores adversarial multi-agent learning in the context of word-based social deduction games. One agent knows the secret word (the impostor) while others must identify them—but the impostor is trying to blend in by giving convincing clues without knowing the actual word.

## Game Rules

- A secret word is chosen (e.g., "ocean")
- All **players** know the word; the **impostor** does not
- Each round, agents give one-word clues related to the secret
- Players try to identify the impostor through suspicious clues
- The impostor tries to blend in using context clues from others

---

## Agent Architectures

### Model 1: Player Agent

| Capability | Description |
|------------|-------------|
| **Clue Generation** | Pick words that prove knowledge without revealing the secret to the impostor |
| **Suspicion Detection** | Analyze clues from others to detect vague or off-target responses |
| **Voting Strategy** | Vote for the most suspicious agent each round |
| **Coordination** | Communicate with other players to build consensus |

### Model 2: Impostor Agent

| Capability | Description |
|------------|-------------|
| **Clue Bluffing** | Generate plausible clues using semantic similarity from observed words |
| **Social Camouflage** | Behave normally in chat to avoid suspicion |
| **Strategic Voting** | Vote to appear normal and eliminate real players |
| **Word Deduction** | Attempt to figure out the secret word from player clues |

---

## Project Structure

```
├── glove_loader.py      # Word vector loading and similarity search
├── agent_game.py        # Three-agent game implementation
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Word Vectors

Place your word vectors file in the project directory:
- `dolma_300_2024_1.2M.100_combined.txt` (primary)
- Or any GloVe-format vectors (e.g., `glove.6B.100d.txt`)

### 3. Run the Demo

```bash
# Test word similarity
python glove_loader.py

# Play the agent game
python agent_game.py
```

---

## How It Works

### Word Embeddings

The framework uses pre-trained word vectors to:
- Find semantically similar words for clue generation
- Measure how "related" a clue is to the secret word
- Help the impostor deduce the word from observed clues

### Game Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. Secret word selected                                │
│  2. Players receive word; Impostor receives hints only  │
│  3. Each agent gives a clue (multiple rounds)           │
│  4. Guesser analyzes clues for authenticity             │
│  5. Vote to identify the impostor                       │
│  6. Impostor wins if undetected OR guesses the word     │
└─────────────────────────────────────────────────────────┘
```

---

## Current Agents

| Agent | Role | Knowledge |
|-------|------|-----------|
| **Knower** | Proves they know the word | Secret word + similar words |
| **Impostor** | Blends in, tries to deduce | Similar words only (no secret) |
| **Guesser** | Identifies who knows | Secret word + analyzes clues |

---

## Roadmap

- [ ] Reinforcement learning training loop
- [ ] Multi-player support (3+ players + 1 impostor)
- [ ] Chat-based coordination between players
- [ ] Suspicion scoring model
- [ ] Tournament evaluation system
- [ ] Web interface for human vs AI play

---

## Research Goals

1. **Adversarial Language Games**: How do agents learn to deceive and detect deception through word choice?
2. **Emergent Communication**: Do player agents develop coordination strategies?
3. **Theory of Mind**: Can agents model what others know or don't know?

---

## License

MIT
