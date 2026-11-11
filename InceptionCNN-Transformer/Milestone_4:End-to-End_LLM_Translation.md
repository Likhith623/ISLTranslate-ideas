# Milestone 4: End-to-End LLM Translation (Stage 2)

Now that the vision model successfully recognizes isolated ASL signs in real-time, the next phase is tackling ASL syntax. ASL uses a Topic-Comment structure that differs heavily from English. 

We will build the NLP Translation layer to convert a sequence of predicted glosses into a grammatically correct English sentence.

---

## Goal

Upgrade our Python prototype to allow you to sign a complete sentence word-by-word. Once you signal the sentence is finished, the app will pass the accumulated ASL glosses to an LLM to generate natural, conversational English.

## User Review Required

> [!IMPORTANT]
> **API Key Setup**
> We will use the **Mistral AI API** (specifically the `mistral-small-2506` model) to handle the translation. 
> 
> You will need to get a Mistral API key from their developer console. Once you have it, you can set it in your Mac's terminal before running the script:
> `export MISTRAL_API_KEY="your_api_key_here"`



## Proposed Changes

### 1. `test/live_test.py` Updates
We will modify the state machine to include a **Sentence Buffer**:
- When you press `Spacebar` and sign a word, the top prediction is appended to a visible list on screen (e.g., `Sentence: [ "STORE", "I", "GO" ]`).
- When you press `Enter / Return`, the app enters a new `TRANSLATING` state.
- It sends the list of glosses to the Mistral API using the `mistralai` Python package.
- The translated English sentence (e.g., `"I am going to the store."`) is displayed prominently on the screen.

### 2. Prompt Engineering
We will write a strict system prompt for the LLM to ensure it acts purely as a translator without adding conversational filler.
**Example Prompt:**
```text
You are an expert American Sign Language (ASL) to English translator. 
I will provide you with a sequence of ASL glosses. 
Your job is to unscramble the ASL syntax (Topic-Comment) and output a single, grammatically perfect, natural-sounding English sentence. 
Do not explain your reasoning. Output ONLY the translated sentence.
```

## Verification Plan

### Manual Verification
1. Run `python test/live_test.py`.
2. Sign a multi-word phrase (e.g., "YOU", "NAME", "WHAT").
3. Verify the words accumulate correctly in the UI buffer.
4. Press `Enter`.
5. Verify the LLM translates it to "What is your name?".
