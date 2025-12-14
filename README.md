# LLM Response Evaluation Pipeline

This repository contains a production-oriented **LLM evaluation pipeline** designed to automatically assess the reliability of AI-generated responses in real time.

The system evaluates AI responses on:
- **Response Relevance & Completeness**
- **Hallucination / Factual Accuracy (Faithfulness)**
- **Latency & Cost**

This project was built as part of the **BeyondChats Internship Assignment**, with a strong emphasis on scalability, low latency, and cost efficiency.

---

## Local Setup Instructions

### Prerequisites
- Python 3.9 or above
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-name>
Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Set Environment Variable
export OPENAI_API_KEY="your_openai_api_key"
Step 4: Run the Evaluation Pipeline
python code.py
The evaluation output will be saved as:
evaluation_results.json

Architecture of the Evaluation Pipeline

The evaluation pipeline follows a hybrid architecture that combines fast heuristic checks with LLM-based evaluation only when necessary.
User Query + AI Response
          │
          ▼
Fast Heuristic Evaluation
- Embedding-based relevance scoring
- Context grounding checks
          │
          ▼
Low Confidence?
   ├── No → Accept heuristic scores
   └── Yes → Invoke LLM-as-a-Judge
          │
          ▼
Final Evaluation Report

Core Components
Preprocessing Layer
Extracts user query, AI response, and relevant context
Uses only vectors_used from the vector database to ensure accurate grounding

Heuristic Evaluation Layer
Computes semantic relevance using embedding similarity
Detects hallucinations by comparing response sentences against retrieved context

LLM-as-a-Judge Layer (Fallback)
Invoked only when heuristic confidence is low
Evaluates relevance, completeness, and faithfulness

Performance Metrics Layer
Estimates latency and cost of the target AI response

Why This Design?
Why not use only an LLM judge?
LLM-based evaluation is expensive
Adds unnecessary latency
Does not scale well for real-time systems handling large volumes

Why not use only heuristic rules?
Heuristics lack nuanced semantic reasoning
Edge cases require deeper contextual understanding

Why this hybrid approach?
Heuristics handle the majority of responses quickly and at zero API cost
LLM judge is used only for low-confidence cases
Balances accuracy, latency, and cost
Closely mirrors real-world, production-grade LLM evaluation systems
Scalability: Latency & Cost Optimization at Scale
If this pipeline is run across millions of daily conversations, the following strategies ensure efficiency:
Latency Optimization
Embedding-based similarity checks are CPU-efficient and fast
The default evaluation path avoids external API calls
LLM judge is triggered only when necessary
Cost Optimization
Minimizes LLM API usage by using confidence thresholds
Deterministic prompts reduce token consumption
Embeddings can be cached for repeated queries or responses
Grounding checks are restricted to vectors_used instead of all retrieved vectors


Tech Stack
Python
OpenAI API (LLM Judge)
SentenceTransformers
scikit-learn

Project Structure
.
├── code.py
├── clean_chat.json
├── clean_context.json
├── evaluation_results.json
├── requirements.txt
└── README.md
Author

Anumula Sri Lakshmi
Final-year Artificial Intelligence & Machine Learning student
Aspiring ML / Prompt Engineer
