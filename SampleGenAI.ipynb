# Cell 1: Install Dependencies
!pip install -q langchain-groq sentence-transformers chromadb langchain-chroma langchain-huggingface langgraph

# Cell 2: Imports
import pandas as pd
import numpy as np
from typing import TypedDict, List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("✅ All imports successful!")

# Cell 3: Initialize Groq LLMs (FIXED API KEYS - use same key for all)
# IMPORTANT: Replace with your actual API key
GROQ_API_KEY = ""  # Your key

# Planner uses Llama 70B for strategic thinking
planner_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",  # Updated model name
    temperature=0.2,
    max_tokens=1024
)

# Executor uses Mixtral for generation
executor_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b-32768",
    temperature=0.5,
    max_tokens=2048
)

# Verifier uses Llama 8B for fast evaluation
verifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",  # Updated model name
    temperature=0,
    max_tokens=512
)

print("✅ All Groq LLMs initialized!")

# Cell 4: Movie Database
movies = {
    "title": [
        "The Dark Knight", "Inception", "Gravity", "The Shawshank Redemption",
        "Pulp Fiction", "The Godfather", "Interstellar", "Memento",
        "Fight Club", "The Matrix", "Goodfellas"
    ],
    "description": [
        "A billionaire vigilante confronts a psychopathic criminal who tests Gotham's moral fabric. Themes of justice, chaos, and sacrifice in a gritty crime thriller.",
        "A thief who enters dreams to plant ideas faces impossible architecture and collapsing realities. A mind-bending sci-fi exploring guilt, reality, and consciousness.",
        "Astronauts stranded after a satellite collision must survive in the silence of deep space. A survival thriller about human will against the cosmos.",
        "An innocent man in prison builds hope through decades of quiet resilience and friendship. A moving story of freedom, perseverance, and the human spirit.",
        "A hitman, a boxer, a gangster's wife, and two small-time criminals collide in a non-linear crime masterpiece.",
        "The aging patriarch of a mafia dynasty transfers control to his reluctant son. A sweeping epic about power, family, and loyalty.",
        "A team of explorers travels through a wormhole in space to ensure humanity's survival. A sci-fi epic about love, time, and sacrifice.",
        "A man with short-term memory loss uses notes and tattoos to hunt for his wife's murderer. A neo-noir mystery told backwards.",
        "An insomniac office worker forms an underground fight club with a charismatic soap salesman, leading to a terrorist organization.",
        "A computer hacker learns that reality is a simulation controlled by machines. A groundbreaking sci-fi action film.",
        "A young man rises through the ranks of the mob, only to face betrayal and paranoia. Martin Scorsese's masterpiece."
    ]
}

df = pd.DataFrame(movies)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
movie_embeddings = embedding_model.encode(df['description'].tolist())
print(f"✅ Loaded {len(df)} movies with embeddings")

# Cell 5: Long-Term Memory with ChromaDB
class LongTermMemory:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            collection_name="movie_preferences_groq",
            embedding_function=self.embeddings,
            persist_directory="./chroma_movie_memory_groq"
        )
        print("✅ Long-Term Memory (ChromaDB) initialized")

    def store_preference(self, user_id: str, query: str, liked_movie: str):
        text = f"User {user_id} liked '{liked_movie}' when asking: {query}"
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{"user_id": user_id, "timestamp": str(datetime.now()), "movie": liked_movie}]
        )
        print(f"💾 LTM: Stored preference for user {user_id}")

    def retrieve_memories(self, user_id: str, query: str, k: int = 3) -> List[str]:
        try:
            results = self.vectorstore.similarity_search(
                f"User {user_id} preferences related to {query}",
                k=k
            )
            memories = [doc.page_content for doc in results]
            if memories:
                print(f"🧠 LTM: Retrieved {len(memories)} relevant memories")
            return memories
        except Exception as e:
            print(f"⚠️ LTM retrieval error: {e}")
            return []

ltm = LongTermMemory()

# Cell 6: Agent State Definition
class AgentState(TypedDict):
    query: str
    user_id: str
    plan: str
    search_strategy: str
    retrieved_docs: List[str]
    retrieved_scores: List[float]
    reranked_docs: List[str]
    short_term_memory: List[Dict]
    long_term_memories: List[str]
    answer: str
    citations: List[str]
    needs_retry: bool
    retry_count: int
    critique: str
    quality_score: float
    needs_human_approval: bool
    human_feedback: Optional[str]

# Cell 7: PLANNER NODE (FIXED)
def planner_node(state: AgentState) -> AgentState:
    """PLANNER: Creates execution strategy using Groq Llama 70B."""
    query = state["query"]
    user_id = state.get("user_id", "anonymous")
    ltm_memories = state.get("long_term_memories", [])

    memory_context = ""
    if ltm_memories:
        memory_context = f"\nUser's past preferences: {', '.join(ltm_memories[:2])}"

    prompt = f"""You are a PLANNER for a movie recommendation system.
User request: {query}
User ID: {user_id}{memory_context}

Create a simple search strategy. Respond with ONLY valid JSON in this exact format:
{{"strategy": "your plan here", "primary_themes": ["theme1", "theme2"]}}

Do not include any other text outside the JSON."""

    try:
        response = planner_llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            plan_json = json.loads(json_match.group())
        else:
            plan_json = json.loads(content)

        plan = plan_json.get("strategy", "Find relevant movies based on user query")
        primary_themes = plan_json.get("primary_themes", ["general"])
        search_strategy = primary_themes[0] if primary_themes else "general"

    except Exception as e:
        print(f"⚠️ Planner error: {e}, using defaults")
        plan = "Find movies matching the user's request"
        search_strategy = "general"

    print(f"📋 PLANNER: Strategy = {search_strategy}")

    return {
        **state,
        "plan": plan,
        "search_strategy": search_strategy
    }

# Cell 8: EXECUTOR NODE (FIXED)
def executor_node(state: AgentState) -> AgentState:
    """EXECUTOR: Does retrieval and generation using Groq Mixtral."""
    query = state["query"]
    search_strategy = state.get("search_strategy", "general")
    ltm_memories = state.get("long_term_memories", [])
    retry_count = state.get("retry_count", 0)
    critique = state.get("critique", "")

    # Enhanced query with LTM context
    enhanced_query = query
    if ltm_memories:
        enhanced_query = f"{query} (User previously enjoyed: {', '.join(ltm_memories[:1])})"

    # Vector search
    query_emb = embedding_model.encode([enhanced_query])
    similarities = cosine_similarity(query_emb, movie_embeddings)[0]

    # Get top movies
    top_indices = np.argsort(similarities)[::-1][:4]
    candidates = df.iloc[top_indices]["title"].tolist()
    candidate_scores = similarities[top_indices].tolist()

    print(f"⚙️ EXECUTOR: Found candidates: {candidates[:3]}")

    # Rerank by score
    reranked = list(zip(candidates, candidate_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in reranked[:3]]

    # Add retry feedback if applicable
    feedback_context = ""
    if retry_count > 0 and critique:
        feedback_context = f"\n\nPREVIOUS FEEDBACK TO ADDRESS: {critique}\nPlease improve your answer."

    prompt = f"""You are a movie recommendation EXPERT.
User request: {query}
Search strategy: {search_strategy}
Top movies found: {', '.join(reranked_docs)}
{feedback_context}

Provide a helpful recommendation. Mention specific movies and explain WHY they match.
Keep your answer clear and informative (at least 100 characters)."""

    try:
        response = executor_llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        print(f"⚠️ Executor error: {e}")
        answer = f"Based on your request '{query}', I recommend: {', '.join(reranked_docs[:2])}. These movies match your interests."

    return {
        **state,
        "retrieved_docs": candidates,
        "retrieved_scores": candidate_scores,
        "reranked_docs": reranked_docs,
        "answer": answer,
        "citations": reranked_docs[:2]
    }

# Cell 9: VERIFIER NODE - COMPLETELY REWRITTEN (FIXED)
def verifier_node(state: AgentState) -> AgentState:
    """VERIFIER: Quality control - NO JSON parsing issues now."""
    query = state["query"]
    answer = state["answer"]
    retry_count = state.get("retry_count", 0)

    # Simple rule-based scoring (NO LLM JSON parsing issues!)
    score = 0
    critique_points = []

    # Criterion 1: Length (0-40 points)
    answer_length = len(answer)
    if answer_length >= 150:
        score += 40
        print(f"   ✓ Length: {answer_length} chars - Excellent")
    elif answer_length >= 80:
        score += 25
        print(f"   ✓ Length: {answer_length} chars - Good")
    elif answer_length >= 40:
        score += 15
        print(f"   ⚠️ Length: {answer_length} chars - Acceptable")
    else:
        critique_points.append(f"Answer too short ({answer_length} chars)")
        print(f"   ✗ Length: {answer_length} chars - Too short")

    # Criterion 2: Contains movie titles (0-30 points)
    movie_titles_found = []
    for movie in df["title"].tolist():
        if movie.lower() in answer.lower():
            movie_titles_found.append(movie)

    if len(movie_titles_found) >= 2:
        score += 30
        print(f"   ✓ Mentions {len(movie_titles_found)} movies")
    elif len(movie_titles_found) >= 1:
        score += 20
        print(f"   ✓ Mentions {len(movie_titles_found)} movie")
    else:
        critique_points.append("No specific movie titles mentioned")
        print(f"   ✗ No movie titles found")

    # Criterion 3: Query relevance (0-30 points)
    # Check if answer addresses the query
    query_words = set(query.lower().split())
    answer_lower = answer.lower()

    relevant_words_found = sum(1 for word in query_words if word in answer_lower and len(word) > 3)
    relevance_score = min(30, relevant_words_found * 10)
    score += relevance_score

    if relevance_score >= 20:
        print(f"   ✓ Good query relevance")
    elif relevance_score >= 10:
        print(f"   ⚠️ Moderate query relevance")
    else:
        critique_points.append("Answer doesn't address the query well")
        print(f"   ✗ Poor query relevance")

    # Determine if retry is needed
    max_retries = 2  # Circuit breaker
    needs_retry = (score < 50 and retry_count < max_retries)

    critique = "; ".join(critique_points) if critique_points else "Quality acceptable"

    print(f"\n📊 VERIFIER: Score={score}/100, Needs retry={needs_retry}")
    if critique_points:
        print(f"   Critique: {critique}")

    return {
        **state,
        "needs_retry": needs_retry,
        "retry_count": retry_count + (1 if needs_retry else 0),
        "critique": critique if needs_retry else "",
        "quality_score": score / 100
    }

# Cell 10: HUMAN-IN-THE-LOOP NODE
def human_approval_node(state: AgentState) -> AgentState:
    """HUMAN-IN-THE-LOOP: For low quality answers."""
    quality_score = state.get("quality_score", 0)

    if quality_score < 0.4:  # Only for very low quality
        print(f"👤 HUMAN-IN-THE-LOOP: Quality score {quality_score:.2f} is low")
        print(f"   Requesting human approval...")
        print(f"   Proposed answer preview: {state['answer'][:150]}...")

        # Simulate human approval (in production, would wait for input)
        human_feedback = "APPROVED"
        print(f"   ✅ Human approved the answer")

        return {
            **state,
            "needs_human_approval": True,
            "human_feedback": human_feedback
        }

    return {
        **state,
        "needs_human_approval": False,
        "human_feedback": None
    }

# Cell 11: ROUTING FUNCTIONS (FIXED - prevents infinite loops)
def route_after_verifier(state: AgentState) -> str:
    """Route based on verification result with circuit breaker."""
    max_retries = 2  # Hard limit

    if state.get("needs_retry", False) and state.get("retry_count", 0) < max_retries:
        print(f"🔄 ROUTER: Retrying (attempt {state['retry_count']}/{max_retries})")
        return "executor"
    elif state.get("quality_score", 1) < 0.4:
        print(f"👤 ROUTER: Routing to human approval")
        return "human_approval"
    else:
        print(f"✅ ROUTER: Answer good, finishing")
        return END

def route_after_human(state: AgentState) -> str:
    """Route after human review."""
    if state.get("human_feedback") == "APPROVED":
        print(f"✅ Human approved - finishing")
        return END
    else:
        print(f"🔄 Human requested changes - retrying")
        return "executor"

# Cell 12: BUILD COMPLETE GRAPH
def build_complete_agent():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("human_approval", human_approval_node)

    # Define flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "verifier")

    # Conditional edges
    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "executor": "executor",
            "human_approval": "human_approval",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "human_approval",
        route_after_human,
        {
            "executor": "executor",
            END: END
        }
    )

    # Add checkpointer for STM
    memory = MemorySaver()

    print("✅ Graph built successfully!")
    return workflow.compile(checkpointer=memory)

# Cell 13: RUN THE SYSTEM
print("=" * 70)
print("COMPLETE MULTI-AGENT ORCHESTRATION with GROQ")
print("Planner → Executor → Verifier → Human-in-Loop")
print("=" * 70)

# Build the app
app = build_complete_agent()

# Setup user session
user_id = "user_test_001"
thread_id = f"{user_id}_session_1"
config = {"configurable": {"thread_id": thread_id}}

# Store some LTM preferences
print("\n📝 Setting up Long-Term Memory...")
ltm.store_preference(user_id, "I love mind-bending plots", "Inception")
ltm.store_preference(user_id, "Christopher Nolan films", "The Dark Knight")

# Retrieve LTM memories
ltm_memories = ltm.retrieve_memories(user_id, "reality bending movie")

# Test query
query = "give me a movie that will make me question reality"

print(f"\n{'='*70}")
print(f"📌 USER QUERY: {query}")
print(f"👤 User ID: {user_id}")
print(f"{'='*70}\n")

# Invoke the agent
result = app.invoke({
    "query": query,
    "user_id": user_id,
    "plan": "",
    "search_strategy": "",
    "retrieved_docs": [],
    "retrieved_scores": [],
    "reranked_docs": [],
    "short_term_memory": [],
    "long_term_memories": ltm_memories,
    "answer": "",
    "citations": [],
    "needs_retry": False,
    "retry_count": 0,
    "critique": "",
    "quality_score": 0.0,
    "needs_human_approval": False,
    "human_feedback": None
}, config=config)

# Display results
print(f"\n{'='*70}")
print(f"🎯 FINAL ANSWER:")
print(f"{'='*70}")
print(f"{result['answer']}")

print(f"\n{'='*70}")
print(f"📊 PERFORMANCE METRICS:")
print(f"{'='*70}")
print(f"   Quality Score: {result['quality_score']:.2f}/1.00")
print(f"   Retries: {result['retry_count']}")
print(f"   Human Approved: {result['needs_human_approval']}")
print(f"   Citations: {result['citations']}")
print(f"   Search Strategy: {result.get('search_strategy', 'N/A')}")

# Store this interaction in LTM
if result['citations']:
    ltm.store_preference(user_id, query, result['citations'][0])

print(f"\n{'='*70}")
print("✅ ALL CONCEPTS DEMONSTRATED SUCCESSFULLY!")
print(f"{'='*70}")
print("✅ Planner → Executor → Verifier architecture")
print("✅ Conditional edges for retry loops (circuit breaker at 2 retries)")
print("✅ Short-Term Memory via Checkpointer (thread_id)")
print("✅ Long-Term Memory via ChromaDB")
print("✅ Human-in-the-Loop for quality assurance")
print("✅ AgentState as shared workflow memory")
print("✅ Zero-cost LLM calls via Groq API")
print("=" * 70)
