# 7. Agentic AI Component 
"""
agent.py
--------
An agentic AI component that performs multi-step reasoning to verify news.
Step 1: Get linguistic prediction from the fine-tuned RoBERTa model.
Step 2: Use a Search Tool (Wikipedia) to find real-world context.
Step 3: Feed both pieces of evidence to an LLM (Gemini) to make a final decision.
"""

import logging                # For logging progress and errors
import wikipedia              # For the Search Tool to retrieve facts from Wikipedia
from google import genai      # For interacting with the new Google GenAI API
import os                     # For environment variable access   
from roberta_inference import RobertaInferencePipeline # Custom module for RoBERTa inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FactCheckingAgent:
    def __init__(self, roberta_model_dir="models/roberta_best", gemini_api_key=None):
        logger.info("Initializing Fact-Checking Agent...")
        
        # 1. Initialize the RoBERTa tool
        self.roberta_pipeline = RobertaInferencePipeline(checkpoint_dir=roberta_model_dir, device="cpu")
        
        # 2. Initialize the LLM API
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Set it as GEMINI_API_KEY env var.")
        
        # Initialize the new Google GenAI client
        self.client = genai.Client(api_key=api_key)
        logger.info("Agent initialization complete.")

    def search_tool(self, query: str, sentences=3):
        """Tool: Search Wikipedia for facts related to the query."""
        logger.info(f"Using Search Tool for query: '{query}'")
        try:
            # Get the top search result page title
            search_results = wikipedia.search(query, results=1)
            if not search_results:
                return "No search results found."
            
            # Fetch the summary of that page
            page = wikipedia.page(search_results[0], auto_suggest=False)
            summary = wikipedia.summary(page.title, sentences=sentences)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Search is too ambiguous. Possible matches: {e.options[:3]}"
        except Exception as e:
            return f"Search failed: {str(e)}"

    def verify_statement(self, statement: str):
        """Multi-step reasoning pipeline."""
        print(f"\n--- Processing: '{statement}' ---")
        
        # Step 1: Run RoBERTa inference
        logger.info("Step 1: Running RoBERTa analysis...")
        roberta_result = self.roberta_pipeline.predict(statement)
        ai_confidence = roberta_result["confidence"]
        ai_label = roberta_result["label"]
        roberta_evidence = f"The stylistic NLP model predicted this is {ai_label} with {ai_confidence:.2%} confidence."
        print(f"RoBERTa Output: {ai_label} ({ai_confidence:.2%})")

        # Step 2: Use Search Tool
        logger.info("Step 2: Retrieving real-world facts...")
        # A real agent might extract keywords from the statement to search, 
        # but passing the whole statement works for a basic implementation.
        search_evidence = self.search_tool(statement)
        print(f"Search Tool Retrieved: {search_evidence[:100]}...")

        # Step 3: LLM Synthesis and Decision Making
        logger.info("Step 3: Synthesizing evidence for final decision...")
        
        prompt = f"""
        You are an expert fact-checking agent. Your job is to classify the following statement as TRUE, FALSE, or UNVERIFIED, and provide a short reasoning.
        
        Statement to evaluate: "{statement}"
        
        Evidence 1 (Linguistic AI pattern): {roberta_evidence}
        Evidence 2 (Wikipedia Search Tool): {search_evidence}
        
        Instructions:
        1. Consider both pieces of evidence.
        2. If the search evidence contradicts the statement, it is FALSE.
        3. Explain your reasoning clearly but concisely (3-4 sentences).
        4. End your response with "FINAL VERDICT: [TRUE/FALSE/UNVERIFIED]"
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        final_decision = response.text
        
        print("\n=== Agent Decision ===")
        print(final_decision)
        print("======================\n")
        
        return {
            "statement": statement,
            "roberta": roberta_result,
            "search_evidence": search_evidence,
            "agent_decision": final_decision
        }

if __name__ == "__main__":
    # You can get a free API key at: https://aistudio.google.com/
    # Run this via: GEMINI_API_KEY="your_key" python src/agent.py
    
    # Let's test it with a statement that sounds structurally real, but is factually false.
    test_statement = "Barack Obama was born in Kenya and served as the 45th President of the United States."
    
    agent = FactCheckingAgent()
    agent.verify_statement(test_statement)