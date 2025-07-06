import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from dotenv import load_dotenv
from typing import Optional, List
import logging
from services.vector_store_service import VectorStoreService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ModelService:
    def __init__(self, vector_store_service: Optional[VectorStoreService] = None):
        """
        Complete model service for Phi-3 with:
        - Automatic device handling
        - Error recovery
        - Context management
        - All original functionality
        """
        self.vector_store = vector_store_service
        self.model_name = os.getenv("MODEL_NAME", "microsoft/phi-3-mini-4k-instruct")
        self.hf_token = os.getenv("HF_TOKEN")
        self.max_context_length = 3000  # Characters for safety
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with accelerate-compatible settings"""
        try:
            logger.info("Initializing model...")
            
            # Tokenizer setup
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                padding_side="left"
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Model loading - let accelerate handle device placement
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                token=self.hf_token,
                low_cpu_mem_usage=True
            )

            # Pipeline without explicit device to avoid conflicts
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True,
                truncation=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Could not initialize model: {str(e)}")

    def _truncate_context(self, context: str) -> str:
        """
        Smart context truncation to fit model limits
        while preserving key information.
        """
        if not context or len(context) <= self.max_context_length:
            return context
        
        # Preserve complete sentences when possible
        truncated = context[:self.max_context_length]
        last_period = truncated.rfind('. ')
        return truncated[:last_period + 1] if last_period > 0 else truncated

    def generate_response(self, prompt: str) -> str:
        """Robust response generation with error handling"""
        try:
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            outputs = self.pipe(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                do_sample=True
            )
            
            # Clean output by removing input prompt
            generated = outputs[0]['generated_text']
            return generated.replace(prompt, "").strip()
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("Memory error - reducing output length")
                return self._handle_memory_error(prompt)
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return "Sorry, I couldn't generate a response."

    def _handle_memory_error(self, prompt: str) -> str:
        """Memory error recovery with reduced requirements"""
        try:
            return self.pipe(
                prompt,
                max_new_tokens=256,  # Reduced length
                temperature=0.7
            )[0]['generated_text'].strip()
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")
            return "The request requires more resources than available."

    def explain_topic(self, topic: str, context: Optional[str] = None) -> str:
        """
        Enhanced topic explanation with automatic context handling.
        Uses vector store if available and no context provided.
        """
        try:
            system_prompt = """You are an expert teacher. Provide clear explanations with examples."""
            
            # Get context if not provided
            if context is None and self.vector_store:
                context = self.vector_store.get_relevant_context(topic)
                context = self._truncate_context(context)

            prompt = f"""[INST] <<SYS>>
            {system_prompt}
            <</SYS>>

            Explain '{topic}'{" using this context:" if context else ""}
            {context if context else ""}
            Provide practical examples. [/INST]"""
            
            return self.generate_response(prompt)
        except Exception as e:
            logger.error(f"Explanation failed: {str(e)}")
            return f"Could not generate explanation: {str(e)}"

    def generate_review_suggestions(self, weak_topics: List[str]) -> str:
        """Generate focused review suggestions with validation"""
        if not weak_topics:
            return "No weak topics identified for review."

        try:
            system_prompt = """You are an expert teacher. Provide specific review suggestions."""
            
            prompt = f"""[INST] <<SYS>>
            {system_prompt}
            <</SYS>>

            Weak topics: {', '.join(weak_topics)}
            Provide key concepts and practice recommendations. [/INST]"""
            
            return self.generate_response(prompt)
        except Exception as e:
            logger.error(f"Review suggestions failed: {str(e)}")
            return "Could not generate review suggestions."

    def clear_resources(self):
        """Clean up model resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del self.model
        del self.pipe