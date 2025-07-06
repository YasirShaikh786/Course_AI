import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import os
from dotenv import load_dotenv

load_dotenv()

class ModelService:
    def __init__(self):
        """
        Initialize the model service with Phi-3 Mini.
        This model is more lightweight and suitable for smaller GPUs.
        """
        self.model_name = os.getenv("MODEL_NAME", "microsoft/phi-3-mini-4k-instruct")
        self.hf_token = os.getenv("HF_TOKEN")  # You should set this in your .env file
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the Phi-3 Mini model and tokenizer with memory-efficient loading and CPU offloading.
        """
        # Use 8-bit quantization with CPU offload enabled
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True  # ✅ This enables safe offloading to CPU
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            token=self.hf_token
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=1024,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )


    def generate_response(self, prompt: str) -> str:
        """
        Generate response using the Phi-3 Mini model.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated response from the model
        """
        return self.pipe(prompt)[0]['generated_text'].strip()

    def explain_topic(self, topic: str, context: str) -> str:
        """
        Generate an explanation for a specific topic using Phi-3 Mini.
        
        Args:
            topic: Topic to explain
            context: Relevant context for the explanation
            
        Returns:
            Detailed explanation of the topic
        """
        system_prompt = """You are an expert teacher. Provide clear, concise explanations with practical examples.
        Keep explanations focused and relevant to the context."""
        
        prompt = f"""{system_prompt}
        
        Explain the following topic in detail: {topic}
        Context: {context}
        
        Provide clear examples and practical applications."""
        return self.generate_response(prompt)

    def generate_review_suggestions(self, weak_topics: list) -> str:
        """
        Generate review suggestions based on weak topics using Phi-3 Mini.
        
        Args:
            weak_topics: List of topics that need review
            
        Returns:
            Focused review suggestions and key concepts
        """
        system_prompt = """You are an expert teacher. Provide focused review suggestions.
        Prioritize the most important concepts and provide practical examples."""
        
        prompt = f"""{system_prompt}
        
        Based on the following weak topics: {', '.join(weak_topics)}
        
        Provide focused review suggestions and key concepts to focus on."""
        return self.generate_response(prompt)
