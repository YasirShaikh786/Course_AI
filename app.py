import gradio as gr
from models.model_service import ModelService
from services.vector_store_service import VectorStoreService
from services.learning_analytics_service import LearningAnalyticsService
from utils.document_processor import DocumentProcessor
import os
import tempfile
from dotenv import load_dotenv
import logging

load_dotenv()

# Initialize services
vector_store_service = VectorStoreService()
model_service = ModelService(vector_store_service=vector_store_service)
learning_analytics_service = LearningAnalyticsService()
document_processor = DocumentProcessor()
logger = logging.getLogger(__name__)

def upload_course_material(files):
    """Robust file upload handler"""
    if not files:
        return "No files uploaded"
    
    results = []
    for file in files:
        tmp_path = None
        try:
            # Validate file
            if not hasattr(file, 'name'):
                results.append("⚠️ Invalid file object received")
                continue
                
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext not in ['.pdf', '.docx', '.txt']:
                results.append(f"⚠️ Skipped {file.name} (unsupported format)")
                continue

            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with open(file.name, 'rb') as src_file:
                    tmp_file.write(src_file.read())

            # Process file
            chunks = document_processor.process_file(tmp_path)
            if not chunks:
                results.append(f"⚠️ No content found in {file.name}")
                continue

            # Store in vector DB
            vector_store_service.add_documents(chunks)
            results.append(f"✅ Processed {file.name} ({len(chunks)} chunks)")

        except Exception as e:
            results.append(f"❌ Failed to process {file.name}: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    return "\n".join(results)
def get_explanation(topic):
    """Get explanation for a topic"""
    # Get relevant context
    context = vector_store_service.get_relevant_context(topic)
    
    # Generate explanation
    return model_service.explain_topic(topic, context)

def provide_feedback(topic, feedback_type):
    """Record user feedback"""
    learning_analytics_service.add_feedback(topic, feedback_type)
    return f"Feedback recorded: {feedback_type} for topic {topic}"

def take_quiz(topic, answers):
    """Process quiz answers and record scores"""
    # TODO: Implement quiz scoring logic
    score = 0.8  # Placeholder score
    learning_analytics_service.record_quiz_score(topic, score)
    return f"Quiz completed. Score: {score * 100:.1f}%"

def get_progress():
    """Get learning progress and generate review suggestions"""
    progress_stats = learning_analytics_service.get_progress_stats()
    weak_topics = progress_stats.get('weak_topics', [])
    
    if weak_topics:
        suggestions = (
            "Recommended Review Topics:\n"
            "• " + "\n• ".join(weak_topics) + "\n\n"
            "Focus on these areas where you've struggled with quizzes or requested slower explanations."
        )
    else:
        suggestions = (
            "No specific review recommendations.\n"
            "Your progress looks good! Consider exploring new topics."
        )
    
    return progress_stats, suggestions

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Phi-3 Mini Teaching Agent") as demo:
        # Header Section
        gr.Markdown("""
        # 📚 Teaching Agent with Phi-3 Mini
        *Your personalized AI learning assistant*
        """)
        
        # Main Tabs
        with gr.Tabs():
            # ===== Upload Tab =====
            with gr.Tab("📤 Upload Course Material", id="upload"):
                gr.Markdown("### Add your learning materials to get started")
                with gr.Row():
                    with gr.Column(scale=3):
                        upload_input = gr.File(
                            label="Upload PDF/DOCX/TXT",
                            file_types=[".pdf", ".docx", ".txt"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("Upload & Process", variant="primary")
                    with gr.Column(scale=2):
                        upload_output = gr.Textbox(
                            label="Processing Status",
                            interactive=False,
                            lines=10,
                            placeholder="Files will appear here after processing..."
                        )
            
            # ===== Learning Tab =====
            with gr.Tab("🎓 Learning Interface", id="learn"):
                # Topic Explanation Section
                with gr.Accordion("🔍 Get Explanation", open=True):
                    with gr.Row():
                        topic_input = gr.Textbox(
                            label="Enter topic/question",
                            placeholder="What would you like to learn about?",
                            lines=2
                        )
                    explain_btn = gr.Button("Explain", variant="primary")
                    explanation_output = gr.Textbox(
                        label="Explanation",
                        lines=8,
                        interactive=False,
                        placeholder="Your explanation will appear here..."
                    )
                
                # Quiz Section
                with gr.Accordion("📝 Take a Quiz", open=False):
                    quiz_input = gr.Textbox(
                        label="Your answer",
                        placeholder="Enter your response here...",
                        lines=3
                    )
                    with gr.Row():
                        quiz_btn = gr.Button("Submit Answer", variant="primary")
                        quiz_output = gr.Textbox(
                            label="Result",
                            interactive=False,
                            show_label=False
                        )
                
                # Feedback Controls
                with gr.Row():
                    gr.Markdown("### Was this helpful?")
                    with gr.Column(scale=1, min_width=100):
                        too_fast_btn = gr.Button("🚀 Too Fast")
                    with gr.Column(scale=1, min_width=100):
                        too_slow_btn = gr.Button("🐢 Too Slow")
                    feedback_output = gr.Textbox(
                        label="Feedback Status",
                        visible=False
                    )
                
                # Progress Tracking
                with gr.Accordion("📊 Your Progress", open=False):
                    with gr.Row():
                        progress_output = gr.JSON(label="Learning Analytics")
                    review_suggestions = gr.Textbox(
                        label="Recommended Review Topics",
                        interactive=False,
                        lines=4
                    )

        # ===== Event Handlers =====
        # Upload
        upload_btn.click(
            fn=upload_course_material,
            inputs=[upload_input],
            outputs=[upload_output]
        )
        
        # Explanation
        explain_btn.click(
            fn=get_explanation,
            inputs=[topic_input],
            outputs=[explanation_output]
        ).then(
            fn=get_progress,
            outputs=[progress_output, review_suggestions]
        )
        
        # Quiz
        quiz_btn.click(
            fn=take_quiz,
            inputs=[topic_input, quiz_input],
            outputs=[quiz_output]
        ).then(
            fn=get_progress,
            outputs=[progress_output, review_suggestions]
        )
        
        # Feedback
        too_fast_btn.click(
            fn=lambda x: provide_feedback(x, "too_fast"),
            inputs=[topic_input],
            outputs=[feedback_output]
        ).then(
            fn=get_progress,
            outputs=[progress_output, review_suggestions]
        )
        
        too_slow_btn.click(
            fn=lambda x: provide_feedback(x, "too_slow"),
            inputs=[topic_input],
            outputs=[feedback_output]
        ).then(
            fn=get_progress,
            outputs=[progress_output, review_suggestions]
        )
        
        # Initial load
        demo.load(
            fn=get_progress,
            outputs=[progress_output, review_suggestions]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()