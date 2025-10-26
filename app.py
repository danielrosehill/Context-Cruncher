"""
Context Cruncher - Gradio Application
Extract structured context data from voice recordings using Gemini AI.
"""
import gradio as gr
import os
from pathlib import Path
import tempfile
from gemini_processor import (
    process_audio_with_gemini,
    create_markdown_file,
    create_json_file
)


def process_audio(
    audio_input,
    uploaded_file,
    api_key: str,
    user_identification: str,
    user_name: str = ""
) -> tuple:
    """
    Process audio from either recording or upload.

    Args:
        audio_input: Audio from microphone recording
        uploaded_file: Uploaded audio file
        api_key: Gemini API key
        user_identification: "name" or "user"
        user_name: User's name if using name identification

    Returns:
        Tuple of (markdown_content, markdown_file, json_file, status_message)
    """
    try:
        # Validate API key
        if not api_key or api_key.strip() == "":
            return (
                "",
                None,
                None,
                "Error: Please provide a Gemini API key"
            )

        # Determine which audio source to use
        audio_path = None
        if audio_input is not None:
            audio_path = audio_input
        elif uploaded_file is not None:
            audio_path = uploaded_file.name

        if audio_path is None:
            return (
                "",
                None,
                None,
                "Error: Please record audio or upload an audio file"
            )

        # Determine user reference
        user_ref = None
        if user_identification == "name":
            if not user_name or user_name.strip() == "":
                return (
                    "",
                    None,
                    None,
                    "Error: Please provide your name when using name identification"
                )
            user_ref = user_name.strip()

        # Process with Gemini
        status_msg = "Processing audio with Gemini API..."
        context_markdown, human_readable_name, snake_case_filename = process_audio_with_gemini(
            audio_path,
            api_key,
            user_ref
        )

        # Create output files
        md_filename, md_content = create_markdown_file(
            context_markdown,
            human_readable_name,
            snake_case_filename
        )

        json_filename, json_content = create_json_file(
            context_markdown,
            human_readable_name,
            snake_case_filename
        )

        # Write files to temp directory for download
        temp_dir = tempfile.mkdtemp()
        md_path = Path(temp_dir) / md_filename
        json_path = Path(temp_dir) / json_filename

        with open(md_path, 'w') as f:
            f.write(md_content)

        with open(json_path, 'w') as f:
            f.write(json_content)

        return (
            md_content,
            str(md_path),
            str(json_path),
            f"Success! Context extracted: {human_readable_name}"
        )

    except Exception as e:
        return (
            "",
            None,
            None,
            f"Error: {str(e)}"
        )


# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main-header {
    text-align: center;
    margin-bottom: 2rem;
}
.status-success {
    color: #10b981;
}
.status-error {
    color: #ef4444;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Context Cruncher") as demo:
    gr.Markdown(
        """
        # 🎙️ Context Cruncher

        Extract structured context data from voice recordings using AI.
        Record your thoughts, preferences, or information naturally, and get clean,
        organized context data ready for AI applications.
        """,
        elem_classes="main-header"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Configuration")

            api_key_input = gr.Textbox(
                label="Gemini API Key",
                placeholder="Enter your Gemini API key",
                type="password",
                info="Get your API key from https://ai.google.dev/"
            )

            user_identification = gr.Radio(
                choices=["user", "name"],
                value="user",
                label="User Identification",
                info="How should you be referred to in the context data?"
            )

            user_name_input = gr.Textbox(
                label="Your Name",
                placeholder="Enter your name",
                visible=False,
                info="Used when 'By Name' is selected"
            )

            gr.Markdown("### 🎤 Audio Input")

            audio_recording = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Audio"
            )

            gr.Markdown("**OR**")

            audio_upload = gr.File(
                label="Upload Audio File",
                file_types=["audio"],
                type="filepath"
            )

            process_btn = gr.Button("🚀 Extract Context", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📄 Extracted Context")

            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                show_label=True
            )

            context_display = gr.Textbox(
                label="Context Data (Markdown)",
                lines=15,
                interactive=False,
                show_copy_button=True
            )

            with gr.Row():
                markdown_download = gr.File(label="Download Markdown")
                json_download = gr.File(label="Download JSON")

    # Show/hide name input based on identification method
    def toggle_name_input(identification_choice):
        return gr.update(visible=identification_choice == "name")

    user_identification.change(
        fn=toggle_name_input,
        inputs=[user_identification],
        outputs=[user_name_input]
    )

    # Process button click
    process_btn.click(
        fn=process_audio,
        inputs=[
            audio_recording,
            audio_upload,
            api_key_input,
            user_identification,
            user_name_input
        ],
        outputs=[
            context_display,
            markdown_download,
            json_download,
            status_output
        ]
    )

    gr.Markdown(
        """
        ---
        ### 💡 How It Works

        1. **Configure**: Enter your Gemini API key and choose how you want to be identified
        2. **Input Audio**: Either record directly in your browser or upload an audio file (MP3, WAV, OPUS)
        3. **Extract**: Click the button and let AI clean up your recording into structured context data
        4. **Download**: Get your context data as Markdown or JSON, or copy directly from the text area

        ### 📚 What is Context Data?

        Context data is specific information about you that AI systems can use to provide more
        personalized results. This tool transforms casual voice recordings into clean, structured
        context by:

        - Removing irrelevant information and tangents
        - Eliminating duplicates
        - Reformatting to third person
        - Organizing hierarchically

        ### 🔐 Privacy

        Your audio is processed using the Gemini API. Make sure you're comfortable with
        Google's privacy policies before using this tool with sensitive information.
        """
    )


if __name__ == "__main__":
    demo.launch()
