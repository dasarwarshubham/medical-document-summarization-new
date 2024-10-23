import os
import boto3
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
# import uuid  # For generating unique filenames

from openai import OpenAI
from mistralai import Mistral
import anthropic

# Part 1 (Setup)
# Load environment variables (AWS credentials)
load_dotenv()

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Configuring Boto3 for retries
retry_config = Config(
    region_name=os.environ.get("AWS_DEFAULT_REGION"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

# Create a boto3 session for accessing Bedrock and Textract
session = boto3.Session()
textract = session.client('textract', config=retry_config)

# Function to upload the document
# def upload_document(uploaded_file):
#     """Save the uploaded document to the 'uploaded_files' folder and return its file path."""
#     if uploaded_file is not None:
#         try:
#             # Create the folder if it doesn't exist
#             upload_folder = "./uploaded_files"
#             if not os.path.exists(upload_folder):
#                 os.makedirs(upload_folder)

#             # Get the original file name and path
#             file_path = os.path.join(upload_folder, uploaded_file.name)

#             # If the file already exists, rename it with a unique ID
#             if os.path.exists(file_path):
#                 file_name, file_extension = os.path.splitext(
#                     uploaded_file.name)
#                 unique_id = str(uuid.uuid4())[:8]  # Generate a short unique ID
#                 file_path = os.path.join(
#                     upload_folder, f"{file_name}_{unique_id}{file_extension}")

#             # Write the file to the folder
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             return file_path
#         except Exception as e:
#             st.error(f"Error uploading file: {e}")
#             return None
#     return None

def upload_document(uploaded_file):
    """Save the uploaded document to the 'uploaded_files' folder and return its file path."""
    if uploaded_file is not None:
        try:
            # Create the folder if it doesn't exist
            upload_folder = "./uploaded_files"
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Get the original file name and path
            file_path = os.path.join(upload_folder, uploaded_file.name)

            # Write the file to the folder (this will overwrite if it already exists)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return file_path
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            return None
    return None


# Function to process the document (Extract text using Textract and split it)
def process_document(file_path, file_type):
    """Extract text from the document (TIFF, PDF, JPG, JPEG) and split it into manageable chunks."""

    # Extract text from TIFF or image (JPG, JPEG)
    def extract_text_from_image(file_path):
        try:
            with open(file_path, 'rb') as document:
                response = textract.detect_document_text(
                    Document={'Bytes': document.read()})

            text = ""
            for item in response["Blocks"]:
                if item["BlockType"] == "LINE":
                    text += item["Text"] + "\n"
            return text
        except ClientError as e:
            st.error(
                f"Amazon Textract error: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return None

    # Extract text from PDF using Textract's `analyze_document` API
    def extract_text_from_pdf(file_path):
        try:
            with open(file_path, 'rb') as document:
                response = textract.analyze_document(
                    Document={'Bytes': document.read()},
                    FeatureTypes=["TABLES", "FORMS"]
                )
            text = ""
            for block in response["Blocks"]:
                if block["BlockType"] == "LINE":
                    text += block["Text"] + "\n"
            return text
        except ClientError as e:
            st.error(
                f"Amazon Textract error: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return None

    # Process the file based on its type
    if file_path and file_type:
        if file_type in ["jpg", "jpeg", "tiff"]:
            extracted_text = extract_text_from_image(file_path)
        elif file_type == "pdf":
            extracted_text = extract_text_from_pdf(file_path)
        else:
            st.error("Unsupported file format.")
            return []

        # If text was successfully extracted, split it into chunks
        if extracted_text:
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                    chunk_overlap=0
                )
                texts = text_splitter.split_text(extracted_text)
                return texts
            except Exception as e:
                st.error(f"Error splitting text into chunks: {e}")
                return []
        else:
            st.error("Failed to extract text from the document.")
            return []
    else:
        st.error("Invalid file path or file type.")
        return []


# Function to generate summary based on selected model
def generate_summary(documents, selected_model, prompt):
    """Generate summary from the provided document chunks using the selected model."""
    try:
        summaries = []
        for chunk in documents:
            try:
                input_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": "Sure, I can help with that. Please provide the text of the medical document."},
                    {"role": "user", "content": chunk}
                ]
                summary_text = ""
                # Open AI GPT
                if selected_model.get("name") == "OpenAI GPT":
                    openai_client = OpenAI(api_key=OPENAI_API_KEY)
                    response = openai_client.chat.completions.create(
                        model=selected_model.get("model"),
                        messages=input_messages,
                        max_tokens=500,
                        temperature=0.3
                    )
                    summary_text = response.choices[0].message.content

                # Mistral
                elif selected_model.get("name") == "Mistral":
                    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
                    response = mistral_client.chat.complete(
                        model=selected_model.get("model"),
                        messages=input_messages
                    )
                    summary_text = response.choices[0].message.content

                # Anthropic
                elif selected_model.get("name") == "Anthropic Claude":
                    anthropic_client = anthropic.Anthropic(
                        api_key=ANTHROPIC_API_KEY)
                    response = anthropic_client.messages.create(
                        model=selected_model.get("model"),
                        max_tokens=1024,
                        messages=input_messages
                    )
                    summary_text = response.content[0].text

                # Self-hosted
                elif selected_model.get("name") == "Self-Hosted Model":
                    model_url = "https://expert-eft-innocent.ngrok-free.app/v1"
                    self_hosted_client = OpenAI(
                        base_url=model_url, api_key='na')
                    response = self_hosted_client.chat.completions.create(
                        model=selected_model.get("model"),
                        messages=input_messages,
                    )
                    summary_text = response.choices[0].message.content

                # Append summary
                summaries.append(summary_text)

            except Exception as e:
                st.error(f"Error invoking {selected_model_text}: {e}")
                return None

        # Return the combined summaries
        return " ".join(summaries)

    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


# Function to check login credentials
def check_login(username, password):
    # You can replace this with a more secure method of handling passwords
    # Example credentials
    return username == os.getenv("LOGIN_USERNAME") and password == os.getenv("LOGIN_PASSWORD")


# Check if the user is logged in
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login Screen
if not st.session_state.logged_in:
    st.subheader("Login")

    # Create a form for username and password
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                # Rerun to show the app after successful login
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
else:
    # If logged in, show the main app
    st.title("Medical Document Summarization")

    # Step 1: Enter your prompt
    prompt = st.text_area("Enter your custom prompt",
                          value="Summarize the medical document.")

    # Step 2: Upload the document
    uploaded_file = st.file_uploader(
        "Upload a medical document (PDF, TIFF, JPG, JPEG)", type=["pdf", "tiff", "jpg", "jpeg"])

    # Step to allow user to select summary display format
    display_format = st.radio(
        "Select the format to display the generated summary:",
        options=["Code Block", "Markdown", "Plain Text"],
        index=1  # Default to "Markdown"
    )

    # Model selection dropdown
    model_options = [
        {"name": "Self-Hosted LLM",
            "model": "qwen2.5-coder-7b-instruct", "status": "Free"},
        {"name": "Mistral", "model": "open-mistral-nemo", "status": "Free"},
        {"name": "Mistral", "model": "mistral-small-latest", "status": "Paid"},
        {"name": "OpenAI", "model": "gpt-4o-mini", "status": "Paid"},
        {"name": "Claude", "model": "claude-3-5-sonnet-20240620", "status": "Paid"},
        {"name": "Claude", "model": "claude-3-haiku-20240307", "status": "Paid"},
    ]
    model_dropdown_options = [
        f"{option['name']} - ({option['model']}) - {option['status']}" for option in model_options
    ]

    selected_model_text = st.selectbox(
        "Select a model to use", model_dropdown_options)

    # Find the selected model's dictionary from model_options
    selected_model = next(
        (model for model in model_options if f"{model['name']} - ({model['model']}) - {model['status']}" == selected_model_text), None)

    # Submit button to trigger processing
    submit_button = st.button("Submit", disabled=(
        uploaded_file is None or selected_model is None))

    if submit_button:
        if uploaded_file is not None and selected_model is not None:
            # Step 3: Process the document
            file_type = uploaded_file.name.split(".")[-1].lower()
            file_path = upload_document(uploaded_file)

            if file_path:
                st.write(f"Processing document: {uploaded_file.name}")
                texts = process_document(file_path, file_type)

                if texts:
                    # Step 4: Generate summary
                    st.write(
                        f"Generating summary using {selected_model_text}...")
                    summary = generate_summary(texts, selected_model, prompt)

                    if summary:
                        st.subheader("Generated Summary")

                        if display_format == "Code Block":
                            st.code(summary, language='text')
                        elif display_format == "Markdown":
                            st.markdown(summary)
                        elif display_format == "Plain Text":
                            st.write(summary)
                    else:
                        st.error("Failed to generate summary.")
                else:
                    st.error("No text extracted from the document.")
        else:
            st.warning(
                "Please upload a document and select a model before submitting.")
    else:
        st.info("Please upload a medical document to proceed.")
