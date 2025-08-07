import streamlit as st
import os
from typing import List, Union
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import re
import json 

# Setting up API key (update this to your method to set the API key)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize Google model
llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=1)

# Streamlit app layout
st.set_page_config(layout="wide", initial_sidebar_state="auto")

# Define Pydantic models
class QuizQuestion(BaseModel):
    question: str
    options: Union[List[str], None] = Field(description="List of options for multiple choice questions, None for True/False")
    correct_answer: str = Field(description="A, B, C, or D for multiple choice; A or B for true/false questions")
    explanation: str
    difficulty: str = Field(default="Medium", description="Easy, Medium, or Hard")

class Quiz(BaseModel):
    questions: List[QuizQuestion]

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Updated function to validate concepts in PDF
def validate_concepts_in_pdf(concepts, pdf_text):
    pdf_text = preprocess_text(pdf_text)
    valid_concepts = []
    for concept in concepts:
        concept = preprocess_text(concept)
        if concept in pdf_text:
            valid_concepts.append(concept)
    return valid_concepts

def extract_json_from_codeblock(text):
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text  # Return as-is if nothing found

# Function to generate quiz
def generate_questions_from_pdf(pdf_text, num_questions, quiz_type, quiz_context):
    output_parser = PydanticOutputParser(pydantic_object=Quiz)
    prompt_template = PromptTemplate(
        template="""
You are an AI-powered quiz generator. Your task is to create a quiz based on the following parameters:
Number of questions: {num_questions}
Quiz type: {quiz_type}
Topic/Context: {quiz_context}
Use the following text as context: {pdf_text}

Guidelines:
1. Generate ONLY {quiz_type} questions based on the content of the uploaded PDF.
2. Ensure all questions are strictly related to the specified topic/context: {quiz_context}
3. Provide a blend of difficulty levels: 20% Easy, 20% Medium, and 60% Hard questions.
4. For each question, provide a detailed explanation of the correct answer.
5. If you lack questions on the specified concept, modify previously used questions by changing their phrasing and making them more complex. DO NOT generate questions outside the specified concept.

Please generate the quiz in the following JSON format:
{{
    "questions": [
        {{
            "question": "Question text here",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "A",
            "explanation": "Explanation here"
            'difficulty':"easy"
        }},
    Here is an example:(Multiple-Choice)
        {{
            "question":"What is the time complexity of a binary search tree?"
            "options":["O(n)", "O(log n)", "O(n^2)", "O(1)"]
            "correct_answer": "B"
            "explanation": "The time complexity of a binary search tree is O(log n). This is because in a balanced binary search tree, each comparison allows the operations to skip about half of the tree, so it takes about log2 n comparisons to find an element, or to insert a new element. This is much faster than the linear time (O(n)) required to find elements by key in an unsorted array, but slower than the constant time (O(1)) of hash tables."
            "difficulty" : "medium"
        }}
        ...
    ]
}}

For true/false questions, use null for options and "A" for True or "B" for False as the correct answer.
    Here's an example:
    {{
        "question": "A binary search tree is always balanced"
        "options": ["True", "False"]
        "correct_answer": "B"
        "explanation": "This statement is false. A binary search tree is not always balanced. While balanced binary search trees (like AVL trees or Red-Black trees) exist, a standard binary search tree can become unbalanced depending on the order of insertions and deletions. An unbalanced tree can degrade to a linked list in the worst case, losing the logarithmic time complexity advantage for operations."
        "difficulty": "easy"
     }}

Difficulty Level Guidelines:
Hard (60% of questions):
Multiple Choice Questions:
   - Ensure distractors (wrong answers) are plausible and based on common misconceptions or errors in understanding.
   - Include answers that require higher-order thinking, such as application of concepts or analysis of information.
   - Consider using "All of the above" or "None of the above" options strategically.
   - For language or writing-related questions, include answers with subtle grammatical or stylistic differences.

True/False Questions:
   - Include statements that require deep understanding of nuances or exceptions to rules.
   - Use complex sentences that combine true and false elements to test careful reading and comprehension.
   - Incorporate statements that challenge common assumptions or misconceptions in the field.

Medium (20% of questions):
Multiple Choice Questions:
   - Include distractors that are plausible but distinguishable from the correct answer with careful thought.
   - Test application of concepts rather than just recall, but avoid overly complex scenarios.
   - Use clear, unambiguous language in both the question stem and answer choices.
   - Occasionally include "All of the above" or "None of the above" options, but not too frequently.

True/False Questions:
   - Create statements that require more than surface-level knowledge to evaluate.
   - Include some statements that have qualifiers (e.g., "always," "never," "sometimes") to test for exceptions.
   - Balance the number of true and false statements.

Easy (20% of questions):
Multiple Choice Questions:
   - Use straightforward language in both the question stem and answer choices.
   - Test basic recall of key concepts, definitions, or facts.
   - Make the correct answer clearly distinguishable from the distractors.
   - Limit the number of answer choices to 3-4 options.

True/False Questions:
   - Create clear, unambiguous statements about fundamental course concepts.
   - Avoid using absolutes like "always" or "never" unless they are definitively true or false.
   - Focus on testing recall of key facts or basic understanding of concepts.

Explanation Guidelines:
1. Provide a clear and concise explanation for why the correct answer is right.
2. If applicable, briefly explain why the other options are incorrect.
3. Include relevant facts, definitions, or concepts from the source material.
4. For harder questions, explain the reasoning or steps to arrive at the correct answer.
5. If the question involves calculations, show the key steps or formulas used.
6. Relate the explanation to the broader context or topic when appropriate.
7. Use simple language and avoid jargon unless it's essential to the subject matter.

Use the following text as context for generating questions, but only if it's relevant to {quiz_context}:
{pdf_text}
""",
        input_variables=["num_questions", "quiz_type", "quiz_context", "pdf_text"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    result = llm_chain.invoke({
        "num_questions": num_questions,
        "quiz_type": quiz_type,
        "quiz_context": quiz_context,
        "pdf_text": pdf_text
    })
    
    # Extract JSON from code block if present
    json_str = extract_json_from_codeblock(result['text'])
    # Parse the JSON output
    try:
        json_output = json.loads(json_str)
        return Quiz(**json_output)
    except json.JSONDecodeError:
        st.error("Failed to generate a valid quiz. Please try again.")
        return None

# Sidebar
st.sidebar.title("📚 About the Quiz Generator")

st.sidebar.write("Welcome to our exciting Adaptive Quiz Generator! This powerful tool creates customized quizzes based on your uploaded PDF content.")

st.sidebar.subheader("🚀 How to Use")
st.sidebar.markdown("""
1. Upload a PDF file
2. Enter the number of questions
3. Select quiz type (multiple-choice or true/false)
4. Enter the topic/context for the quiz
5. Click 'Generate Quiz' to start
""")

st.sidebar.subheader("✨ Key Features")
st.sidebar.markdown("""
- Adaptive difficulty based on your performance
- Blend of easy, medium, and hard questions
- Customizable quiz length and type
- Instant feedback and scoring
""")

st.sidebar.write("Get ready to challenge yourself and learn in an engaging way! 🎓💡")

# Main content
st.title("LLM Adaptive Quiz Generator 📚")
st.markdown("Upload a PDF to generate a quiz.")

pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    pdf_text = get_pdf_text([pdf_file])
    pdf_text = preprocess_text(pdf_text)
    st.session_state.pdf_text = pdf_text

    num_questions = st.number_input("Enter the number of questions for the quiz:", min_value=1, value=5)
    quiz_type = st.selectbox("Select quiz type:", ["multiple-choice", "true-false"])
    
    # Choose between specific topics or entire material
    mode = st.radio(
        "What do you want to be tested on?",
        ("Specific topics/concepts", "Entire material")
    )

    if mode == "Specific topics/concepts":
        quiz_context_input = st.text_input("Enter topic(s)/context(s) for the quiz (comma separated):")
    else:
        quiz_context_input = "entire material"
        st.info("Quiz will be generated from the entire PDF content.")

    if st.button("Generate Quiz"):
        if mode == "Specific topics/concepts":
            if quiz_context_input.strip():  # Check if input is not empty
                quiz_context_preprocessed = [concept.strip() for concept in quiz_context_input.split(',')]
                valid_concepts = validate_concepts_in_pdf(quiz_context_preprocessed, pdf_text)

                if valid_concepts:
                    for concept in valid_concepts:
                        st.success(f"Concept '{concept}' found in the PDF. Generating quiz for this concept.")
                    selected_context = valid_concepts[0]  # Use the first valid concept
                    generated_quiz = generate_questions_from_pdf(pdf_text, num_questions, quiz_type, selected_context)

                    if generated_quiz:
                        st.session_state.generated_quiz = generated_quiz
                        st.session_state.current_question = 0
                        st.session_state.score = 0
                        st.session_state.user_answers = []  # Track user answers
                        st.session_state.answer_submitted = False
                        st.rerun()  # Force rerun to display first question
                else:
                    for concept in quiz_context_preprocessed:
                        st.error(f"Concept '{concept}' not found in the PDF. Please enter valid topics.")
            else:
                st.error("Please enter at least one topic/context for the quiz.")
        
        else:  # mode == "Entire material"
            st.success("Generating quiz from the entire PDF content...")
            # Use a general context that encompasses the whole document
            selected_context = "all content from the document"
            generated_quiz = generate_questions_from_pdf(pdf_text, num_questions, quiz_type, selected_context)
            
            if generated_quiz:
                st.session_state.generated_quiz = generated_quiz
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.user_answers = []  # Track user answers
                st.session_state.answer_submitted = False
                st.rerun()  # Force rerun to display first question
            else:
                st.error("Failed to generate a quiz from the entire material. Please try again.")

# Quiz display section
if 'generated_quiz' in st.session_state and 'current_question' in st.session_state:
    quiz = st.session_state.generated_quiz
    current_question = st.session_state.current_question
    
    if current_question < len(quiz.questions):
        question = quiz.questions[current_question]
        
        # Display current question
        st.markdown(f"### Question {current_question + 1} of {len(quiz.questions)}")
        st.markdown(f"**{question.question}**")
        st.markdown(f"**Difficulty:** {question.difficulty}")

        # Display options
        if question.options:
            options = question.options
            selected_option = st.radio("Choose an option:", options, key=f"question_{current_question}")
        else:
            options = ["True", "False"]
            selected_option = st.radio("Choose:", options, key=f"question_{current_question}")

        # Initialize session states for question flow
        if 'answer_submitted' not in st.session_state:
            st.session_state.answer_submitted = False
        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = []

        # Reset answer_submitted for new question
        if len(st.session_state.user_answers) <= current_question:
            st.session_state.answer_submitted = False

        # Submit answer button - only show if answer not yet submitted
        if not st.session_state.answer_submitted:
            if st.button("Submit Answer", key=f"submit_{current_question}"):
                correct_answer = question.correct_answer
                if question.options:
                    selected_index = options.index(selected_option)
                    correct_index = ord(correct_answer.upper()) - ord('A')
                    is_correct = selected_index == correct_index
                else:
                    is_correct = (selected_option == "True" and correct_answer.upper() == "A") or \
                                 (selected_option == "False" and correct_answer.upper() == "B")

                # Store the user's answer
                st.session_state.user_answers.append({
                    'question_index': current_question,
                    'selected_option': selected_option,
                    'is_correct': is_correct
                })

                if is_correct:
                    st.session_state.score += 1

                st.session_state.answer_submitted = True
                st.rerun()
        
        # Show feedback and next button after answer is submitted
        if st.session_state.answer_submitted and len(st.session_state.user_answers) > current_question:
            user_answer = st.session_state.user_answers[current_question]
            
            if user_answer['is_correct']:
                st.success("Correct!")
            else:
                st.error("Incorrect!")
            
            st.markdown(f"**Explanation:** {question.explanation}")
            
            # Show next question button or finish quiz
            if current_question + 1 < len(quiz.questions):
                if st.button("Next Question", key=f"next_{current_question}"):
                    st.session_state.current_question += 1
                    st.session_state.answer_submitted = False  # Reset for next question
                    st.rerun()
            else:
                if st.button("Finish Quiz", key="finish_quiz"):
                    st.session_state.current_question += 1
                    st.rerun()

    else:
        # Quiz completed
        st.balloons()
        score = st.session_state.score
        total_questions = len(quiz.questions)
        percentage = (score / total_questions) * 100
        
        st.success(f"Quiz completed! Your score: {score}/{total_questions}")
        st.markdown(f"Percentage: {percentage:.2f}%")

        # Feedback based on percentage
        if percentage >= 90:
            st.markdown("""
            ### Outstanding Performance! 🌟
            
            You've demonstrated an exceptional understanding of the material. Your hard work and dedication are clearly paying off. Keep up this level of excellence, and you'll be well-prepared for any challenges ahead!
            """)
        elif percentage >= 80:
            st.markdown("""
            ### Great Job! 👏
            
            You're showing a strong grasp of the subject matter. Your performance is commendable, but there's still room to reach for that perfect score. Review the few questions you missed, and you'll be on your way to mastery!
            """)
        elif percentage >= 70:
            st.markdown("""
            ### Good Progress! 💪
            
            You're on the right track with a solid understanding of many key concepts. Focus on the areas where you had difficulties, and you'll see your scores improve even more. Keep pushing yourself!
            """)
        elif percentage >= 60:
            st.markdown("""
            ### Steady Improvement Needed 📈
            
            You've shown a basic understanding of the material, but there's definitely room for improvement. Don't get discouraged! Identify the areas where you struggled and dedicate some extra time to those topics. You've got this!
            """)
        elif percentage >= 50:
            st.markdown("""
            ### Keep Pushing Forward 🚀
            
            You're halfway there! While you've grasped some concepts, there are still significant areas that need work. Take this as an opportunity to strengthen your understanding. Review the material thoroughly, and don't hesitate to seek help if needed.
            """)
        else:
            st.markdown("""
            ### Time to Regroup and Refocus 🔄
            
            Don't lose heart! Everyone faces challenges when learning new material. This score indicates that you need to revisit the fundamentals of this topic. Consider the following steps:
            
            1. Review all the questions, especially those you got wrong.
            2. Re-read the source material and take detailed notes.
            3. Try explaining concepts to others or use memory techniques to reinforce your learning.
            4. Don't hesitate to ask for help or clarification on difficult topics.
            
            Remember, every expert was once a beginner. Your dedication to improvement is what matters most!
            """)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Review Answers"):
                st.session_state.show_review = True
                st.rerun()
        
        with col2:
            if st.button("Retake Quiz"):
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.user_answers = []
                st.session_state.answer_submitted = False
                if 'show_review' in st.session_state:
                    del st.session_state.show_review
                st.rerun()
        
        with col3:
            if st.button("Start New Quiz"):
                # Clear quiz-related session state but keep PDF
                keys_to_remove = ['generated_quiz', 'current_question', 'score', 'user_answers', 'show_review', 'answer_submitted']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Show review if requested
        if st.session_state.get('show_review', False):
            st.markdown("---")
            st.markdown("## Answer Review")
            for i, question in enumerate(quiz.questions):
                st.markdown(f"### Question {i+1}: {question.question}")
                st.markdown(f"**Difficulty:** {question.difficulty}")
                
                if question.options:
                    st.write("Options:")
                    for j, option in enumerate(question.options):
                        st.write(f"{chr(65+j)}. {option}")
                else:
                    st.write("Options: True / False")
                
                st.write(f"**Correct Answer:** {question.correct_answer}")
                
                # Show user's answer if available
                if 'user_answers' in st.session_state and i < len(st.session_state.user_answers):
                    user_answer = st.session_state.user_answers[i]
                    status = "✅ Correct" if user_answer['is_correct'] else "❌ Incorrect"
                    st.write(f"**Your Answer:** {user_answer['selected_option']} {status}")
                
                st.write(f"**Explanation:** {question.explanation}")
                st.markdown("---")

else:
    if pdf_file is not None:
        st.info("Please select your preferred testing mode and click 'Generate Quiz' to start.")
    else:
        st.info("Please upload a PDF file to begin.")