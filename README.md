# Chatterdoc_Bot


*ChatterDoc_Bot* is an intelligent multi-PDF assistant powered by Google Gemini and LangChain. It allows you to:

- Upload and analyze multiple PDFs
- Ask natural questions from PDF content in multiple languages
- Automatically generate quizzes from uploaded documents
  

  How to Run


1. Clone the repo
git clone <your-repo-link>
cd chatterdoc

2. Install dependencies
pip install -r requirements.txt

3. Add your API key in a .env file
echo GOOGLE_API_KEY=your_google_key_here > .env

4. Run the app
streamlit run app.py
