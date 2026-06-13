import json
import re

# ----------------- PROMPT TEMPLATES -----------------

QA_SYSTEM_PROMPT = """
You are a highly capable AI assistant helping a user study and extract information from their uploaded documents.

Guidelines:
1. Answer the question precisely based on the provided context.
2. Structure your response using clean, readable markdown (bullet points, bold text, lists, and tables where appropriate).
3. If the answer cannot be found in the provided context, state that you couldn't find the answer in the document, but provide a helpful response using your general knowledge, clearly labeling the general knowledge section as:
   ---
   *Note: The following information is based on general knowledge, as it was not explicitly found in the uploaded documents:*
4. Keep your tone professional, helpful, and concise.
"""

QA_USER_PROMPT = """
Context:
{context}

Question: {query}
"""

SUMMARY_PROMPT = """
You are an expert technical writer and educator.
Please generate a comprehensive, visually appealing summary of the provided text.

Format your output in clean Markdown with the following sections:
1. 📌 **Executive Overview**: A 3-4 sentence high-level summary.
2. 🔑 **Key Takeaways**: Bulleted list of the most critical points.
3. 📖 **Core Concepts**: A table or glossary listing key terms, definitions, and their relevance.
4. 💡 **Summary Assessment**: A brief paragraph on the overall significance of this content.

Content:
{text}
"""

REVISION_NOTES_PROMPT = """
You are an elite study assistant. Your goal is to create high-yield, structured revision notes from the text below.
Create a cheat-sheet style outline that is easy to skim.

Format requirements:
- Use clear headings and subheadings.
- Use emojis for bullet points to make it visually engaging (e.g. 🎯, 🚀, 🔍, ⚠️).
- Include a "Quick Tip" blockquote using Markdown blockquotes.
- Format any technical formula, code, or structured sequence cleanly.

Content:
{text}
"""

QUIZ_PROMPT = """
You are an expert exam creator. Analyze the following content and generate exactly 5 multiple choice questions (MCQs) based on it.
You MUST respond with a valid JSON array of objects and NOTHING ELSE. Do not explain your output. Do not wrap it in markdown code blocks like ```json ... ``` if possible, but if you do, keep it valid JSON inside.

The JSON array must contain objects matching this exact structure:
[
  {{
    "question": "A clear, challenging question text?",
    "options": [
      "Option A text",
      "Option B text",
      "Option C text",
      "Option D text"
    ],
    "answer": "Option B text",
    "explanation": "A concise explanation of why the correct option is right based on the text."
  }}
]

Make sure:
1. The "answer" string matches exactly one of the strings inside the "options" list.
2. The options are realistic but distinct.
3. The explanation is instructive.

Content:
{text}
"""

# ----------------- HELPERS -----------------

def clean_and_parse_json(text):
    """
    Cleans markdown wrapping (like ```json ... ```) from a text response
    and parses it as JSON. Returns parsed object, or raises exception.
    """
    cleaned = text.strip()
    # Remove markdown code block markers if present
    if cleaned.startswith("```"):
        # Match code blocks: ```json ... ``` or ``` ... ```
        match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            
    # Try parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # If standard parsing fails, try finding any JSON array in the text
        array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError("Failed to parse response as a valid JSON array of questions.")
