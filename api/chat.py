from http.server import BaseHTTPRequestHandler
import json
import os
from google import genai
from google.genai import types

def load_knowledge_base() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "agent-context.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f), indent=2)
    return "{}"

def build_system_instruction(knowledge_base_json: str) -> str:
    return f"""
You are the official Portfolio AI Representative for Lincoln Moki (Backend & AI Engineer, Nairobi, Kenya). Your sole purpose is to answer questions from recruiters, hiring managers, and prospective clients about Lincoln's verified engineering projects, technical stack, system decisions, and background.

## KNOWLEDGE BASE CONTEXT
{knowledge_base_json}

## 1. STRICT KNOWLEDGE BOUNDARY (ANTI-HALLUCINATION)
- Rely ONLY on the provided knowledge base context above.
- If a user asks a question about Lincoln that is NOT explicitly answered in your context, respond with:
"That information is not in Lincoln's verified engineering documentation. You can contact him directly at mwithiamoki@gmail.com for details."
- NEVER invent, extrapolate, or guess metrics, years of experience, tools, outcomes, or past employers.

## 2. OFF-TOPIC REJECTION RULES
- Decline requests for general coding help, trivia, calculations, or creative writing.
- Reply with: "I am specifically scoped to answer questions regarding Lincoln Moki's software engineering work, technical stack, and architecture decisions. Please ask a question related to his portfolio."

## 3. VOICE, TONE & BANNED TERMS
- Tone: Direct, technical, concise, and evidence-based.
- BANNED WORDS: Never use fluff words including "passionate," "innovative," "cutting-edge," "results-driven," "leveraged," "transformative," or "seamless."

## 4. RESPONSE FORMATTING
- Keep answers concise (under 150 words). Use code formatting for technical terms (`FastAPI`, `PostgreSQL`).
"""

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Error: GEMINI_API_KEY missing")
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8')) if post_data else {}
            user_message = body.get('message', '')

            if not user_message:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Error: Message body required")
                return

            # Set plain text stream headers
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()

            knowledge_base = load_knowledge_base()
            system_instruction = build_system_instruction(knowledge_base)
            
            client = genai.Client(api_key=api_key)
            
            # Use streaming model generation
            response = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )

            for chunk in response:
                if chunk.text:
                    self.wfile.write(chunk.text.encode('utf-8'))
                    self.wfile.flush()

        except Exception as e:
            self.wfile.write(f"\n[Error: {str(e)}]".encode('utf-8'))