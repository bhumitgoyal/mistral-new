from flask import Flask, request, jsonify
import os
import requests
import logging
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# OpenRouter API for Mistral
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

@app.route('/analyze-clauses', methods=['POST'])
def analyze_clauses():
    """
    Endpoint to analyze legal clauses using Mistral model via OpenRouter.
    Accepts a POST request with JSON containing legal text and
    returns categorized clauses as "good" or "bad".
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'legal_text' not in data:
            return jsonify({"error": "Missing 'legal_text' in request body"}), 400
        
        legal_text = data['legal_text']
        
        # Validate input
        if not legal_text or not isinstance(legal_text, str) or legal_text.strip() == "":
            return jsonify({"error": "Invalid or empty legal text provided"}), 400
            
        # Call Mistral via OpenRouter API to analyze the text
        analysis_result = analyze_with_mistral(legal_text)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def analyze_with_mistral(legal_text):
    """
    Send the legal text to Mistral via OpenRouter for analysis and process the response.
    
    Args:
        legal_text (str): The legal text to analyze
        
    Returns:
        dict: Dictionary containing good_clausess and bad_clausess lists
    """
    try:
        # Check if OpenRouter API key is set
        if not OPENROUTER_API_KEY:
            logger.error("OpenRouter API key not found in environment variables")
            return {
                "error": "OpenRouter API key not configured",
                "good_clausess": [],
                "bad_clausess": []
            }
            
        # Prepare the system and user messages for Mistral
        system_message = "You are a legal expert who analyzes contract clauses."
        
        user_message = f"""
        Analyze the following legal text and categorize each clause as either "good" or "bad" for a typical contract participant.
        Good clauses are those that are fair, balanced, and protect both parties' interests.
        Bad clauses are those that are one-sided, potentially exploitative, or have hidden implications.
        
        Format your response as JSON with exactly two keys:
        1. "good_clausess": A list of strings, where each string is a good clause
        2. "bad_clausess": A list of strings, where each string is a bad clause
        
        Respond only with the JSON object, no additional text or explanation.
        
        Legal text to analyze:
        {legal_text}
        """
        
        # Prepare the request payload
        payload = {
            "model": "google/gemini-2.5-pro-exp-03-25:free",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3
        }
        
        # Make the API request to OpenRouter
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            response_content = response_data["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            json_text = extract_json(response_content)
            result = json_text if isinstance(json_text, dict) else json.loads(json_text)
            
            # Ensure the expected keys exist
            if "good_clausess" not in result or "bad_clausess" not in result:
                logger.warning("Mistral response missing expected keys")
                return {
                    "good_clausess": result.get("good_clausess", []),
                    "bad_clausess": result.get("bad_clausess", [])
                }
                
            return {
                "good_clausess": result.get("good_clausess", []),
                "bad_clausess": result.get("bad_clausess", [])
            }
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return {
                "error": f"API request failed: {response.status_code}",
                "good_clausess": [],
                "bad_clausess": []
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": f"Request error: {str(e)}", "good_clausess": [], "bad_clausess": []}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing Mistral response: {str(e)}")
        return {"error": "Failed to parse AI response", "good_clausess": [], "bad_clausess": []}
    except Exception as e:
        logger.error(f"Error in analyze_with_mistral: {str(e)}")
        return {"error": f"Analysis error: {str(e)}", "good_clausess": [], "bad_clausess": []}

def extract_json(text):
    """Extract JSON content from the LLM response"""
    try:
        import re
        import ast
        
        # Remove markdown code block indicators if present
        text = re.sub(r"```(?:json)?", "", text).strip()
        
        # Try to find JSON content between curly braces
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            cleaned = match.group().replace('\n', ' ').replace('\r', '')
            return json.loads(cleaned)
        
        # As a fallback, try literal evaluation
        return ast.literal_eval(text)
    except Exception as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {
            "good_clausess": [],
            "bad_clausess": [],
            "error": "Could not extract valid JSON from the model response."
        }

if __name__ == '__main__':
    # Check if OpenRouter API key is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.warning("OPENROUTER_API_KEY environment variable is not set!")
        print("Warning: OPENROUTER_API_KEY environment variable is not set!")
        print("Set it before running the application: export OPENROUTER_API_KEY='your-api-key'")
    
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment, default to 5000
    app.run(host='0.0.0.0', port=port)
