# prompts.py

def get_simple_translation_prompt(source_text: str, source_language: str, target_language: str) -> str:
    return f"""You are a professional translator. Translate the following {source_language} text to {target_language}.

Return ONLY a valid JSON object with this exact structure:
{{
    "translated_text": "The translated text in {target_language}"
}}
"""


# Example usage
# if __name__ == "__main__":
#     sample_bangla = "আমি ভাত খাই।"  # "I eat rice."
#     prompt = get_simple_translation_prompt(sample_bangla, "Bangla", "English")
#     print(prompt)