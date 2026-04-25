def get_context_aware_translation(source_text: str, source_language: str, target_language: str) -> str:
    return f"""You are a professional translator. Translate the following {source_language} text to {target_language}. Ensure the translation is contextually accurate and grammatically correct.


Return ONLY a valid JSON object with this exact structure:
{{
    "translated_text": "The translated text in {target_language}"
}}
"""
# Example
# if __name__ == "__main__":
#     sample_bangla = "আমি ভাত খাই।"  # "I eat rice."
#     prompt = get_context_aware_translation(sample_bangla, "Bangla", "English")
#     print(prompt)