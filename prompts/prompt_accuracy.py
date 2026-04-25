def get_accuracy_prompt(source_text: str, source_language: str, target_language: str) -> str:
    return f"""You are an expert linguist. Translate the following {source_language} text to {target_language} with high accuracy. 
    Preserve the original tone and meaning.

Return ONLY a valid JSON object with this exact structure:
{{
    "translated_text": "The translated text in {target_language}"
}}
"""

# Example
# if __name__ == "__main__":
#     sample_bangla = "আমি ভাত খাই।"  # "I eat rice."
#     prompt = get_accuracy_prompt(sample_bangla, "Bangla", "English")
#     print(prompt)