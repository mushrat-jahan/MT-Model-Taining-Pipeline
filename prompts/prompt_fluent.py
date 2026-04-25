def get_accuracy_fluent(source_text: str, source_language: str, target_language: str) -> str:
    return f"""You are a professional translator. Traslate the following {source_language} text to {target_language} naturally and fluently as a native speaker would, while preserving the original tone and meaning.

    Return ONLY a valid JSON object with this exact structure:
{{
    "translated_text": "The translated text in {target_language}"
}}
"""


# Example usage
if __name__ == "__main__":
    sample = "I eat rice."  # "আমি ভাত খাই।"
    prompt = get_accuracy_fluent(sample, "English", "Bangla")
    print(prompt)





