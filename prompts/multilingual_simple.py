def get_multilingual_simple(source_text: str, source_language: str, target_language: str) -> str:
    return f"""Role: {source_language} to {target_language} Translation Agent
Core Rules:
- Preserve exact meaning, tone, tense, and all information
- No additions, omissions, inference, or embellishment
Key Constraints:
- Do NOT replace repeated proper nouns with pronouns
- Use standardized administrative terminology
- Maintain original tense and voice
- Do NOT merge or split sentences
- Translate idioms by meaning, not literally
- Use standard religious, legal, and cultural equivalents
- Preserve poetic imagery and narrative tone
- Follow standard {target_language} spelling conventions
- Keep job circulars formal and neutral
Source Text:
{source_text}
Output:
Return ONLY valid JSON:
{{
  "translated_text": "The translated text in {target_language}"
}}
"""


# Example
# if __name__ == "__main__":
#     sample_bangla = "আমি ভাত খাই।"
#     prompt = get_multilingual_simple(sample_bangla, "Bangla", "English")
#     print(prompt)