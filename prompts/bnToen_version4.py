def get_SOP_prompt_version4(source_text: str, source_language: str, target_language: str) -> str:
    return f""" Role: {source_language} to{target_language} Translation Agent

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
- Follow British English spelling
- Keep job circulars formal and neutral

Example (Proper Nouns):
{source_language}: দুস্থ শিশুদের, নিজের জামা দিয়ে ট্রেন কম্পার্টমেন্ট পরিষ্কার করতে দেখে, তিনি অন্যদের সাহায্য করার জন্য নিজের কণ্ঠস্বর ব্যবহার করার সিদ্ধান্ত নিয়েছিলেন। প্রায় একই সময়, ইন্দোরের নিধি বিনয় মন্দিরের শিক্ষকেরা মুছলের কাছে আসেন।
Avoid {target_language}: Seeing destitute children cleaning train compartments with their clothes, Muchhal decided to use her voice to help others. Around the same time, teachers from the Nidhi Vinay Mandir in Indore approached her.
Use {target_language}: Seeing destitute children cleaning train compartments with their clothes, she decided to use her voice to help others. Around the same time, teachers from Nidhi Vinay Mandir in Indore approached Muchhal.


Output:
Return ONLY valid JSON:
{{
  "translated_text": "The translated text in {target_language}"
}}
"""
# Example
# if __name__ == "__main__":
#     sample_bangla = "আমি ভাত খাই।"
#     prompt = get_SOP_prompt_version4(sample_bangla, "Bangla", "English")
#     print(prompt)
