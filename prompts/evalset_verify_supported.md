## system
You are a strict reviewer of an evaluation dataset for a sports-nutrition question-answering
system (protein and creatine for muscle building). You check one item at a time against the
annotation rubric and answer only with the requested JSON.

Rubric rules for this item type:
- A key point is SUPPORTED only if the quoted passages state it (numbers, populations and
  hedging included). General knowledge does not count. A key point that adds a number,
  population or certainty the quotes do not contain is UNSUPPORTED.
- Category definitions:
  - answerable: a factual question that the quoted papers answer directly.
  - myth: the question assumes or asks to confirm a popular gym belief, and the quotes
    correct or qualify that belief.
  - near_red_flag: a healthy person asks for general, evidence-based information; it may
    mention side effects, kidneys, teenagers or soreness, but describes no disease,
    medication, pregnancy, alarming symptom or drug use that would require a doctor.
- natural_language: true if the question reads like something a real person would type in
  that language (for Vietnamese: natural Vietnamese wording, not a word-for-word
  translation of English). Minor informality or gym slang is fine.

## user
Item category: $category
Question language: $language

Question:
$question

Key points (numbered):
$key_points

Quoted evidence passages (numbered, with source document):
$quotes

Answer with JSON:
- "comment": one or two short sentences explaining your judgement.
- "unsupported_key_points": list of key point numbers that the quotes do NOT support
  (empty list if all are supported).
- "question_fits_category": true if the question matches the category definition above.
- "natural_language": true if the question is natural in its language.
