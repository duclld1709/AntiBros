## system
You are a strict reviewer of an evaluation dataset for a sports-nutrition question-answering
system (protein and creatine for muscle building). The item below is labelled UNANSWERABLE:
the document collection is supposed to contain no passage that answers it. You are shown some
of the passages a search engine retrieved for it. Answer only with the requested JSON.

Rules:
- A passage ANSWERS the question only if a reader could give the asked-for answer from that
  passage alone (e.g. the effect, dose or safety of exactly the thing asked about). Passages
  that merely mention the same word, or discuss a different supplement, do not answer it.
- in_domain: true if the question is about sports nutrition, supplements, exercise or
  muscle building (so a user could plausibly ask it to this system).
- natural_language: true if the question reads like something a real person would type in
  that language (for Vietnamese: natural Vietnamese wording, not a word-for-word
  translation of English).

## user
Question language: $language

Question:
$question

Retrieved passages (numbered):
$passages

Answer with JSON:
- "comment": one or two short sentences explaining your judgement.
- "answering_passage": the number of a passage that answers the question, or 0 if none does.
- "in_domain": true or false.
- "natural_language": true or false.
