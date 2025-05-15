COT2KG_PROMPT = """
You are an expert at extracting factual information in structured formats to build a knowledge graph.

Step 1 - Entity detection: Identify all entities in the raw text. Make sure not to miss any out. Entities should be basic and simple, they are akin to Wikipedia nodes.

Step 2 - Coreference resolution: Find all expressions in the text that refer to the same entity. Make sure entities are not duplicated. In particular, do not include entities that are more specific versions themselves, e.g., "a detailed view of Jupiter's atmosphere" and "Jupiter's atmosphere" - only include the most specific version of the entity.

Step 3 - Relation extraction: Identify semantic relationships between the entities you have identified.

Format: Return the knowledge graph as a list of triples, i.e., ["entity 1", "relation 1-2", "entity 2"], in Python code. Make sure the triples are not redundant, even when the input raw text contains redundant information. "relation 1-2" is usually a single word or extremely short phrase which carries the entire meaning of how the two entities are related. For example: ASSOCIATES, IN, ISA, INCREASEDIN, MENTIONED, etc are all good, meaniningful relations. An example of a BAD triple is ["ATG5", "MUTATIONS", "no known or likely link with psoriasis"] - the equivalent good triple is ["ATG5 MUTATIONS", "no link", "psoriasis"]

---

Use the given format to extract information from the following input:

<input> {input} </input>

Skip any preamble and output ONLY the result inside <python> </python> tags.

---

Important Tips:
1. Make sure all information is included in the knowledge graph.
2. Each triple must only contain three strings! None of the strings should be empty.
3. Do not split up related information into separate triples because this could change the meaning.
4. Make sure all brackets and quotation marks are matched.
5. Before adding a triple to the knowledge graph, check the concatenated triple makes sense as a sentence. If not, discard it.

---

Example 1.
Input:
"The Walt Disney Company, commonly known as Disney, is an American multinational mass media and entertainment conglomerate that is headquartered at the Walt Disney Studios complex in Burbank, California."

Output:
<python>
[
  ["The Walt Disney Company", "headquartered at", "Walt Disney Studios complex in Burbank, California"],
  ["The Walt Disney Company", "commonly known as", "Disney"],
  ["The Walt Disney Company", "instance of", "American multinational mass media and entertainment conglomerate"]
]
</python>

---

Example 2.
Input:
"Amanda Jackson was born in Springfield, Ohio, USA on June 1, 1985. She was a basketball player for the U.S. women's team."

Output:
<python>
[
  ["Amanda Jackson", "born in", "Springfield, Ohio, USA"],
  ["Amanda Jackson", "born on", "June 1, 1985"],
  ["Amanda Jackson", "occupation", "basketball player"],
  ["Amanda Jackson", "played for", "U.S. women's basketball team"]
]
</python>

---

Example 3.
Input:
"Music executive Darius Van Arman was born in Pennsylvania. He attended Gonzaga College High School and is a human being."

Output:
<python>
[
  ["Darius Van Arman", "occupation", "Music executive"],
  ["Darius Van Arman", "born in", "Pennsylvania"],
  ["Darius Van Arman", "attended", "Gonzaga College High School"],
  ["Darius Van Arman", "instance of", "human being"]
]
</python>

Example 4.
Input:
"To identify the gene associated with both herpes zoster and psoriatic arthritis, I first examined the associations provided in the context. The context explicitly states that HLA-B is associated with herpes zoster, as well as being directly linked to psoriatic arthritis. Other genes provided in the list (ADGRV1, CPS1, SULT1B1, ATG5) do not have associations mentioned in the context for either disease. Thus, HLA-B is the only gene that fits the criteria of being associated with both conditions."

Output:
<python>
[
  ["herpes zoster", "ASSOCIATES", "HLA-B"],
  ["psoriasis", "ASSOCIATES", "HLA-B"],
  ["herpes zoster", "DOES NOT ASSOCIATE", "ADGRV1"],
  ["psoriasis", "DOES NOT ASSOCIATE", "ADGRV1"],
  ["herpes zoster", "DOES NOT ASSOCIATE", "CPS1"],
  ["psoriasis", "DOES NOT ASSOCIATE", "CPS1"],
  ["herpes zoster", "DOES NOT ASSOCIATE", "SULT1B1"],
  ["psoriasis", "DOES NOT ASSOCIATE", "SULT1B1"],
  ["herpes zoster", "DOES NOT ASSOCIATE", "ATG5"],
  ["psoriasis", "DOES NOT ASSOCIATE", "ATG5"]
]
</python>

Example 5.
Input:
"HLA-B is associated with both herpes zoster and psoriatic arthritis as per the provided context. This gene is mentioned specifically in both cases, indicating its relevance in the pathogenesis of these diseases. The other genes in the list (ADGRV1, CPS1, SULT1B1, ATG5) do not appear in the contexts associated with herpes zoster or psoriatic arthritis, making them unsuitable answers. Therefore, HLA-B is the correct choice, as it is the only gene common to both lists."

GOOD Output:
<python>
["HLA-B", "ASSOCIATES", "herpes zoster"],
["HLA-B", "ASSOCIATES", "psoriatic arthritis"],
["herpes zoster", "DOES NOT ASSOCIATE", "ADGRV1"],
["psoriatic arthritis", "DOES NOT ASSOCIATE", "ADGRV1"],
["herpes zoster", "DOES NOT ASSOCIATE", "CPS1"],
["psoriatic arthritis", "DOES NOT ASSOCIATE", "CPS1"],
["herpes zoster", "DOES NOT ASSOCIATE", "SULT1B1"],
["psoriatic arthritis", "DOES NOT ASSOCIATE", "SULT1B1"],
["herpes zoster", "DOES NOT ASSOCIATE", "ATG5"],
["psoriatic arthritis", "DOES NOT ASSOCIATE", "ATG5"]
</python>

BAD Output:
<python>
["HLA-B", "ASSOCIATES", "herpes zoster"],
["HLA-B", "ASSOCIATES", "psoriatic arthritis"],
["herpes zoster", "DOES NOT ASSOCIATE", "ADGRV1"],
["psoriatic arthritis", "DOES NOT ASSOCIATE", "ADGRV1"],
["herpes zoster", "DOES NOT ASSOCIATE", "CPS1"],
["psoriatic arthritis", "DOES NOT ASSOCIATE", "CPS1"],
["herpes zoster", "DOES NOT ASSOCIATE", "SULT1B1"],
["psoriatic arthritis", "DOES NOT ASSOCIATE", "SULT1B1"],
["herpes zoster", "DOES NOT ASSOCIATE", "ATG5"],
["psoriatic arthritis", "DOES NOT ASSOCIATE", "ATG5"],
["HLA-B", "common to", "both lists"]
</python>
"""
