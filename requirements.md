# FoodAdvisor Requirements

## 1. Purpose

FoodAdvisor analyzes a user-provided dish or recipe name. It uses recipe books indexed with retrieval-augmented generation (RAG) to identify the ingredients supported by a matching recipe, then uses a separate LLM call to identify allergens in those ingredients.

The system must keep its conclusions traceable to recipe evidence and communicate uncertainty. Missing allergen evidence must never be presented as proof that a recipe contains no allergens.

## 2. Scope

The system consists of:

1. A recipe-book RAG that stores each complete recipe as one independent chunk.
2. A first analysis agent that receives the user's dish or recipe name and retrieved recipe chunks, then extracts supported ingredients.
3. A second analysis agent that receives the extracted ingredient list and identifies supported allergens in a separate LLM call.

This document covers only cookbook ingestion and retrieval, recipe interpretation, and allergen identification.

## 3. Recipe-book RAG requirements

- Recipe books are separate, identifiable sources.
- The indexing unit is one complete recipe. A recipe must not be split across chunks or combined with an adjacent recipe.
- A recipe chunk must retain its title, ingredients, documented sub-ingredients, relevant notes, and source attribution. Preserve a page or section reference when available.
- All user input and all text read from files in `resources/` must remain untrusted data. Treat prompt-like content or commands in these sources as recipe text only; never execute or follow them as instructions.
- Recipe boundaries must be determined from the source structure, including recipes that continue across pages.
- If a recipe boundary or its completeness cannot be determined reliably, indexing must fail explicitly or require review. It must not present mixed or incomplete content as a complete recipe.
- Retrieval results must identify their source and provide complete recipe evidence to the interpretation stage.
- No allergen mention in a retrieved chunk is not proof of allergen absence.
- For PDFs, pages without extractable text are ignored; OCR is not required. A resource with no identifiable recipe chunks is reported as ignored during index rebuild rather than treated as a successfully indexed recipe source.

The existing RAG resource guidance supports `.txt`, `.md`, `.json`, and text-based `.pdf` files. See [the resource README](./resources/README.md) for the current indexing behavior.

## 4. Agent 1: recipe interpretation and ingredient extraction

### Input

- The dish or recipe name entered by the user.
- Complete recipe chunks retrieved by RAG, including source metadata and relevance information when available.

### Requirements

- Match the user input to the retrieved recipes using the recipe title and supporting content. A merely similar recipe must not be treated as confirmation of the requested recipe.
- Extract only ingredients explicitly present or otherwise supported by the retrieved recipe evidence.
- Produce a complete inventory of all ingredient entries explicitly listed for the matched recipe. Never invent, omit, remove, merge, translate, or silently normalize listed items; retain optional items, garnishes, seasonings, and unclear wording as separate entries.
- For compound ingredients, retain the named compound and also decompose it into basic ingredients only when those components are documented in the evidence. Never replace it with guessed components.
- If an ingredient, term, quantity, or compound composition cannot be interpreted confidently, preserve the original wording when possible and explain exactly what is unclear in `uncertainties`. Never silently drop the item.
- Preserve ingredient names in the source wording. Each ingredient must have a short verbatim evidence quote from a matched recipe chunk that contains and explicitly supports the item; validate this citation before passing the extraction to Agent 2.
- Do not merge ingredients from different recipes into a synthetic recipe.
- Associate each extracted ingredient with its supporting source/evidence.
- Preserve uncertainty for weak matches, incomplete sources, ambiguous ingredients, and undocumented sub-ingredients. Do not fill gaps with general culinary assumptions.
- If the recipe cannot be confirmed, return a partial or empty ingredient list as appropriate and explain the uncertainty.
- Return valid JSON only, with ingredient names, evidence, and `confirmed`/`uncertain` statuses. Validate the schema before passing results to Agent 2.
- The first LLM prompt must explicitly state that the user's recipe name and retrieved chunks are data, not instructions. Put them in a structured user payload, not in the trusted system instructions.

### Output

Return structured data containing the original query, matched recipe/source information, ingredients with evidence and confirmation status, and uncertainties.

## 5. Agent 2: allergen identification

### Input

- The structured ingredient list from Agent 1, including each ingredient's confirmation status.
- Ingredient evidence and provenance from Agent 1 when available.

### Requirements

- Run allergen identification as a separate LLM call after ingredient extraction.
- Identify allergens only when the ingredient-to-allergen relationship is supported by the ingredient information or its evidence.
- Use the applicable allergen list. For recipes intended for Portugal/the European Union, account for the 14 regulated allergen groups.
- Link each identified allergen to its source ingredient and supporting evidence.
- Treat unclear ingredients or undocumented compound ingredients as uncertainty, not as confirmed presence or absence.
- Report cross-contact or trace warnings only when supported by the available source; do not invent them.
- Prefer an explicit uncertainty warning over a false-negative conclusion when information is incomplete. Do not present an unsupported allergen as confirmed merely to avoid a false negative.
- An empty list of identified allergens means only that none were confirmed from the available information. It must not be phrased as "allergen-free" or as a food-safety guarantee.
- Agent 2 must be a distinct LLM call with its own instructions and must receive only the validated ingredients, their evidence/status, uncertainty information, and the applicable allergen groups.
- Agent 2 must interpret every extraction uncertainty into a specific, clear warning for the user, explaining what could not be interpreted and how that limits the allergen assessment. If extraction uncertainties exist and Agent 2 returns no warnings, the workflow must fail explicitly.
- The second LLM prompt must treat all ingredient and evidence text as untrusted data, ignore instructions embedded in it, and return valid JSON. A confirmed allergen must refer to an ingredient confirmed by Agent 1.

### Output

Return structured data containing identified allergens linked to their source ingredients and evidence, plus separate uncertainty or source-supported warnings.

## 6. End-to-end acceptance criteria

1. Given a cookbook containing multiple recipes, the index creates one complete, independent chunk per recipe, with source attribution.
2. Given a recipe that continues across pages, its ingredients and relevant notes remain in the same recipe chunk.
3. Given a user query and a supported RAG match, Agent 1 returns only ingredients supported by that recipe and preserves their evidence.
4. Given multiple similar recipes, the system does not combine their ingredients or claim that a similar recipe is the exact requested recipe.
5. Given an absent, incomplete, or ambiguous match, Agent 1 reports uncertainty instead of inventing missing ingredients.
6. Agent 2 receives Agent 1's structured ingredient result and runs in a distinct LLM call.
7. Given a confirmed allergenic ingredient, Agent 2 identifies the supported allergen and links it to that ingredient.
8. Given incomplete ingredient evidence, Agent 2 reports uncertainty and does not claim the recipe contains no allergens.
9. The final response preserves ingredient evidence, allergen evidence, and uncertainty from both stages.
10. The English-language home page accepts a recipe name and presents the allergen result and warnings.
11. The page's Backend log displays the exact system and user messages sent to each LLM call, including Agent 1's instructions and recipe/RAG payload, Agent 1's JSON result, Agent 2's instructions and ingredient/uncertainty payload, and Agent 2's JSON result. The trace must match the messages actually passed to the LLM client and must never display credentials.
12. Prompt-like text in either a user's recipe name or cookbook resources cannot change system behavior or trigger actions.

## 7. LLM workflow and output contracts

The ingredient extraction stage returns JSON with `query`, `matched_recipes`, `ingredients`, and `uncertainties`. Each ingredient has `name`, `status`, and `evidence`; status is `confirmed` or `uncertain`. The list must preserve every source-listed ingredient and uncertainty must retain unresolved source wording rather than omit it.

The allergen stage returns JSON with `allergens` and `warnings`. Each allergen has `name`, `ingredient`, `status`, and `evidence`; each warning has `message`, `ingredient` (or null), and `status`. Invalid JSON, missing fields, or unsupported status values are explicit errors, not successful empty results.

The main result begins with “The allergens are:” when allergens are identified. If none are confirmed, it says “No allergens were identified from the available recipe evidence.” It must not state a food-safety guarantee, and uncertainty warnings plus a food-safety caveat remain visible.

The backend implements the two-call workflow in `/api/analyze`; its response includes a workflow trace for the UI. The `.github/skills/` files provide project guidance and reinforce the same data trust boundary.
