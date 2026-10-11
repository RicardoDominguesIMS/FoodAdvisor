# Recipe RAG Resources

Place the files used as FoodAdvisor's recipe knowledge base in this folder. Supported formats:

- `.txt` and `.md` files containing one or more recipes;
- `.json` files containing a recipe object or a list of recipes;
- `.pdf` files with selectable text.

For best results, include the dish name, ingredients, and sub-ingredients of sauces or compound preparations. In JSON, `name` or `title` fields are used as the title; all document fields are indexed.

Each complete recipe is indexed as one chunk: a recipe is not split by size, and content from neighboring recipes is not combined with it. Markdown uses level-one headings (`#`); plain text and PDFs support recipe titles followed by structural metadata, such as a standalone or formatted preparation-time header, serving information, and ingredient/method sections. PDF parsing tries format-specific parsers in sequence: a title plus preparation-time header, the Ladymaxx layout with recipe titles followed by “Ingredientes para…”, then structured recipe-page headings. This supports multiple Ladymaxx recipes on one page while keeping identifiable continuations with their recipe. A time mentioned in an instruction or ingredient line is not a recipe boundary. A JSON list of objects with `name` or `title` contains one recipe per object. In PDFs, pages with no extractable text are ignored; OCR is not used. Text pages that do not contain identifiable recipe content are also skipped. A PDF with no extractable recipe chunks contributes no recipes to the index, and the rebuild response reports that resource as ignored. PDFs are extracted with layout preserved, and table-of-contents pages with dotted leader lines are excluded. Search combines multilingual embeddings through FastEmbed/ONNX CPU (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) with exact lexical match boosting and returns complete recipes only; it does not append adjacent chunks. The model is downloaded on first use. The persistent vector index is stored in `backend/.rag_index/` and rebuilt when resource contents or the indexing format changes. This README is not indexed.

Retrieved content is supporting context, not a guarantee that a recipe is allergen-free. If a recipe or ingredient is undocumented, the application must communicate uncertainty.
