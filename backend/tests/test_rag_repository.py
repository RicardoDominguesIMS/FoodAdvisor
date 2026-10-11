import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from foodadvisor.rag_repository import FastEmbedder, RecipeRAGRepository


class KeywordTestEmbedder:
    TERMS = ("pasta", "pesto", "frango", "salada")

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vector = np.array(
                [
                    float(term in normalized or (term == "pasta" and "massa" in normalized))
                    for term in self.TERMS
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.stack(vectors)


class RecipeRAGRepositoryTests(unittest.TestCase):
    def test_fast_embedder_does_not_start_multiprocessing_workers(self) -> None:
        model = Mock()
        model.embed.return_value = iter([np.array([3.0, 4.0], dtype=np.float32)])
        embedder = FastEmbedder.__new__(FastEmbedder)
        embedder.model = model

        vectors = embedder.encode(["receita"])

        model.embed.assert_called_once_with(
            ["receita"], batch_size=32, parallel=None
        )
        np.testing.assert_allclose(vectors, [[0.6, 0.8]])

    def test_markdown_indexes_each_complete_recipe_without_neighboring_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "resources"
            resources.mkdir()
            first_recipe = (
                "# Pães com vegetais\n"
                "Ingredientes: cenoura e courgette.\n"
                + "Preparação: misture e leve ao forno. " * 30
            )
            second_recipe = (
                "\n\n# Pão de milho\n"
                "Ingredientes: farinha de milho e alecrim."
            )
            (resources / "paes.md").write_text(
                first_recipe + second_recipe, encoding="utf-8"
            )
            repository = RecipeRAGRepository(
                resources_dir=resources,
                index_dir=root / "index",
                embedder=KeywordTestEmbedder(),
                min_similarity=0.0,
            )

            matches = repository.search("Pães com vegetais")

            self.assertEqual(len(repository.chunks), 2)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["title"], "Pães com vegetais")
            self.assertGreater(len(matches[0]["content"]), 500)
            self.assertIn("courgette", matches[0]["content"])
            self.assertNotIn("Pão de milho", matches[0]["content"])
            self.assertNotIn("alecrim", matches[0]["content"])

    def test_pdf_recipe_boundaries_keep_adjacent_recipes_separate(self) -> None:
        pages = [
            (
                23,
                "Pães com vegetais\n40 SEG. | 12 UNIDADES\n"
                "100g cenoura\n50g courgette\nMétodo com courgette.\n\n"
                "Pão de milho\n7 MIN. 10 SEG. | 535G\n"
                "200g farinha de milho\n",
            ),
            (24, "Método do pão de milho com alecrim.\n"),
        ]

        recipes = RecipeRAGRepository._split_pdf_into_recipes(pages)

        self.assertEqual([recipe[0] for recipe in recipes], ["Pães com vegetais", "Pão de milho"])
        self.assertEqual(recipes[0][2], 23)
        self.assertIn("cenoura", recipes[0][1])
        self.assertNotIn("farinha de milho", recipes[0][1])
        self.assertIn("farinha de milho", recipes[1][1])
        self.assertIn("alecrim", recipes[1][1])

    def test_pdf_recipe_title_excludes_overlapping_quick_guide_column(self) -> None:
        title = "Empadão de carne" + " " * 70 + "GUIA EXPRESS"
        recipes = RecipeRAGRepository._split_pdf_into_recipes(
            [(89, f"{title}\n44 MIN. 35 SEG. | 8 PESSOAS\n180g de cebola")]
        )

        self.assertEqual([recipe[0] for recipe in recipes], ["Empadão de carne"])
        self.assertNotIn("GUIA EXPRESS", recipes[0][1])

    def test_pdf_parser_falls_back_to_ladymaxx_sections_and_splits_same_page(self) -> None:
        ladymaxx_page_6 = (
            "Sopa de laranja\n"
            "e cenoura\n"
            "Ingredientes para 4 pessoas\n"
            "400 g de cenouras\n"
            "100 ml de sumo de laranja\n"
            "Preparação\n"
            "Deitar a cebola na lâmina e refogar durante 1 min.\n"
            "Creme fino de\n"
            "espargos\n"
            "Ingredientes para 4 pessoas\n"
            "500 g de espargos brancos\n"
            "250 ml de natas batidas\n"
            "Preparação\n"
            "Descascar os espargos e deixar cozer durante 15 min."
        )
        pages = [(6, ladymaxx_page_6)]

        self.assertIsNone(RecipeRAGRepository._split_pdf_with_time_headers(pages))
        recipes = RecipeRAGRepository._split_pdf_into_recipes(pages)

        self.assertEqual(
            [(recipe[0], recipe[2]) for recipe in recipes],
            [("Sopa de laranja e cenoura", 6), ("Creme fino de espargos", 6)],
        )
        self.assertIn("sumo de laranja", recipes[0][1])
        self.assertIn("refogar durante 1 min", recipes[0][1])
        self.assertNotIn("espargos", recipes[0][1])
        self.assertIn("natas batidas", recipes[1][1])
        self.assertIn("cozer durante 15 min", recipes[1][1])
        self.assertNotIn("sumo de laranja", recipes[1][1])

    def test_ladymaxx_parser_keeps_continuation_and_separates_last_recipe(self) -> None:
        pages = [
            (
                77,
                "Sorvete de\n"
                "pêssego\n"
                "Ingredientes para 4 pessoas\n"
                "600 g de pêssegos\n"
                "Preparação\n"
                "Triturar a fruta e reservar.",
            ),
            (
                78,
                "Junte as natas e misture durante 2 min.\n"
                "Gelado de\n"
                "chantili\n"
                "Ingredientes para 4 pessoas\n"
                "200 ml de natas\n"
                "Preparação\n"
                "Bater as natas e servir.",
            ),
        ]

        recipes = RecipeRAGRepository._split_pdf_into_recipes(pages)

        self.assertEqual(
            [(recipe[0], recipe[2]) for recipe in recipes],
            [("Sorvete de pêssego", 77), ("Gelado de chantili", 78)],
        )
        self.assertIn("Junte as natas", recipes[0][1])
        self.assertNotIn("Gelado de chantili", recipes[0][1])
        self.assertIn("200 ml de natas", recipes[1][1])
        self.assertNotIn("pêssegos", recipes[1][1])

    def test_ladymaxx_recipe_section_without_a_title_fails_explicitly(self) -> None:
        from foodadvisor.rag_repository import RAGUnavailableError

        with self.assertRaisesRegex(RAGUnavailableError, "sem título identificável"):
            RecipeRAGRepository._split_pdf_into_recipes(
                [(6, "Ingredientes para 4 pessoas\n400 g de cenouras")]
            )

    def test_pdf_recipe_continuation_lines_with_durations_are_not_new_recipes(self) -> None:
        pages = [
            (
                29,
                "Massa de fartos ou choux\n"
                "6 MIN. 25 SEG. | 22 UNIDADES\n"
                "170g farinha tipo 55\n"
                "Deixe a massa descansar entre 15 a 30 minutos .\n"
                "5 ovos S* 40 seg 4\n"
                "Coloque a massa no tabuleiro e leve ao forno.\n",
            ),
            (
                33,
                "Cenouras à algarvia\n"
                "20 MIN. 10 SEG. | 6 PESSOAS | sem leite sem glúten vegan\n"
                "500g cenouras\n"
                "50g azeite\n"
                "80g vinagre\n"
                "5 seg 5\n"
                "Deite o preparado sobre as cenouras e misture.\n"
                "Hummus\n"
                "40 SEG. | ± 420G | sem leite vegan\n"
                "Grão-de-bico e sementes de sésamo.\n",
            ),
            (
                79,
                "Arroz de mariscos\n"
                "30 MIN. 7 SEG. | 6 PESSOAS | sem leite\n"
                "800g mistura de mariscos\n"
                "40g azeite 5 min 120ºC 2\n"
                "Pese o caldo reservado e junte água até atingir 900g.\n"
                "Junte os mariscos ao arroz e sirva.\n",
            ),
        ]

        recipes = RecipeRAGRepository._split_pdf_into_recipes(pages)

        self.assertEqual(
            [recipe[0] for recipe in recipes],
            ["Massa de fartos ou choux", "Cenouras à algarvia", "Hummus", "Arroz de mariscos"],
        )
        choux = recipes[0][1]
        carrots = recipes[1][1]
        seafood_rice = recipes[3][1]
        self.assertIn("5 ovos S* 40 seg 4", choux)
        self.assertIn("Coloque a massa no tabuleiro", choux)
        self.assertIn("50g azeite", carrots)
        self.assertIn("Deite o preparado sobre as cenouras", carrots)
        self.assertIn("Pese o caldo reservado", seafood_rice)
        self.assertIn("Junte os mariscos ao arroz", seafood_rice)

    def test_pdf_without_detectable_recipe_boundaries_is_not_combined(self) -> None:
        recipes = RecipeRAGRepository._split_pdf_into_recipes(
            [(1, "Receita sem título identificável e ingredientes variados.")]
        )

        self.assertEqual(recipes, [])

    def test_pdf_pages_without_text_are_ignored_without_breaking_recipe_continuation(self) -> None:
        pages = [
            (
                1,
                "TARTE DE MAÇÃ\n"
                "PORÇÕES: 6 GRAU DE DIFICULDADE: FÁCIL\n"
                "INGREDIENTES MODO DE PREPARAÇÃO\n"
                "200 g de farinha\n"
                "Misture os ingredientes e leve ao forno.",
            ),
            (2, ""),
            (
                3,
                "Continue a cozedura durante 20 minutos.\n"
                "Retire do forno e sirva.",
            ),
            (4, ""),
        ]

        recipes = RecipeRAGRepository._split_pdf_into_recipes(pages)

        self.assertEqual(len(recipes), 1)
        self.assertEqual(recipes[0][0], "TARTE DE MAÇÃ")
        self.assertEqual(recipes[0][2], 1)
        self.assertIn("Continue a cozedura", recipes[0][1])
        self.assertIn("Retire do forno e sirva", recipes[0][1])

    def test_pdf_detects_recipe_pages_without_preparation_time_headers(self) -> None:
        pages = [
            (
                20,
                "RECEITAS CONTEMPORÂNEAS\n"
                "Texto introdutório sobre sobremesas e ingredientes.",
            ),
            (
                21,
                "CREME DE LEITE DE AMÊNDOA E ABÓBORA\n"
                "Uma sobremesa elegante.\n"
                "PORÇÕES: 5 GRAU DE DIFICULDADE: MÉDIO\n"
                "INGREDIENTES MODO DE PREPARAÇÃO\n"
                "200 ml de bebida de amêndoa\n"
                "1 colher de sopa de abóbora\n"
                "4 colheres de açúcar\n"
                "Leve ao lume e misture bem.\n"
                "Coloque numa travessa e sirva frio.",
            ),
            (
                22,
                "PISTÁCIOS\n"
                "O xarope de agave é uma alternativa ao açúcar.\n"
                "1 colher de sopa de bebida de aveia\n"
                "60 g de amido de milho\n"
                "60 ml de xarope de agave\n"
                "Numa tigela junte os ingredientes.\n"
                "Misture muito bem e aqueça até engrossar.\n"
                "Transfira para tacinhas e decore.",
            ),
        ]

        recipes = RecipeRAGRepository._split_pdf_into_recipes(pages)

        self.assertEqual(
            [recipe[0] for recipe in recipes],
            ["CREME DE LEITE DE AMÊNDOA E ABÓBORA", "PISTÁCIOS"],
        )
        self.assertEqual([recipe[2] for recipe in recipes], [21, 22])
        self.assertIn("200 ml de bebida de amêndoa", recipes[0][1])
        self.assertIn("sirva frio", recipes[0][1])
        self.assertIn("60 g de amido de milho", recipes[1][1])
        self.assertIn("decore", recipes[1][1])

    def test_pdf_structured_recipe_pages_keep_unheaded_continuations_together(self) -> None:
        pages = [
            (
                10,
                "TARTE DE MAÇÃ\n"
                "PORÇÕES: 6 GRAU DE DIFICULDADE: FÁCIL\n"
                "INGREDIENTES MODO DE PREPARAÇÃO\n"
                "200 g de farinha\n"
                "Misture os ingredientes e leve ao forno.",
            ),
            (
                11,
                "Continue a cozedura durante 20 minutos.\n"
                "Retire do forno e sirva.",
            ),
            (
                12,
                "BOLO DE PERA\n"
                "PORÇÕES: 8 GRAU DE DIFICULDADE: MÉDIO\n"
                "INGREDIENTES MODO DE PREPARAÇÃO\n"
                "3 peras e 150 g de açúcar\n"
                "Bata os ovos e leve ao forno.",
            ),
        ]

        recipes = RecipeRAGRepository._split_pdf_into_recipes(pages)

        self.assertEqual([recipe[0] for recipe in recipes], ["TARTE DE MAÇÃ", "BOLO DE PERA"])
        self.assertIn("Continue a cozedura", recipes[0][1])
        self.assertIn("sirva", recipes[0][1])
        self.assertNotIn("BOLO DE PERA", recipes[0][1])
        self.assertIn("3 peras", recipes[1][1])

    def test_plain_text_does_not_split_on_durations_in_recipe_instructions(self) -> None:
        recipes = RecipeRAGRepository._plain_text_recipes(
            "Massa de fartos ou choux\n"
            "6 MIN. 25 SEG.\n"
            "Deixe a massa descansar entre 15 a 30 minutos.\n"
            "5 ovos S* 40 seg 4\n"
            "Coloque no forno por 10 minutos.\n"
            "Cenouras à algarvia\n"
            "20 MIN. 10 SEG.\n"
            "500g cenouras e 50g azeite.",
            "recipes",
        )

        self.assertEqual(
            [recipe[0] for recipe in recipes],
            ["Massa de fartos ou choux", "Cenouras à algarvia"],
        )
        self.assertIn("5 ovos S* 40 seg 4", recipes[0][1])
        self.assertIn("Coloque no forno por 10 minutos", recipes[0][1])

    def test_plain_text_splits_recipes_at_title_and_preparation_time(self) -> None:
        recipes = RecipeRAGRepository._plain_text_recipes(
            "Pasta al pesto\n15 min\nMassa e manjericão.\n\n"
            "Salada de frango\n20 min\nFrango e alface.",
            "receitas",
        )

        self.assertEqual([recipe[0] for recipe in recipes], ["Pasta al pesto", "Salada de frango"])
        self.assertNotIn("Frango", recipes[0][1])
        self.assertNotIn("Pesto", recipes[1][1])

    def test_exact_recipe_title_outranks_semantically_similar_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "resources"
            resources.mkdir()
            (resources / "chourico.txt").write_text(
                "Pães com chouriço e queijo.", encoding="utf-8"
            )
            (resources / "vegetais.txt").write_text(
                "Pães com vegetais, cenoura e courgette.", encoding="utf-8"
            )
            repository = RecipeRAGRepository(
                resources_dir=resources,
                index_dir=root / "index",
                embedder=KeywordTestEmbedder(),
                min_similarity=0.2,
            )

            matches = repository.search("Pães com vegetais")

            self.assertEqual(matches[0]["source"], "vegetais.txt")
            self.assertIn("cenoura", matches[0]["content"])

    def test_search_returns_semantically_relevant_resource_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "resources"
            resources.mkdir()
            (resources / "pesto.md").write_text(
                "# Pasta al pesto\nMassa, manjericão e pinhões.", encoding="utf-8"
            )
            (resources / "salada.txt").write_text(
                "Salada de frango com alface e tomate.", encoding="utf-8"
            )
            embedder = KeywordTestEmbedder()
            repository = RecipeRAGRepository(
                resources_dir=resources,
                index_dir=root / "index",
                embedder=embedder,
                min_similarity=0.2,
            )

            matches = repository.search("massa com pesto")

            self.assertEqual(matches[0]["source"], "pesto.md")
            self.assertEqual(matches[0]["title"], "Pasta al pesto")
            self.assertGreaterEqual(matches[0]["score"], 0.2)
            self.assertIn("[Fonte: pesto.md", repository.format_context(matches))

    def test_index_is_rebuilt_when_resource_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "resources"
            resources.mkdir()
            (resources / "pasta.txt").write_text("Pasta com pesto.", encoding="utf-8")
            embedder = KeywordTestEmbedder()
            repository = RecipeRAGRepository(
                resources_dir=resources,
                index_dir=root / "index",
                embedder=embedder,
                min_similarity=0.0,
            )

            repository.search("pasta")
            first_embedding_call_count = embedder.calls
            (resources / "frango.txt").write_text("Salada de frango.", encoding="utf-8")
            matches = repository.search("frango")

            self.assertGreater(embedder.calls, first_embedding_call_count)
            self.assertTrue(any(match["source"] == "frango.txt" for match in matches))

    def test_rebuild_index_forces_resource_reload_and_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "resources"
            resources.mkdir()
            resource = resources / "recipes.md"
            resource.write_text("# Pasta al pesto\nMassa e manjericão.", encoding="utf-8")
            embedder = KeywordTestEmbedder()
            repository = RecipeRAGRepository(
                resources_dir=resources,
                index_dir=root / "index",
                embedder=embedder,
                min_similarity=0.0,
            )
            repository.search("pasta")
            first_embedding_call_count = embedder.calls

            resource.write_text(
                "# Salada de frango\nFrango, alface e tomate.", encoding="utf-8"
            )
            result = repository.rebuild_index()

            self.assertEqual(
                result,
                {"resource_files": 1, "recipe_chunks": 1, "ignored_resources": []},
            )
            self.assertGreater(embedder.calls, first_embedding_call_count)
            matches = repository.search("frango")
            self.assertEqual(matches[0]["title"], "Salada de frango")

    def test_rebuild_reports_resources_without_extractable_recipe_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "resources"
            resources.mkdir()
            (resources / "recipes.pdf").write_bytes(b"test pdf placeholder")
            (resources / "pasta.txt").write_text("Pasta al pesto.", encoding="utf-8")
            repository = RecipeRAGRepository(
                resources_dir=resources,
                index_dir=root / "index",
                embedder=KeywordTestEmbedder(),
            )

            with patch.object(repository, "_read_pdf_resource", return_value=[]):
                result = repository.rebuild_index()

            self.assertEqual(result["resource_files"], 2)
            self.assertEqual(result["recipe_chunks"], 1)
            self.assertEqual(result["ignored_resources"], ["recipes.pdf"])

    def test_search_requires_at_least_one_resource(self) -> None:
        from foodadvisor.rag_repository import RAGUnavailableError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "resources"
            resources.mkdir()
            (resources / "README.md").write_text("Instructions only.", encoding="utf-8")
            repository = RecipeRAGRepository(
                resources_dir=resources,
                index_dir=root / "index",
                embedder=KeywordTestEmbedder(),
            )

            with self.assertRaises(RAGUnavailableError):
                repository.search("pasta")


if __name__ == "__main__":
    unittest.main()
