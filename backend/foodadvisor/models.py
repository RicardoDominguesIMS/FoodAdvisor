from dataclasses import dataclass, field


@dataclass
class RecipeAnalysisRequest:
    recipe: str


@dataclass
class RecipeAnalysisResult:
    recipe: str
    ingredients: list[dict] = field(default_factory=list)
    allergens: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    uncertainties: list[str | dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    workflow_log: dict = field(default_factory=dict)

    @property
    def allergen_message(self) -> str:
        confirmed = list(
            dict.fromkeys(
                allergen["name"]
                for allergen in self.allergens
                if allergen["status"] == "confirmed"
            )
        )
        uncertain = list(
            dict.fromkeys(
                allergen["name"]
                for allergen in self.allergens
                if allergen["status"] == "uncertain"
            )
        )
        if confirmed:
            return "The allergens are: " + ", ".join(confirmed)
        if uncertain:
            return "Potential allergens need confirmation: " + ", ".join(uncertain)
        if self.warnings or self.uncertainties:
            return "The allergen assessment is incomplete; review the warnings below."
        return "No allergens were identified from the available recipe evidence."

    def to_dict(self) -> dict:
        return {
            "recipe": self.recipe,
            "ingredients": self.ingredients,
            "allergens": self.allergens,
            "allergen_message": self.allergen_message,
            "warnings": self.warnings,
            "uncertainties": self.uncertainties,
            "sources": self.sources,
            "workflow_log": self.workflow_log,
            "safety_note": (
                "This document-based analysis cannot guarantee food safety or the "
                "absence of cross-contact, contamination, or undeclared ingredients."
            ),
        }
