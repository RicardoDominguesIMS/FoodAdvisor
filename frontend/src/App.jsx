import { useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function JsonLog({ title, value, emptyMessage }) {
  return (
    <section className="log-card">
      <h3>{title}</h3>
      {value == null ? (
        <p className="log-empty">{emptyMessage}</p>
      ) : (
        <pre>{JSON.stringify(value, null, 2)}</pre>
      )}
    </section>
  )
}

function App() {
  const [recipe, setRecipe] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [workflow, setWorkflow] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const analyzeRecipe = async (event) => {
    event.preventDefault()
    if (!recipe.trim()) {
      setError('Enter a recipe name to continue.')
      return
    }

    setLoading(true)
    setError('')
    setAnalysis(null)
    setWorkflow(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ recipe }),
      })
      const data = await response.json()
      if (!response.ok) {
        if (data.workflow_log) setWorkflow(data.workflow_log)
        throw new Error(data.detail || 'The recipe analysis failed.')
      }
      setAnalysis(data)
      setWorkflow(data.workflow_log)
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? 'Could not connect to the backend. Check that the API is running and try again.'
          : requestError.message,
      )
    } finally {
      setLoading(false)
    }
  }

  const ingredientStage = workflow?.ingredient_extraction
  const allergenStage = workflow?.allergen_analysis

  return (
    <main className="app-shell">
      <div className="app-container">
        <header className="page-header">
          <div className="brand-mark" aria-hidden="true">F</div>
          <div>
            <p className="eyebrow">Recipe intelligence</p>
            <h1>FoodAdvisor</h1>
            <p className="page-subtitle">
              Evidence-based allergen insights from your recipe books.
            </p>
          </div>
        </header>

        <div className="workspace-grid">
          <section className="analysis-column">
            <form className="recipe-card" onSubmit={analyzeRecipe}>
              <label htmlFor="recipe-input">Select a recipe</label>
              <p className="field-hint">
                Enter the recipe name exactly as you know it.
              </p>
              <input
                id="recipe-input"
                type="text"
                value={recipe}
                onChange={(event) => setRecipe(event.target.value)}
                placeholder="e.g. Orange and carrot soup"
                autoComplete="off"
              />
              <button type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner" aria-hidden="true" />
                    Analyzing recipe…
                  </>
                ) : (
                  'Analyze allergens'
                )}
              </button>
              {error && <p className="error-message" role="alert">{error}</p>}
            </form>

            <section className="result-card" aria-live="polite">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Analysis result</p>
                  <h2>Allergen summary</h2>
                </div>
                <span className="result-icon" aria-hidden="true">✓</span>
              </div>

              {!analysis ? (
                <p className="result-placeholder">
                  Your confirmed allergens and evidence-based warnings will appear here.
                </p>
              ) : (
                <>
                  <p className="allergen-summary">{analysis.allergen_message}</p>
                  {analysis.allergens?.length > 0 && (
                    <ul className="allergen-list">
                      {analysis.allergens.map((allergen, index) => (
                        <li key={`${allergen.name}-${allergen.ingredient}-${index}`}>
                          <span>{allergen.name}</span>
                          <small>From {allergen.ingredient}</small>
                        </li>
                      ))}
                    </ul>
                  )}
                  {analysis.warnings?.length > 0 && (
                    <div className="warning-box">
                      <h3>Uncertainty and warnings</h3>
                      <ul>
                        {analysis.warnings.map((warning, index) => (
                          <li key={`${warning.message}-${index}`}>{warning.message}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <p className="safety-note">{analysis.safety_note}</p>
                  <details className="ingredients-details">
                    <summary>View extracted ingredients</summary>
                    <ul>
                      {analysis.ingredients.map((ingredient, index) => (
                        <li key={`${ingredient.name}-${index}`}>
                          <strong>{ingredient.name}</strong>
                          <span className={`status-tag ${ingredient.status}`}>
                            {ingredient.status}
                          </span>
                          <p>{ingredient.evidence}</p>
                        </li>
                      ))}
                    </ul>
                  </details>
                </>
              )}
            </section>
          </section>

          <details className="backend-log" aria-label="Backend log">
            <summary className="log-header">
              <div className="log-indicator" aria-hidden="true" />
              <h2>Backend log</h2>
            </summary>
            <div className="backend-log-content">
              <p className="log-description">
                Inspect the exact system instructions and data sent to both LLM stages.
              </p>

              <div className="log-stage">
                <span className="stage-number">01</span>
                <div>
                  <h3>Ingredient extraction</h3>
                  <p>Recipe name and selected RAG chunks</p>
                </div>
              </div>
              <JsonLog
                title="Input sent to LLM 1"
                value={ingredientStage?.request}
                emptyMessage="The selected recipe and retrieved chunks will appear here."
              />
              <JsonLog
                title="LLM 1 response"
                value={ingredientStage?.response}
                emptyMessage="The structured ingredient extraction will appear here."
              />

              <div className="log-stage">
                <span className="stage-number">02</span>
                <div>
                  <h3>Allergen identification</h3>
                  <p>Ingredients and supporting evidence</p>
                </div>
              </div>
              <JsonLog
                title="Input sent to LLM 2"
                value={allergenStage?.request}
                emptyMessage="The validated ingredients will be passed to the second LLM."
              />
              <JsonLog
                title="LLM 2 response"
                value={allergenStage?.response}
                emptyMessage="The allergen results and warnings will appear here."
              />
            </div>
          </details>
        </div>
        <footer className="page-footer">
          Recipe text and cookbook content are analyzed as data, never as instructions.
        </footer>
      </div>
    </main>
  )
}

export default App
