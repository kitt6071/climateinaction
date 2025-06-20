function setupKnowledgeTransferControls() {
    const analyzeBtn = document.getElementById('analyzeKnowledgeTransferBtn');
    const speciesInput = document.getElementById('ktTargetSpecies');

    if (analyzeBtn) {
        analyzeBtn.removeEventListener('click', performKnowledgeTransferAnalysis);
        analyzeBtn.addEventListener('click', performKnowledgeTransferAnalysis);
    }

    if (speciesInput) {
        setupSpeciesAutocomplete(speciesInput);
    }
}

function setupKnowledgeTransferSliders() {
    const similaritySlider = document.getElementById('ktSimilarityThreshold');
    const similarityValue = document.getElementById('ktSimilarityValue');

    if (similaritySlider && similarityValue) {
        similaritySlider.addEventListener('input', () => {
            similarityValue.textContent = similaritySlider.value;
        });
    }
}

function setupSpeciesAutocomplete(input) {
    let speciesList = [];

    if (window.allTriplets) {
        const uniqueSpecies = new Set(window.allTriplets.map(t => t.subject).filter(Boolean));
        speciesList = Array.from(uniqueSpecies).sort();
    }

    input.addEventListener('input', () => {
        const value = input.value.toLowerCase();
        const suggestions = speciesList.filter(species =>
            species.toLowerCase().includes(value)
        ).slice(0, 10);

        showAutocompleteSuggestions(input, suggestions);
    });
}

function showAutocompleteSuggestions(input, suggestions) {
    document.querySelector('.autocomplete-suggestions')?.remove();

    if (suggestions.length === 0) return;

    const suggestionsDiv = document.createElement('div');
    suggestionsDiv.className = 'autocomplete-suggestions';
    Object.assign(suggestionsDiv.style, {
        position: 'absolute',
        backgroundColor: 'white',
        border: '1px solid #ddd',
        borderRadius: '4px',
        maxHeight: '200px',
        overflowY: 'auto',
        zIndex: '1000',
        width: `${input.offsetWidth}px`
    });

    suggestions.forEach(suggestion => {
        const suggestionItem = document.createElement('div');
        suggestionItem.textContent = suggestion;
        Object.assign(suggestionItem.style, {
            padding: '8px 12px',
            cursor: 'pointer',
            borderBottom: '1px solid #eee'
        });

        suggestionItem.addEventListener('mouseenter', () => suggestionItem.style.backgroundColor = '#f5f5f5');
        suggestionItem.addEventListener('mouseleave', () => suggestionItem.style.backgroundColor = 'white');
        suggestionItem.addEventListener('click', () => {
            input.value = suggestion;
            suggestionsDiv.remove();
        });

        suggestionsDiv.appendChild(suggestionItem);
    });

    input.parentNode.style.position = 'relative';
    input.parentNode.appendChild(suggestionsDiv);

    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !suggestionsDiv.contains(e.target)) {
            suggestionsDiv.remove();
        }
    }, { once: true });
}

async function performKnowledgeTransferAnalysis() {
    const targetSpecies = document.getElementById('ktTargetSpecies').value.trim();
    if (!targetSpecies) {
        alert('Please enter a target species name');
        return;
    }

    const similarityThreshold = parseFloat(document.getElementById('ktSimilarityThreshold').value) || 0.1;
    const minEvidenceCount = parseInt(document.getElementById('ktMinEvidence').value) || 1;
    const resultsContainer = document.getElementById('knowledgeTransferResults');

    if (resultsContainer) {
        resultsContainer.innerHTML = `<div class="loading">Analyzing knowledge transfer for ${targetSpecies}...</div>`;
    }

    try {
        const response = await fetch('/api/knowledge_transfer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_species: targetSpecies,
                similarity_threshold: similarityThreshold,
                min_evidence_count: minEvidenceCount
            })
        });

        if (!response.ok) {
            if (response.status === 404) {
                const errorData = await response.json();
                displaySpeciesNotFoundError(errorData);
                return;
            }
            const errorText = await response.text();
            throw new Error(`API request failed: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        displayKnowledgeTransferResults(data);

    } catch (error) {
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="error">
                    <h3>Analysis Error</h3>
                    <p>${error.message}</p>
                    <p><strong>Tip:</strong> Try common species names or use the Explorer tab to browse available species.</p>
                </div>`;
        }
    }
}

function displaySpeciesNotFoundError(errorData) {
    const container = document.getElementById('knowledgeTransferResults');
    if (!container) return;

    const suggestionsHtml = errorData.suggestions && errorData.suggestions.length > 0 
        ? `<div class="species-suggestions">
               <h4>Did you mean one of these species?</h4>
               <div class="suggestion-list">
                   ${errorData.suggestions.map(species => 
                       `<button class="suggestion-btn" onclick="selectSuggestedSpecies('${species.replace(/'/g, "\\'")}')">
                           ${species}
                       </button>`
                   ).join('')}
               </div>
           </div>`
        : '';

    container.innerHTML = `
        <div class="species-not-found">
            <h3>Species Not Found</h3>
            <p>${errorData.error}</p>
            <p>${errorData.message}</p>
            <div class="dataset-info">
                <p><strong>Dataset contains ${errorData.total_species_count} species.</strong></p>
            </div>
            ${suggestionsHtml}
            <div class="search-tips">
                <h4>Search Tips:</h4>
                <ul>
                    <li>Use scientific names (e.g., "Gallus gallus" instead of "chicken")</li>
                    <li>Try common names without modifiers (e.g., "eagle" instead of "bald eagle")</li>
                    <li>Check spelling and capitalization</li>
                    <li>Browse available species in the Explorer tab</li>
                </ul>
            </div>
        </div>`;
}

function selectSuggestedSpecies(speciesName) {
    const input = document.getElementById('ktTargetSpecies');
    if (input) {
        input.value = speciesName;
        performKnowledgeTransferAnalysis();
    }
}

function displayKnowledgeTransferResults(data) {
    const container = document.getElementById('knowledgeTransferResults');
    if (!container) return;

    if (data.similar_species_count === 0) {
        container.innerHTML = `
            <div class="no-results">
                <h3>No Similar Species Found</h3>
                <p>No species with sufficient similarity to ${data.target_species} found.</p>
                <p>Try adjusting the filters.</p>
            </div>`;
        return;
    }

    const potentialThreatsCount = data.knowledge_transfer_candidates.reduce((sum, c) => sum + c.transferable_threats.length, 0);

    container.innerHTML = `
        <div class="knowledge-transfer-results">
            <h3>Analysis for ${data.target_species}</h3>
            <div class="summary-stats">
                <div class="stat-card">
                    <div>${data.current_threat_count}</div>
                    <div>Current Threats</div>
                </div>
                <div class="stat-card">
                    <div>${data.similar_species_count}</div>
                    <div>Similar Species</div>
                </div>
                <div class="stat-card">
                    <div>${potentialThreatsCount}</div>
                    <div>Potential Threats</div>
                </div>
            </div>
            
            <div class="transfer-sections">
                <div class="section">
                    <h4>Research Recommendations</h4>
                    ${renderResearchRecommendations(data.research_recommendations)}
                </div>
                <div class="section">
                    <h4>Similar Species Analysis</h4>
                    ${renderSimilarSpeciesAnalysis(data.knowledge_transfer_candidates)}
                </div>
                <div class="section">
                    <h4>Knowledge Gaps</h4>
                    ${renderKnowledgeGaps(data.knowledge_gaps)}
                </div>
                <div class="section">
                    <h4>Current Threats</h4>
                    <div class="current-threats">
                        ${data.current_threats.map(threat => `<span class="threat-tag">${threat}</span>`).join('')}
                    </div>
                </div>
            </div>
        </div>`;
}

function renderResearchRecommendations(recommendations) {
    if (!recommendations?.length) {
        return '<p>No specific research recommendations generated.</p>';
    }

    return recommendations.map(rec => `
        <div class="recommendation-card priority-${rec.priority.toLowerCase()}">
            <h5>${rec.title}</h5>
            <span class="priority-badge priority-${rec.priority.toLowerCase()}">${rec.priority}</span>
            <p>${rec.description}</p>
            <ul>
                ${rec.specific_actions.map(action => `<li>${action}</li>`).join('')}
            </ul>
        </div>
    `).join('');
}

function renderSimilarSpeciesAnalysis(candidates) {
    if (!candidates?.length) {
        return '<p>No similar species found.</p>';
    }

    return candidates.slice(0, 5).map(candidate => `
        <div class="similar-species-card">
            <h5>${candidate.similar_species}</h5>
            <div class="similarity-scores">
                <span>Similarity: ${(candidate.combined_similarity * 100).toFixed(1)}%</span>
                <span>${candidate.shared_threat_count} shared threats</span>
            </div>
            
            <h6>Potential Threats:</h6>
            <div class="transferable-threats">
                ${candidate.transferable_threats.slice(0, 3).map(threat => `
                    <div class="transferable-threat">
                        <div>
                            <strong>${threat.threat}</strong>
                            <span>Score: ${(threat.transferability_score * 100).toFixed(1)}%</span>
                        </div>
                        <div>${threat.evidence_count} studies</div>
                        <div>${threat.transfer_reasoning}</div>
                        <div>
                            ${threat.suggested_research.map(suggestion => `<div class="suggestion">${suggestion}</div>`).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
            
            <details>
                <summary>Shared Threats (${candidate.shared_threat_count})</summary>
                <div>
                    ${candidate.shared_threats.map(threat => `<span class="threat-tag shared">${threat}</span>`).join('')}
                </div>
            </details>
        </div>
    `).join('');
}

function renderKnowledgeGaps(gaps) {
    if (!gaps) {
        return '<p>No knowledge gap analysis available.</p>';
    }

    const sections = [
        { key: 'threat_categories_missing', title: 'Missing Threat Categories' },
        { key: 'impact_mechanisms_understudied', title: 'Understudied Impact Mechanisms' },
        { key: 'geographic_coverage_gaps', title: 'Geographic Coverage Gaps' },
        { key: 'temporal_coverage_gaps', title: 'Temporal Coverage Gaps' }
    ];

    return sections.map(section => {
        const items = gaps[section.key] || [];
        if (items.length === 0) return '';

        return `
            <div class="gap-section">
                <h6>${section.title}</h6>
                <div class="gap-items">
                    ${items.map(item => `<span class="gap-tag">${item}</span>`).join('')}
                </div>
            </div>`;
    }).filter(Boolean).join('');
}

(function() {
    'use strict';

    function initializeKnowledgeTransfer() {
        setupKnowledgeTransferControls();
        setupKnowledgeTransferSliders();
        window.performKnowledgeTransferAnalysis = performKnowledgeTransferAnalysis;
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeKnowledgeTransfer);
    } else {
        initializeKnowledgeTransfer();
    }
})();
