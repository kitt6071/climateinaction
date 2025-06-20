(function() {
    'use strict';

    function initializeWhenReady() {
        if (typeof window.AppState === 'undefined' || typeof window.AppUtils === 'undefined') {
            setTimeout(initializeWhenReady, 100);
            return;
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeChainBuilder);
        } else {
            setTimeout(initializeChainBuilder, 200);
        }
    }

    function initializeChainBuilder() {
        let currentChain = [];

        const chainElementIds = ['toggleChainModeButton', 'chainDisplayArea', 'chainNextStepOptionsArea', 'mainContentArea', 'chainBuilderSection'];
        chainElementIds.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                window.AppState.elements[id] = element;
            }
        });

        const toggleChainModeButton = window.AppUtils.getElement('toggleChainModeButton');
        const chainDisplayArea = window.AppUtils.getElement('chainDisplayArea');
        const chainNextStepOptionsArea = window.AppUtils.getElement('chainNextStepOptionsArea');
        const mainContentArea = window.AppUtils.getElement('mainContentArea');
        const chainBuilderSection = window.AppUtils.getElement('chainBuilderSection');

        if (!chainDisplayArea || !chainNextStepOptionsArea) {
            console.error('Critical chain builder elements missing.');
        }

        function toggleChainMode() {
            window.AppState.chainBuilder.chainModeActive = !window.AppState.chainBuilder.chainModeActive;
            const isActive = window.AppState.chainBuilder.chainModeActive;

            if (mainContentArea) mainContentArea.style.display = isActive ? 'none' : 'block';
            if (chainBuilderSection) chainBuilderSection.style.display = isActive ? 'block' : 'none';
            if (toggleChainModeButton) toggleChainModeButton.textContent = isActive ? 'Exit Chain Mode' : 'Start Exploration Chain';
        }

        if (toggleChainModeButton) {
            toggleChainModeButton.addEventListener('click', toggleChainMode);
        }

        function startNewChainWithSpecies(speciesName, initialTripletId) {
            currentChain = [];
            window.AppState.chainBuilder.currentChain = currentChain;
            if (!window.AppState.chainBuilder.chainModeActive) {
                toggleChainMode();
            }
            addToChain('species', { name: speciesName, id: initialTripletId });
            displayThreatsForSpeciesInChain(speciesName);
        }

        function addToChain(type, data) {
            currentChain.push({ type, data });
            window.AppState.chainBuilder.currentChain = currentChain;
            renderChainVisual();
        }

        function renderChainVisual() {
            if (!chainDisplayArea) {
                console.error("chainDisplayArea not found for rendering chain visual.");
                return;
            }

            if (currentChain.length === 0) {
                chainDisplayArea.innerHTML = '<em>Chain is empty. Start by selecting a species.</em>';
                return;
            }

            chainDisplayArea.innerHTML = currentChain.map(item => {
                if (item.type === 'species') {
                    const name = item.data.name || 'Unknown Species';
                    return `<span class="chain-item species" title="${name}">${name}</span>`;
                } else if (item.type === 'threat') {
                    const desc = item.data.description || 'Unknown Threat';
                    const displayDesc = desc.length > 40 ? desc.substring(0, 40) + '...' : desc;
                    return `<span class="chain-item threat" title="${desc}">${displayDesc}</span>`;
                }
                return '';
            }).join('<span class="chain-arrow">➔</span>');
        }

        function displayThreatsForSpeciesInChain(speciesName) {
            if (!chainNextStepOptionsArea) {
                console.error("chainNextStepOptionsArea not found for displaying threats.");
                return;
            }

            const speciesThreats = window.AppState.allTripletsData.filter(t => t.subject === speciesName);
            const uniqueThreatsMap = new Map();
            speciesThreats.forEach(t => {
                const threatKey = `${t.predicate || 'N/A'} (${t.object || 'N/A'})`;
                if (!uniqueThreatsMap.has(threatKey)) {
                    uniqueThreatsMap.set(threatKey, t.id);
                }
            });

            let listHtml;
            if (uniqueThreatsMap.size > 0) {
                const items = Array.from(uniqueThreatsMap.entries()).map(([threatDescription, tripletId]) => {
                    const originalTriplet = window.AppUtils.getTripletById(tripletId);
                    const fullThreatDesc = originalTriplet ? `${originalTriplet.predicate || 'N/A'} (${originalTriplet.object || 'N/A'})` : threatDescription;
                    return `<li data-threat-triplet-id="${tripletId}" data-threat-desc="${fullThreatDesc}">${threatDescription}</li>`;
                }).join('');
                listHtml = `<ul>${items}</ul>`;
            } else {
                listHtml = '<p>No specific threats found for this species to continue the chain.</p>';
            }

            chainNextStepOptionsArea.innerHTML = `<h3>Select a Threat for ${speciesName || 'Selected Species'}:</h3>${listHtml}`;

            chainNextStepOptionsArea.querySelectorAll('li[data-threat-triplet-id]').forEach(item => {
                item.addEventListener('click', (event) => {
                    const { threatTripletId, threatDesc } = event.currentTarget.dataset;
                    addToChain('threat', { id: threatTripletId, description: threatDesc });
                    fetchAndDisplayLinkingSpecies(threatTripletId);
                });
            });
        }

        function fetchAndDisplayLinkingSpecies(sourceThreatTripletId) {
            if (!chainNextStepOptionsArea) {
                console.error("chainNextStepOptionsArea not found for linking species.");
                return;
            }
            chainNextStepOptionsArea.innerHTML = '<p>Finding other species affected by similar threats...</p>';

            fetch(`/api/similar_threats?id=${sourceThreatTripletId}&top_n=10`)
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
                })
                .then(similarItems => {
                    if (similarItems.error) {
                        chainNextStepOptionsArea.innerHTML = `<p>Error: ${similarItems.error}</p>`;
                        return;
                    }

                    const speciesOfOriginThreat = currentChain.length >= 2 ? currentChain[currentChain.length - 2].data.name : null;
                    const uniqueLinkingSpecies = new Map();

                    similarItems.forEach(item => {
                        const { subject: speciesName, score, id: representativeTripletId, predicate, object } = item;
                        if (typeof speciesName === 'string' && speciesName !== speciesOfOriginThreat) {
                            const numericScore = (typeof score === 'number' && !isNaN(score)) ? score : 0;
                            if (!uniqueLinkingSpecies.has(speciesName) || numericScore > uniqueLinkingSpecies.get(speciesName).score) {
                                uniqueLinkingSpecies.set(speciesName, {
                                    representativeTripletId,
                                    score: numericScore,
                                    predicate: predicate || 'N/A',
                                    object: object || 'N/A'
                                });
                            }
                        }
                    });

                    let html;
                    if (uniqueLinkingSpecies.size > 0) {
                        const sortedSpecies = Array.from(uniqueLinkingSpecies.entries())
                            .sort(([, aData], [, bData]) => bData.score - aData.score);
                        
                        const listItems = sortedSpecies.map(([speciesName, data]) => {
                            return `<li data-species-name="${speciesName}" 
                                        data-triplet-id="${data.representativeTripletId}" 
                                        class="option-species">
                                        ${speciesName} <span class="similarity-score">(Score: ${data.score.toFixed(4)})</span>
                                        <br><small>Linked via: "${data.predicate} (${data.object})"</small>
                                    </li>`;
                        }).join('');
                        html = `<h3>Select a Linked Species:</h3><ul>${listItems}</ul>`;
                    } else {
                        html = '<h3>Select a Linked Species:</h3><p>No other distinct species found affected by similar threats.</p>';
                    }
                    chainNextStepOptionsArea.innerHTML = html;

                    chainNextStepOptionsArea.querySelectorAll('li[data-species-name]').forEach(item => {
                        item.addEventListener('click', (event) => {
                            const { speciesName, tripletId } = event.currentTarget.dataset;
                            addToChain('species', { name: speciesName, id: tripletId });
                            displayThreatsForSpeciesInChain(speciesName);
                        });
                    });
                })
                .catch(error => {
                    console.error('Error fetching linking species:', error);
                    if (chainNextStepOptionsArea) {
                        chainNextStepOptionsArea.innerHTML = `<p>Error fetching linking species: ${error.message}</p>`;
                    }
                });
        }

        window.startNewChainWithSpecies = startNewChainWithSpecies;
        window.toggleChainMode = toggleChainMode;
        window.addToChain = addToChain;
        window.renderChainVisual = renderChainVisual;

        if (window.AppFunctions) {
            Object.assign(window.AppFunctions, {
                startNewChainWithSpecies,
                toggleChainMode,
                addToChain,
                renderChainVisual,
                displayThreatsForSpeciesInChain,
                fetchAndDisplayLinkingSpecies
            });
        }
    }

    initializeWhenReady();
})();