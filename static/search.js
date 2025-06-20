(function() {
    'use strict';
    
    function initializeWhenReady() {
        if (typeof window.AppState === 'undefined' || typeof window.AppUtils === 'undefined') {
            setTimeout(initializeWhenReady, 100);
            return;
        }
        
        initializeSearchModule();
    }
    
    function initializeSearchModule() {
        function switchTab(tabName) {
            const tabMapping = {
                'explorer': 'explorerTab',
                'profiling': 'profilingTab', 
                'systemic': 'systemicTab',
                'knowledge-transfer': 'knowledge-transferTab',
                'embeddings': 'embeddingsTab'
            };
            
            const tabId = tabMapping[tabName] || tabName;
            
            document.querySelectorAll('.tab-content').forEach(content => {
                content.style.display = 'none';
                content.classList.remove('active');
            });
            
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });
            
            const selectedContent = window.AppUtils.getElement(tabId);
            if (selectedContent) {
                selectedContent.style.display = 'block';
                selectedContent.classList.add('active');
            }
            
            const selectedButton = document.querySelector(`[data-tab="${tabName}"]`);
            if (selectedButton) {
                selectedButton.classList.add('active');
            }
            
            window.AppState.search.activeTab = tabId;
        }

        function displayTriplets(triplets) {
            const tripletsContainer = window.AppUtils.getElement('tripletsContainer');
            if (!tripletsContainer) return;
            
            tripletsContainer.innerHTML = ''; 
            if (!triplets || triplets.length === 0) {
                tripletsContainer.innerHTML = '<p>No data to display. Try a different search.</p>';
                return;
            }
            
            const fragment = document.createDocumentFragment();
            
            triplets.forEach(triplet => {
                const card = document.createElement('div');
                card.classList.add('triplet-card');
                card.innerHTML = `
                    <h3>${triplet.subject || 'N/A'}</h3>
                    <p><strong>Threat:</strong> ${triplet.predicate || 'N/A'}</p>
                    <p><strong>Category:</strong> ${triplet.object || 'N/A'}</p>
                    <p><a href="https://doi.org/${triplet.doi}" target="_blank" class="doi-link">DOI: ${triplet.doi || 'N/A'}</a></p>
                    <div class="triplet-actions">
                        <button class="similar-button" data-id="${triplet.id}">Find Similar Threats</button>
                        <button class="explore-details-button" data-triplet-id="${triplet.id}">Explore Details</button> 
                        <button class="start-chain-button" data-species-name="${triplet.subject}" data-triplet-id="${triplet.id}">Start Chain Here</button> 
                    </div>
                `;
                fragment.appendChild(card);
            });
            
            tripletsContainer.appendChild(fragment);
            tripletsContainer.addEventListener('click', handleTripletCardClick);
        }

        function handleTripletCardClick(event) {
            const target = event.target;
            
            if (target.classList.contains('similar-button')) {
                handleFindSimilar(event);
            } else if (target.classList.contains('explore-details-button')) {
                handleExploreDetailsClick(event);
            } else if (target.classList.contains('start-chain-button')) {
                const speciesName = target.dataset.speciesName;
                const tripletId = target.dataset.tripletId;
                
                if (typeof window.startNewChainWithSpecies === 'function') {
                    window.startNewChainWithSpecies(speciesName, tripletId);
                } else if (window.AppFunctions && typeof window.AppFunctions.startNewChainWithSpecies === 'function') {
                    window.AppFunctions.startNewChainWithSpecies(speciesName, tripletId);
                }
            }
        }

        function performSearch() {
            const searchInput = window.AppUtils.getElement('searchInput');
            const searchQuery = searchInput ? searchInput.value.trim() : '';
            
            if (!searchQuery) {
                displayTriplets(window.AppState.allTripletsData);
                updateSearchStats(window.AppState.allTripletsData.length, window.AppState.allTripletsData.length);
                return;
            }
            
            window.AppState.search.currentQuery = searchQuery;
            window.AppState.search.lastSearchTime = Date.now();
            
            if (!window.AppState.fuse) return;
            
            try {
                const searchResults = window.AppState.fuse.search(searchQuery);
                const triplets = searchResults.map(result => result.item);
                
                window.AppState.search.lastResults = triplets;
                window.AppState.search.resultCount = triplets.length;
                
                displayTriplets(triplets);
                updateSearchStats(triplets.length, window.AppState.allTripletsData.length);
                
            } catch (error) {
                displayTriplets([]);
                updateSearchStats(0, window.AppState.allTripletsData.length);
            }
        }

        function updateSearchStats(resultCount, totalCount) {
            const searchStats = window.AppUtils.getElement('searchStats');
            if (searchStats) {
                if (window.AppState.search.currentQuery) {
                    searchStats.textContent = `Found ${resultCount} results out of ${totalCount} total records`;
                } else {
                    searchStats.textContent = `Showing all ${totalCount} records`;
                }
            }
        }

        function performSpeciesSearch() {
            const speciesSearchInput = window.AppUtils.getElement('speciesSearchInput');
            const searchQuery = speciesSearchInput ? speciesSearchInput.value.trim() : '';
            
            if (!searchQuery) {
                populateSpeciesDropdown(window.AppState.allSpecies);
                return;
            }
            
            if (!window.AppState.speciesFuse) return;
            
            try {
                const searchResults = window.AppState.speciesFuse.search(searchQuery);
                const species = searchResults.map(result => result.item);
                populateSpeciesDropdown(species);
                
            } catch (error) {
                populateSpeciesDropdown([]);
            }
        }

        function populateSpeciesDropdown(species) {
            const speciesSelect = window.AppUtils.getElement('speciesSelect');
            if (!speciesSelect) return;
            
            while (speciesSelect.children.length > 1) {
                speciesSelect.removeChild(speciesSelect.lastChild);
            }
            
            species.forEach(speciesName => {
                const option = document.createElement('option');
                option.value = speciesName;
                option.textContent = speciesName;
                speciesSelect.appendChild(option);
            });
        }

        function initializeTabs() {
            document.querySelectorAll('.tab-button').forEach(button => {
                button.addEventListener('click', function() {
                    const tabName = this.getAttribute('data-tab');
                    if (tabName) {
                        switchTab(tabName);
                    }
                });
            });
            
            switchTab('explorer');
        }

        function clearFilters() {
            const filterElements = ['filterSpecies', 'filterThreat', 'filterCategory', 'filterDOI'];
            filterElements.forEach(id => {
                const element = window.AppUtils.getElement(id);
                if (element) element.value = '';
            });
            
            const searchInput = window.AppUtils.getElement('searchInput');
            if (searchInput) searchInput.value = '';
            
            window.AppState.search.currentQuery = '';
            window.AppState.search.activeFilters = {};
            window.AppState.search.lastResults = window.AppState.allTripletsData;
            
            displayTriplets(window.AppState.allTripletsData);
            updateSearchStats(window.AppState.allTripletsData.length, window.AppState.allTripletsData.length);
        }

        function applyFilters() {
            const filters = {
                species: window.AppUtils.getElement('filterSpecies')?.value || '',
                threat: window.AppUtils.getElement('filterThreat')?.value || '',
                category: window.AppUtils.getElement('filterCategory')?.value || '',
                doi: window.AppUtils.getElement('filterDOI')?.value || ''
            };
            
            let filteredData = window.AppState.allTripletsData;
            
            Object.entries(filters).forEach(([key, value]) => {
                if (value.trim()) {
                    filteredData = filteredData.filter(triplet => {
                        const fieldValue = triplet[key === 'species' ? 'subject' : 
                                                   key === 'threat' ? 'predicate' : 
                                                   key === 'category' ? 'object' : key] || '';
                        return fieldValue.toLowerCase().includes(value.toLowerCase());
                    });
                }
            });
            
            window.AppState.search.lastResults = filteredData;
            window.AppState.search.resultCount = filteredData.length;
            window.AppState.search.activeFilters = filters;
            
            displayTriplets(filteredData);
            updateSearchStats(filteredData.length, window.AppState.allTripletsData.length);
        }

        function handleFindSimilar(event) {
            const tripletId = event.target.dataset.id; 
            if (!tripletId) return;
            
            const similarThreatsContent = window.AppUtils.getElement('similarThreatsContent');
            const similarThreatsModal = window.AppUtils.getElement('similarThreatsModal');
            
            if (!similarThreatsContent || !similarThreatsModal) return;

            similarThreatsContent.innerHTML = '<p>Loading similar threats...</p>';
            similarThreatsModal.style.display = 'block';
            
            fetch(`/api/similar_threats?id=${tripletId}&top_n=5`)
                .then(response => { 
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`); 
                    return response.json(); 
                })
                .then(similarItems => {
                    if (similarItems.error) { 
                        similarThreatsContent.innerHTML = `<p>Error: ${similarItems.error}.</p>`; 
                        return; 
                    }
                    renderSimilarItems(similarItems);
                })
                .catch(error => { 
                    similarThreatsContent.innerHTML = `<p>Error fetching similar threats: ${error.message}.</p>`; 
                });
        }

        function renderSimilarItems(items) {
            const similarThreatsContent = window.AppUtils.getElement('similarThreatsContent');
            if (!similarThreatsContent) return;
            
            if (!items || items.length === 0) { 
                similarThreatsContent.innerHTML = '<p>No significantly similar threats found.</p>'; 
                return; 
            }
            
            let html = '<ul>';
            items.forEach(item => {
                html += `<li class="similar-item">
                    <p><strong>Species:</strong> ${item.subject || 'N/A'}</p>
                    <p><strong>Threat:</strong> ${item.predicate || 'N/A'} (${item.object || 'N/A'})</p>
                    <p><a href="https://doi.org/${item.doi}" target="_blank" class="doi-link">DOI: ${item.doi || 'N/A'}</a></p>
                    <p class="score">Similarity Score: ${typeof item.score === 'number' ? item.score.toFixed(4) : 'N/A'}</p>
                    <button class="explore-details-button" data-triplet-id="${item.id}">Explore Details</button>
                </li>`;
            });
            html += '</ul>';
            similarThreatsContent.innerHTML = html;
            
            document.querySelectorAll('#similarThreatsContent .explore-details-button').forEach(button => {
                button.addEventListener('click', handleExploreDetailsClick);
            });
        }

        function handleExploreDetailsClick(event) {
            const tripletId = event.target.dataset.tripletId; 
            if (typeof tripletId === 'undefined') {
                const detailedInfoContent = window.AppUtils.getElement('detailedInfoContent');
                const detailedInfoModal = window.AppUtils.getElement('detailedInfoModal');
                if (detailedInfoContent) detailedInfoContent.innerHTML = "<p>Error: Could not get item details (ID missing).</p>";
                if (detailedInfoModal) detailedInfoModal.style.display = 'block';
                return;
            }
            
            const selectedTriplet = window.AppUtils.getTripletById(tripletId);
            if (!selectedTriplet) {
                const detailedInfoContent = window.AppUtils.getElement('detailedInfoContent');
                const detailedInfoModal = window.AppUtils.getElement('detailedInfoModal');
                if (detailedInfoContent) detailedInfoContent.innerHTML = "<p>Error: Could not load details (item not found).</p>";
                if (detailedInfoModal) detailedInfoModal.style.display = 'block';
                return;
            }
            
            const speciesB = selectedTriplet.subject;
            const threatX_predicate = selectedTriplet.predicate;
            const threatX_object = selectedTriplet.object;
            
            const detailedInfoTitle = window.AppUtils.getElement('detailedInfoTitle');
            const detailedInfoContent = window.AppUtils.getElement('detailedInfoContent');
            const detailedInfoModal = window.AppUtils.getElement('detailedInfoModal');
            
            if (detailedInfoTitle) detailedInfoTitle.textContent = `Details for: ${speciesB}`;
            
            let contentHtml = '';
            const otherThreatsForSpeciesB = window.AppState.allTripletsData.filter(t => t.subject === speciesB && t.id !== selectedTriplet.id);
            contentHtml += `<div class="detail-section"><h3>Other Threats for ${speciesB}</h3>`;
            
            if (otherThreatsForSpeciesB.length > 0) {
                const uniqueThreats = new Map();
                otherThreatsForSpeciesB.forEach(t => { 
                    const threatKey = `${t.predicate || ''} (${t.object || ''})`; 
                    if (!uniqueThreats.has(threatKey)) { 
                        uniqueThreats.set(threatKey, t.doi); 
                    }
                });
                
                if (uniqueThreats.size > 0) {
                    contentHtml += '<ul>';
                    uniqueThreats.forEach((doi, threatDesc) => { 
                        contentHtml += `<li>${threatDesc} <a href="https://doi.org/${doi}" target="_blank" class="doi-link">[DOI]</a></li>`; 
                    });
                    contentHtml += '</ul>';
                    contentHtml += `<p class="stat">${speciesB} has ${uniqueThreats.size} other distinct documented threat(s) in this dataset.</p>`;
                } else { 
                    contentHtml += '<p>No other distinct threats found for this species in the dataset.</p>'; 
                }
            } else { 
                contentHtml += '<p>No other distinct threats found for this species in the dataset.</p>'; 
            }
            contentHtml += `</div>`;
            
            const otherSpeciesWithThreatX = window.AppState.allTripletsData.filter(t => t.predicate === threatX_predicate && t.object === threatX_object && t.subject !== speciesB);
            contentHtml += `<div class="detail-section"><h3>Other Species Affected by: "${threatX_predicate} (${threatX_object})"</h3>`;
            
            if (otherSpeciesWithThreatX.length > 0) {
                contentHtml += '<ul>';
                const uniqueOtherSpecies = [...new Set(otherSpeciesWithThreatX.map(t => t.subject))];
                uniqueOtherSpecies.forEach(speciesName => { 
                    contentHtml += `<li>${speciesName}</li>`; 
                });
                contentHtml += '</ul>';
                contentHtml += `<p class="stat">This specific threat combination affects ${uniqueOtherSpecies.length} other species in this dataset.</p>`;
            } else { 
                contentHtml += '<p>No other species found with this exact threat combination in the dataset.</p>'; 
            }
            contentHtml += `</div>`;
            
            if (detailedInfoContent) detailedInfoContent.innerHTML = contentHtml;
            if (detailedInfoModal) detailedInfoModal.style.display = 'block';
        }

        window.switchTab = switchTab;
        window.performSearch = performSearch;
        window.applyFilters = applyFilters;
        window.clearFilters = clearFilters;
        window.displayTriplets = displayTriplets;
        window.initializeTabs = initializeTabs;
        window.handleFindSimilar = handleFindSimilar;
        window.handleExploreDetailsClick = handleExploreDetailsClick;
        window.renderSimilarItems = renderSimilarItems;

        if (window.AppFunctions) {
            window.AppFunctions.handleFindSimilar = handleFindSimilar;
            window.AppFunctions.handleExploreDetailsClick = handleExploreDetailsClick;
            window.AppFunctions.renderSimilarItems = renderSimilarItems;
        }
    }
    
    initializeWhenReady();
})();