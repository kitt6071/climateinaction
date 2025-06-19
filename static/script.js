window.AppState = {
    allTripletsData: [],
    allSpecies: [],
    fuse: null,
    speciesFuse: null,
    search: {
        currentQuery: '',
        lastResults: [],
        resultCount: 0,
        activeTab: 'explorerTab',
        activeFilters: {},
        lastSearchTime: 0
    },
    elements: {},
    charts: {
        topThreatsChart: null,
        speciesThreatCountChart: null,
        speciesThreatCategoriesChart: null,
        threatImpactChart: null,
        vulnerabilityDistributionChart: null,
        threatPropagationChart: null,
        instances: new Map(),
        configurations: new Map()
    },
    chainModeActive: false,
    currentChain: [],
    networkData: { nodes: [], links: [] },
    similarityMatrix: new Map(),
    speciesNetworkMetrics: new Map(),
    currentNetworkAnalysis: null,
    d3NetworkSimulation: null,
    knowledgeGraph: {
        nodes: new Map(),
        edges: new Map(),
        speciesNodes: new Map(),
        threatNodes: new Map(),
        impactNodes: new Map(),
        semanticClusters: new Map(),
        clusters: [],
        initialized: false
    },
    data: {
        allTriplets: [],
        speciesList: [],
        threatsList: [],
        currentDataVersion: 0
    },
    ui: {
        currentView: 'explorer',
        loadingStates: new Map(),
        activeModals: [],
        currentSort: 'alphabetical'
    },
    chainBuilder: {
        chainModeActive: false,
        currentChain: [],
        lastChainElement: null,
        chainHistory: []
    },
    analysis: {
        networkMetrics: null,
        criticalNodes: [],
        vulnerabilityCorridors: [],
        indirectImpacts: [],
        lastAnalysisTime: 0
    }
};

// Global Configuration
window.AppConfig = {
    fuseOptions: {
        includeScore: true,
        threshold: 0.4, 
        distance: 100, 
        minMatchCharLength: 2, 
        keys: ["subject", "predicate", "object"]
    },
    speciesFuseOptions: {
        includeScore: true,
        threshold: 0.3,
        keys: ["name"]
    }
};

// Utility functions for accessing global state
window.AppUtils = {
    getElement: (id) => {
        if (!window.AppState.elements[id]) {
            window.AppState.elements[id] = document.getElementById(id);
        }
        return window.AppState.elements[id];
    },
    getTripletById: (id) => {
        if (!window.AppState.allTripletsData || !id) return null;
        
        const numericId = typeof id === 'string' ? parseInt(id.replace('triplet_', '')) : id;
        
        let triplet = window.AppState.allTripletsData.find(t => t.id === numericId || t.id === id);
        
        if (!triplet) {
            const index = numericId - 1;
            if (index >= 0 && index < window.AppState.allTripletsData.length) {
                triplet = window.AppState.allTripletsData[index];
            }
        }
        
        return triplet;
    }
};

// Main Application Initialization
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing application...');
    cacheMainElements();
    setupMainEventListeners();
    loadInitialData();

    setTimeout(() => {
        if (typeof window.initializeTabs === 'function') {
            window.initializeTabs();
        }
        
        setTimeout(() => {
            if (window.AppState.allTripletsData?.length > 0 && 
                typeof window.initializeKnowledgeGraph === 'function') {
                console.log(`Initializing KG...`);
                try {
                    const limitedData = window.AppState.allTripletsData.slice(0, 2000);
                    window.initializeKnowledgeGraph(limitedData);
                } catch (error) {
                    console.error('Error initializing KG:', error);
                }
            } else {
                console.log('KG initialization skipped.');
            }
        }, 500);
    }, 500);
});

function cacheMainElements() {
    const elementIds = [
        'tripletsContainer', 'searchInput', 'searchButton',
        'speciesSearchInput', 'speciesDropdown', 'analyzeSpeciesButton',
        'speciesAnalysisResults', 'selectedSpeciesName', 'totalThreatsCount',
        'semanticClustersCount', 'impactCategoriesCount', 'threatClustersContainer',
        'impactAnalysisContainer', 'speciesProfileContainer',
        'networkAnalysisType', 'runNetworkAnalysis', 'similarityThreshold',
        'similarityValue', 'networkCanvas', 'networkLegend',
        'focalSpeciesSelect', 'findIndirectImpacts', 'pathwayVisualization',
        'pathwayList', 'executeCustomQuery', 'queryResultsTable',
        'similarThreatsModal', 'similarThreatsContent', 'closeSimilarThreatsModalBtn',
        'detailedInfoModal', 'detailedInfoTitle', 'detailedInfoContent',
        'closeDetailedInfoModalBtn', 'mainContentArea', 'chainBuilderSection',
        'chainDisplayArea', 'chainNextStepOptionsArea', 'toggleChainModeButton'
    ];
    elementIds.forEach(id => {
        window.AppState.elements[id] = document.getElementById(id);
    });
    
    const chartContexts = [
        'topThreatsChart', 'speciesThreatCountChart', 'speciesThreatCategoriesChart',
        'threatImpactChart', 'vulnerabilityDistributionChart', 'threatPropagationChart'
    ];
    chartContexts.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            window.AppState.elements[id + 'Ctx'] = element.getContext('2d');
        }
    });
}

function setupMainEventListeners() {
    const searchButton = window.AppUtils.getElement('searchButton');
    const searchInput = window.AppUtils.getElement('searchInput');
    if (searchButton) searchButton.addEventListener('click', performSearch);
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performSearch();
        });
    }
    
    const analyzeSpeciesButton = window.AppUtils.getElement('analyzeSpeciesButton');
    if (analyzeSpeciesButton) {
        analyzeSpeciesButton.addEventListener('click', analyzeSpecies);
    }
    
    setupModalControls();
    setupSystemicAnalysisListeners();
    
    const toggleChainModeButton = window.AppUtils.getElement('toggleChainModeButton');
    if (toggleChainModeButton) {
        toggleChainModeButton.addEventListener('click', toggleChainMode);
    }
}

function setupModalControls() {
    const closeSimilarThreatsModalBtn = window.AppUtils.getElement('closeSimilarThreatsModalBtn');
    const closeDetailedInfoModalBtn = window.AppUtils.getElement('closeDetailedInfoModalBtn');
    const similarThreatsModal = window.AppUtils.getElement('similarThreatsModal');
    const detailedInfoModal = window.AppUtils.getElement('detailedInfoModal');
    
    if (closeSimilarThreatsModalBtn) {
        closeSimilarThreatsModalBtn.onclick = () => {
            if (similarThreatsModal) similarThreatsModal.style.display = 'none';
        };
    }
    
    if (closeDetailedInfoModalBtn) {
        closeDetailedInfoModalBtn.onclick = () => {
            if (detailedInfoModal) detailedInfoModal.style.display = 'none';
        };
    }
    
    window.onclick = (event) => {
        if (event.target == similarThreatsModal && similarThreatsModal) {
            similarThreatsModal.style.display = 'none';
        }
        if (event.target == detailedInfoModal && detailedInfoModal) {
            detailedInfoModal.style.display = 'none';
        }
    };
}

function setupSystemicAnalysisListeners() {
}

async function loadInitialData() {
    try {
        console.log('Loading data...');
        const response = await fetch('/api/triplets');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        if (data.error) {
            console.error('Error fetching triplets:', data.error);
            const tripletsContainer = window.AppUtils.getElement('tripletsContainer');
            if (tripletsContainer) {
                tripletsContainer.innerHTML = `<p>Error loading data: ${data.error}.</p>`;
            }
            return;
        }
        
        window.AppState.allTripletsData = data;
        console.log(`${data.length} triplets loaded.`);
        
        if (typeof Fuse !== 'undefined') {
            window.AppState.fuse = new Fuse(data, window.AppConfig.fuseOptions);
            console.log('Fuse search initialized.');
        } else { 
            console.error("Fuse.js not loaded.");
        }
        
        try {
            const initialDisplayData = data.slice(0, 50);
            displayTriplets(initialDisplayData);
            console.log(`Displaying first ${initialDisplayData.length} triplets.`);
        } catch (error) {
            console.error('Error displaying triplets:', error);
        }
        
        if (data.length > 0) {
            setTimeout(() => {
                try {
                    processAndRenderCharts(data);
                } catch (error) {
                    console.error('Error processing charts:', error);
                }
            }, 100);
            
            setTimeout(() => {
                try {
                    loadSpeciesProfilingData();
                } catch (error) {
                    console.error('Error loading species profiling data:', error);
                }
            }, 200);
        }
        
    } catch (error) {
        console.error('Error loading initial data:', error);
        const tripletsContainer = window.AppUtils.getElement('tripletsContainer');
        if (tripletsContainer) {
            tripletsContainer.innerHTML = `<p>Failed to load data. ${error.message}.</p>`;
        }
    }
}

function loadSpeciesProfilingData() {
    try {
        const speciesSet = new Set();
        
        if (window.AppState.allTripletsData && window.AppState.allTripletsData.length > 0) {
            const dataToProcess = window.AppState.allTripletsData.slice(0, 5000);
            console.log(`Extracting species from ${dataToProcess.length} triplets.`);
            
            dataToProcess.forEach(triplet => {
                if (triplet.subject && typeof triplet.subject === 'string') {
                    speciesSet.add(triplet.subject.trim());
                }
            });
            
            const uniqueSpecies = Array.from(speciesSet).sort();
            
            const limitedSpecies = uniqueSpecies.slice(0, 1000);
            console.log(`Found ${uniqueSpecies.length} unique species, using ${limitedSpecies.length}.`);
            
            window.AppState.allSpecies = limitedSpecies.map(name => ({ name: name }));
            
            if (window.AppState.allSpecies.length > 0) {
                window.AppState.speciesFuse = new Fuse(window.AppState.allSpecies, window.AppConfig.speciesFuseOptions);
            }
            
            setTimeout(() => {
                try {
                    populateSpeciesDropdown();
                    setupSpeciesSearch();
                    populateFocalSpeciesDropdown();
                    console.log('Species profiling ready.');
                } catch (error) {
                    console.error('Error setting up species UI:', error);
                }
            }, 100);
            
        } else {
            console.error('No triplets data for species profiling.');
        }
    } catch (error) {
        console.error('Error loading species profiling data:', error);
    }
}

function setupSpeciesSearch() {
    const speciesSearchInput = window.AppUtils.getElement('speciesSearchInput');
    const speciesDropdown = window.AppUtils.getElement('speciesDropdown');
    
    if (!speciesSearchInput || !speciesDropdown) {
        console.error('Species search elements not found.');
        return;
    }
    
    speciesSearchInput.addEventListener('input', function() {
        const query = this.value.toLowerCase().trim();
        
        clearTimeout(this.searchTimeout);
        this.searchTimeout = setTimeout(() => {
            if (query === '') {
                populateSpeciesDropdown();
            } else {
                const filteredSpecies = window.AppState.allSpecies
                    .filter(species => species.name.toLowerCase().includes(query))
                    .map(species => species.name)
                    .sort();
                
                speciesDropdown.innerHTML = '<option value="">Select a species...</option>';
                filteredSpecies.forEach(species => {
                    const option = document.createElement('option');
                    option.value = species;
                    option.textContent = species;
                    speciesDropdown.appendChild(option);
                });
            }
        }, 150);
    });
    
    speciesDropdown.addEventListener('change', function() {
        if (this.value) {
            analyzeSpecies(this.value);
        }
    });
}

function populateSpeciesDropdown() {
    const speciesDropdown = window.AppUtils.getElement('speciesDropdown');
    
    if (!speciesDropdown || !window.AppState.allSpecies || window.AppState.allSpecies.length === 0) {
        console.error('Cannot populate species dropdown.');
        return;
    }
    
    speciesDropdown.innerHTML = '<option value="">Select a species...</option>';
    
    const sortedSpecies = window.AppState.allSpecies
        .map(species => species.name)
        .sort();
    
    sortedSpecies.forEach(speciesName => {
        const option = document.createElement('option');
        option.value = speciesName;
        option.textContent = speciesName;
        speciesDropdown.appendChild(option);
    });
}

function populateFocalSpeciesDropdown() {
    const focalSpeciesSelect = document.getElementById('focalSpeciesSelect');
    
    console.log('Attempting to populate focal species dropdown...');
    console.log('focalSpeciesSelect element:', focalSpeciesSelect);
    console.log('window.AppState.allSpecies:', window.AppState.allSpecies);
    
    if (!focalSpeciesSelect) {
        console.error('focalSpeciesSelect element not found.');
        return;
    }
    
    if (!window.AppState.allSpecies || window.AppState.allSpecies.length === 0) {
        console.error('No species data available.');
        setTimeout(() => {
            if (window.AppState.allSpecies && window.AppState.allSpecies.length > 0) {
                console.log('Retrying focal species dropdown population...');
                populateFocalSpeciesDropdown();
            }
        }, 500);
        return;
    }
    
    focalSpeciesSelect.innerHTML = '<option value="">Select a species...</option>';
    
    const sortedSpecies = window.AppState.allSpecies
        .map(species => species.name)
        .sort();
    
    const limitedSpecies = sortedSpecies.slice(0, 100);
    
    limitedSpecies.forEach(speciesName => {
        const option = document.createElement('option');
        option.value = speciesName;
        option.textContent = speciesName;
        focalSpeciesSelect.appendChild(option);
    });
    
    console.log(`Successfully populated focal species dropdown with ${limitedSpecies.length} species.`);
}

async function analyzeSpecies() {
    const speciesDropdown = window.AppUtils.getElement('speciesDropdown');
    const speciesSearchInput = window.AppUtils.getElement('speciesSearchInput');
    
    const selectedSpecies = (speciesDropdown && speciesDropdown.value) || 
                          (speciesSearchInput && speciesSearchInput.value.trim());
    
    if (!selectedSpecies) {
        alert('Please select a species to analyze.');
        return;
    }

    const speciesThreats = window.AppState.allTripletsData.filter(triplet => triplet.subject === selectedSpecies);
    
    if (speciesThreats.length === 0) {
        alert('No threat data found for this species.');
        return;
    }

    await performSemanticAnalysis(selectedSpecies, speciesThreats);
}

async function performSemanticAnalysis(speciesName, threats) {
    try {
        const speciesAnalysisResults = window.AppUtils.getElement('speciesAnalysisResults');
        if (speciesAnalysisResults) {
            speciesAnalysisResults.style.display = 'block';
        }
        
        const selectedSpeciesName = window.AppUtils.getElement('selectedSpeciesName');
        if (selectedSpeciesName) {
            selectedSpeciesName.textContent = `Analysis: ${speciesName}`;
        }

        const response = await fetch('/api/species_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ species_name: speciesName, threats: threats })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const analysisData = await response.json();
        
        if (analysisData.error) {
            throw new Error(analysisData.error);
        }

        updateSummaryCards(analysisData);
        renderThreatClusters(analysisData.semantic_clusters);
        renderImpactAnalysis(analysisData.impact_analysis);
        renderComprehensiveProfile(analysisData.comprehensive_profile);
        renderSpeciesCharts(analysisData);

    } catch (error) {
        console.error('Error in species analysis:', error);
        alert(`Error analyzing species: ${error.message}`);
    }
}

// Make functions available globally for other modules
window.AppFunctions = {
    loadSpeciesProfilingData,
    setupSpeciesSearch,
    populateSpeciesDropdown,
    analyzeSpecies,
    performSemanticAnalysis,
    updateSummaryCards,
    renderThreatClusters,
    renderImpactAnalysis,
    renderComprehensiveProfile,
    renderSpeciesCharts,
    processAndRenderCharts: window.processAndRenderCharts,
    performNetworkAnalysis,
    performIndirectImpactsAnalysis,
    executeKnowledgeGraphQuery,
    toggleChainMode,
    displayTriplets,
    performSearch,
    initializeTabs
};

function updateSummaryCards(data) {
    const elements = ['totalThreatsCount', 'semanticClustersCount', 'impactCategoriesCount'];
    const values = [data.total_threats || 0, data.semantic_clusters?.length || 0, data.impact_categories_count || 0];
    
    elements.forEach((id, index) => {
        const element = window.AppUtils.getElement(id);
        if (element) element.textContent = values[index];
    });
}

function renderThreatClusters(clusters) {
    const container = window.AppUtils.getElement('threatClustersContainer');
    if (!container || !clusters) return;
    
    container.innerHTML = '';
    
    clusters.forEach((cluster, index) => {
        const clusterDiv = document.createElement('div');
        clusterDiv.className = 'threat-cluster';
        clusterDiv.innerHTML = `
            <h6>Cluster ${index + 1}: ${cluster.category || 'Uncategorized'}</h6>
            <div class="cluster-items">${cluster.threats.slice(0, 3).join(', ')}${cluster.threats.length > 3 ? '...' : ''}</div>
            <div class="cluster-count">${cluster.threats.length} threats</div>
        `;
        container.appendChild(clusterDiv);
    });
}

function renderImpactAnalysis(impactData) {
    const container = window.AppUtils.getElement('impactAnalysisContainer');
    if (!container || !impactData) return;
    
    container.innerHTML = '';
    
    const impactArray = Array.isArray(impactData) ? impactData : Object.entries(impactData).map(([key, value]) => ({ category: key, ...value }));
    
    impactArray.forEach((data) => {
        const impactDiv = document.createElement('div');
        impactDiv.className = 'impact-category';
        
        const category = data.category || data.name || 'Unknown';
        const count = data.count || data.frequency || 0;
        const percentage = data.percentage || 0;
        
        let severity = 'Low';
        let severityClass = 'severity-low';
        if (count > 10 || percentage > 20) {
            severity = 'High';
            severityClass = 'severity-high';
        } else if (count > 5 || percentage > 10) {
            severity = 'Medium';
            severityClass = 'severity-medium';
        }
        
        const commonImpacts = data.common_impacts ? 
            (Array.isArray(data.common_impacts) ? data.common_impacts.join(', ') : data.common_impacts) :
            'Various';
        
        impactDiv.innerHTML = `
            <h6>${category}</h6>
            <div class="impact-severity">
                <span class="severity-label">Severity:</span>
                <span class="severity-indicator ${severityClass}">${severity}</span>
            </div>
            <div class="impact-description"><strong>Impacts:</strong> ${commonImpacts}</div>
            <div class="impact-frequency"><strong>Frequency:</strong> ${count} (${percentage ? `${percentage.toFixed(1)}%` : 'N/A'})</div>
        `;
        container.appendChild(impactDiv);
    });
}

function renderComprehensiveProfile(profileData) {
    const container = window.AppUtils.getElement('speciesProfileContainer');
    if (!container || !profileData) return;
    
    let profileHtml = `
        <div class="profile-section">
            <h5>Threat-Impact</h5>
            <p>The idea here is that for each category of threat, we can see what proportion of the threats lead to what impact in the triplets:</p>
            <div class="threat-impact-matrix">
                ${Object.entries(profileData.threat_impact_probabilities || {}).map(([threat, impacts]) => 
                    Object.entries(impacts).map(([impact, probability]) => `
                        <div class="matrix-cell">
                            <div class="threat-name">${threat}</div>
                            <div class="impact-name">${impact}</div>
                            <div class="impact-probability">${(probability * 100).toFixed(1)}%</div>
                        </div>
                    `).join('')
                ).join('')}
            </div>
        </div>
    `;
    
    if (profileData.vulnerability_score) {
        profileHtml += `
            <div class="profile-section">
                <h5>Vulnerability Assessment</h5>
                <p><strong>Overall Score:</strong> ${profileData.vulnerability_score.toFixed(2)}/10</p>
                <p><strong>Primary Risks:</strong> ${profileData.primary_risks?.join(', ') || 'N/A'}</p>
            </div>
        `;
    }
    
    container.innerHTML = profileHtml;
}

function renderSpeciesCharts(data) {
    const speciesThreatCategoriesChartCtx = window.AppUtils.getElement('speciesThreatCategoriesChartCtx');
    if (speciesThreatCategoriesChartCtx && data.threat_categories) {
        if (window.AppState.charts.speciesThreatCategoriesChart) {
            window.AppState.charts.speciesThreatCategoriesChart.destroy();
        }
        
        window.AppState.charts.speciesThreatCategoriesChart = new Chart(speciesThreatCategoriesChartCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(data.threat_categories),
                datasets: [{
                    data: Object.values(data.threat_categories),
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }
}

function displayTriplets(triplets) {
    const container = window.AppUtils.getElement('tripletsContainer');
    if (!container) { 
        console.error("tripletsContainer not found."); 
        return; 
    }
    
    container.innerHTML = ''; 
    if (!triplets || triplets.length === 0) {
        container.innerHTML = '<p>No data to display. Try a different search.</p>';
        return;
    }
    
    triplets.forEach(triplet => {
        const card = document.createElement('div');
        card.classList.add('triplet-card');
        card.innerHTML = `
            <h3>${triplet.subject || 'N/A'}</h3>
            <p><strong>Threat:</strong> ${triplet.predicate || 'N/A'}</p>
            <p><strong>Impact:</strong> ${triplet.object || 'N/A'}</p>
            <p><a href="https://doi.org/${triplet.doi}" target="_blank" class="doi-link">DOI: ${triplet.doi || 'N/A'}</a></p>
            <button class="similar-button" data-id="${triplet.id}">Similar</button>
            <button class="explore-details-button" data-triplet-id="${triplet.id}">Details</button> 
            <button class="start-chain-button" data-species-name="${triplet.subject}" data-triplet-id="${triplet.id}">Start Chain</button> 
        `;
        container.appendChild(card);
    });

    document.querySelectorAll('#tripletsContainer .similar-button').forEach(button => {
        button.addEventListener('click', window.handleFindSimilar || (() => console.log('handleFindSimilar not loaded.')));
    });
    document.querySelectorAll('#tripletsContainer .explore-details-button').forEach(button => {
        button.addEventListener('click', window.handleExploreDetailsClick || (() => console.log('handleExploreDetailsClick not loaded.')));
    });
    document.querySelectorAll('#tripletsContainer .start-chain-button').forEach(button => {
        button.addEventListener('click', (event) => {
            const speciesName = event.target.dataset.speciesName;
            const tripletId = event.target.dataset.tripletId;
            if (typeof window.startNewChainWithSpecies === 'function') {
                window.startNewChainWithSpecies(speciesName, tripletId);
            } else {
                console.log('Chain builder not loaded.');
            }
        });
    });
}

function performSearch() {
    console.log('performSearch needs to be implemented by search.js');
}

function initializeTabs() {
    const tabs = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    let systemicAnalysisHasRun = false;

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const targetTab = tab.getAttribute('data-tab');
            tabContents.forEach(content => {
                if (content.id === `${targetTab}Tab`) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });

            if (targetTab === 'systemic' && !systemicAnalysisHasRun) {
                console.log('Systemic tab opened, running analysis...');
                if (typeof performNetworkAnalysis === 'function') {
                    performNetworkAnalysis();
                    systemicAnalysisHasRun = true;
                } else {
                    console.error('performNetworkAnalysis function not found.');
                }
            }
        });
    });
}

function processAndRenderCharts(data) {
    console.log('processAndRenderCharts needs to be implemented by chart_util.js');
}

function performIndirectImpactsAnalysis() {
    console.log('performIndirectImpactsAnalysis needs to be implemented by analysis.js');
}

function executeKnowledgeGraphQuery(queryType) {
    console.log('executeKnowledgeGraphQuery needs to be implemented by kg_query.js');
}

function toggleChainMode() {
    console.log('toggleChainMode needs to be implemented by chain_build.js');
}
