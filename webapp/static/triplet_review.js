let currentTriplet = null;
let reviewProgress = {
    completed: 0,
    totalRating: 0
};
let reviewSession = null;
let currentGroup = null;

function generateSessionId() {
    const timestamp = Date.now().toString(36);
    const randomPart = Math.random().toString(36).substr(2, 9);
    return `review_${timestamp}_${randomPart}`;
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, checking for review elements...');
    const reviewBtn = document.getElementById('loadRandomTriplet');
    const reviewTab = document.getElementById('reviewTab');
    console.log('Review tab found:', !!reviewTab);
    console.log('Load button found:', !!reviewBtn);
    
    if (reviewBtn) {
        console.log('Initializing review system on DOM load');
        initializeReviewSystem();
    } else {
        console.log('Review elements not found on DOM load');
    }
});

function initializeReviewSystem() {
    if (window.reviewSystemInitialized) {
        return;
    }
    
    const startBtn = document.getElementById('startReviewSession');
    const endBtn = document.getElementById('endSession');
    const loadBtn = document.getElementById('loadRandomTriplet');
    const submitBtn = document.getElementById('submitReview');
    const skipBtn = document.getElementById('skipTriplet');
    
    if (!startBtn || !endBtn || !loadBtn || !submitBtn || !skipBtn) {
        return;
    }
    
    startBtn.addEventListener('click', startSession);
    endBtn.addEventListener('click', endSession);
    loadBtn.addEventListener('click', loadRandomTriplet);
    submitBtn.addEventListener('click', submitReview);
    skipBtn.addEventListener('click', skipTriplet);
    
    window.reviewSystemInitialized = true;
}

async function loadRandomTriplet() {
    const status = document.getElementById('reviewStatus');
    status.textContent = 'Loading new triplet group...';
    
    if (!reviewSession || !reviewSession.id) {
        status.textContent = 'Error: No active review session';
        return;
    }
    
    try {
        const response = await fetch(`/api/random-triplet?session_id=${encodeURIComponent(reviewSession.id)}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            currentGroup = data.group;
            displayTripletGroup(data.group);
            status.textContent = `Loaded ${data.group.triplets.length} triplets from ${data.group.doi}`;
        } else {
            throw new Error(data.message || 'Failed to load triplet group');
        }
        
    } catch (error) {
        console.error('Error loading random triplet group:', error);
        status.textContent = `Error: ${error.message}`;
    }
}

function displayTripletGroup(group) {
    if (!group) return;
    currentGroup = group; 
    
    document.getElementById('tripletReviewSection').style.display = 'block';
    
    const doiElement = document.getElementById('reviewDOI');
    doiElement.textContent = group.doi;
    doiElement.href = `https://doi.org/${group.doi}`;
    document.getElementById('reviewTitle').textContent = group.title || 'Title not available';
    
    document.getElementById('reviewAbstract').textContent = group.abstract || 'Abstract not available';
    
    const displayContainer = document.getElementById('tripletDisplayList');
    displayContainer.innerHTML = '';
    
    document.getElementById('tripletsHeader').textContent = `Extracted Triplets`;

    if (group.triplets && group.triplets.length > 0) {
        group.triplets.forEach((triplet, index) => {
            const tripletEl = document.createElement('div');
            tripletEl.className = 'triplet-item';
            tripletEl.dataset.tripletId = triplet.id; 

            tripletEl.innerHTML = `
                <div class="triplet-item-number">${index + 1}</div>
                <div class="triplet-item-content">
                    <div class="triplet-component">
                        <div class="triplet-text"><strong>Subject:</strong> <span>${triplet.subject}</span></div>
                        <div class="triplet-validation-control">
                            <input type="checkbox" id="triplet-${triplet.id}-subject-valid" class="triplet-component-valid-checkbox" data-part="subject" checked>
                            <label for="triplet-${triplet.id}-subject-valid">Valid</label>
                        </div>
                    </div>
                    <div class="triplet-component">
                        <div class="triplet-text"><strong>Predicate:</strong> <span>${triplet.predicate}</span></div>
                        <div class="triplet-validation-control">
                            <input type="checkbox" id="triplet-${triplet.id}-predicate-valid" class="triplet-component-valid-checkbox" data-part="predicate" checked>
                            <label for="triplet-${triplet.id}-predicate-valid">Valid</label>
                        </div>
                    </div>
                    <div class="triplet-component">
                        <div class="triplet-text"><strong>Object:</strong> <span>${triplet.object}</span></div>
                        <div class="triplet-validation-control">
                            <input type="checkbox" id="triplet-${triplet.id}-object-valid" class="triplet-component-valid-checkbox" data-part="object" checked>
                            <label for="triplet-${triplet.id}-object-valid">Valid</label>
                        </div>
                    </div>
                </div>
            `;
            displayContainer.appendChild(tripletEl);
        });
    } else {
        displayContainer.innerHTML = '<div class="triplet-item-content">No triplets found for this abstract.</div>';
    }
    
    resetReviewForm();
    
    document.getElementById('tripletReviewSection').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

function selectRating(rating) {
    document.querySelectorAll('.rating-btn').forEach(btn => {
        btn.classList.remove('selected');
        if (parseInt(btn.dataset.rating, 10) === rating) {
            btn.classList.add('selected');
        }
    });
}

function resetReviewForm() {
    document.querySelectorAll('.rating-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    
    document.querySelectorAll('.checkbox-group input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    
    document.getElementById('reviewComments').value = '';
    
    if (currentTriplet) {
        delete currentTriplet.rating;
    }
}

async function submitReview() {
    if (!currentGroup) {
        alert('No triplet group loaded for review');
        return;
    }

    if (!reviewSession || !reviewSession.name) {
        alert("Please start a review session first.");
        document.getElementById('startReviewBtn').focus();
        return;
    }

    const reviewData = {
        group_doi: currentGroup.doi,
        triplets: currentGroup.triplets.map(t => {
            const tripletItemEl = document.querySelector(`.triplet-item[data-triplet-id="${t.id}"]`);
            
            const isSubjectValid = tripletItemEl ? tripletItemEl.querySelector('[data-part="subject"]').checked : true;
            const isPredicateValid = tripletItemEl ? tripletItemEl.querySelector('[data-part="predicate"]').checked : true;
            const isObjectValid = tripletItemEl ? tripletItemEl.querySelector('[data-part="object"]').checked : true;

            return {
                id: t.id,
                subject: t.subject,
                predicate: t.predicate,
                object: t.object,
                validity: {
                    subject: isSubjectValid,
                    predicate: isPredicateValid,
                    object: isObjectValid
                }
            };
        }),
        comments: document.getElementById('reviewComments').value.trim(),
        reviewer: reviewSession ? {
            name: reviewSession.name,
            session_id: reviewSession.id
        } : {
            name: 'Anonymous',
            expertise: 'Unknown'
        },
        timestamp: new Date().toISOString()
    };
    
    try {
        const response = await fetch('/api/submit-review', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(reviewData)
        });
        
        const result = await response.json();

        if (result.success) {
            alert("Review submitted successfully!");
            document.getElementById('reviewComments').value = '';
            loadRandomTriplet();
            updateReviewProgress();
        } else {
            alert(`Submission failed: ${result.message}`);
        }
    } catch (error) {
        console.error("Error submitting review:", error);
        alert("An error occurred during submission. See console for details.");
    }
}

function skipTriplet() {
    document.getElementById('reviewStatus').textContent = 'Triplet skipped';
    resetReviewForm();
    loadRandomTriplet();
}

function storeReviewLocally(reviewData) {
    try {
        let reviews = JSON.parse(localStorage.getItem('tripletReviews') || '[]');
        reviews.push(reviewData);
        
        if (reviews.length > 100) {
            reviews = reviews.slice(-100);
        }
        
        localStorage.setItem('tripletReviews', JSON.stringify(reviews));
    } catch (error) {
        console.error('Error storing review locally:', error);
    }
}

function updateProgress(rating) {
    reviewProgress.completed++;
    reviewProgress.totalRating += rating;
    
    document.getElementById('reviewsCompleted').textContent = reviewProgress.completed;
    saveProgress();
}

function loadProgress() {
    try {
        const saved = localStorage.getItem('reviewProgress');
        if (saved) {
            reviewProgress = JSON.parse(saved);
            document.getElementById('reviewsCompleted').textContent = reviewProgress.completed;
        }
    } catch (error) {
        console.error('Error loading progress:', error);
    }
}

function saveProgress() {
    try {
        localStorage.setItem('reviewProgress', JSON.stringify(reviewProgress));
    } catch (error) {
        console.error('Error saving progress:', error);
    }
}

// Export reviews function (for data export)
function exportReviews() {
    try {
        const reviews = localStorage.getItem('tripletReviews');
        if (!reviews) {
            alert('No reviews to export');
            return;
        }
        
        const blob = new Blob([reviews], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `triplet_reviews_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error exporting reviews:', error);
        alert('Error exporting reviews');
    }
}

// Session Management Functions
function startSession() {
    const name = document.getElementById('sessionReviewerName').value.trim();
    
    if (!name) {
        alert('Please enter your name or reviewer ID');
        return;
    }
    
    reviewSession = {
        id: generateSessionId(),
        name: name,
        startTime: new Date().toISOString(),
        reviews: 0
    };
    
    saveSession();
    
    updateSessionUI();
    
    document.getElementById('reviewStatus').textContent = `Session started for ${name}. Ready to review triplets!`;
}

function endSession() {
    if (!reviewSession) {
        return;
    }
    
    const reviewCount = reviewSession.reviews || 0;
    const sessionName = reviewSession.name;
    
    reviewSession = null;
    localStorage.removeItem('reviewSession');
    
    document.getElementById('reviewerSetup').style.display = 'block';
    document.getElementById('activeSession').style.display = 'none';
    document.getElementById('tripletReviewSection').style.display = 'none';
    
    document.getElementById('sessionReviewerName').value = '';
    
    document.getElementById('reviewStatus').textContent = 
        `Session ended for ${sessionName}. Completed ${reviewCount} reviews. Thank you!`;
}

function loadSession() {
    try {
        const saved = localStorage.getItem('reviewSession');
        if (saved) {
            reviewSession = JSON.parse(saved);
            updateSessionUI();
        }
    } catch (error) {
        console.error('Error loading session:', error);
        localStorage.removeItem('reviewSession');
    }
}

function saveSession() {
    try {
        localStorage.setItem('reviewSession', JSON.stringify(reviewSession));
    } catch (error) {
        console.error('Error saving session:', error);
    }
}

function updateSessionUI() {
    if (reviewSession) {
        document.getElementById('reviewerSetup').style.display = 'none';
        document.getElementById('activeSession').style.display = 'block';
        
        document.getElementById('activeReviewerName').textContent = reviewSession.name;
        
        document.getElementById('reviewStatus').textContent = 
            `Session active for ${reviewSession.name}. Click "Load Random Triplet" to start reviewing.`;
        loadReviewProgress();
    } else {
        document.getElementById('reviewerSetup').style.display = 'block';
        document.getElementById('activeSession').style.display = 'none';
        document.getElementById('tripletReviewSection').style.display = 'none';
    }
}

async function loadReviewProgress() {
    try {
        const response = await fetch('/api/review-progress');
        if (response.ok) {
            const progress = await response.json();
            if (progress.success) {
                const statusEl = document.getElementById('reviewStatus');
                statusEl.innerHTML = `
                    Session: ${reviewSession.name}<br>
                    Progress: ${progress.reviewed}/${progress.total_abstracts} reviewed (${progress.progress_percentage}%)<br>
                    Available: ${progress.available} abstracts ready for review
                `;
            }
        }
    } catch (error) {
        console.error('Error loading progress:', error);
    }
}



const originalLoadRandomTriplet = loadRandomTriplet;
loadRandomTriplet = async function() {
    if (!reviewSession) {
        alert('Please start a review session first by entering your information above.');
        return;
    }
    
    await originalLoadRandomTriplet();
};

const originalSubmitReview = submitReview;
submitReview = async function() {
    if (!reviewSession) {
        alert('Please start a review session first');
        return;
    }
    
    await originalSubmitReview();
    
    if (reviewSession) {
        reviewSession.reviews = (reviewSession.reviews || 0) + 1;
        saveSession();
    }
};

async function updateReviewProgress() {
    try {
        const response = await fetch('/api/reviews/stats');
        const data = await response.json();
        if (data.success) {
            document.getElementById('reviewsCompleted').textContent = data.reviews_completed;
        } else {
            console.error("Failed to fetch review stats:", data.message);
        }
    } catch (error) {
        console.error("Error fetching review stats:", error);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const progressSection = document.getElementById('reviewProgress');
    if (progressSection) {
        const exportBtn = document.createElement('button');
        exportBtn.textContent = 'Export Reviews';
        exportBtn.className = 'export-btn';
        exportBtn.onclick = exportReviews;
        progressSection.appendChild(exportBtn);
    }

    const clearReviewsBtn = document.getElementById('clearReviewsBtn');
    if (clearReviewsBtn) {
        clearReviewsBtn.addEventListener('click', async () => {
            if (confirm("Are you sure you want to permanently delete all review data? This action cannot be undone.")) {
                try {
                    const response = await fetch('/api/clear-reviews', {
                        method: 'POST',
                    });
                    const result = await response.json();
                    
                    if (result.success) {
                        alert(result.message);
                        updateReviewProgress();
                    } else {
                        alert(`Error: ${result.message}`);
                    }
                } catch (error) {
                    console.error("Failed to clear reviews:", error);
                    alert("An error occurred while trying to clear reviews. See console for details.");
                }
            }
        });
    }

    // Initial fetch of review progress from the server
    updateReviewProgress();
}); 