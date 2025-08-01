let currentTriplet = null;
let reviewProgress = {
    completed: 0,
    totalRating: 0
};
let reviewSession = null;
let currentGroup = null;

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

setTimeout(function() {
    console.log('Delayed initialization check...');
    if (document.getElementById('loadRandomTriplet') && !window.reviewSystemInitialized) {
        console.log('Initializing review system on delay');
        initializeReviewSystem();
        window.reviewSystemInitialized = true;
    }
}, 1000);

window.addEventListener('tripletsLoaded', function() {
    if (document.getElementById('loadRandomTriplet')) {
        console.log('Triplets loaded, initializing review system');
        initializeReviewSystem();
    }
});

function initializeReviewSystem() {
    if (window.reviewSystemInitialized) {
        console.log('Review system already initialized, skipping...');
        return;
    }
    
    console.log('Initializing review system...');
    
    const startBtn = document.getElementById('startReviewSession');
    const endBtn = document.getElementById('endSession');
    const loadBtn = document.getElementById('loadRandomTriplet');
    
    console.log('Review elements found:', {
        startBtn: !!startBtn,
        endBtn: !!endBtn, 
        loadBtn: !!loadBtn
    });
    
    if (!startBtn || !endBtn || !loadBtn) {
        console.warn('Review system elements not found, skipping initialization');
        return;
    }
    
    startBtn.addEventListener('click', startSession);
    endBtn.addEventListener('click', endSession);
    loadBtn.addEventListener('click', loadRandomTriplet);
    
    document.querySelectorAll('.rating-btn').forEach(button => {
        button.addEventListener('click', () => {
            const rating = parseInt(button.dataset.rating, 10);
            selectRating(rating);
        });
    });
    
    document.getElementById('submitReview').addEventListener('click', submitReview);
    document.getElementById('skipTriplet').addEventListener('click', skipTriplet);
    
    loadProgress();
    
    loadSession();
    
    window.reviewSystemInitialized = true;
    console.log('Review system initialization complete');
}

async function loadRandomTriplet() {
    const status = document.getElementById('reviewStatus');
    status.textContent = 'Loading new triplet group...';
    
    try {
        const response = await fetch('/api/random-triplet');
        
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
    document.getElementById('tripletReviewSection').style.display = 'block';
    
    const doiElement = document.getElementById('reviewDOI');
    doiElement.textContent = group.doi;
    doiElement.href = `https://doi.org/${group.doi}`;
    document.getElementById('reviewTitle').textContent = group.title || 'Title not available';
    
    document.getElementById('reviewAbstract').textContent = group.abstract || 'Abstract not available';
    
    const displayContainer = document.getElementById('tripletDisplay');
    displayContainer.innerHTML = '';
    
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

    // No longer needed since rating is removed.
    // const overallRating = document.querySelector('.rating-btn.selected') ? document.querySelector('.rating-btn.selected').dataset.rating : null;
    // if (!overallRating) {
    //     alert("Please select an overall accuracy rating.");
    //     return;
    // }

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
        
        if (response.ok) {
            console.log('Review submitted to backend');
        } else {
            console.log('Backend not available, storing locally');
        }
    } catch (error) {
        console.log('Backend not available, storing locally:', error.message);
    }
    
    storeReviewLocally(reviewData);
    
    updateProgress(reviewData.rating);
    
    document.getElementById('reviewStatus').textContent = `Review submitted! Rating: ${reviewData.rating}/5`;
    
    resetReviewForm();
    
    setTimeout(() => {
        loadRandomTriplet();
    }, 1500);
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
            if (reviewProgress.completed > 0) {
                document.getElementById('averageRating').textContent = 
                    (reviewProgress.totalRating / reviewProgress.completed).toFixed(1);
            }
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
    } else {
        document.getElementById('reviewerSetup').style.display = 'block';
        document.getElementById('activeSession').style.display = 'none';
        document.getElementById('tripletReviewSection').style.display = 'none';
    }
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
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

document.addEventListener('DOMContentLoaded', function() {
    const progressSection = document.getElementById('reviewProgress');
    if (progressSection) {
        const exportBtn = document.createElement('button');
        exportBtn.textContent = 'Export Reviews';
        exportBtn.className = 'export-btn';
        exportBtn.onclick = exportReviews;
        progressSection.appendChild(exportBtn);
    }
}); 