let currentTriplet = null;
let reviewProgress = {
    completed: 0,
    totalRating: 0
};
let reviewSession = null;

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
    
    const ratingButtons = document.querySelectorAll('.rating-btn');
    ratingButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            selectRating(this.dataset.rating);
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
    const button = document.getElementById('loadRandomTriplet');
    const status = document.getElementById('reviewStatus');
    
    button.textContent = '⏳ Loading...';
    button.disabled = true;
    status.textContent = 'Fetching random triplet...';
    
    try {
        const response = await fetch('/api/random-triplet');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            currentTriplet = data.triplet;
            displayTriplet(data.triplet);
            status.textContent = `Loaded triplet from ${data.triplet.doi}`;
        } else {
            throw new Error(data.message || 'Failed to load triplet');
        }
        
    } catch (error) {
        console.error('Error loading random triplet:', error);
        status.textContent = `Error: ${error.message}`;
        
        const localData = window.AppState?.allTripletsData || window.triplets;
        console.log('API failed, checking local data. Available triplets:', localData ? localData.length : 'undefined');
        if (localData && localData.length > 0) {
            const randomIndex = Math.floor(Math.random() * localData.length);
            const triplet = localData[randomIndex];
            console.log('Selected random triplet:', triplet);
            
            currentTriplet = {
                subject: triplet.subject,
                predicate: triplet.predicate,
                object: triplet.object,
                doi: triplet.doi,
                abstract: triplet.abstract || triplet.threat_sentence || "Abstract not available in local data."
            };
            displayTriplet(currentTriplet);
            status.textContent = 'Loaded triplet from local data';
        } else {
            status.textContent = 'Unable to load triplet. Please load data first using the "Load Data from Cloud" button.';
        }
    }
    
    button.textContent = 'Load Random Triplet';
    button.disabled = false;
}

function displayTriplet(triplet) {
    document.getElementById('tripletReviewSection').style.display = 'block';
    
    const doiElement = document.getElementById('reviewDOI');
    doiElement.textContent = triplet.doi;
    doiElement.href = `https://doi.org/${triplet.doi}`;
    
    document.getElementById('reviewTitle').textContent = triplet.title || 'Title not available';
    
    document.getElementById('reviewAbstract').textContent = triplet.abstract || 'Abstract not available';
    
    document.getElementById('reviewSubject').textContent = triplet.subject;
    document.getElementById('reviewPredicate').textContent = triplet.predicate;
    document.getElementById('reviewObject').textContent = triplet.object;
    
    resetReviewForm();
    
    document.getElementById('tripletReviewSection').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

function selectRating(rating) {
    document.querySelectorAll('.rating-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    
    document.querySelector(`[data-rating="${rating}"]`).classList.add('selected');
    
    currentTriplet.rating = parseInt(rating);
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
    if (!currentTriplet) {
        alert('No triplet loaded for review');
        return;
    }
    
    if (!currentTriplet.rating) {
        alert('Please select an overall accuracy rating');
        return;
    }
    
    const reviewData = {
        triplet: {
            subject: currentTriplet.subject,
            predicate: currentTriplet.predicate,
            object: currentTriplet.object,
            doi: currentTriplet.doi
        },
        rating: currentTriplet.rating,
        validation: {
            speciesCorrect: document.getElementById('speciesCorrect').checked,
            threatCorrect: document.getElementById('threatCorrect').checked,
            relationshipCorrect: document.getElementById('relationshipCorrect').checked,
            abstractSupports: document.getElementById('abstractSupports').checked,
            conservationRelevant: document.getElementById('conservationRelevant').checked
        },
        comments: document.getElementById('reviewComments').value.trim(),
        reviewer: reviewSession ? {
            name: reviewSession.name,
            expertise: reviewSession.expertise,
            institution: reviewSession.institution,
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
    document.getElementById('averageRating').textContent = 
        (reviewProgress.totalRating / reviewProgress.completed).toFixed(1);
    
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
    const expertise = document.getElementById('sessionExpertise').value;
    const institution = document.getElementById('sessionInstitution').value.trim();
    
    if (!name) {
        alert('Please enter your name or reviewer ID');
        return;
    }
    
    if (!expertise) {
        alert('Please select your area of expertise');
        return;
    }
    
    reviewSession = {
        id: generateSessionId(),
        name: name,
        expertise: expertise,
        institution: institution,
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
    document.getElementById('sessionExpertise').value = '';
    document.getElementById('sessionInstitution').value = '';
    
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
        document.getElementById('activeExpertise').textContent = reviewSession.expertise;
        
        const institution = reviewSession.institution ? ` (${reviewSession.institution})` : '';
        document.getElementById('reviewStatus').textContent = 
            `Session active for ${reviewSession.name}${institution}. Click "Load Random Triplet" to start reviewing.`;
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