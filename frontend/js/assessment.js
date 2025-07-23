/**
 * File: frontend/js/assessment.js
 * Assessment functionality for MathemAI
 */

// Assessment state
const assessmentState = {
    currentStep: 1,
    totalSteps: 5,
    startTime: null,
    endTime: null,
    responses: {},
    stepTimings: {}
};

/**
 * Initialize the assessment
 */
function initializeAssessment() {
    assessmentState.startTime = new Date();
    updateProgressBar();
    
    // Set up event listeners for all rating inputs
    const ratingInputs = document.querySelectorAll('.rating input');
    ratingInputs.forEach(input => {
        input.addEventListener('change', function() {
            const name = this.name;
            const value = parseInt(this.value);
            assessmentState.responses[name] = value;
        });
    });
    
    // Set up event listeners for all text inputs
    const textInputs = document.querySelectorAll('input[type="text"], input[type="number"], select');
    textInputs.forEach(input => {
        input.addEventListener('change', function() {
            const name = this.id.replace('student-', '');
            let value = this.value;
            
            // Convert numeric values to numbers
            if (this.type === 'number') {
                value = parseInt(value);
            }
            
            assessmentState.responses[name] = value;
        });
    });
}

/**
 * Update the progress bar based on current step
 */
function updateProgressBar() {
    const progressBar = document.getElementById('assessment-progress');
    if (progressBar) {
        const progressPercentage = (assessmentState.currentStep / assessmentState.totalSteps) * 100;
        progressBar.style.width = `${progressPercentage}%`;
        progressBar.setAttribute('aria-valuenow', progressPercentage);
    }
}

/**
 * Move to the next step in the assessment
 * @param {number} currentStepNumber - The current step number
 */
function nextStep(currentStepNumber) {
    // Save timing information for the current step
    assessmentState.stepTimings[`step${currentStepNumber}`] = {
        endTime: new Date()
    };
    
    // Validate the current step
    if (!validateStep(currentStepNumber)) {
        showNotification('Please complete all fields before continuing.', 'warning');
        return;
    }
    
    // Hide current step
    document.getElementById(`step-${currentStepNumber}`).style.display = 'none';
    
    // Show next step
    const nextStepNumber = currentStepNumber + 1;
    const nextStepElement = document.getElementById(`step-${nextStepNumber}`);
    
    if (nextStepElement) {
        nextStepElement.style.display = 'block';
        assessmentState.currentStep = nextStepNumber;
        
        // Record start time for the next step
        assessmentState.stepTimings[`step${nextStepNumber}`] = {
            startTime: new Date()
        };
        
        // Update progress bar
        updateProgressBar();
        
        // Scroll to top of form
        window.scrollTo({
            top: document.getElementById('assessment-form').offsetTop - 20,
            behavior: 'smooth'
        });
    } else {
        // If no next step, complete the assessment
        completeAssessment();
    }
}

/**
 * Move to the previous step in the assessment
 * @param {number} currentStepNumber - The current step number
 */
function prevStep(currentStepNumber) {
    // Hide current step
    document.getElementById(`step-${currentStepNumber}`).style.display = 'none';
    
    // Show previous step
    const prevStepNumber = currentStepNumber - 1;
    if (prevStepNumber >= 1) {
        document.getElementById(`step-${prevStepNumber}`).style.display = 'block';
        assessmentState.currentStep = prevStepNumber;
        
        // Update progress bar
        updateProgressBar();
    }
}

/**
 * Validate the current step
 * @param {number} stepNumber - The step number to validate
 * @return {boolean} Whether the step is valid
 */
function validateStep(stepNumber) {
    let isValid = true;
    
    // Validation logic for each step
    switch (stepNumber) {
        case 1: // Basic information
            if (!assessmentState.responses.age || !assessmentState.responses.grade) {
                isValid = false;
            }
            break;
            
        case 2: // Number skills
            if (!assessmentState.responses.number_recognition || 
                !assessmentState.responses.number_comparison || 
                !assessmentState.responses.counting_skills) {
                isValid = false;
            }
            break;
            
        case 3: // Calculation skills
            if (!assessmentState.responses.calculation_accuracy || 
                !assessmentState.responses.calculation_fluency || 
                !assessmentState.responses.arithmetic_facts_recall) {
                isValid = false;
            }
            break;
            
        case 4: // Problem solving
            if (!assessmentState.responses.word_problem_solving || 
                !assessmentState.responses.place_value) {
                isValid = false;
            }
            break;
            
        case 5: // Cognitive factors
            if (!assessmentState.responses.math_anxiety_level || 
                !assessmentState.responses.working_memory_score || 
                !assessmentState.responses.visual_spatial_score || 
                !assessmentState.responses.attention_score) {
                isValid = false;
            }
            break;
    }
    
    return isValid;
}

/**
 * Complete the assessment and submit data
 */
async function completeAssessment() {
    assessmentState.endTime = new Date();
    
    // Calculate response time
    const totalTimeMs = assessmentState.endTime - assessmentState.startTime;
    const responseTime = Math.round(totalTimeMs / 1000); // Convert to seconds
    
    // Determine response time category
    let responseTimeCategory;
    if (responseTime < 180) { // Less than 3 minutes
        responseTimeCategory = 'fast';
    } else if (responseTime < 360) { // 3-6 minutes
        responseTimeCategory = 'average';
    } else if (responseTime < 600) { // 6-10 minutes
        responseTimeCategory = 'slow';
    } else {
        responseTimeCategory = 'very_slow';
    }
    
    // Prepare the assessment data
    const assessmentData = {
        ...assessmentState.responses,
        response_time: responseTimeCategory
    };
    
    // Show loading indicator
    document.getElementById('assessment-form').innerHTML = `
        <div class="text-center">
            <h3>Processing Results</h3>
            <p>Please wait while we analyze your assessment data...</p>
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;
    
    try {
        // Submit the assessment for screening
        const results = await api.screenForDyscalculia(assessmentData);
        
        // Save the assessment data
        await api.saveAssessment(assessmentData);
        
        // Store results in local storage for the results page
        localStorage.setItem('mathemat_assessment_results', JSON.stringify(results));
        localStorage.setItem('mathemat_assessment_data', JSON.stringify(assessmentData));
        
        // Redirect to results page
        window.location.href = 'results.html';
    } catch (error) {
        console.error('Error completing assessment:', error);
        
        // Show error message
        document.getElementById('assessment-form').innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Error Processing Assessment</h4>
                <p>There was a problem processing your assessment. Please try again later.</p>
                <hr>
                <p class="mb-0">Error details: ${error.message}</p>
                <button class="btn btn-primary mt-3" onclick="window.location.reload()">Try Again</button>
            </div>
        `;
    }
}

// Initialize the assessment when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const assessmentForm = document.getElementById('assessment-form');
    if (assessmentForm) {
        initializeAssessment();
    }
});